"""SkillOpt rollout for the video2frames task.

The skill content (the tuned text) is the fixed instruction prompt from the
old APO project. Each rollout appends `<frame n | Xs>` placeholder labels
interleaved with the frame images (SAS URLs or base64 data URIs) as one
multimodal user message, calls the target deployment through
[chat_target_messages][skillopt.model.azure_openai.chat_target_messages], and
scores the JSON output with [evaluate][video2frames_env.evaluator.evaluate].

`run_batch` writes `results.jsonl` incrementally under `out_root` so an
interrupted batch resumes without re-running finished tasks. Tasks known to be
blocked by the Azure content filter (probe cache) are short-circuited to score
0 with `fail_reason="content_filter"` instead of burning retries.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, cast

from openai import AzureOpenAI

from blob_utils import BlobConfig, blob_config_from_env, blob_sas_url, load_env
from video2frames_env.evaluator import DEFAULT_HARD_THRESHOLD, evaluate
from video2frames_env.tasks import DATA_DIR, FrameTask

logger = logging.getLogger(__name__)

CONTENT_FILTER_CACHE_PATH = DATA_DIR / "content_filter_cache.json"

_CONTENT_FILTER_MARKERS = ("content_policy_violation", "content_filter", "content management policy")


def frame_placeholder(index: int, seconds_per_frame: int) -> str:
    """Placeholder for frame `index` (1-based): `<frame n | Xs>` with X = (n-1)*step."""
    return f"<frame {index} | {(index - 1) * seconds_per_frame}s>"


def download_as_data_uri(url: str) -> str:
    """Download an image and encode it as a base64 data URI."""
    import httpx

    response = httpx.get(url, timeout=60.0)
    response.raise_for_status()
    encoded = base64.b64encode(response.content).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def build_messages(
    skill_content: str,
    task: FrameTask,
    config: BlobConfig,
    use_base64: bool = False,
) -> List[Dict[str, Any]]:
    """One multimodal user message: skill text, then frame labels interleaved with images.

    Identical content layout to the old APO agent so target behavior is comparable.
    """
    parts: List[Dict[str, Any]] = [
        {"type": "text", "text": f"{skill_content}\n\n### FRAMES"},
    ]
    for index, blob_path in enumerate(task["frame_blobs"], start=1):
        url = blob_sas_url(config, blob_path)
        if use_base64:
            url = download_as_data_uri(url)
        parts.append({"type": "text", "text": frame_placeholder(index, task["seconds_per_frame"])})
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "user", "content": parts}]


def load_blocked_ids(path: Any = None) -> Set[str]:
    """Task ids marked blocked in the content-filter probe cache (empty when absent)."""
    cache_path = CONTENT_FILTER_CACHE_PATH if path is None else path
    if not os.path.exists(str(cache_path)):
        return set()
    cache = cast(Dict[str, bool], json.loads(open(cache_path, encoding="utf-8").read()))
    return {task_id for task_id, blocked in cache.items() if blocked}


def is_content_filter_error(error: BaseException) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in _CONTENT_FILTER_MARKERS)


def _base_result(task: FrameTask) -> Dict[str, Any]:
    return {
        "id": str(task["id"]),
        "task_type": task["family"],
        "task_description": (
            f"Describe the video from {task['num_frames']} frames "
            f"({task['seconds_per_frame']}s apart) as structured JSON."
        ),
        "hard": 0,
        "soft": 0.0,
        "scene_match": 0,
        "courier_match": 0,
        "judge_score": 0.0,
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
        "num_frames": task["num_frames"],
        "video": task["video"],
    }


def process_one(
    task: FrameTask,
    out_root: str,
    skill_content: str,
    *,
    judge_client: Optional[AzureOpenAI] = None,
    config: Optional[BlobConfig] = None,
    blocked_ids: Optional[Set[str]] = None,
    exec_timeout: int = 180,
    max_completion_tokens: int = 2048,
    hard_threshold: float = DEFAULT_HARD_THRESHOLD,
    use_base64: bool = False,
) -> Dict[str, Any]:
    """Run and score one task; never raises (errors land in `fail_reason`)."""
    result = _base_result(task)
    task_id = result["id"]

    if blocked_ids and task_id in blocked_ids:
        result["fail_reason"] = "content_filter"
        result["agent_ok"] = True
        logger.warning("Task %s: known content-filter block, scoring 0 without a request.", task_id)
        return result

    try:
        from skillopt.model import chat_target_messages

        if config is None:
            config = blob_config_from_env()
        if judge_client is None:
            judge_client = AzureOpenAI()

        messages = build_messages(skill_content, task, config, use_base64=use_base64)
        logger.info("Task %s (%s): sending %d frames to the target model", task_id, task["family"], task["num_frames"])
        response_text, _usage = chat_target_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=3,
            stage="rollout",
            timeout=exec_timeout,
        )
        result["response"] = str(response_text or "")
        result["agent_ok"] = True

        scores = evaluate(judge_client, result["response"], task["solution"], hard_threshold=hard_threshold)
        result.update(scores)

        pred_dir = os.path.join(out_root, "predictions", task_id)
        os.makedirs(pred_dir, exist_ok=True)
        with open(os.path.join(pred_dir, "rollout.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id": task_id,
                    "skill_chars": len(skill_content),
                    "response": result["response"],
                    "solution": task["solution"],
                    "scores": scores,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except RuntimeError as e:
        if is_content_filter_error(e):
            result["fail_reason"] = "content_filter"
            result["agent_ok"] = True
            logger.warning("Task %s: rejected by the content filter, scoring 0.", task_id)
        else:
            result["fail_reason"] = f"error: {e}"
            logger.error("Task %s: rollout failed: %s", task_id, e)
    except Exception as e:  # noqa: BLE001 — one bad task must not kill the batch
        result["fail_reason"] = f"error: {type(e).__name__}: {e}"
        logger.error("Task %s: unexpected rollout error: %s", task_id, e)
    return result


def run_batch(
    tasks: List[FrameTask],
    out_root: str,
    skill_content: str,
    *,
    workers: int = 4,
    exec_timeout: int = 180,
    max_completion_tokens: int = 2048,
    hard_threshold: float = DEFAULT_HARD_THRESHOLD,
    use_base64: bool = False,
) -> List[Dict[str, Any]]:
    """Roll out a batch of tasks in parallel with `results.jsonl` resume support."""
    load_env()
    os.makedirs(out_root, exist_ok=True)
    results_path = os.path.join(out_root, "results.jsonl")

    existing: List[Dict[str, Any]] = []
    done_ids: Set[str] = set()
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = cast(Dict[str, Any], json.loads(line))
                except json.JSONDecodeError:
                    continue
                existing.append(row)
                done_ids.add(str(row["id"]))

    pending = [task for task in tasks if str(task["id"]) not in done_ids]
    if existing:
        logger.info("Resuming rollout batch: %d/%d already done", len(existing), len(existing) + len(pending))
    if not pending:
        return existing

    config = blob_config_from_env()
    judge_client = AzureOpenAI()
    blocked_ids = load_blocked_ids()

    results = list(existing)
    completed = len(existing)
    total = len(existing) + len(pending)
    with open(results_path, "a", encoding="utf-8") as outf:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_one,
                    task,
                    out_root,
                    skill_content,
                    judge_client=judge_client,
                    config=config,
                    blocked_ids=blocked_ids,
                    exec_timeout=exec_timeout,
                    max_completion_tokens=max_completion_tokens,
                    hard_threshold=hard_threshold,
                    use_base64=use_base64,
                ): task
                for task in pending
            }
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                completed += 1
                print(
                    f"    [rollout] {completed}/{total} id={row['id']} "
                    f"hard={row['hard']} soft={row['soft']:.3f}"
                    + (f" ({row['fail_reason']})" if row["fail_reason"] else ""),
                    flush=True,
                )
                outf.write(json.dumps(row, ensure_ascii=False) + "\n")
                outf.flush()
    return results
