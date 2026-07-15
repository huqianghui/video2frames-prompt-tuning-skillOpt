"""Tests for video2frames_env.rollout (offline: target and judge mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, cast

import pytest

import video2frames_env.rollout as rollout_module
from blob_utils import BlobConfig
from video2frames_env.evaluator import JudgeResponse
from video2frames_env.rollout import (
    build_messages,
    frame_placeholder,
    is_content_filter_error,
    load_blocked_ids,
    process_one,
    run_batch,
)
from video2frames_env.tasks import FrameTask

CONFIG = BlobConfig(
    blob_endpoint="https://example.blob.core.windows.net/",
    sas_token="sig=abc",
    container_name="process-videos",
    frames_folder="training/frame",
)

GOOD_OUTPUT = json.dumps(
    {
        "english_detail": "A courier leaves a parcel at the door.",
        "brief": "Courier drops parcel.",
        "title": "Parcel delivery",
        "scene_type": "outdoor",
        "is_courier_action": True,
    }
)


def make_task(task_id: str = "t-001", num_frames: int = 2) -> FrameTask:
    return cast(
        FrameTask,
        {
            "id": task_id,
            "video": f"/workspace/videos/Charades/{task_id}.mp4",
            "family": "Charades",
            "frame_blobs": [f"training/frame/Charades/{task_id}.mp4_frame/{i}.jpg" for i in range(num_frames)],
            "num_frames": num_frames,
            "seconds_per_frame": 4,
            "solution": {
                "english_detail": "A courier leaves a parcel at the door.",
                "brief": "Courier drops parcel.",
                "title": "Parcel delivery",
                "scene_type": "outdoor",
                "is_courier_action": True,
            },
        },
    )


def fake_judge(score: float = 1.0) -> Any:
    parsed = JudgeResponse(reason="fixed", score=score)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **_: completion)))


def patch_target(monkeypatch: pytest.MonkeyPatch, response: str) -> Dict[str, Any]:
    """Patch skillopt.model.chat_target_messages and capture the call kwargs."""
    import skillopt.model as skillopt_model

    captured: Dict[str, Any] = {}

    def fake_chat(**kwargs: Any) -> tuple:
        captured.update(kwargs)
        return response, {"total_tokens": 1}

    monkeypatch.setattr(skillopt_model, "chat_target_messages", fake_chat)
    return captured


def test_frame_placeholder_timing() -> None:
    assert frame_placeholder(1, 4) == "<frame 1 | 0s>"
    assert frame_placeholder(3, 4) == "<frame 3 | 8s>"


def test_build_messages_layout() -> None:
    task = make_task(num_frames=2)
    messages = build_messages("SKILL TEXT", task, CONFIG)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    parts = messages[0]["content"]
    assert parts[0] == {"type": "text", "text": "SKILL TEXT\n\n### FRAMES"}
    assert parts[1]["text"] == "<frame 1 | 0s>"
    assert parts[2]["type"] == "image_url"
    assert parts[2]["image_url"]["url"].startswith("https://example.blob.core.windows.net/process-videos/")
    assert parts[2]["image_url"]["url"].endswith("?sig=abc")
    assert parts[3]["text"] == "<frame 2 | 4s>"
    assert len(parts) == 1 + 2 * task["num_frames"]


def test_load_blocked_ids(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"a": True, "b": False, "c": True}), encoding="utf-8")
    assert load_blocked_ids(cache) == {"a", "c"}
    assert load_blocked_ids(tmp_path / "missing.json") == set()


def test_is_content_filter_error() -> None:
    assert is_content_filter_error(RuntimeError("LLM message call failed: content_policy_violation"))
    assert is_content_filter_error(RuntimeError("filtered due to content management policy"))
    assert not is_content_filter_error(RuntimeError("connection reset"))


def test_process_one_blocked_short_circuit(tmp_path: Path) -> None:
    result = process_one(
        make_task("blocked-1"),
        str(tmp_path),
        "SKILL",
        judge_client=fake_judge(),
        config=CONFIG,
        blocked_ids={"blocked-1"},
    )
    assert result["hard"] == 0
    assert result["soft"] == 0.0
    assert result["fail_reason"] == "content_filter"
    assert result["agent_ok"] is True
    conversation = json.loads((tmp_path / "predictions" / "blocked-1" / "conversation.json").read_text("utf-8"))
    assert "content_filter" in conversation[-1]["content"]


def test_process_one_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = patch_target(monkeypatch, GOOD_OUTPUT)
    result = process_one(
        make_task("ok-1"),
        str(tmp_path),
        "SKILL",
        judge_client=fake_judge(1.0),
        config=CONFIG,
        blocked_ids=set(),
    )
    assert result["agent_ok"] is True
    assert result["soft"] == pytest.approx(1.0)
    assert result["hard"] == 1
    assert result["scene_match"] == 1 and result["courier_match"] == 1
    assert captured["stage"] == "rollout"
    assert (tmp_path / "predictions" / "ok-1" / "rollout.json").exists()
    assert result["reference_text"].startswith("Expected ground-truth output:")
    conversation = json.loads((tmp_path / "predictions" / "ok-1" / "conversation.json").read_text("utf-8"))
    assert [m["role"] for m in conversation] == ["user", "assistant", "system"]
    assert conversation[1]["content"] == GOOD_OUTPUT


def test_process_one_content_filter_runtime_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import skillopt.model as skillopt_model

    def raise_filter(**_: Any) -> tuple:
        raise RuntimeError("LLM message call failed after 3 retries: content_policy_violation")

    monkeypatch.setattr(skillopt_model, "chat_target_messages", raise_filter)
    result = process_one(
        make_task("cf-1"), str(tmp_path), "SKILL", judge_client=fake_judge(), config=CONFIG, blocked_ids=set()
    )
    assert result["soft"] == 0.0 and result["hard"] == 0
    assert result["fail_reason"] == "content_filter"


def test_process_one_invalid_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patch_target(monkeypatch, "sorry, no JSON here")
    result = process_one(
        make_task("bad-1"), str(tmp_path), "SKILL", judge_client=fake_judge(), config=CONFIG, blocked_ids=set()
    )
    assert result["soft"] == 0.0 and result["hard"] == 0
    assert result["fail_reason"].startswith("invalid_json")
    conversation = json.loads((tmp_path / "predictions" / "bad-1" / "conversation.json").read_text("utf-8"))
    assert conversation[1]["content"] == "sorry, no JSON here"
    assert "invalid_json" in conversation[-1]["content"]


def test_run_batch_writes_and_resumes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patch_target(monkeypatch, GOOD_OUTPUT)
    monkeypatch.setattr(rollout_module, "load_env", lambda: None)
    monkeypatch.setattr(rollout_module, "blob_config_from_env", lambda: CONFIG)
    monkeypatch.setattr(rollout_module, "AzureOpenAI", lambda: fake_judge(1.0))
    monkeypatch.setattr(rollout_module, "load_blocked_ids", lambda: set())

    tasks = [make_task(f"t-{i}") for i in range(3)]
    results = run_batch(tasks, str(tmp_path), "SKILL", workers=2)
    assert len(results) == 3
    assert all(row["hard"] == 1 for row in results)

    results_path = tmp_path / "results.jsonl"
    assert len(results_path.read_text(encoding="utf-8").strip().splitlines()) == 3

    # Second run resumes: nothing new is executed, existing rows are returned.
    def explode(**_: Any) -> tuple:
        raise AssertionError("target must not be called on resume")

    import skillopt.model as skillopt_model

    monkeypatch.setattr(skillopt_model, "chat_target_messages", explode)
    resumed = run_batch(tasks, str(tmp_path), "SKILL", workers=2)
    assert {row["id"] for row in resumed} == {"t-0", "t-1", "t-2"}
