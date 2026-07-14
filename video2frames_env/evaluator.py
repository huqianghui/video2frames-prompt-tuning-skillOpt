"""Scoring for video2frames rollouts, ported unchanged from the APO project.

The soft score is the old APO reward so runs stay directly comparable:

    soft = 0.2 * scene_type exact match
         + 0.2 * is_courier_action exact match
         + 0.6 * LLM-judge semantic score over english_detail/brief/title

SkillOpt additionally needs a binary `hard` signal per task; we derive it as
`soft >= hard_threshold` (default 0.8). Invalid JSON output scores 0.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, cast

from openai import AzureOpenAI
from pydantic import BaseModel, Field

from video2frames_env.tasks import judge_model

logger = logging.getLogger(__name__)

SCENE_WEIGHT = 0.2
COURIER_WEIGHT = 0.2
JUDGE_WEIGHT = 0.6
DEFAULT_HARD_THRESHOLD = 0.8


class JudgeResponse(BaseModel):
    reason: str = Field(description="The reason for the score. No more than 100 characters.")
    score: float = Field(description="The score for the semantic match on a 0-1 scale. Be critical.")


def parse_model_output(raw: str) -> Dict[str, Any]:
    """Parse the model's JSON output, tolerating markdown code fences.

    Raises `ValueError` when the output is not a JSON object.
    """
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"Model output is not a JSON object: {type(parsed)}")
    return cast(Dict[str, Any], parsed)


def judge_text_fields(client: AzureOpenAI, generated: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """LLM-judge semantic similarity of english_detail/brief/title on a 0-1 scale."""
    judge_prompt = (
        "You are a strict grader of video event descriptions.\n"
        "Compare the generated fields against the expected ground truth. "
        "Judge whether they describe the same subjects and actions; wording may differ.\n\n"
        f"Generated:\n"
        f"- english_detail: {generated.get('english_detail')}\n"
        f"- brief: {generated.get('brief')}\n"
        f"- title: {generated.get('title')}\n\n"
        f"Expected:\n"
        f"- english_detail: {expected['english_detail']}\n"
        f"- brief: {expected['brief']}\n"
        f"- title: {expected['title']}\n\n"
        "Score the semantic match on a 0-1 scale. Be critical; partial credit is allowed."
    )
    completion = client.chat.completions.parse(
        model=judge_model(),
        messages=[{"role": "user", "content": judge_prompt}],
        response_format=JudgeResponse,
        temperature=0.0,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        logger.warning("Judge returned no parsed response; scoring 0.")
        return 0.0
    logger.info("Judge score %.2f: %s", parsed.score, parsed.reason)
    return max(0.0, min(1.0, parsed.score))


def compute_scores(generated: Dict[str, Any], expected: Dict[str, Any], judge_score: float) -> Dict[str, Any]:
    """Exact-match components plus the weighted soft score (old APO reward)."""
    scene_match = str(generated.get("scene_type", "")).strip().lower() == expected["scene_type"]
    courier_value = generated.get("is_courier_action")
    if isinstance(courier_value, str):
        courier_value = courier_value.strip().lower() == "true"
    courier_match = bool(courier_value) == expected["is_courier_action"]
    soft = SCENE_WEIGHT * scene_match + COURIER_WEIGHT * courier_match + JUDGE_WEIGHT * judge_score
    return {
        "scene_match": int(scene_match),
        "courier_match": int(courier_match),
        "judge_score": judge_score,
        "soft": soft,
    }


def evaluate(
    client: AzureOpenAI,
    raw_output: str,
    solution: Dict[str, Any],
    hard_threshold: float = DEFAULT_HARD_THRESHOLD,
) -> Dict[str, Any]:
    """Score one raw model output against the ground-truth solution.

    Returns a dict with `hard`, `soft`, the score components, and a
    `fail_reason` describing invalid JSON output (empty on success).
    """
    try:
        generated = parse_model_output(raw_output)
    except ValueError as e:
        logger.warning("Invalid JSON output, scoring 0: %s", e)
        return {
            "scene_match": 0,
            "courier_match": 0,
            "judge_score": 0.0,
            "soft": 0.0,
            "hard": 0,
            "fail_reason": f"invalid_json: {e}",
        }
    judge_score = judge_text_fields(client, generated, solution)
    scores = compute_scores(generated, solution, judge_score)
    scores["hard"] = int(scores["soft"] >= hard_threshold)
    scores["fail_reason"] = ""
    return scores
