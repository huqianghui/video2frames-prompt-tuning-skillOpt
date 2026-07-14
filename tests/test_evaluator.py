"""Tests for video2frames_env.evaluator (offline, judge mocked)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, cast

import pytest
from openai import AzureOpenAI

from video2frames_env.evaluator import (
    COURIER_WEIGHT,
    JUDGE_WEIGHT,
    SCENE_WEIGHT,
    JudgeResponse,
    compute_scores,
    evaluate,
    parse_model_output,
)

SOLUTION: Dict[str, Any] = {
    "english_detail": "A courier leaves a parcel at the door.",
    "brief": "Courier drops parcel.",
    "title": "Parcel delivery",
    "scene_type": "outdoor",
    "is_courier_action": True,
}


class FakeJudgeClient:
    """Stands in for AzureOpenAI: returns a fixed JudgeResponse score."""

    def __init__(self, score: float) -> None:
        parsed = JudgeResponse(reason="fixed", score=score)
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=lambda **_: completion))


def fake_judge(score: float) -> AzureOpenAI:
    return cast(AzureOpenAI, FakeJudgeClient(score))


def generated(scene: str = "outdoor", courier: Any = True) -> Dict[str, Any]:
    return {
        "english_detail": "A courier leaves a parcel.",
        "brief": "Courier drops parcel.",
        "title": "Parcel delivery",
        "scene_type": scene,
        "is_courier_action": courier,
    }


def test_parse_model_output_plain_json() -> None:
    assert parse_model_output('{"a": 1}') == {"a": 1}


def test_parse_model_output_fenced_json() -> None:
    raw = "Here you go:\n```json\n{\"scene_type\": \"indoor\"}\n```"
    assert parse_model_output(raw) == {"scene_type": "indoor"}


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", ""])
def test_parse_model_output_rejects_non_objects(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_model_output(raw)


def test_compute_scores_all_match() -> None:
    scores = compute_scores(generated(), SOLUTION, judge_score=1.0)
    assert scores["scene_match"] == 1
    assert scores["courier_match"] == 1
    assert scores["soft"] == pytest.approx(SCENE_WEIGHT + COURIER_WEIGHT + JUDGE_WEIGHT)


def test_compute_scores_mismatches() -> None:
    scores = compute_scores(generated(scene="indoor", courier="false"), SOLUTION, judge_score=0.5)
    assert scores["scene_match"] == 0
    assert scores["courier_match"] == 0
    assert scores["soft"] == pytest.approx(JUDGE_WEIGHT * 0.5)


def test_compute_scores_courier_string_true() -> None:
    scores = compute_scores(generated(courier="True"), SOLUTION, judge_score=0.0)
    assert scores["courier_match"] == 1


def test_evaluate_success_and_hard_threshold() -> None:
    raw = json.dumps(generated())
    result = evaluate(fake_judge(1.0), raw, SOLUTION)
    assert result["hard"] == 1
    assert result["soft"] == pytest.approx(1.0)
    assert result["fail_reason"] == ""

    result = evaluate(fake_judge(0.5), raw, SOLUTION, hard_threshold=0.8)
    assert result["soft"] == pytest.approx(0.7)
    assert result["hard"] == 0


def test_evaluate_invalid_json_scores_zero() -> None:
    result = evaluate(fake_judge(1.0), "not json at all", SOLUTION)
    assert result["hard"] == 0
    assert result["soft"] == 0.0
    assert result["fail_reason"].startswith("invalid_json")
