"""Tests for video2frames_env.adapter (offline, rollout mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

import video2frames_env.adapter as adapter_module
from video2frames_env.adapter import Video2FramesAdapter

from tests.test_dataloader import write_splits


@pytest.fixture()
def adapter(tmp_path: Path) -> Video2FramesAdapter:
    split_dir = write_splits(tmp_path, {"train": 8, "val": 5, "test": 6})
    adapter = Video2FramesAdapter(split_dir=str(split_dir), workers=2, max_completion_tokens=1024)
    adapter.setup({})
    return adapter


def test_build_train_env_returns_items(adapter: Video2FramesAdapter) -> None:
    items = adapter.build_train_env(batch_size=4, seed=7)
    assert len(items) == 4
    assert all("frame_blobs" in item for item in items)


def test_build_eval_env_uses_alias_split(adapter: Video2FramesAdapter) -> None:
    items = adapter.build_eval_env(env_num=3, split="valid_unseen", seed=7)
    assert [item["id"] for item in items] == [t["id"] for t in adapter.dataloader.test_items[:3]]


def test_rollout_forwards_adapter_settings(
    adapter: Video2FramesAdapter, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: Dict[str, Any] = {}

    def fake_run_batch(tasks: List[Dict[str, Any]], out_root: str, skill_content: str, **kwargs: Any) -> list:
        captured.update(kwargs, n_tasks=len(tasks), out_root=out_root, skill=skill_content)
        return [{"id": str(t["id"]), "hard": 1, "soft": 1.0} for t in tasks]

    monkeypatch.setattr(adapter_module, "run_batch", fake_run_batch)
    items = adapter.build_train_env(batch_size=2, seed=1)
    results = adapter.rollout(items, "SKILL", str(tmp_path / "out"))
    assert len(results) == 2
    assert captured["n_tasks"] == 2
    assert captured["skill"] == "SKILL"
    assert captured["workers"] == 2
    assert captured["max_completion_tokens"] == 1024


def test_reference_text_exposes_solution(adapter: Video2FramesAdapter) -> None:
    item = adapter.dataloader.train_items[0]
    text = adapter.build_reference_text(item)
    assert text.startswith("Expected ground-truth output:")
    assert json.loads(text.split(":", 1)[1]) == item["solution"]
    assert adapter.build_reference_text({"id": "x"}) == ""


def test_task_types_are_families(adapter: Video2FramesAdapter) -> None:
    assert adapter.get_task_types() == ["Charades"]


def test_judge_model_kwarg_sets_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    split_dir = write_splits(tmp_path, {"train": 2, "val": 1, "test": 1})
    monkeypatch.setenv("JUDGE_MODEL", "stale-from-env")
    Video2FramesAdapter(split_dir=str(split_dir), judge_model="gpt-5.4")
    assert os.environ["JUDGE_MODEL"] == "gpt-5.4"


def test_judge_model_default_leaves_env_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    split_dir = write_splits(tmp_path, {"train": 2, "val": 1, "test": 1})
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    Video2FramesAdapter(split_dir=str(split_dir))
    assert "JUDGE_MODEL" not in os.environ
