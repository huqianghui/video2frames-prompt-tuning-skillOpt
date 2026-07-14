"""Tests for video2frames_env.dataloader (offline, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from video2frames_env.dataloader import FrameDataLoader


def make_task(task_id: str, family: str = "Charades") -> Dict[str, Any]:
    return {
        "id": task_id,
        "video": f"/workspace/videos/{family}/{task_id}.mp4",
        "family": family,
        "frame_blobs": [f"training/frame/{family}/{task_id}.mp4_frame/0.jpg"],
        "num_frames": 1,
        "seconds_per_frame": 4,
        "solution": {"english_detail": "d", "brief": "b", "title": "t", "scene_type": "indoor", "is_courier_action": "no"},
    }


def write_splits(root: Path, counts: Dict[str, int]) -> Path:
    split_dir = root / "splits"
    for name, count in counts.items():
        (split_dir / name).mkdir(parents=True)
        items = [make_task(f"{name}-{i:03d}") for i in range(count)]
        (split_dir / name / "items.json").write_text(json.dumps(items), encoding="utf-8")
    return split_dir


@pytest.fixture()
def loader(tmp_path: Path) -> FrameDataLoader:
    split_dir = write_splits(tmp_path, {"train": 8, "val": 5, "test": 6})
    loader = FrameDataLoader(split_dir=str(split_dir))
    loader.setup({})
    return loader


def test_loads_all_splits(loader: FrameDataLoader) -> None:
    assert len(loader.train_items) == 8
    assert len(loader.val_items) == 5
    assert len(loader.test_items) == 6
    assert loader.get_train_size() == 8


def test_split_aliases_map_to_val_and_test(loader: FrameDataLoader) -> None:
    assert loader.get_split_items("valid_seen") == loader.val_items
    assert loader.get_split_items("selection") == loader.val_items
    assert loader.get_split_items("valid_unseen") == loader.test_items
    assert loader.get_split_items("test") == loader.test_items


def test_defaults_use_split_dir_mode() -> None:
    loader = FrameDataLoader()
    assert loader.split_mode == "split_dir"
    assert loader.split_dir.endswith("data/splits")


def test_missing_keys_raise(tmp_path: Path) -> None:
    split_dir = write_splits(tmp_path, {"train": 1, "val": 1, "test": 1})
    broken = [{"id": "x"}]
    (split_dir / "train" / "items.json").write_text(json.dumps(broken), encoding="utf-8")
    loader = FrameDataLoader(split_dir=str(split_dir))
    with pytest.raises(ValueError, match="missing keys"):
        loader.setup({})


def test_train_batches_cover_epoch(loader: FrameDataLoader) -> None:
    batches = loader.plan_train_epoch(epoch=0, steps_per_epoch=2, accumulation=1, batch_size=4, seed=42)
    assert len(batches) == 2
    ids: List[str] = [item["id"] for batch in batches for item in batch.payload]
    assert sorted(ids) == sorted(t["id"] for t in loader.train_items)


def test_eval_batch_respects_env_num(loader: FrameDataLoader) -> None:
    batch = loader.build_eval_batch(env_num=3, split="test", seed=1)
    assert batch.batch_size == 3
    assert [item["id"] for item in batch.payload] == [t["id"] for t in loader.test_items[:3]]


def test_real_splits_when_present() -> None:
    """Sanity-check the actual prepared dataset when it exists (40/24/30)."""
    from video2frames_env.tasks import SPLITS_DIR

    if not (SPLITS_DIR / "train" / "items.json").exists():
        pytest.skip("data/splits not prepared")
    loader = FrameDataLoader()
    loader.setup({})
    assert len(loader.train_items) == 40
    assert len(loader.val_items) == 24
    assert len(loader.test_items) == 30
