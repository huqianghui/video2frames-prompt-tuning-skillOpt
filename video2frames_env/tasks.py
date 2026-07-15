"""Shared task definitions for the video2frames SkillOpt environment.

A task is one video whose frames were pre-extracted to Azure Blob Storage,
paired with the customer-provided ground-truth solution. Tasks are produced
by `prepare_data.py` and stored as `data/{train,val,test}.jsonl` plus the
SkillOpt split mirror `data/splits/<split>/items.json`.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, TypedDict, cast

from blob_utils import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"


class FrameTask(TypedDict):
    """One task record produced by `prepare_data.py`."""

    id: str
    video: str
    family: str
    frame_blobs: List[str]
    num_frames: int
    seconds_per_frame: int
    solution: Dict[str, Any]


def judge_model() -> str:
    """Azure OpenAI deployment used to grade generated descriptions.

    Read from JUDGE_MODEL at call time; Video2FramesAdapter sets it from the
    YAML `env.judge_model` key (the single source of truth for model choice).
    """
    return os.environ.get("JUDGE_MODEL", "gpt-4.1-mini")


def load_tasks(split: str, limit: Optional[int] = None) -> List[FrameTask]:
    """Load a dataset split written by `prepare_data.py`, optionally truncated to `limit` tasks."""
    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python prepare_data.py` first.")
    tasks: List[FrameTask] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(cast(FrameTask, json.loads(line)))
    if limit is not None:
        tasks = tasks[:limit]
    return tasks
