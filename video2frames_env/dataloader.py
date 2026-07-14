"""SkillOpt dataloader for the video2frames task splits.

Consumes the `data/splits/{train,val,test}/items.json` mirror written by
`prepare_data.py`. The parent [SplitDataLoader][skillopt.datasets.base.SplitDataLoader]
already reads `items.json` JSON arrays; this subclass only pins the defaults
(`split_mode="split_dir"` on `data/splits`) and validates that every item is a
well-formed task record.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from skillopt.datasets.base import SplitDataLoader

from video2frames_env.tasks import SPLITS_DIR

logger = logging.getLogger(__name__)

REQUIRED_KEYS = ("id", "video", "family", "frame_blobs", "num_frames", "seconds_per_frame", "solution")


class FrameDataLoader(SplitDataLoader):
    """Split-directory dataloader over the pre-extracted video frame tasks."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("split_mode", "split_dir")
        kwargs.setdefault("split_dir", str(SPLITS_DIR))
        super().__init__(**kwargs)

    def load_split_items(self, split_path: str) -> List[Dict[str, Any]]:
        items = super().load_split_items(split_path)
        for item in items:
            missing = [key for key in REQUIRED_KEYS if key not in item]
            if missing:
                raise ValueError(
                    f"Task {item.get('id', '<no id>')!r} in {split_path} is missing keys {missing}. "
                    "Re-run `python prepare_data.py --mirror-only`."
                )
        logger.debug("Loaded %d tasks from %s", len(items), split_path)
        return items
