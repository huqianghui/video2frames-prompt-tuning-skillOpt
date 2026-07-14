#!/usr/bin/env python3
"""Train the video2frames skill (prompt) with SkillOpt's ReflACT trainer.

Registers [Video2FramesAdapter][video2frames_env.adapter.Video2FramesAdapter]
under the env name `video2frames` in the installed `scripts.train` registry
and delegates to its CLI.

Usage:
    python train.py --config configs/video2frames/default.yaml \
        [--cfg-options train.num_epochs=1 train.batch_size=2 env.limit=4 env.workers=1]
"""

from __future__ import annotations

import os

from blob_utils import load_env


def bootstrap_env() -> None:
    """Load .env and set skillopt model env vars.

    Must run before any `skillopt`/`scripts` import: skillopt.model reads
    AZURE_OPENAI_* / *_DEPLOYMENT / AUTH_MODE at module import time.
    """
    load_env()
    os.environ.setdefault("AZURE_OPENAI_AUTH_MODE", "api_key")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    os.environ.setdefault("TARGET_DEPLOYMENT", deployment)
    os.environ.setdefault("OPTIMIZER_DEPLOYMENT", deployment)

    from install_prompts import ensure_prompts

    ensure_prompts()


def main() -> None:
    bootstrap_env()

    import scripts.train as skillopt_train

    from video2frames_env.adapter import Video2FramesAdapter

    skillopt_train._ENV_REGISTRY["video2frames"] = Video2FramesAdapter
    skillopt_train.main()


if __name__ == "__main__":
    main()
