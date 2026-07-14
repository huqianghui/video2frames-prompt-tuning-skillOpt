#!/usr/bin/env python3
"""Evaluate a single skill (prompt) on a video2frames split without training.

Registers [Video2FramesAdapter][video2frames_env.adapter.Video2FramesAdapter]
in the installed `scripts.eval_only` registry and delegates to its CLI.

Usage:
    python eval.py --config configs/video2frames/default.yaml \
        --skill outputs/<run>/best_skill.md --split valid_unseen
"""

from __future__ import annotations

from train import bootstrap_env


def main() -> None:
    bootstrap_env()

    import scripts.eval_only as skillopt_eval

    from video2frames_env.adapter import Video2FramesAdapter

    skillopt_eval._ENV_REGISTRY["video2frames"] = Video2FramesAdapter
    skillopt_eval.main()


if __name__ == "__main__":
    main()
