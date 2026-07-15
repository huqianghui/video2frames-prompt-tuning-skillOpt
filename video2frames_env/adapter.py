"""SkillOpt EnvAdapter wiring the video2frames dataloader and rollout together.

Registered under the env name `video2frames` by `train.py`/`eval.py`. The
adapter forwards batches from [FrameDataLoader][video2frames_env.dataloader.FrameDataLoader]
to [run_batch][video2frames_env.rollout.run_batch] and exposes the ground-truth
solution as hidden reference text so reflection can compare the model output
against the expected description without leaking it into the rollout prompt.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter

from video2frames_env.dataloader import FrameDataLoader
from video2frames_env.evaluator import DEFAULT_HARD_THRESHOLD
from video2frames_env.rollout import run_batch


class Video2FramesAdapter(EnvAdapter):
    """Adapter for the video2frames frame-description task."""

    def __init__(
        self,
        split_dir: str = "",
        split_mode: str = "split_dir",
        data_path: str = "",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        seed: int = 42,
        limit: int = 0,
        workers: int = 4,
        exec_timeout: int = 180,
        max_completion_tokens: int = 2048,
        hard_threshold: float = DEFAULT_HARD_THRESHOLD,
        use_base64: bool = False,
        judge_model: str = "",
        analyst_workers: int = 4,
        failure_only: bool = False,
        minibatch_size: int = 4,
        edit_budget: int = 4,
    ) -> None:
        self.workers = workers
        self.exec_timeout = exec_timeout
        self.max_completion_tokens = int(max_completion_tokens)
        self.hard_threshold = float(hard_threshold)
        self.use_base64 = bool(use_base64)
        if judge_model:
            # Same pattern skillopt uses for target/optimizer deployments:
            # evaluator.judge_model() reads JUDGE_MODEL at call time.
            os.environ["JUDGE_MODEL"] = judge_model
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.dataloader = FrameDataLoader(
            split_dir=split_dir,
            split_mode=split_mode,
            data_path=data_path,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )

    def setup(self, cfg: Dict[str, Any]) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self) -> FrameDataLoader:
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs: Any) -> List[Dict[str, Any]]:
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs: Any) -> List[Dict[str, Any]]:
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs: Any) -> List[Dict[str, Any]]:
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def rollout(self, env_manager: Any, skill_content: str, out_dir: str, **kwargs: Any) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = env_manager
        return run_batch(
            tasks,  # type: ignore[arg-type]
            out_dir,
            skill_content,
            workers=self.workers,
            exec_timeout=self.exec_timeout,
            max_completion_tokens=self.max_completion_tokens,
            hard_threshold=self.hard_threshold,
            use_base64=self.use_base64,
        )

    def build_reference_text(self, item: Dict[str, Any]) -> str:
        solution = item.get("solution")
        if not isinstance(solution, dict):
            return ""
        return "Expected ground-truth output:\n" + json.dumps(solution, ensure_ascii=False, indent=2)

    def get_task_types(self) -> List[str]:
        seen: List[str] = []
        for item in self.dataloader.train_items + self.dataloader.val_items + self.dataloader.test_items:
            family = str(item.get("family") or "video2frames")
            if family not in seen:
                seen.append(family)
        return seen or ["video2frames"]
