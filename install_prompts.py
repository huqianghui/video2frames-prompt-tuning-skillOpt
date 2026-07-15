#!/usr/bin/env python3
"""Install SkillOpt reflection prompts into the installed `skillopt` package.

The `skillopt` 0.2.0 wheel omits the `skillopt/prompts/*.md` package data
(no `[tool.setuptools.package-data]` upstream), so `load_prompt(...)` raises
`FileNotFoundError: Prompt 'analyst_success' not found` during training.

Prompt sources, in priority order:

1. `custom_prompt/<name>` — project-specific overrides (edit these to tailor
   the optimizer's reflection behavior to the task);
2. `skillopt_prompts/<name>` — the vendored upstream defaults (pinned commit);
3. GitHub raw download — fallback only when a file is missing from both dirs.

`train.py` calls [ensure_prompts][install_prompts.ensure_prompts] on every
run: a prompt is (re)written into `site-packages/skillopt/prompts/` whenever
it is missing or its content differs from the resolved source, so edits under
`custom_prompt/` take effect on the next training run automatically.

Usage:
    python install_prompts.py [--force]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
CUSTOM_DIR = REPO_ROOT / "custom_prompt"
VENDORED_DIR = REPO_ROOT / "skillopt_prompts"

PROMPTS_REPO = "huqianghui/SkillOpt"
PROMPTS_REF = "50fed2958b71ae24266a179ba8e791b4e00007bd"  # matches skillopt 0.2.0
RAW_BASE_URL = f"https://raw.githubusercontent.com/{PROMPTS_REPO}/{PROMPTS_REF}/skillopt/prompts"

# The generic reflection prompts referenced by skillopt.engine.trainer /
# skillopt.gradient at v0.2.0. Kept explicit so no listing API call is needed.
PROMPT_NAMES: List[str] = [
    "analyst_error.md",
    "analyst_error_full_rewrite.md",
    "analyst_error_rewrite.md",
    "analyst_success.md",
    "analyst_success_full_rewrite.md",
    "analyst_success_rewrite.md",
    "lr_autonomous.md",
    "merge_failure.md",
    "merge_failure_full_rewrite.md",
    "merge_failure_rewrite.md",
    "merge_final.md",
    "merge_final_full_rewrite.md",
    "merge_final_rewrite.md",
    "merge_success.md",
    "merge_success_full_rewrite.md",
    "merge_success_rewrite.md",
    "meta_skill.md",
    "ranking.md",
    "ranking_rewrite.md",
    "rewrite_skill.md",
    "slow_update.md",
]


def prompts_dir() -> Path:
    """Directory of the installed `skillopt.prompts` package."""
    import skillopt.prompts

    return Path(skillopt.prompts.__file__).resolve().parent


def resolve_local(name: str) -> Optional[Path]:
    """Local source for a prompt: custom override first, then vendored default."""
    for directory in (CUSTOM_DIR, VENDORED_DIR):
        path = directory / name
        if path.exists():
            return path
    return None


def ensure_prompts(force: bool = False) -> int:
    """Sync prompt files into site-packages from local sources (GitHub as fallback).

    A file is written when it is missing, its content differs from the resolved
    source, or `force` is set. Returns the number of files written.
    """
    target = prompts_dir()
    written = 0
    to_download: List[str] = []

    for name in PROMPT_NAMES:
        source = resolve_local(name)
        installed = target / name
        if source is None:
            if force or not installed.exists():
                to_download.append(name)
            continue
        content = source.read_text(encoding="utf-8")
        if force or not installed.exists() or installed.read_text(encoding="utf-8") != content:
            installed.write_text(content, encoding="utf-8")
            written += 1
            origin = "custom" if source.parent == CUSTOM_DIR else "vendored"
            logger.info("  installed %s (%s, %d chars)", name, origin, len(content))

    if to_download:
        import httpx

        logger.info("Downloading %d prompt files missing locally", len(to_download))
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            for name in to_download:
                response = client.get(f"{RAW_BASE_URL}/{name}")
                response.raise_for_status()
                (target / name).write_text(response.text, encoding="utf-8")
                written += 1
                logger.info("  fetched %s (%d chars)", name, len(response.text))

    if written:
        logger.info("Installed %d SkillOpt prompt files into %s", written, target)
    else:
        logger.debug("SkillOpt prompts already up to date in %s", target)
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Sync SkillOpt reflection prompts into site-packages")
    parser.add_argument("--force", action="store_true", help="Rewrite all files even if content matches")
    args = parser.parse_args()
    written = ensure_prompts(force=args.force)
    print(f"Done: {written} file(s) written to {prompts_dir()}")


if __name__ == "__main__":
    main()
