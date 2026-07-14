#!/usr/bin/env python3
"""Install SkillOpt reflection prompts into the installed `skillopt` package.

The `skillopt` 0.2.0 wheel omits the `skillopt/prompts/*.md` package data
(no `[tool.setuptools.package-data]` upstream), so `load_prompt(...)` raises
`FileNotFoundError: Prompt 'analyst_success' not found` during training.
This script downloads the prompt files from the SkillOpt GitHub repo (pinned
commit) into `site-packages/skillopt/prompts/`.

Usage:
    python install_prompts.py [--force]

`train.py` calls [ensure_prompts][install_prompts.ensure_prompts]
automatically, so a manual run is only needed to refresh with `--force`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

PROMPTS_REPO = "huqianghui/SkillOpt"
PROMPTS_REF = "50fed2958b71ae24266a179ba8e791b4e00007bd"  # matches skillopt 0.2.0
PROMPTS_API_URL = f"https://api.github.com/repos/{PROMPTS_REPO}/contents/skillopt/prompts?ref={PROMPTS_REF}"
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


def missing_prompts(target: Path) -> List[str]:
    return [name for name in PROMPT_NAMES if not (target / name).exists()]


def ensure_prompts(force: bool = False) -> int:
    """Download prompt files that are missing (or all, with `force`).

    Returns the number of files written.
    """
    import httpx

    target = prompts_dir()
    names = PROMPT_NAMES if force else missing_prompts(target)
    if not names:
        logger.debug("SkillOpt prompts already present in %s", target)
        return 0

    logger.info("Installing %d SkillOpt prompt files into %s", len(names), target)
    written = 0
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for name in names:
            url = f"{RAW_BASE_URL}/{name}"
            response = client.get(url)
            response.raise_for_status()
            (target / name).write_text(response.text, encoding="utf-8")
            written += 1
            logger.info("  fetched %s (%d chars)", name, len(response.text))
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Install SkillOpt reflection prompts into site-packages")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()
    written = ensure_prompts(force=args.force)
    print(f"Done: {written} file(s) written to {prompts_dir()}")


if __name__ == "__main__":
    main()
