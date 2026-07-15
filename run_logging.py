"""Per-run log files: mirror stdout/stderr into `logs/<prefix>_<timestamp>.log`.

SkillOpt's trainer reports progress with `print()`, so the reliable way to
keep a complete per-run record is to tee both streams. `logging` output
(judge retries, rollout warnings) goes through the same tee because the
handlers are configured after the streams are replaced.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import IO

LOGS_DIR = Path(__file__).resolve().parent / "logs"


class Tee:
    """Write-through wrapper: everything written goes to both streams."""

    def __init__(self, primary: IO[str], secondary: IO[str]) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        self._secondary.write(data)
        return self._primary.write(data)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def __getattr__(self, name: str):
        return getattr(self._primary, name)


def setup_run_logging(prefix: str) -> Path:
    """Tee stdout/stderr into a fresh timestamped file under `logs/`."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.log"
    # line-buffered so the file is readable while the run is in flight
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    print(f"[run_logging] mirroring output to {log_path}")
    return log_path
