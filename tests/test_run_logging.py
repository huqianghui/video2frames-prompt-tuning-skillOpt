"""Offline tests for the per-run log tee."""

import logging
import sys

import run_logging as rl


def test_tee_writes_both_streams(tmp_path):
    log_path = tmp_path / "out.log"
    with open(log_path, "w", encoding="utf-8") as f:
        captured = []

        class Fake:
            def write(self, data):
                captured.append(data)
                return len(data)

            def flush(self):
                pass

        tee = rl.Tee(Fake(), f)
        tee.write("hello\n")
        tee.flush()
    assert captured == ["hello\n"]
    assert log_path.read_text(encoding="utf-8") == "hello\n"


def test_setup_run_logging_creates_file_and_tees(monkeypatch, tmp_path):
    monkeypatch.setattr(rl, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(logging, "basicConfig", lambda **_: None)

    log_path = rl.setup_run_logging("train")
    print("stdout line")
    print("stderr line", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()

    assert log_path.parent == tmp_path / "logs"
    assert log_path.name.startswith("train_")
    content = log_path.read_text(encoding="utf-8")
    assert "mirroring output to" in content
    assert "stdout line" in content
    assert "stderr line" in content
