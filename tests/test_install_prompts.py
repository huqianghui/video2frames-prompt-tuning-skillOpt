"""Offline tests for the prompt sync logic (no network)."""

from pathlib import Path

import pytest

import install_prompts as ip


@pytest.fixture()
def dirs(monkeypatch, tmp_path):
    custom = tmp_path / "custom_prompt"
    vendored = tmp_path / "skillopt_prompts"
    target = tmp_path / "site-packages-prompts"
    for d in (custom, vendored, target):
        d.mkdir()
    monkeypatch.setattr(ip, "CUSTOM_DIR", custom)
    monkeypatch.setattr(ip, "VENDORED_DIR", vendored)
    monkeypatch.setattr(ip, "prompts_dir", lambda: target)
    monkeypatch.setattr(ip, "PROMPT_NAMES", ["a.md", "b.md"])
    return custom, vendored, target


def test_vendored_installed_when_missing(dirs):
    custom, vendored, target = dirs
    (vendored / "a.md").write_text("vendored a", encoding="utf-8")
    (vendored / "b.md").write_text("vendored b", encoding="utf-8")
    assert ip.ensure_prompts() == 2
    assert (target / "a.md").read_text(encoding="utf-8") == "vendored a"


def test_custom_overrides_vendored(dirs):
    custom, vendored, target = dirs
    (vendored / "a.md").write_text("vendored a", encoding="utf-8")
    (vendored / "b.md").write_text("vendored b", encoding="utf-8")
    (custom / "a.md").write_text("custom a", encoding="utf-8")
    ip.ensure_prompts()
    assert (target / "a.md").read_text(encoding="utf-8") == "custom a"
    assert (target / "b.md").read_text(encoding="utf-8") == "vendored b"


def test_noop_when_up_to_date(dirs):
    custom, vendored, target = dirs
    (vendored / "a.md").write_text("a", encoding="utf-8")
    (vendored / "b.md").write_text("b", encoding="utf-8")
    assert ip.ensure_prompts() == 2
    assert ip.ensure_prompts() == 0


def test_custom_edit_propagates(dirs):
    custom, vendored, target = dirs
    (vendored / "a.md").write_text("vendored a", encoding="utf-8")
    (vendored / "b.md").write_text("vendored b", encoding="utf-8")
    ip.ensure_prompts()
    (custom / "a.md").write_text("custom a v2", encoding="utf-8")
    assert ip.ensure_prompts() == 1
    assert (target / "a.md").read_text(encoding="utf-8") == "custom a v2"


def test_repo_vendored_prompts_complete():
    vendored = Path(ip.__file__).parent / "skillopt_prompts"
    missing = [n for n in ip.PROMPT_NAMES if not (vendored / n).exists()]
    assert missing == []
