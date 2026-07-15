"""Offline tests for the content-filter probe cache (no network, no credentials)."""

from types import SimpleNamespace
from typing import Any, Dict

import httpx
import pytest
from openai import BadRequestError

import probe_content_filter as pcf


def make_task(task_id: str = "42") -> Dict[str, Any]:
    return {
        "id": task_id,
        "video": "/workspace/home/azureuser/data/sft_data/videos/Charades/X.mp4",
        "family": "Charades",
        "frame_blobs": ["training/frame/Charades/X.mp4_frame/0.jpg"],
        "num_frames": 1,
        "seconds_per_frame": 3,
        "solution": {},
    }


def test_load_probe_cache_missing_returns_empty(tmp_path):
    assert pcf.load_probe_cache(tmp_path / "missing.json") == {}


def test_save_and_load_probe_cache_round_trip(tmp_path):
    path = tmp_path / "cache.json"
    pcf.save_probe_cache({"1": True, "2": False}, path)
    assert pcf.load_probe_cache(path) == {"1": True, "2": False}


def test_probe_task_cached_hits_cache_without_probing(monkeypatch):
    def fail_probe(*args: Any) -> bool:
        raise AssertionError("probe_task must not be called on a cache hit")

    monkeypatch.setattr(pcf, "probe_task", fail_probe)
    cache = {"42": True}
    assert pcf.probe_task_cached(None, make_task("42"), None, cache, "gpt-4.1-mini") is True  # type: ignore[arg-type]
    assert cache == {"42": True}


def test_probe_task_cached_probes_once_and_persists(monkeypatch, tmp_path):
    calls: list[str] = []
    saved_path = tmp_path / "cache.json"

    def fake_probe(client: Any, task: Dict[str, Any], config: Any, model: str) -> bool:
        calls.append(task["id"])
        return False

    real_save = pcf.save_probe_cache
    monkeypatch.setattr(pcf, "probe_task", fake_probe)
    monkeypatch.setattr(pcf, "save_probe_cache", lambda cache: real_save(cache, saved_path))

    cache: Dict[str, bool] = {}
    assert pcf.probe_task_cached(None, make_task("7"), None, cache, "gpt-4.1-mini") is False  # type: ignore[arg-type]
    assert pcf.probe_task_cached(None, make_task("7"), None, cache, "gpt-4.1-mini") is False  # type: ignore[arg-type]
    assert calls == ["7"]
    assert cache == {"7": False}
    assert pcf.load_probe_cache(saved_path) == {"7": False}


def make_bad_request(message: str, code: Any = None) -> BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "https://example.test/openai"))
    return BadRequestError(message, response=response, body={"code": code} if code else None)


def make_client(errors: list[BaseException]) -> Any:
    """Client whose create() raises the queued errors, then succeeds."""

    def create(**kwargs: Any) -> Any:
        if errors:
            raise errors.pop(0)
        return SimpleNamespace()

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.fixture()
def no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(pcf.time, "sleep", sleeps.append)
    monkeypatch.setattr(pcf, "blob_sas_url", lambda config, blob: "https://example.test/frame.jpg")
    return sleeps


TIMEOUT_MSG = "Timed out while downloading image from https://example.test/frame.jpg"


def test_probe_task_retries_transient_download_timeout(no_sleep):
    client = make_client([make_bad_request(TIMEOUT_MSG), make_bad_request(TIMEOUT_MSG)])
    assert pcf.probe_task(client, make_task(), None, "gpt-4.1-mini") is False  # type: ignore[arg-type]
    assert no_sleep == [5, 10]


def test_probe_task_persistent_download_timeout_excludes(no_sleep):
    client = make_client([make_bad_request(TIMEOUT_MSG)] * 3)
    assert pcf.probe_task(client, make_task(), None, "gpt-4.1-mini") is True  # type: ignore[arg-type]
    assert no_sleep == [5, 10]


def test_probe_task_content_filter_blocked_without_retry(no_sleep):
    client = make_client([make_bad_request("filtered due to the content management policy", code="content_filter")])
    assert pcf.probe_task(client, make_task(), None, "gpt-4.1-mini") is True  # type: ignore[arg-type]
    assert no_sleep == []


def test_probe_task_other_bad_request_raises(no_sleep):
    client = make_client([make_bad_request("Invalid image format")])
    with pytest.raises(BadRequestError):
        pcf.probe_task(client, make_task(), None, "gpt-4.1-mini")  # type: ignore[arg-type]
