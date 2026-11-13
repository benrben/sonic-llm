"""Model downloads against a local fake Hub: progress, cancel, resume (with and without range support), the .part file and its
rename, the disk-space refusal, the checksum, gated files and the token."""

import asyncio
import hashlib
import shutil
from collections import namedtuple

import pytest
from fakehub import gguf_bytes

from sonic_playground import downloads, settings
from sonic_playground.downloads import DownloadError, Downloads

SIZE = 1_000_000
REPO, NAME = "owner/repo", "tiny-Q4_0.gguf"


@pytest.fixture
def served(hub_server):
    hub_server.content = gguf_bytes(SIZE)
    hub_server.add_file(REPO, NAME, hub_server.content)
    return hub_server


async def test_a_download_is_written_to_a_part_file_and_renamed_when_complete(served, cfg):
    dl = Downloads()
    d = dl.start(REPO, NAME, hashlib.sha256(served.content).hexdigest())
    await d.task
    assert (d.state, d.error) == ("done", None) and d.bytes_done == d.bytes_total == SIZE
    assert (cfg.models_dir / NAME).read_bytes() == served.content and not (cfg.models_dir / (NAME + ".part")).exists()
    assert d.to_dict()["percent"] == 100.0


async def test_progress_shows_bytes_percent_and_speed_while_it_runs(served, cfg):
    served.delay = 0.02  # 16 pieces of 64 KB: about 0.3 s
    d = Downloads().start(REPO, NAME)
    seen = []
    while not d.task.done():
        seen.append(d.to_dict())
        await asyncio.sleep(0.03)
    mid = [s for s in seen if s["state"] == "downloading" and 0 < s["bytes_done"] < SIZE]
    assert mid and all(s["bytes_total"] == SIZE and 0 < s["percent"] < 100 for s in mid)
    assert any(s["speed_bps"] > 0 for s in mid)
    assert (cfg.models_dir / (NAME + ".part")).exists() or d.state == "done"


async def test_cancel_stops_the_download_and_keeps_the_part_file(served, cfg):
    served.delay = 0.05
    d = Downloads().start(REPO, NAME)
    while d.bytes_done < 200_000:
        await asyncio.sleep(0.01)
    d.task.cancel()
    await asyncio.gather(d.task, return_exceptions=True)
    assert d.state == "cancelled"
    part = cfg.models_dir / (NAME + ".part")
    assert 0 < part.stat().st_size < SIZE and not (cfg.models_dir / NAME).exists()


async def test_starting_again_resumes_with_a_range_request_and_appends(served, cfg):
    served.delay = 0.05
    dl = Downloads()
    d = dl.start(REPO, NAME)
    while d.bytes_done < 300_000:
        await asyncio.sleep(0.01)
    await dl.cancel(d.id)
    kept = (cfg.models_dir / (NAME + ".part")).stat().st_size
    served.delay = 0
    served.requests.clear()
    d2 = dl.start(REPO, NAME)
    await d2.task
    assert d2.state == "done" and d2.resumed and (cfg.models_dir / NAME).read_bytes() == served.content  # content is intact
    assert served.requests[0]["headers"]["Range"] == f"bytes={kept}-" and d2.bytes_total == SIZE


async def test_a_server_without_range_support_starts_over(served, cfg):
    (cfg.models_dir).mkdir()
    (cfg.models_dir / (NAME + ".part")).write_bytes(served.content[:5000])
    served.ranges = False
    d = Downloads().start(REPO, NAME)
    await d.task
    assert d.state == "done" and not d.resumed and (cfg.models_dir / NAME).read_bytes() == served.content


async def test_a_part_file_that_is_already_the_whole_file_is_finished_without_a_new_download(served, cfg):
    cfg.models_dir.mkdir()
    (cfg.models_dir / (NAME + ".part")).write_bytes(served.content)
    d = Downloads().start(REPO, NAME)
    await d.task
    assert d.state == "done" and (cfg.models_dir / NAME).read_bytes() == served.content


async def test_a_part_file_longer_than_the_file_is_dropped_and_downloaded_again(served, cfg):
    cfg.models_dir.mkdir()
    (cfg.models_dir / (NAME + ".part")).write_bytes(b"x" * (SIZE + 10))
    d = Downloads().start(REPO, NAME)
    await d.task
    assert d.state == "done" and (cfg.models_dir / NAME).read_bytes() == served.content


async def test_not_enough_disk_space_refuses_before_writing(served, cfg, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10**12, 10**12 - 1_500_000, 1_500_000))  # file + 1 GiB will not fit
    d = Downloads().start(REPO, NAME)
    await d.task
    assert d.state == "failed" and "Not enough disk space" in d.error and "spare" in d.error
    assert not (cfg.models_dir / NAME).exists()


async def test_only_the_missing_part_counts_against_the_free_space(served, cfg, monkeypatch):
    cfg.models_dir.mkdir()
    (cfg.models_dir / (NAME + ".part")).write_bytes(served.content[:SIZE - 100])
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10**12, 0, downloads.RESERVE_BYTES + 200))
    d = Downloads().start(REPO, NAME)
    await d.task
    assert d.state == "done"


async def test_a_wrong_checksum_removes_the_file(served, cfg):
    d = Downloads().start(REPO, NAME, "0" * 64)
    await d.task
    assert d.state == "failed" and "checksum" in d.error
    assert not list(cfg.models_dir.iterdir())


async def test_a_file_that_is_not_a_gguf_is_refused(served, cfg):
    served.add_file(REPO, "page-Q4_0.gguf", b"<html>sign in</html>" * 100)
    d = Downloads().start(REPO, "page-Q4_0.gguf")
    await d.task
    assert d.state == "failed" and "not a GGUF" in d.error and not list(cfg.models_dir.iterdir())


async def test_a_gated_file_says_what_to_do_and_the_token_is_sent_but_not_kept(served, cfg, monkeypatch):
    served.token = "good"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    d = Downloads().start(REPO, NAME)
    await d.task
    assert d.state == "failed" and "gated or private" in d.error and "HF_TOKEN" in d.error
    monkeypatch.setenv("HF_TOKEN", "good")
    d2 = Downloads().start(REPO, NAME)
    await d2.task
    assert d2.state == "done" and served.requests[-1]["headers"]["Authorization"] == "Bearer good"
    assert "good" not in repr(d2.to_dict()) and "good" not in repr(vars(settings.get()))


async def test_a_missing_file_a_second_start_and_an_existing_file_are_refused(served, cfg):
    dl = Downloads()
    d = dl.start(REPO, "nothing-Q4_0.gguf")
    await d.task
    assert d.state == "failed" and "not found" in d.error
    served.delay = 0.05
    first = dl.start(REPO, NAME)
    with pytest.raises(DownloadError, match="already being downloaded"):
        dl.start(REPO, NAME)
    await dl.cancel(first.id)
    (cfg.models_dir / "have-Q4_0.gguf").write_bytes(b"x")
    with pytest.raises(DownloadError, match="already in the models folder"):
        dl.start(REPO, "have-Q4_0.gguf")
    with pytest.raises(DownloadError, match="must be a .gguf"):
        dl.start(REPO, "model.bin")


async def test_offline_mode_refuses_downloads(served, cfg, monkeypatch):
    monkeypatch.setattr(settings, "_current", settings.Settings(**{**settings.get().__dict__, "offline": True}))
    with pytest.raises(DownloadError, match="offline"):
        Downloads().start(REPO, NAME)


async def test_a_file_in_a_folder_of_the_repo_is_saved_by_its_plain_name(served, cfg):
    served.add_file(REPO, "sub/dir/deep-Q4_0.gguf", gguf_bytes(5000))
    d = Downloads().start(REPO, "sub/dir/deep-Q4_0.gguf")
    await d.task
    assert d.state == "done" and (cfg.models_dir / "deep-Q4_0.gguf").exists()


async def test_a_lost_connection_keeps_the_part_and_says_how_to_continue(served, cfg):
    served.close()
    d = Downloads().start(REPO, NAME)
    await d.task
    assert d.state == "failed" and "Start the download again" in d.error
