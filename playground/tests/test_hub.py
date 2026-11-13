"""The Hub client against a local fake server: search, repository facts, the file tree with pages, gated and rate-limited answers,
timeouts, an unreachable server, the cache, and where the token goes."""

import pytest

from sonic_playground import hub
from sonic_playground.hub import HubClient, HubError

TREE = [{"type": "file", "oid": "a1", "size": 3083, "path": ".gitattributes"},
        {"type": "file", "oid": "b2", "size": 2375773472, "lfs": {"oid": "c" * 64, "size": 2375773472, "pointerSize": 135}, "path": "Qwen3-4B-Q4_0.gguf"},
        {"type": "file", "oid": "b3", "size": 4000000000, "lfs": {"oid": "d" * 64, "size": 4000000000}, "path": "Qwen3-4B-Q8_0.gguf"},
        {"type": "file", "oid": "b4", "size": 100, "lfs": {"oid": "e" * 64}, "path": "mmproj-F16.gguf"},
        {"type": "directory", "oid": "f", "size": 0, "path": "split"},
        {"type": "file", "oid": "b5", "size": 9, "lfs": {"oid": "f" * 64}, "path": "split/big-00001-of-00003.gguf"},
        {"type": "file", "oid": "b6", "size": 9, "lfs": {"oid": "f" * 64}, "path": "split/big-00002-of-00003.gguf"}]


async def test_search_asks_for_the_most_downloaded_gguf_repos_and_keeps_a_few_fields(hub_server):
    rows = await HubClient(hub_server.url).search("qwen3 4b")
    assert rows == [{"id": "unsloth/Qwen3-4B-GGUF", "downloads": 5000, "likes": 100, "tags": ["gguf", "qwen3", "license:apache-2.0"]}]
    q = hub_server.requests[0]["query"]
    assert "search=qwen3%204b" in q and "filter=gguf" in q and "sort=downloads" in q and "direction=-1" in q and "limit=20" in q


async def test_an_empty_search_lists_the_top_repositories_without_a_search_term(hub_server):
    await HubClient(hub_server.url).search("  ")
    assert "search=" not in hub_server.requests[0]["query"] and "filter=gguf" in hub_server.requests[0]["query"]


async def test_results_are_kept_for_a_few_minutes(hub_server):
    now = [1000.0]
    c = HubClient(hub_server.url, ttl=300, clock=lambda: now[0])
    await c.search("a")
    await c.search("a")
    assert len(hub_server.requests) == 1
    now[0] += 301
    await c.search("a")
    assert len(hub_server.requests) == 2


async def test_a_repository_gives_its_architecture_and_gated_state(hub_server):
    info = await HubClient(hub_server.url).repo("unsloth/Qwen3-4B-GGUF")
    assert info == {"id": "unsloth/Qwen3-4B-GGUF", "private": False, "gated": False, "downloads": 5000, "likes": 100,
                    "arch": "qwen3", "context_length": 40960}


async def test_the_file_tree_follows_its_pages_and_keeps_files_only(hub_server):
    hub_server.tree_pages = [TREE[:3], TREE[3:]]
    files = await HubClient(hub_server.url).files("unsloth/Qwen3-4B-GGUF")
    assert [f["path"] for f in files] == [".gitattributes", "Qwen3-4B-Q4_0.gguf", "Qwen3-4B-Q8_0.gguf", "mmproj-F16.gguf",
                                          "split/big-00001-of-00003.gguf", "split/big-00002-of-00003.gguf"]
    assert files[1]["size"] == 2375773472 and files[1]["sha256"] == "c" * 64 and files[0]["sha256"] is None
    assert len(hub_server.requests) == 2


def test_model_files_offers_single_gguf_files_with_quant_and_fit_and_says_what_it_hides():
    files = [{"path": t["path"], "size": t["size"], "sha256": (t.get("lfs") or {}).get("oid")} for t in TREE if t["type"] == "file"]
    out = hub.model_files(files, total_ram=16 * 2**30)
    assert [(f["name"], f["quant"], f["fit"]) for f in out["files"]] == [("Qwen3-4B-Q4_0.gguf", "Q4_0", "fits easily"),
                                                                         ("Qwen3-4B-Q8_0.gguf", "Q8_0", "fits easily")]
    assert out["hidden"] == {"split": 2, "projector": 1}
    big = hub.model_files([{"path": "x-Q4_0.gguf", "size": 25 * 10**9, "sha256": None}], 48 * 2**30)["files"][0]
    assert big["fit"] == "tight" and hub.model_files([{"path": "x-Q4_0.gguf", "size": 70 * 10**9, "sha256": None}], 48 * 2**30)["files"][0]["fit"] == "too large"
    assert hub.model_files(files, None)["files"][0]["fit"] == "unknown"


async def test_a_gated_repository_says_so_and_how_to_get_access(hub_server, monkeypatch):
    hub_server.gated_repos = {"meta/secret-GGUF"}
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(HubError, match="gated or private.*HF_TOKEN") as e:
        await HubClient(hub_server.url).repo("meta/secret-GGUF")
    assert e.value.kind == "gated"
    monkeypatch.setenv("HF_TOKEN", "wrong")
    with pytest.raises(HubError, match="your HF_TOKEN does not have"):
        await HubClient(hub_server.url).repo("meta/secret-GGUF")
    monkeypatch.setenv("HF_TOKEN", "good")
    assert (await HubClient(hub_server.url).repo("meta/secret-GGUF"))["arch"] == "qwen3"


async def test_the_token_goes_only_to_the_hosts_that_may_have_it(hub_server, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setattr(hub, "TOKEN_HOSTS", ("huggingface.co",))
    await HubClient(hub_server.url).search("x")
    assert "Authorization" not in hub_server.requests[0]["headers"]
    assert hub.auth_headers("https://huggingface.co/api/models") == {"Authorization": "Bearer secret"}
    assert hub.auth_headers("https://cdn-lfs.huggingface.co/x") == {"Authorization": "Bearer secret"}
    assert hub.auth_headers("https://evilhuggingface.co/x") == {} and hub.auth_headers("https://example.com/huggingface.co") == {}


async def test_a_missing_repository_and_a_rate_limit_have_plain_messages(hub_server):
    hub_server.rate_limited = True
    with pytest.raises(HubError, match="limiting requests. Try again in 30 seconds") as e:
        await HubClient(hub_server.url).search("x")
    assert e.value.kind == "rate_limited"
    assert hub.error_for(404, {}, "a/b", False).kind == "not_found"
    assert hub.error_for(500, {}, "a/b", False).kind == "http"


async def test_a_server_that_does_not_answer_in_time_is_a_timeout(hub_server):
    import httpx

    hub_server.hang = 1.0
    with pytest.raises(HubError, match="did not answer in time") as e:
        await HubClient(hub_server.url, timeout=httpx.Timeout(0.2)).search("x")
    assert e.value.kind == "timeout"


async def test_an_unreachable_hub_says_offline(hub_server):
    url = hub_server.url
    hub_server.close()
    with pytest.raises(HubError, match="Are you offline") as e:
        await HubClient(url).search("x")
    assert e.value.kind == "offline"
