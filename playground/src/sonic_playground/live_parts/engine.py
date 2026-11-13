"""One engine of the live playground (live.py): an sllm-server or llama-server process, its HTTP client, its counters and
its streamed completions; and the record of the engine processes the playground started (settings.engines_file), so that a
reload that skipped the exit hook does not leave them running."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx

from .. import footprint, live, settings  # live: its log and BUS are looked up there when used (tests patch them)
from .config import BLOCK_PAGE, DEFER_GIVE_UP_S, LLAMA_SLOTS, SERIES


# -- one engine (a server process) -----------------------------------------------------------------------------------
START_TIMEOUT_S = 300.0  # an engine not healthy by then is stopped and its start fails
STOP_GONE_S = 5.0  # after SIGKILL a stop waits this long for the process group to be gone (then logs it)


def free_port() -> int:
    """A TCP port nothing listens on now."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Engine:
    def __init__(self, rt: str, gguf: Path, bos: bool):
        self.rt, self.gguf, self.bos = rt, gguf, bos
        # which implementation runs the model (sllm only): "next" is Sonic LLM's own engine, which the answer levels were
        # measured on; the session sets it from the engine's /caps once it is up ("llama" for a model the own engine cannot run)
        self.kind = "next"
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self.ready_s: float | None = None
        self.client: httpx.AsyncClient | None = None
        self.tokens = 0  # generated tokens since start (the sampler differentiates it)
        self.active = 0
        self.log_path: Path | None = None
        self.peak_mb = 0.0  # footprint peak of the current run (begin_run resets it)
        self.session_peak_mb = 0.0  # footprint peak since this engine started
        self.max_running = 0  # most requests the scheduler ran at once in this run (sllm /metrics scheduler.running, sampled)
        self.kv_used_peak_cells = 0  # most KV cells in use at once in this run (sllm /metrics kv.sequences[].cells, sampled)
        self.cells = 65536  # KV cells of the whole cache (set by the session from the model's bytes per token)
        self.ctx = 16384  # sllm-server: cells of one sequence (the session sets it per scenario)
        self.extra: list[str] = []  # sllm-server: more flags for one scenario (the session sets them)
        # reads of the running server (/metrics every tick, /caps): a failure is logged once when it starts, not on every
        # poll, and not while the engine starts or stops (then failing reads are expected)
        self.up = False
        self.reads_ok: dict[str, bool] = {}
        self.failed: str | None = None  # why its last restart failed (the session's next run starts it again); None: it did not

    def _read_worked(self, what: str) -> None:
        self.reads_ok[what] = True

    def _read_failed(self, what: str, ex: Exception) -> None:
        """Log a failing read of `what` once: the first failure since the engine came up or since the read last worked."""
        if not self.up or not self.reads_ok.get(what, True):
            return
        self.reads_ok[what] = False
        live.log(f"{self.label}: reading {what} failed ({type(ex).__name__}: {ex}); logged again only after it has worked again")

    def max_parallel(self, ctx: int | None = None) -> int:
        """Agents that run at once with `ctx` cells each (sllm-server: its own --ctx)."""
        return max(1, min(LLAMA_SLOTS, self.cells // (ctx or self.ctx)))

    @property
    def label(self) -> str:
        return SERIES[self.rt]

    def argv(self) -> list[str]:
        cfg = settings.get()
        if self.rt == "sllm":
            # the block stays the session's (a comparison gives both engines the same one); --auto-mode adds the rest; the
            # engine is the default one
            return [str(cfg.server_bin), "--model", str(self.gguf), "--host", "127.0.0.1", "--port", str(self.port), "--auto-mode",
                    "--ctx", str(self.ctx), "--kv-pages", str(self.cells // BLOCK_PAGE), *cfg.engine_args, *self.extra]  # fmt: skip
        return [str(cfg.llama_bin), "-m", str(self.gguf), "--host", "127.0.0.1", "--port", str(self.port), "-ngl", "99",
                "-fa", "on", "-c", str(self.cells), "-np", str(LLAMA_SLOTS), "-kvu", "--metrics"]  # fmt: skip

    async def start(self) -> None:
        self.port = free_port()
        logs = settings.get().logs_dir
        logs.mkdir(parents=True, exist_ok=True)
        self.log_path = logs / f"{self.rt}-{time.strftime('%Y%m%dT%H%M%S')}.log"
        t0 = time.monotonic()
        self.proc = subprocess.Popen(self.argv(), stdout=open(self.log_path, "wb"), stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)  # fmt: skip
        _remember_engine(self.proc.pid)
        # one connection per request in flight: the default pool (100) would hold the other agents inside the client, and
        # that wait would count as time to first token
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0),
                                        limits=httpx.Limits(max_connections=None, max_keepalive_connections=64))
        while time.monotonic() - t0 < START_TIMEOUT_S:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{SERIES[self.rt]} exited during startup (code {self.proc.returncode}); see {self.log_path}")
            if await self._healthy():
                self.ready_s = time.monotonic() - t0
                self.up, self.reads_ok = True, {}
                return
            await asyncio.sleep(0.1)
        await self.stop()  # a server that never became healthy does not keep running beside the next try
        raise TimeoutError(f"{SERIES[self.rt]} not ready after {START_TIMEOUT_S:g} s; see {self.log_path}")

    async def _healthy(self) -> bool:
        """Whether /health answers 200 (a server that does not answer yet is not healthy)."""
        with contextlib.suppress(httpx.HTTPError):
            r = await self.client.get(f"http://127.0.0.1:{self.port}/health", timeout=2)
            return r.status_code == 200
        return False

    async def stop(self) -> None:
        self.up = False
        if self.client is not None:
            try:
                await self.client.aclose()
            except Exception as ex:  # the process is stopped below all the same
                live.log(f"{self.label}: closing its HTTP client failed ({type(ex).__name__}: {ex})")
        p, self.proc = self.proc, None
        if p is None:
            return
        _forget_engine(p.pid)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGTERM)
        for _ in range(100):
            if p.poll() is not None:
                break
            await asyncio.sleep(0.1)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGKILL)
        if not await _group_gone(p, STOP_GONE_S):  # the caller gives the GPU lock back next: say when that is early
            live.log(f"{self.label}: process group {p.pid} still running {STOP_GONE_S:g} s after SIGKILL")

    def footprint_mb(self) -> float | None:
        if self.proc is None:
            return None
        used = footprint.footprint_bytes(self.proc.pid)
        if used is None:
            return None
        mb = used / 1048576
        self.peak_mb = max(self.peak_mb, mb)
        self.session_peak_mb = max(self.session_peak_mb, mb)
        return round(mb, 1)

    def begin_run(self) -> None:
        """Start the per-run peaks (memory, requests at once, KV cells in use) from now."""
        self.peak_mb, self.max_running, self.kv_used_peak_cells = 0.0, 0, 0
        self.footprint_mb()

    def note_metrics(self, j) -> None:
        """Peaks from one sllm-server /metrics document: requests running (scheduler.running) and KV cells in use."""
        sched = j.get("scheduler") if isinstance(j, dict) else None
        if isinstance(sched, dict) and isinstance(sched.get("running"), int):
            self.max_running = max(self.max_running, sched["running"])
        kv = j.get("kv") if isinstance(j, dict) else None
        seqs = kv.get("sequences") if isinstance(kv, dict) else None
        if isinstance(seqs, list):
            used = sum(x["cells"] for x in seqs if isinstance(x, dict) and isinstance(x.get("cells"), int))
            self.kv_used_peak_cells = max(self.kv_used_peak_cells, used)

    async def real_kv_bytes_per_token(self) -> int | None:
        """KV bytes of one token for the KV type the running sllm-server really uses (/caps engine.profile
        kv_bytes_per_token_per_seq.<kv type>.total_once)."""
        try:
            caps = (await self.client.get(f"http://127.0.0.1:{self.port}/caps", timeout=5)).json()["engine"]
            got = int(caps["profile"]["kv_bytes_per_token_per_seq"][caps["kv"]["type"]]["total_once"])
        except (httpx.HTTPError, RuntimeError, KeyError, TypeError, ValueError) as ex:  # RuntimeError: the client is closed (a stop)
            self._read_failed("/caps KV bytes per token", ex)
            return None
        self._read_worked("/caps KV bytes per token")
        return got

    async def tokenize(self, text: str) -> list[int]:
        r = await self.client.post(f"http://127.0.0.1:{self.port}/tokenize",
                                   json={"content": text, "add_special": self.bos, "parse_special": True})  # fmt: skip
        r.raise_for_status()
        return list(r.json()["tokens"])

    async def prompt_tokens_read(self) -> int | None:
        """Prompt tokens the server has read since it started, from its own counters: sllm-server /metrics
        (prompt_tokens_evaluated), llama-server /metrics (llamacpp:prompt_tokens_total). None when unreadable."""
        try:
            r = await self.client.get(f"http://127.0.0.1:{self.port}/metrics", timeout=0.4)
            if self.rt == "sllm":
                j = r.json()
                self.note_metrics(j)
                got = _find_key(j, "prompt_tokens_evaluated")
            else:
                m = re.search(r"^llamacpp:prompt_tokens_total\s+([0-9.e+]+)", r.text, re.M)
                got = int(float(m.group(1))) if m else None
        except (httpx.HTTPError, RuntimeError, ValueError) as ex:  # RuntimeError: the client is closed (a stop)
            self._read_failed("/metrics", ex)  # polled every tick: logged once, not per tick
            return None
        self._read_worked("/metrics")
        return got

    async def generate(self, ids: list[int], n: int, *, lane: str | None, on_token=None, ignore_eos: bool = False) -> dict:
        """One streamed completion. Returns text, token ids, TTFT, decode tok/s, inter-token gaps, cached tokens."""
        body = self._completion_body(ids, n, lane, ignore_eos)
        out: dict = {"text": "", "ids": [], "prompt_tokens": len(ids)}
        times: list[int] = []
        t0 = time.monotonic_ns()
        self.active += 1
        try:
            while not await self._stream_completion(body, out, times, t0, on_token):
                continue  # deferred: asked again after the wait
        finally:
            self.active -= 1
        _note_timings(out, times, t0)
        return out

    def _completion_body(self, ids: list[int], n: int, lane: str | None, ignore_eos: bool) -> dict:
        """The /completion request: greedy, streamed, the prompt cached; the lane on sllm-server only."""
        body = {"prompt": ids, "n_predict": n, "temperature": 0, "stream": True, "cache_prompt": True,
                "return_tokens": True, "ignore_eos": ignore_eos}  # fmt: skip
        if lane and self.rt == "sllm":
            body["lane"] = lane
        return body

    async def _stream_completion(self, body: dict, out: dict, times: list[int], t0: int, on_token) -> bool:
        """One POST /completion, its stream read into `out` (and a time per token into `times`). False when the server
        deferred it: then it waited the server's retry time and the request is to be sent again."""
        async with self.client.stream("POST", f"http://127.0.0.1:{self.port}/completion", json=body) as resp:
            if resp.status_code != 200:
                await self._wait_if_deferred(resp, out, t0)
                return False
            async for line in resp.aiter_lines():
                d = _sse_json(line)
                if d is None:
                    continue
                self._take_chunk(d, out, times, on_token)
                if d.get("stop"):
                    t = d.get("timings") or {}
                    out["cached_tokens"] = t.get("cache_n", d.get("tokens_cached"))
                    out["stop"] = d.get("stop_type")
                    break
        return True

    async def _wait_if_deferred(self, resp, out: dict, t0: int) -> None:
        """A failed /completion: a sllm-server 503 "deferred" (a full lane) is counted in out["deferred"] and waited out
        until DEFER_GIVE_UP_S; any other failure raises."""
        raw = await resp.aread()
        wait = deferred_wait_s(resp.status_code, resp.headers, raw) if self.rt == "sllm" else None
        if wait is not None and (time.monotonic_ns() - t0) / 1e9 < DEFER_GIVE_UP_S:
            out["deferred"] = out.get("deferred", 0) + 1  # a full lane: ask again after retry_after (the wait is in TTFT)
            await asyncio.sleep(wait)
            return
        raise RuntimeError(f"{SERIES[self.rt]} HTTP {resp.status_code}: {raw[:200]!r}")

    def _take_chunk(self, d: dict, out: dict, times: list[int], on_token) -> None:
        """One streamed chunk into `out`: its tokens and text, a time for each token."""
        toks, text = d.get("tokens") or [], d.get("content", "")
        if toks or text:
            now = time.monotonic_ns()
            k = max(1, len(toks))
            times += [now] * k
            out["ids"] += toks
            out["text"] += text
            self.tokens += k
            if on_token is not None:
                on_token(text)


async def _group_gone(p: subprocess.Popen, wait_s: float) -> bool:
    """Whether the process group led by `p` is gone, waiting up to `wait_s` for it. The leader is reaped first: a zombie
    still counts as a member of its group."""
    end = time.monotonic() + wait_s
    while True:
        p.poll()
        try:
            os.killpg(p.pid, 0)
        except (ProcessLookupError, PermissionError):  # no member left that is ours
            return True
        if time.monotonic() >= end:
            return False
        await asyncio.sleep(0.05)


def _sse_json(line: str) -> dict | None:
    """The JSON of one server-sent `data:` line; None for any other line, an empty one and [DONE]."""
    payload = line[5:].strip() if line.startswith("data:") else ""
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


def _note_timings(out: dict, times: list[int], t0: int) -> None:
    """The timings of a finished completion into `out`: tokens, wall time, TTFT, decode tok/s, p95 inter-token gap."""
    n_out = len(times)
    out["n"] = n_out
    out["wall_ms"] = (time.monotonic_ns() - t0) / 1e6
    if times:
        out["ttft_ms"] = (times[0] - t0) / 1e6
        out["start_s"], out["first_s"], out["last_s"] = t0 / 1e9, times[0] / 1e9, times[-1] / 1e9  # time.monotonic() seconds
        span = (times[-1] - times[0]) / 1e9
        out["decode_tok_s"] = (n_out - 1) / span if span > 0 and n_out > 1 else None
        gaps = sorted((b - a) / 1e6 for a, b in zip(times, times[1:]) if b > a)
        out["itl_p95_ms"] = gaps[int(0.95 * (len(gaps) - 1))] if gaps else None


def deferred_wait_s(status: int, headers, raw: bytes) -> float | None:
    """Seconds to wait before asking again after a sllm-server 503 "deferred" (error.retry_after_ms, else the Retry-After
    header); None for any other failure."""
    if status != 503:
        return None
    try:
        err = json.loads(raw).get("error") or {}
    except (ValueError, AttributeError):  # not JSON, or JSON that is not an object: a failure that is not a deferral
        err = {}
    if err.get("type") != "deferred":
        return None
    if isinstance(err.get("retry_after_ms"), (int, float)):
        return max(0.01, err["retry_after_ms"] / 1000)
    try:
        return max(0.01, float(headers.get("retry-after")))
    except (TypeError, ValueError):
        return 1.0


def _find_val(d, key: str):
    """The first value of `key` anywhere in a nested JSON value, whatever its type (None when absent)."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        vals = d.values()
    elif isinstance(d, list):
        vals = d
    else:
        return None
    for v in vals:
        x = _find_val(v, key)
        if x is not None:
            return x
    return None


def _find_key(d, key: str):
    """The first value of `key` anywhere in a nested JSON value (sllm-server nests its counters)."""
    if isinstance(d, dict):
        if isinstance(d.get(key), (int, float)):
            return int(d[key])
        vals = d.values()
    elif isinstance(d, list):
        vals = d
    else:
        return None
    for v in vals:
        x = _find_key(v, key)
        if x is not None:
            return x
    return None


def _engine_pids() -> list[int]:
    try:
        return [int(x) for x in json.loads(settings.get().engines_file.read_text())]
    except (OSError, ValueError):  # no file yet, or one that is not a list of pids: no engine is remembered
        return []


def _write_engine_pids(pids) -> None:
    path = settings.get().engines_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(set(pids))))


def _remember_engine(pid: int) -> None:
    _write_engine_pids({*_engine_pids(), pid})


def _forget_engine(pid: int) -> None:
    _write_engine_pids(p for p in _engine_pids() if p != pid)


def _is_engine_command(cmd: str) -> bool:
    """Whether a process command line is one of the engines this playground starts."""
    cfg = settings.get()
    names = {cfg.server_bin.name, "sllm-server", "llama-server"} | ({cfg.llama_bin.name} if cfg.llama_bin else set())
    return any(name in cmd for name in names)


def stop_orphans() -> list[int]:
    """At start: stop the engines an earlier run of this playground left behind (a reload that skipped the exit hook).
    Only pids from the engines file whose parent is gone (ppid 1) and that still run an engine."""
    stopped, keep = [], []
    for pid in _engine_pids():
        out = subprocess.run(["ps", "-o", "ppid=,command=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        ppid, _, cmd = out.partition(" ")
        if not out or not _is_engine_command(cmd):
            continue  # gone, or the pid is another program now
        if ppid.strip() == "1":
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)
                stopped.append(pid)
        else:
            keep.append(pid)  # still has its playground (e.g. the old worker of a reload has not exited yet)
    _write_engine_pids(keep)
    return stopped
