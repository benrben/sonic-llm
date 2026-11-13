"""The page's event stream (live_parts/events.py, GET /api/live/events): a subscriber that fell behind is told to resync
instead of silently missing run_done; a reconnect whose missed events are gone from the replay buffer is told the same;
event ids keep growing across a server restart; every (re)connect starts with a snapshot of /state.

No GPU, no server."""

import asyncio
import json
import time

import pytest

from sonic_playground.live_parts.config import REPLAY
from sonic_playground import live
from sonic_playground.live_parts import events as events_mod


def frames(chunks):
    """The data of each event frame, with its id (None when the frame has none)."""
    out = []
    for c in chunks:
        if "data: " not in c:
            continue
        sid = c.split("\n", 1)[0][4:] if c.startswith("id: ") else None
        out.append((sid, json.loads(c.split("data: ", 1)[1])))
    return out


async def drain(agen, until_type=None, limit_s=2.0):
    out = []
    end = time.monotonic() + limit_s
    async for c in agen:
        out.append(c)
        if until_type and f'"type": "{until_type}"' in c or time.monotonic() > end:
            break
    return out


async def test_a_subscriber_that_fell_behind_gets_resync_and_not_a_silent_gap():
    bus = live.Bus()
    agen = bus.stream(lifetime_s=3)
    assert (await agen.__anext__()).startswith("retry:")  # subscribed
    for i in range(2100):  # the page stopped reading for a while: more than its queue holds
        bus.publish({"type": "token", "i": i})
    bus.publish({"type": "run_done", "run": "r1", "stopped": True})
    evs = [ev for _, ev in frames(await drain(agen, until_type="resync"))]
    await agen.aclose()
    assert evs[-1]["type"] == "resync"  # it refreshes from /state instead of waiting for a run_done that never comes
    assert len(evs) <= 2001


async def test_after_the_resync_the_subscriber_gets_events_again():
    bus = live.Bus()
    agen = bus.stream(lifetime_s=3)
    await agen.__anext__()
    for i in range(2100):
        bus.publish({"type": "token", "i": i})
    await drain(agen, until_type="resync")
    bus.publish({"type": "run_done", "run": "r2"})
    evs = [ev for _, ev in frames(await drain(agen, until_type="run_done"))]
    await agen.aclose()
    assert evs[-1] == {"type": "run_done", "run": "r2", "t": evs[-1]["t"]}


async def test_a_reconnect_past_the_replay_buffer_is_told_to_resync():
    bus = live.Bus()
    for i in range(REPLAY + 10):
        bus.publish({"type": "tick", "i": i})
    agen = bus.stream(lifetime_s=0.3, last_id=3)  # events 4 .. 10 are gone from the buffer
    evs = frames(await drain(agen, until_type="tick"))
    await agen.aclose()
    assert evs[0] == (None, {"type": "resync", "t": evs[0][1]["t"]})
    assert evs[1][1]["type"] == "tick"


async def test_a_reconnect_with_every_missed_event_in_the_buffer_gets_no_resync():
    bus = live.Bus()
    for i in range(5):
        bus.publish({"type": "tick", "i": i})
    evs = frames(await drain(bus.stream(lifetime_s=0.2, last_id=2)))
    assert [ev["i"] for _, ev in evs] == [2, 3, 4]  # ids 3 .. 5


async def test_event_ids_keep_growing_across_a_server_restart():
    old = live.Bus(start=events_mod.epoch_start())
    for i in range(5):
        old.publish({"type": "tick", "i": i})
    last = old.seq
    await asyncio.sleep(0.002)
    new = live.Bus(start=events_mod.epoch_start())  # the restarted server
    for i in range(7):
        new.publish({"type": "tick", "i": 100 + i})
    evs = frames(await drain(new.stream(lifetime_s=0.2, last_id=last)))
    ticks = [ev["i"] for _, ev in evs if ev["type"] == "tick"]
    assert ticks == list(range(100, 107))  # everything the new server said, not the tail after the old id
    assert live.BUS.seq > 10**12  # the live bus counts from its start time


async def test_a_last_event_id_ahead_of_the_server_replays_nothing_and_streams_from_now():
    bus = live.Bus()
    bus.publish({"type": "tick", "i": 1})
    agen = bus.stream(lifetime_s=0.5, last_id=10**9)
    await agen.__anext__()
    bus.publish({"type": "tick", "i": 2})
    evs = frames(await drain(agen, until_type="tick"))
    await agen.aclose()
    assert [ev for _, ev in evs if ev["type"] != "resync"][0]["i"] == 2


@pytest.fixture
def lib(cfg, monkeypatch):
    cfg.samples_dir.mkdir()
    (cfg.samples_dir / "a.md").write_text("hello")
    monkeypatch.setattr(live, "SESSION", None)


async def test_every_connect_starts_with_a_snapshot_of_the_state(lib):
    resp = await live.get_events(None)
    agen = resp.body_iterator
    chunks = [await agen.__anext__(), await agen.__anext__()]
    await agen.aclose()
    assert chunks[0].startswith("retry:")
    (sid, ev), = frames(chunks[1:])
    assert sid is None and ev["type"] == "snapshot"  # no id: the browser's Last-Event-ID stays the last real event
    assert ev["state"] == await live.get_state()
