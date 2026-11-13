"""The limits and timings of a live session (live.py). Folders and binaries are not here: they come from `settings`."""

from __future__ import annotations

IDLE_STOP_S = 15 * 60  # a loaded model nobody uses for this long is unloaded
TICK_S = 0.5
RATE_WINDOW_S = 2.0  # rates are averaged over this window: the counters move once per engine step, and one step can
# take longer than a tick (big prompt batches, big models), which drew 0 / peak / 0 saw teeth
RATE_MOVES = 4  # ... and over at least this many moves of the counter, so that a slow model's long steps do not draw teeth
AGENT_TICK_S = 1.0  # how often an agent that is writing tells the page its token count
AGENT_REPLY_CHARS = 200_000  # the most of one agent's reply the page gets
REPLAY = 5000  # events kept for a browser that reconnects (it sends Last-Event-ID and gets what it missed)
CTX = 1 << 20  # no own cap on one request: the model's trained context and the KV cache are the limits
MAX_AGENTS = 256  # sllm-server tracks at most 256 sequences at once (its seq-id cap)
MANY_AGENTS = 16  # above this a step leaves the exact kernels: the engine takes its many-agent setup (--many-agents)
LLAMA_SLOTS = MAX_AGENTS
# Both engines get the same KV memory, as much as fits (half each when both are loaded). llama-server shares it as one
# pool over its slots (-kvu); sllm-server gives every sequence its own stream of --ctx cells, so it runs cells / ctx
# agents at once: its --ctx follows the test and the number of agents asked for.

DEFER_GIVE_UP_S = 900.0  # a deferred request is asked again until it has waited this long
LONG_STREAMS = 4  # chat and the long document: one long context plus room for the few other requests they make
BLOCK_PAGE = 256  # cells of one KV page (--kv-pages counts these)
ANSWER_ROOM = 256  # when a prompt has to be cut to fit the context, at least this much is left for the answer
DOC_CHARS = None  # agents read whole documents (fit_prompt cuts one only when it is longer than the agent's context)
SERIES = {"sllm": "Sonic LLM", "llama": "llama.cpp"}  # what a person reads for each engine
