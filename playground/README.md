# Sonic LLM playground

A web page that runs live tests on your own models with Sonic LLM, and, if you have it, side by side with llama.cpp's
`llama-server`. Pick a model, pick a test, press Run, and watch the speed, the agents and the memory move.

The numbers on the page are one live run on your machine. They are not a benchmark: there is one repetition, the two
engines run one after the other, and nothing is rotated or repeated to average out noise. For measured results see
[`docs/benchmarks.md`](../docs/benchmarks.md).

## Start it

Build the engine once (from the repository root), then start the playground from this folder:

```sh
cd engine && cargo build --release -p sllm-server && cd ..
cd playground
uv run --locked sonic-playground
```

Open the address it prints (http://127.0.0.1:8000). It needs [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer.
The page tells you in plain words if the engine is not built yet or if there is no model.

Models: click **Get models** on the page to download one, or put `.gguf` files in `~/models` (or the folder you set
below). Nothing is loaded until you press Load, or press Send and let the page load the model for you.

## Settings

Each setting is a flag, an environment variable or a default, in that order.

| Flag | Environment variable | Default |
|------|----------------------|---------|
| `--server-bin` | `SONIC_SERVER_BIN` | `../engine/target/release/sllm-server` (next to this folder) |
| `--models` | `SONIC_MODELS` | `~/models`; every `.gguf` under it, searched recursively |
| `--llama-bin` | `LLAMA_SERVER_BIN` | `llama-server` on your `PATH`; without it the page runs Sonic LLM alone |
| `--data` | `SONIC_PLAYGROUND_DATA` | `playground/.data` (run history, logs, your uploaded documents; not in git) |
| `--engine-args` | `SONIC_ENGINE_ARGS` | none; extra flags for `sllm-server`, for example `--threads 8` |
| `--host`, `--port` | `SONIC_PLAYGROUND_HOST`, `SONIC_PLAYGROUND_PORT` | `127.0.0.1`, `8000` |
| `--offline` | `SONIC_PLAYGROUND_OFFLINE=1` | off; turns model search and downloads off (local models still work) |
| (environment only) | `HF_TOKEN` | none; for models that need a Hugging Face account. Sent only to huggingface.co, never stored or logged |
| (environment only) | `HF_ENDPOINT` | `https://huggingface.co` |

## The page

The top bar holds the model picker with its load state, the **Compare with llama.cpp** switch, **Get models**, the history of
earlier runs, the engine details and a light/dark switch. The left side is the list of tests, in four groups, each a card that
says in one line what you will see; picking a card shows only the controls that test needs, and one **Run** button starts it.
The right side reads top to bottom like the charts in the README: the headline numbers (total time, first word, read speed,
write speed), then the agents as a grid of tiles that fill as they write, then live charts, then the answers, then a table of
every number. Before the first run it explains itself in three steps. Everything works from the keyboard, in light and dark,
down to a phone's width, and offline: the page uses no external fonts, scripts or images.

The numbers are one live run on this machine, not a benchmark, and the page says so under the headline numbers.

## Compare with llama.cpp

Install llama.cpp so that `llama-server` is on your `PATH` (for example `brew install llama.cpp`), or set
`LLAMA_SERVER_BIN`, then restart the playground. The **Compare with llama.cpp** switch in the top bar then loads the same
model in both engines, with the same memory, and runs every test on both. Each headline number then shows Sonic LLM (blue) and
llama.cpp (orange) and a ratio in words ("3.6x faster", "about the same" within 5%). Without `llama-server` the switch is off and
the (i) button next to it says why. Comparison is never required.

## Models

**Get models** opens a window with three tabs and a strip for the downloads going on:

- **On this Mac:** your models with their size and which engine runs them; **Load** one, or **Remove** it (the button asks once
  more first; only files inside your models folder are ever removed).
- **Recommended:** the eight models the benchmarks use, each with its size on disk and roughly the memory it needs.
- **Search Hugging Face:** type a name to search the most downloaded GGUF repositories (nothing typed shows the top ones),
  pick one, and see its files with their quantization, size and whether they fit this Mac. Files split in several parts and
  vision add-ons (mmproj) are not shown, because the playground loads one file. **Have a link? Paste it** takes a link to a
  `.gguf` file, or `owner/repo` and the file name.
- **Downloads:** a progress bar with speed and a Cancel button.

Every file shows two labelled hints, never colour alone: *Memory: fits easily / tight fit / too large for this Mac*, and
*Engine: Sonic LLM* or *Engine: fallback, slower*.

Downloads go to a `.part` file and are renamed when complete and checked against the hash the Hub reports. Cancelling or a
lost connection keeps the part; start the same download again and it continues where it stopped. The playground checks the
free disk space first. A finished model shows up in the picker at once. Sonic LLM's own engine runs some architectures
natively; any other GGUF still works on the fallback engine inside `sllm-server`, and the page says which one you get.

## The tests

Fifteen tests in four groups. Each card on the page carries the same one-line promise.

| Group | Test | What you will see |
|-------|------|-------------------|
| Talk | Chat | One question, one answer: how soon the first word comes and how fast the rest is written. Pick an answer level (off, sonic, fast, normal, deep). |
| Talk | Long document | A long text, a question about it, then the same question again from memory. |
| Many agents | Agent swarm | One agent per document reads it at the same time, then one answer merges what they found. |
| Many agents | Research team | Several questions; an agent per document answers from its own text or says NOT FOUND; a manager merges the reports. |
| Many agents | Follow-up question | A new question for the agents of the last swarm or team, which still remember their documents. |
| Many agents | Relay | Fresh agents with an empty memory work through a plan of small stories. |
| Many agents | Agents taking turns | Many agents take turns, a few at a time; the others are parked on disk. Sonic LLM only. |
| Documents | Many questions, one document | Many agents ask different questions about the same long document, read only once. |
| Documents | Classify | One agent per document sorts it under one of your labels. |
| Documents | Extract | One agent per document pulls the fields you name into JSON. |
| Documents | Summarize | One agent per document writes its own short summary. |
| Documents | Compress | One agent per document rewrites it shorter, keeping names, numbers and dates. |
| Checks | Chat under load | A chat question alone, then again while many agents work. |
| Checks | Throughput burst | Many agents write at the same time: words per second the machine writes. |
| Checks | Exactness | The same question alone and beside other agents: are the words identical? |

The tests read **documents**: nine sample files come with the playground (`samples/`), and you can add your own text files from
the documents window (they are kept in `<data>/documents/`, and you can remove them again there; the samples stay). Every run
is saved, and the clock button in the top bar lists the earlier runs; **Show numbers** puts one back on the page (the numbers
are kept, the answers are not).

## Develop

```sh
uv run --locked pytest -q                                   # unit tests, no model and no GPU
SONIC_SERVER_BIN=../engine/target/release/sllm-server \
  uv run --locked pytest -q tests/test_browser.py           # the whole page with real clicks, on the engine's --fake mode
uv run playwright install chromium                          # once, for the browser test
```

The browser test starts the playground with a fake model file, the engine in `--fake` mode and a fake Hugging Face on a local
port. No test downloads a real model or reaches the network.
