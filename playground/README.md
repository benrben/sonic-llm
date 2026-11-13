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

## Compare with llama.cpp

Install llama.cpp so that `llama-server` is on your `PATH` (for example `brew install llama.cpp`), or set
`LLAMA_SERVER_BIN`, then restart the playground. The **Compare with llama.cpp** switch in Options then loads the same model in
both engines, with the same memory, and runs every test on both. Without `llama-server` the switch is off and says why.
Comparison is never required.

## Models

**Get models** opens a window with four parts:

- **Recommended:** the eight models the benchmarks use, each with its size on disk and roughly the memory it needs.
- **Search Hugging Face:** type a name to search the most downloaded GGUF repositories (nothing typed shows the top ones),
  pick one, and see its files with their quantization, size and whether they fit this Mac. Files split in several parts and
  vision add-ons (mmproj) are not shown, because the playground loads one file.
- **Your own link:** paste a Hugging Face link to a `.gguf` file, or `owner/repo` and the file name.
- **Downloads and installed models:** progress with speed, Cancel, and Remove (with a confirmation; only files inside your
  models folder are ever removed).

Downloads go to a `.part` file and are renamed when complete and checked against the hash the Hub reports. Cancelling or a
lost connection keeps the part; start the same download again and it continues where it stopped. The playground checks the
free disk space first. A finished model shows up in the picker at once. Sonic LLM's own engine runs some architectures
natively; any other GGUF still works on the fallback engine inside `sllm-server`, and the page says which one you get.

## The tests

| Test | What it shows |
|------|---------------|
| Chat | One chat turn; attach files to ask about them. Pick an answer level (off, sonic, fast, normal, deep). |
| Agent swarm | One agent per file reads it at the same time, then the chat merges their notes. |
| Many questions, one document | Many agents ask the same long document different questions; it is read once. |
| Chat under load | The chat question alone, then again while a swarm runs. |
| Exactness | The same question alone and beside other agents: are the words identical? |
| Throughput | N agents each write 256 tokens at once: tokens per second when many agents write. |
| Research team | Several questions over your files; one agent per file answers or says NOT FOUND; a manager merges. |
| Ask the readers again | A follow-up to the agents of the last swarm or team, from their documents still in memory. |
| Relay of fresh agents | Laps of fresh agents with empty memory work through a plan of small stories. |
| Long document | The selected files joined into one long prompt, asked twice. |
| Agents taking turns | Many agents take turns, a few at a time; parked on disk in between. Sonic LLM only. |
| Classify | One agent per file puts it under one of your labels. |
| Extract | One agent per file pulls fields you name into JSON. |
| Compress | One agent per file rewrites it shorter, keeping names, numbers and decisions. |
| Summarize | One agent per file summarizes it. |

The tests read **documents**: nine sample files come with the playground (`samples/`), and you can add your own text files
from the file picker (they are kept in `<data>/documents/`).

## Develop

```sh
uv run --locked pytest -q                                   # unit tests, no model and no GPU
SONIC_SERVER_BIN=../engine/target/release/sllm-server \
  uv run --locked pytest -q tests/test_browser.py           # the whole page with real clicks, on the engine's --fake mode
uv run playwright install chromium                          # once, for the browser test
```

The browser test starts the playground with a fake model file, the engine in `--fake` mode and a fake Hugging Face on a local
port. No test downloads a real model or reaches the network.
