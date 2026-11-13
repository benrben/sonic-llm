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
| Documents | Many questions, one document | Many agents ask different questions about the same document, read only once. |
| Documents | Classify | One agent per document sorts it under one of your labels. |
| Documents | Extract | One agent per document pulls the fields you name into JSON. |
| Documents | Summarize | One agent per document writes its own short summary. |
| Documents | Compress | One agent per document rewrites it shorter, keeping names, numbers and dates. |
| Checks | Chat under load | A chat question alone, then again while many agents work. |
| Checks | Throughput burst | Many agents write at the same time: words per second the machine writes. |
| Checks | Exactness | The same question alone and beside other agents: are the words identical? |

The tests read **documents**: 256 sample files come with the playground (`samples/`, see below), and you can add your own text files
from the documents window (they are kept in `<data>/documents/`, and you can remove them again there; the samples stay). The
window has shortcuts to pick the first 8, 32 or 64 documents, or all of them, and counts what is picked. Every run is saved, and
the clock button in the top bar lists the earlier runs with the score of graded ones; **Show numbers** puts one back on the page
(the numbers are kept, the answers are not).

## The sample library and grading

`samples/` holds 256 invented documents, 2.5 to 4 KB each, in ten kinds: meeting notes, product manual, trip plan, recipe,
incident report, invoice, lease summary, lab report, club newsletter and project status. 256 is the most agents the engine runs at
once, so "all samples" fills it. They are made by `tools/make_samples.py` (standard library only, a fixed seed: the same files on
every run; a test checks that the committed files are exactly what it writes) from templates and word lists. Every document states
its facts once, with distractor numbers, dates and names around them. All people, places and companies are made up. The generated
files are committed, so nobody needs to run the generator. `samples/answers.json` is the answer key: per document its kind, about
five fields with their true values, and about five questions with the answers a grader accepts ("14 March 2027", "March 14, 2027",
"2027-03-14"; "1,250" and "1250").

A run is **graded** when the test has a right answer, every document it reads has a key and you typed no question of your own:

| Test | What each agent gets | Right means |
|------|----------------------|-------------|
| Agent swarm, Research team | one keyed question about its own document (the five questions are spread over the agents) | the answer holds an accepted form |
| Many questions, one document | the document's keyed questions in turn (more agents than questions repeat them; the result says so) | the same |
| Follow-up question | a second keyed question, never the first | the same |
| Classify | the ten kinds as labels | the reply names exactly one label, and it is the right one |
| Extract | its document's keyed fields | the reply is a JSON object and each field matches |

Every other test (summarize, compress, relay, long document, chat, turns, burst, load, exactness) says "Not graded" and shows no
score, and so does a run with your own question, labels or documents.

The rules are in `grading.py`, deterministic, with no model as judge. Case, punctuation, quotes, thousands separators, leading
articles and common date forms do not matter. An accepted form must appear as whole words or numbers: `12500` is not `1250`.
Two different dates, codes or numbers put forward as the answer ("1,250 or 1,520 euros") count as hedging and fail. The thinking of
a model (`<think>...</think>` or reasoning) is never graded, only the final answer; no final answer is "No answer". The page shows
**Correct** with the score and the percentage for each engine (blue Sonic LLM, orange llama.cpp), a plain line such as "Both
engines got 61 of 64 right", a check, cross or dash in every agent tile, and, in the answers, a row per agent that opens to the
document, the question, the whole answer, the expected answer and the verdict, with a filter (All, Wrong only, No answer) and 50
rows to a page. The run record keeps `graded`, `correct`, `wrong` and `empty` per engine and, per agent, the question, the
expected answer and the verdict.

What "correct" means, and does not: the agent's answer holds the exact fact written in the text, found by rules. It is one run, so
the percentage is one reading, not a measure of the model. A right fact next to a wrong claim passes, a correct answer worded in
a way the accepted forms do not cover fails, and a name is checked only for the name (two names given together pass if one is
right). A verbose answer that mentions another number next to the right one passes unless that number is offered as the answer.

## How many agents fit

Next to the **Agents** field the page says "This Mac can run up to N agents at once on this test" for the loaded model and the test
you picked, and **Max** sets it (for the tests with one agent per document it picks that many documents). N comes from the engine's
own memory plan: the largest count whose context still holds a whole document and the answer, never more than 256. The defaults
stay modest (8 agents for the document tests); Max is your choice. Past N the rest wait for a free slot or read less of their
document, and the plan line under the test says so. With **Compare with llama.cpp** on, a run of more than 64 agents carries a
warning that it takes much longer.

## Develop

```sh
uv run --locked pytest -q                                   # unit tests, no model and no GPU
python tools/make_samples.py                                # write the 256 samples and answers.json again (identical output)
SONIC_SERVER_BIN=../engine/target/release/sllm-server \
  uv run --locked pytest -q tests/test_browser.py           # the whole page with real clicks, on the engine's --fake mode
uv run playwright install chromium                          # once, for the browser test
```

The browser test starts the playground with a fake model file, the engine in `--fake` mode and a fake Hugging Face on a local
port. No test downloads a real model or reaches the network.
