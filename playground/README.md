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
| Talk | Long document | A long text, a question about it, then the same question again from memory. On picked samples it is graded: six questions about one long document, an update and a negative among them. |
| Many agents | Agent swarm | One agent per document reads it at the same time, then one answer merges what they found. |
| Many agents | Research team | Several questions; an agent per document answers from its own text or says NOT FOUND; a manager merges the reports. Picked documents that make up a whole case get the case's questions too. |
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
window has shortcuts to pick the first 8, 32 or 64 documents, or all of them, only the short, medium or long samples, or the documents of one case, and counts what is picked. Every run is saved, and
the clock button in the top bar lists the earlier runs with the score of graded ones; **Show numbers** puts one back on the page
(the numbers are kept, the answers are not).

## The sample library and grading

`samples/` holds 256 invented documents in eight kinds (project plan, meeting minutes, invoice, incident report, status report, event
plan, service agreement, test report) and three sizes on purpose: 86 short (about 3 to 5 KB), 85 medium (about 6 to 13 KB) and 85 long
(about 17 to 25 KB, several sections, an appendix, markdown tables). The library is about 3.7 MB. They are made by
`tools/make_samples.py` (standard library only, a fixed seed: the same files on every run; a test checks that the committed files are
exactly what it writes) from structured data: a project with teams and their leads, a table of items (quantity, unit price, due date),
a budget in euros and in US dollars, room bookings, change requests, a revision and an erratum. Every answer is **computed** from that
data, never typed, and the numbers a question asks for (a total, a difference, a weekday, a deadline) are never printed in the text.
All people, places and companies are made up. The generated files are committed, so nobody needs to run the generator.

**Levels.** Every question has a level, **medium** or **hard**; the **Difficulty** buttons of the graded tests pick Medium, Hard or Mixed
(the default: both, in turns). A document has about eight questions, four of each. (There are no easy questions in this library;
the grader and the key can carry a level "easy", but nothing in the page offers it.)

**Skills.** Every question also has a skill, and "All numbers" shows the score by skill:

| Skill | What it asks | Example (shape) |
|-------|--------------|-----------------|
| lookup | a fact with a look-alike next to it | who leads a team whose lead shares a surname with another lead; the budget in dollars, when the text also says euros |
| table | a cell of a table with two nearly equal codes | the unit price of KX-5230, when KX-5203 is in the table too |
| multi-hop | two facts from different sections | who leads the team that owns the delayed item; the room booked on the day of the audit |
| arithmetic | a result that is not in the text | quantity times unit price; a total of the items with one status; how much more one line costs than another; by what percentage one price is above another |
| compare | the cheapest, latest or largest of five items, or the highest total | which of five items has the lowest unit price |
| date-math | days between two dates, the weekday of a date, a deadline "three weeks after" | on what day of the week is the audit |
| update | a value stated early and revised later: "revised to", an erratum, a later email; the old value is the trap | the current due date; the total cost with the corrected unit price |
| negative | the document does not say; the answer is `NOT STATED` | the telephone number of a team lead; who leads a team the document never mentions |
| cross-document | a fact that needs two or more documents of a case | see Cases |

Every question has an `accept` list ("14 March 2027", "March 14, 2027", "2027-03-14"; "3,400 euros", "3400"), a `reject` list where a
trap is known (the superseded value, the look-alike, the other currency) and a `how` that says how the answer follows from the data.

**Cases.** `answers.json` also holds 16 cases: sets of 4 to 8 documents that belong to one project (plan, minutes, invoices, incident,
status report) and 3 or 4 questions each that only the whole set answers: the total of all invoices, who approved the change that caused
the incident, how many days after the audit the incident happened, who leads the team that owns the item the status report shows as
delayed. The key lists the facts each question needs and in which document; a test checks that no single document holds them all. The
documents window can pick a case. **Research team** on the documents of exactly one whole case sends the case's questions to every agent
(after its own) and to the manager, which writes one `Q<n>: value` line per question; those lines are graded as "cross-document" and
counted in the score. Picking only part of a case gives the ordinary team test.

A run is **graded** when the test has a right answer, every document it reads has a key and you typed no question of your own:

| Test | What each agent gets | Right means |
|------|----------------------|-------------|
| Agent swarm, Research team | one keyed question about its own document, of the difficulty picked (Mixed: medium and hard in turns); on a whole case the manager also gets the case's questions | the answer is right by its type, below |
| Many questions, one document | the document's keyed questions in turn (more agents than questions repeat them; the result says so) | the same |
| Follow-up question | a second keyed question about the same document, always **hard**, never the first | the same |
| Long document | the picked long sample, six questions: its update, its negative, then the difficulty's own; the first is asked twice (read, then from memory), the rest read only their question | the same |
| Classify | the eight kinds as labels; on **Hard**, the rating inside the kind (an incident's severity, an invoice's status paid, overdue or disputed, a project's size, a status report's green, amber or red ...), worked out from the figures and the rule printed in the document | the reply names exactly one label, and it is the right one |
| Extract | its document's six keyed fields, the last of them as revised | the reply is a JSON object and every field matches (all or nothing); the page also counts the fields right, for example "31 of 40 fields right" |

Every other test (summarize, compress, relay, chat, turns, burst, load, exactness) says "Not graded" and shows no score, and so does a run
with your own question, labels or documents.

**What every graded question tells the model** (one place, `FORMAT` in `live_parts/graded.py`): answer from the document only, with just the
value on one line; numbers as digits, with the unit if the question names one; dates as written in the document or as YYYY-MM-DD; a
duration as a number of days; and exactly `NOT STATED` when the document does not say.

**How grading decides** (`grading.py`: deterministic rules, no model as judge). Case, punctuation, quotes, thousands separators, a
currency sign or word, `.00` and common date forms do not matter. An accepted form must appear as whole words or numbers: `12500` is not `1250`.
The type of the question says more:

- **number, date, text:** an accepted form is in the final answer, and the answer names no second answer of the same kind ("1,250 or 1,520 euros", two dates, two weekdays) which is hedging and fails.
- **weekday:** the name or its first three letters.
- **days (a duration):** "17 days", "seventeen days" and "2 weeks and 3 days" are all 17 days (words up to ninety-nine, weeks and days in one phrase); "2 weeks" is 14. Only questions that ask for days read durations this way.
- **percent:** within 0.5 of the key (the key has one decimal). This tolerance is only for the questions that say "about" and ask for a percentage.
- **NOT STATED:** the reply says "not stated", and gives no value: no number, date or code, and at most 15 other words. An invented value fails. Synonyms ("not mentioned", "I don't know") do not pass; they get the reason "did not say NOT STATED".
- The thinking of a model (`<think>...</think>` or reasoning) is never graded, only the final answer; no final answer is "No answer".

A wrong answer says why: *gave the old value*, *gave the amount in the other currency*, *gave the look-alike* (when it contains a `reject`
form and no accepted one), *wrong arithmetic*, *wrong date math*, *gave two answers*, *invented an answer*, *did not say NOT STATED*,
*wrong answer* or *no answer*. The page shows **Correct** with the score and the percentage for each engine (blue Sonic LLM, orange
llama.cpp), under it the right/asked by level that were asked ("medium 18/24 · hard 9/20"), a plain line such as "Both engines got 61 of 64
right", a check, cross or dash in every agent tile, and, in the answers, a row per agent that opens to the document, the question, its level
and skill, the whole answer, the expected answer, the verdict and the reason, with a filter (All, Wrong only, No answer) and 50 rows to a
page. "All numbers" adds the score by skill. The history drawer shows the score and the hard score of every graded run. The run record
keeps `graded`, `correct`, `wrong`, `empty`, `by_level`, `by_skill`, `reasons` and, per agent, the question, expected answer, verdict and reason.

**What a score means, and does not.** It is one run on generated documents. It measures the **model**, not the engine: both engines
should score the same on the same model, and a difference between them is noise or a bug, not speed. The percentage is one reading, not
a benchmark. A right fact next to a wrong claim passes, a correct answer worded in a way the accepted forms do not cover fails (the
format rule in the prompt exists to prevent that), a name is checked for the full name only, and a verbose answer that shows its working
("40 units at 85 euros is 3,400 euros") fails as hedging because it names more than one number. A model that is told to answer with just
the value and does will score what it knows.

## How many agents fit

Next to the **Agents** field the page says "This Mac can run up to N agents at once on this test" for the loaded model and the test
you picked, and **Max** sets it (for the tests with one agent per document it keeps that many of the documents you picked, or of the library when none are picked). N comes from the engine's
own memory plan: the largest count whose context still holds a whole document and the answer, never more than 256. It follows the
documents you picked (the longest of them; with none picked, the biggest of the library), so long documents leave room for fewer agents
and "Short only" for more (there are 86 short samples, so 86 is the most agents of one-per-document tests on them; the many-questions
test can ask one short document with more). The defaults
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
