# Compatibility with llama.cpp's llama-server

Sonic LLM aims to be a drop-in for `llama-server`: the same command-line spellings, the same HTTP paths, the same JSON in and out. This page says exactly how far that goes today.

It was checked by sending 40 identical requests to `llama-server` (build 11669) and to `sllm-server` with the same model (Qwen3 4B, Q4_0) and comparing status codes and the shape of every answer, field by field. The probe script is `conf.py` in the results folder of the project; rerun it after any change to the interface.

| Result | Probes | Share |
|---|---|---|
| Same status and same fields | 19 | 48% |
| Both refuse, same status | 6 | 15% |
| Same fields except `reasoning_content` (see "Thinking") | 10 | 25% |
| Not supported here | 4 | 10% |
| Different on purpose (`/metrics`) | 1 | 2% |

## Command line

| llama-server | Sonic LLM | Note |
|---|---|---|
| `-m`, `--model` | same | |
| `-c`, `--ctx-size` | same | context per sequence |
| `-t`, `--threads` | same | |
| `--host`, `--port` | same | |
| `-a`, `--alias` | same | the first name is the model id |
| `--api-key`, `--api-key-file` | same | `/health` stays public |
| `-ub`, `--ubatch-size` | same | |
| `--reasoning-format` | `auto`, `deepseek`, `none` | |
| `-np`, `--parallel` | accepted | Sizes the cache only together with `--auto-mode`. It does not cap how many requests run and does not split the context between slots. |
| `-ngl`, `--jinja`, `-fa`, `-b`, `--mlock`, `--no-mmap`, `--mmap`, `-kvu`, `--slots`, `--metrics`, `-cb` | accepted and ignored, with a note on stderr | what they switch on is always on, or has no meaning here |
| anything else | error | an unknown flag is never ignored |

## Endpoints

| Path | Status |
|---|---|
| `GET /health`, `/v1/health` | same |
| `GET /props` | same keys, except the nine `chat_template_caps` flags |
| `GET /slots` | same keys; a slot here is a scheduler lane slot |
| `GET /v1/models`, `/models` | same, plus a few extra fields |
| `GET /lora-adapters` | always `[]` |
| `POST /completion`, `/completions` | same, including streaming, `grammar`, `json_schema`, `stop`, `n_probs` (up to 2) |
| `POST /v1/completions` | same, including `timings` and `system_fingerprint` |
| `POST /v1/chat/completions`, `/chat/completions` | same, including streaming, usage in the stream, tools, `response_format` (`json_object`, `json_schema`), `logprobs` (not streamed, `top_logprobs` up to 2) |
| `POST /tokenize`, `/detokenize`, `/apply-template` | same, including `with_pieces` |
| `POST /v1/messages`, `/v1/messages/count_tokens` | same (Anthropic interface) |
| `POST /embedding`, `/v1/embeddings`, `/v1/rerank` and their aliases | 501 "not supported", the answer llama-server gives when started without those modes |
| `GET /metrics` | **different on purpose:** JSON, not Prometheus text |
| `POST /infill` | not supported (404) |
| `POST /v1/responses` | not supported (404) |
| `POST /slots/{id}?action=save\|restore\|erase` | not supported (404); the agent interface's suspend and wake do this job |
| image input | not supported |

## Differences in behaviour you should know

**Sampling is greedy.** The engine always picks the most likely token. `temperature`, `top_k`, `top_p` and `seed` are accepted so that existing clients keep working, and they have no effect: the same request always gives the same answer. `repeat_penalty` and the other penalties are refused with an error. If your use needs varied answers, this is the main gap today.

**Thinking is off unless you ask.** llama-server lets a thinking model think by default; Sonic LLM renders the chat template with thinking off. To turn it on send `"chat_template_kwargs": {"enable_thinking": true}`. The thinking text then arrives in `message.reasoning_content` (and `delta.reasoning_content` when streaming), as in llama-server. This is why ten probes differ by exactly that one field.

**Unknown request fields are refused.** llama-server ignores a field it does not know. Sonic LLM answers 400 and names the field, so a setting you rely on is never dropped without notice. The sampling fields above are the one exception.

**Schemas the grammar cannot enforce are refused.** `pattern`, `allOf` and integer `minimum`/`maximum` in a JSON schema give a 400, as do a schema together with `tools` and a schema together with a grammar.

**Some invalid values answer 422** where llama-server answers 400 (for example `n_predict` below -1). A missing or mistyped field answers 400 on both.

**`/tokenize` does not parse special tokens by default** (`parse_special` is false; llama-server's default is true).

## Extra, not in llama-server

`lane` (`interactive` or `background`) and `level` (`sonic`, `fast`, `normal`, `deep`) on chat requests, the agent interface under `/v1/agents`, the helper endpoints under `/v1/helper`, and `/caps`, which lists every request field the server accepts and refuses.
