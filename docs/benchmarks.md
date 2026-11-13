# Benchmarks: Sonic LLM vs llama.cpp's llama-server

Measured on 2026-11-13 on one machine: Apple M5 Pro, 48 GB, macOS. Eight models from 2B to 20B parameters, fourteen ways of working, both engines loaded with the same GGUF file and given the same requests.

How to read these numbers:

- **One run per cell.** Differences below about 10% are inside the run-to-run noise of a warm laptop. Treat 1.0x-1.1x as "the same".
- llama-server is build 11669 with Metal, all layers on the GPU, one slot per agent and a unified cache (`-np N -kvu`).
- Every request asked for at most 128 answer tokens with thinking off. Documents are 2.5-6 KB text files.
- "Read" is prompt processing, "write" is generation. With several agents, write is the sum over agents.
- Memory is not in these tables: the two servers account for model weights differently, and we have not yet measured it in a way that is fair to both.
- Models too large to load twice on 48 GB (Qwen3.6 35B, Qwen3.8 27B) are not here; they have their own section in the README.

## Summary: how many times faster than llama-server (total time per mode)

| Mode | Gemma 4 E2B | Qwen3 4B | Gemma 3 4B | Phi 3.5 mini | Mistral 7B | Llama 3.1 8B | Gemma 4 12B | gpt-oss 20B |
|---|---|---|---|---|---|---|---|---|
| 16 questions about one document | **5.6x** | **4.8x** | **5.8x** | **3.8x** | **5.2x** | **5.0x** | **6.3x** | **3.7x** |
| Burst of 32 short requests | **3.9x** | **3.6x** | **3.2x** | **2.9x** | **2.1x** | **3.4x** | **3.0x** | **2.0x** |
| Chat while 8 agents work | **317x** | **264x** | **164x** | **128x** | did not finish | **634x** | **346x** | **919x** |
| Follow-up question, 8 agents | **2.1x** | 1.21x | **1.6x** | 1.02x | 1.11x | 1.13x | **1.9x** | **1.6x** |
| One long document | **1.9x** | 1.34x | 1.42x | **1.7x** | 1.12x | 1.34x | 1.21x | **1.5x** |
| One chat turn | 1.13x | 1.08x | 1.09x | 1.03x | 1.05x | 1.03x | 1.05x | 1.09x |
| Research team, 8 agents | **2.3x** | 1.26x | 1.20x | 1.03x | 1.08x | 1.09x | **2.2x** | **1.8x** |
| Swarm, 8 agents x 8 documents | **1.8x** | 1.22x | 1.22x | 1.23x | 1.02x | 1.12x | 1.09x | 1.47x |
| Classify 8 documents | **2.2x** | 1.16x | 1.16x | **1.5x** | 0.83x | 1.16x | 1.02x | **1.9x** |
| Extract from 8 documents | **1.9x** | 1.22x | 1.17x | 1.18x | 1.02x | 1.06x | 1.17x | 1.34x |
| Summarize 8 documents | **1.8x** | 1.15x | 1.10x | 1.27x | 1.18x | 1.00x | 1.27x | 1.35x |
| Compress 8 documents | **1.8x** | 1.21x | 1.11x | 1.18x | 0.94x | 0.98x | 1.34x | 1.44x |
| Relay, 8 agents | **1.9x** | 1.21x | 0.98x | 1.06x | 1.03x | 1.06x | 1.03x | 1.35x |
| Exactness run | **2.1x** | 1.30x | 1.09x | 1.28x | 0.87x | 1.06x | 0.99x | 1.47x |

Above 1.00x Sonic LLM is faster. Cells: 103 faster, 4 within 2%, 4 slower, 1 did not finish in 150 s, of 112.

## Every mode in detail

Each cell is `Sonic LLM vs llama-server (ratio)`; a ratio above 1.00x means Sonic LLM is better on that measure.

### 16 questions about one document

16 agents get the same document with different questions: the shared start is read once.

| Model | Total time, s | Read, tokens/s | Write, tokens/s | First token, s |
|---|---|---|---|---|
| Gemma 4 E2B | 2.05 vs 11.6 (**5.6x**) | n/a | 691 vs 124 (**5.6x**) | 0.27 vs 5.12 (**19.1x**) |
| Qwen3 4B | 3.70 vs 17.9 (**4.8x**) | n/a | 400 vs 83.4 (**4.8x**) | 0.73 vs 6.48 (**8.9x**) |
| Gemma 3 4B | 3.16 vs 18.2 (**5.8x**) | n/a | 605 vs 105 (**5.8x**) | 0.67 vs 7.75 (**11.6x**) |
| Phi 3.5 mini | 5.67 vs 21.6 (**3.8x**) | n/a | 331 vs 94.8 (**3.5x**) | 0.81 vs 6.67 (**8.2x**) |
| Mistral 7B | 6.89 vs 35.7 (**5.2x**) | n/a | 268 vs 49.1 (**5.5x**) | 1.94 vs 14.8 (**7.6x**) |
| Llama 3.1 8B | 5.71 vs 28.6 (**5.0x**) | n/a | 249 vs 50.2 (**5.0x**) | 1.40 vs 11.2 (**8.0x**) |
| Gemma 4 12B | 7.35 vs 46.2 (**6.3x**) | n/a | 168 vs 27.1 (**6.2x**) | 2.24 vs 22.2 (**9.9x**) |
| gpt-oss 20B | 6.19 vs 23.2 (**3.7x**) | n/a | 245 vs 57.5 (**4.3x**) | 0.59 vs 10.2 (**17.2x**) |

### Burst of 32 short requests

32 agents each get a short prompt and write at once.

| Model | Total time, s | Read, tokens/s | Write, tokens/s | First token, s |
|---|---|---|---|---|
| Gemma 4 E2B | 4.22 vs 16.3 (**3.9x**) | n/a | 1,954 vs 503 (**3.9x**) | 0.48 vs 0.49 (1.00x) |
| Qwen3 4B | 5.75 vs 20.9 (**3.6x**) | n/a | 1,432 vs 392 (**3.7x**) | 0.50 vs 0.63 (1.26x) |
| Gemma 3 4B | 6.06 vs 19.6 (**3.2x**) | n/a | 1,359 vs 418 (**3.3x**) | 0.34 vs 0.55 (**1.6x**) |
| Phi 3.5 mini | 6.66 vs 19.2 (**2.9x**) | n/a | 1,236 vs 427 (**2.9x**) | 0.48 vs 0.64 (1.35x) |
| Mistral 7B | 17.3 vs 35.6 (**2.1x**) | n/a | 475 vs 230 (**2.1x**) | 1.35 vs 1.29 (0.95x) |
| Llama 3.1 8B | 9.21 vs 31.4 (**3.4x**) | n/a | 892 vs 261 (**3.4x**) | 0.29 vs 1.76 (**6.0x**) |
| Gemma 4 12B | 16.9 vs 50.2 (**3.0x**) | n/a | 486 vs 163 (**3.0x**) | 1.15 vs 1.95 (**1.7x**) |
| gpt-oss 20B | 13.3 vs 25.9 (**2.0x**) | n/a | 618 vs 316 (**2.0x**) | 0.29 vs 2.03 (**7.0x**) |

### Chat while 8 agents work

The chat question alone, then again while 8 agents work. The time shown is the wait for the chat's first token while they work.

| Model | Total time, s | Read, tokens/s | Write, tokens/s | First token, s |
|---|---|---|---|---|
| Gemma 4 E2B | 0.06 vs 20.5 (**317x**) | n/a | n/a | 0.06 vs 20.5 (**317x**) |
| Qwen3 4B | 0.12 vs 32.5 (**264x**) | n/a | n/a | 0.12 vs 32.5 (**264x**) |
| Gemma 3 4B | 0.14 vs 22.6 (**164x**) | n/a | n/a | 0.14 vs 22.6 (**164x**) |
| Phi 3.5 mini | 0.14 vs 17.7 (**128x**) | n/a | n/a | 0.14 vs 17.7 (**128x**) |
| Mistral 7B | no data | n/a | n/a | no data |
| Llama 3.1 8B | 0.08 vs 51.7 (**634x**) | n/a | n/a | 0.08 vs 51.7 (**634x**) |
| Gemma 4 12B | 0.11 vs 37.5 (**346x**) | n/a | n/a | 0.11 vs 37.5 (**346x**) |
| gpt-oss 20B | 0.05 vs 43.1 (**919x**) | n/a | n/a | 0.05 vs 43.1 (**919x**) |

### Follow-up question, 8 agents

A second question to 8 agents that already read a document each: only the new question should be read.

| Model | Total time, s | Read, tokens/s | Write, tokens/s | First token, s |
|---|---|---|---|---|
| Gemma 4 E2B | 1.42 vs 2.96 (**2.1x**) | n/a | 523 vs 401 (1.30x) | 0.06 vs 1.23 (**19.1x**) |
| Qwen3 4B | 2.36 vs 2.85 (1.21x) | n/a | 326 vs 282 (1.15x) | 0.12 vs 0.24 (**2.1x**) |
| Gemma 3 4B | 2.91 vs 4.80 (**1.6x**) | n/a | 363 vs 350 (1.04x) | 0.13 vs 1.99 (**14.9x**) |
| Phi 3.5 mini | 4.28 vs 4.37 (1.02x) | n/a | 242 vs 227 (1.06x) | 0.16 vs 0.32 (**2.0x**) |
| Mistral 7B | 4.47 vs 4.95 (1.11x) | n/a | 228 vs 219 (1.04x) | 0.15 vs 0.27 (**1.8x**) |
| Llama 3.1 8B | 4.51 vs 5.11 (1.13x) | n/a | 217 vs 195 (1.11x) | 0.16 vs 0.27 (**1.7x**) |
| Gemma 4 12B | 7.21 vs 13.5 (**1.9x**) | n/a | 112 vs 98.2 (1.14x) | 0.99 vs 6.45 (**6.5x**) |
| gpt-oss 20B | 4.56 vs 7.29 (**1.6x**) | n/a | 158 vs 152 (1.04x) | 0.13 vs 2.69 (**20x**) |

### One long document

8 files joined into one long prompt, asked twice: read speed, then the second ask from memory.

| Model | Total time, s | Read, tokens/s | Write, tokens/s | First token, s |
|---|---|---|---|---|
| Gemma 4 E2B | 3.80 vs 7.31 (**1.9x**) | 8,811 vs 3,416 (**2.6x**) | 109 vs 103 (1.06x) | 1.24 vs 3.19 (**2.6x**) |
| Qwen3 4B | 8.52 vs 11.4 (1.34x) | 1,971 vs 1,548 (1.27x) | 62.2 vs 63.1 (0.99x) | 5.04 vs 6.42 (1.27x) |
| Gemma 3 4B | 6.78 vs 9.64 (1.42x) | 3,075 vs 2,375 (1.29x) | 80.9 vs 82.6 (0.98x) | 3.54 vs 4.59 (1.29x) |
| Phi 3.5 mini | 11.4 vs 19.6 (**1.7x**) | 1,627 vs 947 (**1.7x**) | 39.5 vs 36.3 (1.09x) | 7.53 vs 12.9 (**1.7x**) |
| Mistral 7B | 20.4 vs 22.9 (1.12x) | 850 vs 797 (1.07x) | 36.8 vs 40.6 (0.91x) | 14.1 vs 15.0 (1.07x) |
| Llama 3.1 8B | 15.1 vs 20.3 (1.34x) | 1,088 vs 874 (1.24x) | 43.0 vs 42.8 (1.00x) | 8.72 vs 10.9 (1.24x) |
| Gemma 4 12B | 23.5 vs 28.5 (1.21x) | 734 vs 749 (0.98x) | 30.5 vs 29.3 (1.04x) | 14.9 vs 14.6 (0.98x) |
| gpt-oss 20B | 8.04 vs 12.2 (**1.5x**) | 2,425 vs 1,466 (**1.7x**) | 76.9 vs 75.0 (1.02x) | 3.94 vs 6.51 (**1.7x**) |

### One chat turn

A single short chat message, nothing else running.

| Model | Total time, s | Read, tokens/s | Write, tokens/s | First token, s |
|---|---|---|---|---|
| Gemma 4 E2B | 0.98 vs 1.11 (1.13x) | n/a | 135 vs 122 (1.10x) | 0.03 vs 0.07 (**2.2x**) |
| Qwen3 4B | 1.30 vs 1.41 (1.08x) | n/a | 100 vs 94.1 (1.06x) | 0.03 vs 0.06 (**1.7x**) |
| Gemma 3 4B | 1.39 vs 1.51 (1.09x) | n/a | 95.1 vs 89.1 (1.07x) | 0.05 vs 0.08 (**1.6x**) |
| Phi 3.5 mini | 1.26 vs 1.30 (1.03x) | n/a | 106 vs 104 (1.02x) | 0.05 vs 0.08 (**1.5x**) |
| Mistral 7B | 2.33 vs 2.44 (1.05x) | n/a | 55.8 vs 54.3 (1.03x) | 0.05 vs 0.10 (**2.1x**) |
| Llama 3.1 8B | 2.32 vs 2.38 (1.03x) | n/a | 56.6 vs 55.6 (1.02x) | 0.08 vs 0.10 (1.26x) |
| Gemma 4 12B | 3.83 vs 4.03 (1.05x) | n/a | 34.7 vs 33.2 (1.04x) | 0.16 vs 0.21 (1.27x) |
| gpt-oss 20B | 1.51 vs 1.64 (1.09x) | n/a | 92.5 vs 83.9 (1.10x) | 0.14 vs 0.13 (0.91x) |

### Research team, 8 agents

One agent per document answers the questions from it; a manager merges the reports.

| Model | Total time, s | Read, tokens/s | Write, tokens/s | First token, s |
|---|---|---|---|---|
| Gemma 4 E2B | 1.53 vs 3.59 (**2.3x**) | 9,036 vs 3,646 (**2.5x**) | 160 vs 103 (**1.6x**) | 0.73 vs 2.82 (**3.8x**) |
| Qwen3 4B | 4.15 vs 5.23 (1.26x) | 2,802 vs 2,230 (1.26x) | 50.9 vs 35.6 (1.43x) | 2.99 vs 3.20 (1.07x) |
| Gemma 3 4B | 4.61 vs 5.52 (1.20x) | 2,737 vs 2,331 (1.17x) | 53.0 vs 66.2 (0.80x) | 2.48 vs 4.22 (**1.7x**) |
| Phi 3.5 mini | 9.28 vs 9.52 (1.03x) | 2,466 vs 1,758 (1.40x) | 141 vs 54.3 (**2.6x**) | 4.00 vs 4.89 (1.22x) |
| Mistral 7B | 13.0 vs 14.1 (1.08x) | 1,121 vs 1,020 (1.10x) | 29.0 vs 29.8 (0.97x) | 7.07 vs 7.03 (0.99x) |
| Llama 3.1 8B | 9.36 vs 10.2 (1.09x) | 1,170 vs 1,110 (1.05x) | 20.7 vs 18.2 (1.14x) | 5.66 vs 6.26 (1.11x) |
| Gemma 4 12B | 15.2 vs 33.6 (**2.2x**) | 795 vs 360 (**2.2x**) | 11.9 vs 16.3 (0.73x) | 8.86 vs 29.2 (**3.3x**) |
| gpt-oss 20B | 4.10 vs 7.20 (**1.8x**) | 3,209 vs 1,648 (**1.9x**) | 46.7 vs 37.3 (1.25x) | 2.52 vs 5.93 (**2.4x**) |

### Swarm, 8 agents x 8 documents

One agent per document reads it at the same time, then one answer merges their notes.

| Model | Total time, s | Read, tokens/s | Write, tokens/s | First token, s |
|---|---|---|---|---|
| Gemma 4 E2B | 2.68 vs 4.71 (**1.8x**) | 9,394 vs 3,809 (**2.5x**) | 359 vs 258 (1.39x) | 0.71 vs 2.55 (**3.6x**) |
| Qwen3 4B | 5.73 vs 6.97 (1.22x) | 3,021 vs 2,465 (1.23x) | 199 vs 144 (1.38x) | 2.75 vs 2.79 (1.01x) |
| Gemma 3 4B | 6.62 vs 8.05 (1.22x) | 2,661 vs 2,167 (1.23x) | 187 vs 194 (0.96x) | 2.62 vs 4.60 (**1.8x**) |
| Phi 3.5 mini | 8.64 vs 10.6 (1.23x) | 2,751 vs 2,003 (1.37x) | 149 vs 119 (1.25x) | 3.51 vs 4.02 (1.14x) |
| Mistral 7B | 13.9 vs 14.2 (1.02x) | 1,274 vs 1,257 (1.01x) | 89.8 vs 92.4 (0.97x) | 6.03 vs 6.48 (1.08x) |
| Llama 3.1 8B | 11.7 vs 13.2 (1.12x) | 1,322 vs 1,185 (1.12x) | 104 vs 90.5 (1.15x) | 4.53 vs 6.06 (1.34x) |
| Gemma 4 12B | 20.6 vs 22.4 (1.09x) | 809 vs 759 (1.07x) | 56.9 vs 65.2 (0.87x) | 8.41 vs 13.0 (**1.6x**) |
| gpt-oss 20B | 8.48 vs 12.4 (1.47x) | 3,081 vs 1,651 (**1.9x**) | 143 vs 117 (1.23x) | 2.58 vs 5.55 (**2.1x**) |

### Classify 8 documents

One agent per document puts it under one of the given labels.

| Model | Total time, s | Read, tokens/s | Write, tokens/s | First token, s |
|---|---|---|---|---|
| Gemma 4 E2B | 1.33 vs 2.89 (**2.2x**) | 8,556 vs 3,952 (**2.2x**) | 13.8 vs 15.6 (0.88x) | 0.77 vs 2.47 (**3.2x**) |
| Qwen3 4B | 3.81 vs 4.41 (1.16x) | 2,725 vs 2,353 (1.16x) | 6.40 vs 5.80 (1.10x) | 2.94 vs 2.89 (0.98x) |
| Gemma 3 4B | 3.82 vs 4.44 (1.16x) | 2,968 vs 2,566 (1.16x) | 6.50 vs 12.1 (0.54x) | 2.39 vs 3.87 (**1.6x**) |
| Phi 3.5 mini | 4.96 vs 7.61 (**1.5x**) | 2,583 vs 2,033 (1.27x) | 6.70 vs 23.3 (0.29x) | 3.72 vs 3.86 (1.04x) |
| Mistral 7B | 15.6 vs 13.0 (0.83x) | 923 vs 1,072 (0.86x) | 12.2 vs 10.6 (1.15x) | 8.26 vs 6.65 (0.80x) |
| Llama 3.1 8B | 8.67 vs 10.0 (1.16x) | 1,162 vs 1,004 (1.16x) | 2.20 vs 2.60 (0.85x) | 5.12 vs 6.71 (1.31x) |
| Gemma 4 12B | 14.0 vs 14.3 (1.02x) | 811 vs 793 (1.02x) | 1.20 vs 2.90 (0.41x) | 8.53 vs 12.6 (1.47x) |
| gpt-oss 20B | 3.28 vs 6.11 (**1.9x**) | 3,198 vs 1,703 (**1.9x**) | 5.40 vs 6.90 (0.78x) | 2.53 vs 5.40 (**2.1x**) |

### Extract from 8 documents

One agent per document pulls named fields into a JSON object.

| Model | Total time, s | Read, tokens/s | Write, tokens/s | First token, s |
|---|---|---|---|---|
| Gemma 4 E2B | 2.81 vs 5.21 (**1.9x**) | 8,755 vs 3,691 (**2.4x**) | 400 vs 270 (1.48x) | 0.75 vs 2.63 (**3.5x**) |
| Qwen3 4B | 6.12 vs 7.45 (1.22x) | 2,718 vs 2,262 (1.20x) | 208 vs 137 (**1.5x**) | 2.85 vs 3.11 (1.09x) |
| Gemma 3 4B | 6.55 vs 7.66 (1.17x) | 2,660 vs 2,297 (1.16x) | 180 vs 192 (0.94x) | 2.55 vs 4.26 (**1.7x**) |
| Phi 3.5 mini | 9.37 vs 11.1 (1.18x) | 2,388 vs 1,869 (1.28x) | 141 vs 116 (1.21x) | 4.10 vs 4.43 (1.08x) |
| Mistral 7B | 17.1 vs 17.4 (1.02x) | 973 vs 1,026 (0.95x) | 74.5 vs 69.3 (1.08x) | 8.07 vs 7.15 (0.89x) |
| Llama 3.1 8B | 13.2 vs 14.0 (1.06x) | 1,063 vs 1,133 (0.94x) | 76.6 vs 66.1 (1.16x) | 5.98 vs 6.15 (1.03x) |
| Gemma 4 12B | 20.2 vs 23.7 (1.17x) | 772 vs 685 (1.13x) | 45.6 vs 47.6 (0.96x) | 9.12 vs 14.0 (**1.5x**) |
| gpt-oss 20B | 7.54 vs 10.1 (1.34x) | 2,802 vs 1,645 (**1.7x**) | 93.7 vs 86.6 (1.08x) | 2.89 vs 5.88 (**2.0x**) |

### Summarize 8 documents

One agent per document summarizes it.

| Model | Total time, s | Read, tokens/s | Write, tokens/s | First token, s |
|---|---|---|---|---|
| Gemma 4 E2B | 2.51 vs 4.49 (**1.8x**) | 7,930 vs 3,692 (**2.1x**) | 292 vs 222 (1.31x) | 0.84 vs 2.68 (**3.2x**) |
| Qwen3 4B | 6.13 vs 7.07 (1.15x) | 2,504 vs 2,221 (1.13x) | 166 vs 122 (1.36x) | 3.24 vs 2.75 (0.85x) |
| Gemma 3 4B | 6.54 vs 7.19 (1.10x) | 2,496 vs 2,251 (1.11x) | 135 vs 163 (0.83x) | 2.75 vs 4.38 (**1.6x**) |
| Phi 3.5 mini | 9.20 vs 11.7 (1.27x) | 2,225 vs 1,703 (1.31x) | 116 vs 120 (0.97x) | 4.36 vs 4.64 (1.06x) |
| Mistral 7B | 17.4 vs 20.6 (1.18x) | 943 vs 882 (1.07x) | 61.7 vs 59.9 (1.03x) | 7.98 vs 8.01 (1.00x) |
| Llama 3.1 8B | 12.2 vs 12.2 (1.00x) | 1,116 vs 1,177 (0.95x) | 83.0 vs 76.6 (1.08x) | 6.33 vs 6.09 (0.96x) |
| Gemma 4 12B | 21.5 vs 27.4 (1.27x) | 680 vs 515 (1.32x) | 41.5 vs 47.9 (0.87x) | 10.5 vs 19.3 (**1.8x**) |
| gpt-oss 20B | 8.92 vs 12.0 (1.35x) | 2,572 vs 1,648 (**1.6x**) | 130 vs 112 (1.16x) | 3.17 vs 5.46 (**1.7x**) |

### Compress 8 documents

One agent per document rewrites it shorter, keeping names, numbers and decisions.

| Model | Total time, s | Read, tokens/s | Write, tokens/s | First token, s |
|---|---|---|---|---|
| Gemma 4 E2B | 2.99 vs 5.27 (**1.8x**) | 8,341 vs 3,718 (**2.2x**) | 385 vs 277 (1.39x) | 0.76 vs 2.69 (**3.5x**) |
| Qwen3 4B | 6.38 vs 7.74 (1.21x) | 2,607 vs 2,230 (1.17x) | 212 vs 152 (1.39x) | 3.18 vs 3.15 (0.99x) |
| Gemma 3 4B | 7.01 vs 7.78 (1.11x) | 2,437 vs 2,253 (1.08x) | 180 vs 196 (0.92x) | 2.94 vs 4.46 (**1.5x**) |
| Phi 3.5 mini | 9.90 vs 11.7 (1.18x) | 2,198 vs 1,711 (1.28x) | 127 vs 121 (1.05x) | 4.15 vs 5.17 (1.25x) |
| Mistral 7B | 19.8 vs 18.5 (0.94x) | 868 vs 986 (0.88x) | 66.5 vs 72.6 (0.92x) | 9.03 vs 7.46 (0.83x) |
| Llama 3.1 8B | 13.8 vs 13.5 (0.98x) | 1,039 vs 1,154 (0.90x) | 91.4 vs 87.7 (1.04x) | 6.40 vs 6.18 (0.97x) |
| Gemma 4 12B | 22.6 vs 30.3 (1.34x) | 709 vs 512 (1.38x) | 59.0 vs 65.6 (0.90x) | 10.0 vs 19.6 (**2.0x**) |
| gpt-oss 20B | 8.74 vs 12.6 (1.44x) | 2,721 vs 1,651 (**1.6x**) | 144 vs 114 (1.26x) | 2.99 vs 5.54 (**1.8x**) |

### Relay, 8 agents

A plan of small tasks; each lap a fresh agent with an empty memory takes the next one.

| Model | Total time, s | Read, tokens/s | Write, tokens/s | First token, s |
|---|---|---|---|---|
| Gemma 4 E2B | 2.58 vs 4.98 (**1.9x**) | n/a | 187 vs 132 (1.42x) | 0.91 vs 3.03 (**3.3x**) |
| Qwen3 4B | 6.58 vs 7.93 (1.21x) | n/a | 168 vs 127 (1.32x) | 3.26 vs 3.03 (0.93x) |
| Gemma 3 4B | 7.69 vs 7.55 (0.98x) | n/a | 120 vs 138 (0.87x) | 3.64 vs 4.92 (1.35x) |
| Phi 3.5 mini | 11.3 vs 11.9 (1.06x) | n/a | 120 vs 92.0 (1.30x) | 5.36 vs 4.85 (0.90x) |
| Mistral 7B | 20.8 vs 21.5 (1.03x) | n/a | 63.7 vs 64.4 (0.99x) | 10.4 vs 7.06 (0.68x) |
| Llama 3.1 8B | 13.8 vs 14.7 (1.06x) | n/a | 79.3 vs 66.6 (1.19x) | 6.61 vs 6.09 (0.92x) |
| Gemma 4 12B | 24.7 vs 25.4 (1.03x) | n/a | 50.8 vs 54.4 (0.93x) | 12.4 vs 14.7 (1.19x) |
| gpt-oss 20B | 9.57 vs 12.9 (1.35x) | n/a | 127 vs 102 (1.24x) | 3.27 vs 6.35 (**1.9x**) |

### Exactness run

The same question alone and next to 7 other agents; the tokens must be identical.

| Model | Total time, s | Read, tokens/s | Write, tokens/s | First token, s |
|---|---|---|---|---|
| Gemma 4 E2B | 10.8 vs 23.0 (**2.1x**) | n/a | same tokens on both | n/a |
| Qwen3 4B | 29.2 vs 38.0 (1.30x) | n/a | same tokens on both | n/a |
| Gemma 3 4B | 22.7 vs 24.7 (1.09x) | n/a | same tokens on both | n/a |
| Phi 3.5 mini | 18.7 vs 24.0 (1.28x) | n/a | same tokens on both | n/a |
| Mistral 7B | 75.0 vs 65.0 (0.87x) | n/a | same tokens on both | n/a |
| Llama 3.1 8B | 51.5 vs 54.8 (1.06x) | n/a | same tokens on both | n/a |
| Gemma 4 12B | 42.9 vs 42.4 (0.99x) | n/a | same tokens on both | n/a |
| gpt-oss 20B | 33.6 vs 49.2 (1.47x) | n/a | same tokens on both | n/a |

