# ggml-b11669

Official llama.cpp b11669 ggml 0.27.0 (decisions.md D028). Nothing here is rebuilt or modified.

- `include/`: headers from `research/inference_runtime/vendor/llama-src/ggml/include` (tarball 620d222). Committed.
- `lib/`: the dylibs from `research/inference_runtime/vendor/llama-bin`. Not committed; run `./fetch.sh`.
- `SHA256SUMS`: hashes of the real dylib files and the headers; `fetch.sh` verifies them.

`libggml-rpc` is copied although `sllm-core` does not use it: `libggml.0.dylib` (the backend registry) has a
load command for it, so dyld fails without it.

Task 4a adds `libllama.0.7.0.dylib` (the llama.cpp b11669 library, `lib/`) and the headers `include/llama.h` and
`include/ggml-opt.h` (from `llama-src/include` and `llama-src/ggml/include`) for the `llama-ref` crate.
`libllama-common` is not needed and is not copied.
