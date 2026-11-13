# ggml-rowexact-b11669

The OFFICIAL llama.cpp b11669 Metal backend (tarball 620d222, the pin of `../ggml-b11669`) plus the patches in `patches/`: an
opt-in **row-exact mode** (0010, most of this README), the qmm kernels (0001-0003) and partial stream copies (0011, below).
Every other dylib is the byte-identical official one. Two kernel sources gain SLLM kernels
and change no official kernel's code path (`mul_mv`: a rows-per-threadgroup loop that is off in every official pipeline;
`mul_mv_mma`: the one-row kernel below); `build.sh` checks that the other 34 embedded kernel sources equal the official
dylib's. With the switch off every decision in `ggml-metal-ops.cpp` is the official one, so the default behaviour is the
official lib's.

## What the mode guarantees

With the switch on, a token row's logits do not depend on the other rows of its step as long as the step's graph has at
most `maxn` (16) rows and the KV cache is one stream per sequence (`kv_unified = false`; with a unified pool the cell
layout of other sequences changes the flash-attention block grouping and nothing helps). Measured with
`engine/sllm-server/examples/spec_probe.rs`: 5-row and 3-row verify rows are 100% bitwise equal to one-row decode, also
in steps where sequences hold different row counts (libllama then splits them into equal micro-batches).

- weight mat-muls (`ggml_metal_op_mul_mat_rowexact`): ggml's few-row MMA kernel (`mul_mv_mma`) for every row count from 2
  to `maxn` (stock: from 2..6 rows by type, mat-vec for 1); types it cannot take (Q8_0, K not a multiple of its step) use the
  `mul_mv_ext` kernel with fixed nsg/nxpsg, or `mul_mv` when K is not a multiple of 128. Above `maxn` rows: stock paths.
  With `maxn` 32 (`--many-agents`) a type qmm does not take (Gemma 4's Q4_K output layer) used to fall to `mul_mv_ext`
  above 16 rows: 10.5 ms per 32-row step against 4.5 ms at 128 rows. On MMA, Gemma 4 E2B writes 32 agents at 1,453 tok/s
  instead of 1,093, same tokens (2026-11-04).
- one Q4_0 row (`kernel_sllm_mme_q4_0_f32`): the MMA kernel's exact arithmetic at mat-vec speed. On Apple GPUs
  `simdgroup_multiply_accumulate` is a sequential fma chain over its 8 k onto the accumulator (measured on 2.56M random
  outputs, 0 mismatches), so the MMA result of a row is a fixed sequence of IEEE operations: per block a 32-step fma chain
  in the MMA's step order, per K slice (the MMA's nsg for the weight's shape) `fma(d, block, acc)` in block order, then the
  slices summed in order. Same bits as the MMA kernel at any row count; Qwen3-4B decode alone 10.75 ms (official 10.52,
  MMA 12.56). `SLLM_ROW_EXACT_MM=mma` keeps the MMA kernel for one row.
- one token per sequence against a 2-D weight (`[d, 1, n]`, recurrent and hybrid layers): n rows of one MMA slice, so
  the weights are read once, not once per sequence (Qwen3.5-4B, 8 sequences: 23.2 ms, official 25.4, per slice 28.4).
- flash attention: the vec kernel for up to 16 query rows (stock: below 20), baseline config (Q = 1) and a constant `nsg`
  (default 2) instead of ones picked from the number of rows and the K/V view length.
- not covered: mixture-of-experts mat-muls (`mul_mat_id`); the engine turns speculation off for them.

## Use

- Build: `./build.sh` (after `../ggml-b11669/fetch.sh`; ~5 s, bit-for-bit reproducible), then build the server with
  `SLLM_GGML_METAL=rowexact cargo build --release -p sllm-server`.
- Switch: `SLLM_ROW_EXACT=1` in the environment, or `ggml_backend_metal_sllm_set_row_exact(int)` (exported; the server
  calls it for `--row-exact on` and `--spec auto|on`).
- Experiments: `SLLM_ROW_EXACT_MAXN`, `SLLM_ROW_EXACT_NSG`, `SLLM_ROW_EXACT_PARTS` (1 mat-muls, 2 attention),
  `SLLM_ROW_EXACT_MM=mv|ext|mvr|mma`, `SLLM_MME_NR0` (weight rows per simdgroup, 1), `SLLM_MME_SG` (simdgroups per
  threadgroup, 2).
- `apply_edits.py` + `mkpatch.sh` regenerate `patches/0010-row-exact.patch` from the pristine tarball (on top of 0001-0003).

## Partial stream copies (`patches/0011-sllm-copy-rows.patch`)

With per-sequence KV streams (`kv_unified = false`) libllama's `seq_cp` between streams queues a copy of each layer's whole
stream view (`[row, kv_size]`) and runs it at the start of the next `llama_decode` of that context, through
`ggml_backend_tensor_copy` (a memcpy for shared Metal buffers): at 40k cells gigabytes, on the caller's thread.
`ggml_backend_metal_sllm_set_copy_rows(n)` (exported) makes such a copy stop after `n` rows; 0, the default, is the official
whole copy, so a lib without a caller of it behaves as the official one. The caller promises that every row the copied streams use is
below `n`; the rows past it hold stale data (finite: only rows the model wrote, or zeros) that the cell metadata masks, and a masked key
adds exactly 0 to the attention sum, so no result depends on them.

- The hint is **thread-local**: the thread that calls `llama_decode` sets it right before and clears it right after (the server
  does, `ffi::CopyHint`). A copy of another context applies inside that context's own `llama_decode`, so it is never inside
  the window; a context on another thread never sees it.
- It cuts only a copy between two views of one cache tensor (`src->view_src == dst->view_src`) with the same type and shape,
  2-D and contiguous (libllama's `k_stream` / `v_stream`). Every other copy, and a copy with `n` at or past the stream's rows,
  moves all of its bytes. A transposed V cache (flash attention off) has no cells as rows: the server sets the hint only
  with flash attention on.
- Host code only (`ggml-metal-device.m`): `verify_embed.py` still finds every kernel source unchanged.
- Server side: `SLLM_COPY_PARTIAL=0` turns it off; `/metrics` `stream_runs` counts `copies_partial`, `copy_rows`, `copies_whole`, and
  from `ggml_backend_metal_sllm_copy_stats` the copies ggml really cut (`copies_cut`) and `copy_bytes_saved`.

`0011` is kept by hand (it is a small diff of one function against the tree after 0001-0003 and 0010): to change it, apply
the earlier patches to a pristine tree the way `build.sh` does, edit `ggml-metal-device.m`, and `diff -ruN` against a copy.
`mkpatch.sh` regenerates 0010 and then builds with all of them, which checks that 0011 still applies.

## Prefill attention (`patches/0012-sllm-pfa-port.patch`)

The GQA-packed prefill attention of `../sllm-metal-b11669` (its patches 0007 and 0011: `kernels/prefill.metal`, `ggml-metal-pfa.{h,m}`),
ported here without its hint: libllama's graph never set one, so the kernel never ran. It now takes every FLASH_ATTN_EXT that
ggml itself would run on its tensor-API kernel (f16 K/V, head size 128 or 256, 2 or 4 query heads per KV head, at least 20
rows), and its init self-test checks it bitwise against that kernel, or it stays off. `SLLM_PREFILL_CASCADE=0` turns it off,
`SLLM_CASCADE_TRACE=1` prints every dispatch and why an op was not taken. Qwen3-4B reads a 16k-token document 36% faster with
the same tokens. Gemma 4 (head size 512, 8 query heads per KV head) is not bitwise ggml's there and ran slower, so it is left out.
verify_embed.py accepts the new kernel kind `prefill`.

