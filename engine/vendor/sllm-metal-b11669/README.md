# sllm-metal-b11669

SLLM's patched `libggml-metal` (decisions.md D042): the official llama.cpp b11669 / ggml 0.27.0 Metal backend plus
two SLLM kernels and the code that dispatches them: the batched quantized matmul (`qmm`, D048, from the phase 2 qmm
spike), the shared-prefix decode attention (`cascade`, from the phase 2 cascade spike, `bench/phase2-cascade`) and the
GQA-packed prefill attention (`prefill`, from the phase 2 prefillattn spike, `bench/phase2-prefillattn`).
sllm-core links this directory by default. Everything except `libggml-metal` is a byte-identical copy of the
official binary in `../ggml-b11669/lib`, which stays untouched and is what the llama.cpp baselines use.

- `build.sh`: the reproducible build (no cmake here). It extracts `ggml/` from the pinned source tarball
  (`research/inference_runtime/vendor/llama-source.tar.gz`, commit 620d222, sha256 checked), applies `patches/`,
  embeds the kernel sources the way the CMake `GGML_METAL_EMBED_LIBRARY` step does, compiles with release flags,
  links against the official `libggml-base`, copies the official set beside it and writes `MANIFEST.json`.
  Run `../ggml-b11669/fetch.sh` first. Takes about 5 s. The output is bit-for-bit reproducible.
- `verify_embed.py`: run by `build.sh`. Checks that all 36 official kernel sources embedded in the patched dylib are
  byte-identical to the official dylib's, and that the only extra ones are `cascade`, `prefill` and `qmm`.
- `patches/0001-sllm-qmm-kernel.patch`: new files only: `kernels/qmm.metal` (the kernel and the split-K reduce) and
  `ggml-metal-qmm.{h,m}` (dispatch table, plan, encode, the init self-test, the exported status).
- `patches/0002-sllm-qmm-hooks.patch`: the hooks into ggml-metal (about 40 lines):
  - `CMakeLists.txt`: `kernels/qmm.metal` in the kernel source list and `ggml-metal-qmm.m` in the sources;
  - `ggml-metal-device.m`: the `QMM` library kind; a failed compile of `qmm.metal` disables only qmm; the
    self-test call after the library is built;
  - `ggml-metal-ops.cpp`: the dispatch branch at the top of `ggml_metal_op_mul_mat` (after the FWHT check);
  - `ggml-metal.cpp`: the split-K scratch in `get_alloc_size`.
- `patches/0003-sllm-cascade-kernel.patch`: new files only: `kernels/cascade.metal` (scan, fused prefix/suffix pass,
  merge) and `ggml-metal-cascade.{h,m}` (the hint, the plan, the encoder, the init self-test, the exported status).
- `patches/0004-sllm-cascade-hooks.patch`: the hooks (about 30 lines): the kernel and source in `CMakeLists.txt`; the
  `CASCADE` library kind, its compile-failure fallback and the self-test call in `ggml-metal-device.m`; the dispatch
  branch at the top of `ggml_metal_op_flash_attn_ext`; the scratch in `get_alloc_size`.
- `patches/0005-sllm-cb-error.patch`: failed command buffers (`ggml-metal-context.m`, `ggml-metal.cpp`). Stock ggml
  only logs a failed command buffer in `ggml_metal_synchronize` (which returns nothing) and then refuses every later
  graph (`graph_compute` status -1) until the backend is recreated. The patch records the failure (status, Metal's
  error domain, code and text) and exports `ggml_backend_metal_sllm_take_error(backend, msg, n)`, which returns it and
  clears the error state; sllm-core calls it after every step and fails only that step. Also
  `ggml_backend_metal_sllm_inject_error(backend)` (tests only: the next synchronize reports a failure). Seen in
  sllm-server when several processes together hit the GPU's wired-memory limit (kernel log:
  `IOGPUScheduler::freeToWireMemory: failed to wire down memory`).
- `patches/0006-sllm-cascadeq.patch` (phase 2 kvmixed, adds lines only): the mixed-precision cascade for sllm-core's
  opt-in `KvLayout::OwnQ8` (shared prefix and tail pages f16, each agent's full own pages q8_0).
  - `kernels/cascade.metal`: `kernel_sllm_cascade_suffixq8_d{128,256}_g{2,4}`, appended: the cascade's suffix pass
    reading q8_0 K/V views directly (ggml's block layout, any strides), optionally rotating its queries when the q8_0
    keys are stored Hadamard-rotated (D034).
  - `ggml-metal-cascadeq.{h,m}`: the op contract (f16 views in src[1]/src[2], q8_0 views in src[5]/src[6], the mask
    over both, the mark `op_params[8] = "SLMX"`, prefix range in `[9..10]`, key rotation in `[11]`), the plan, the
    encoder (the cascade's scan twice, its fused dispatch over the f16 part with the q8_0 suffix pass beside it, its
    merge over S + 2 partials), a self-test (3 cases against a double reference on the dequantised data, rotated and
    not, 1 to 9 rows; plus a bitwise repeat-determinism check), and the status export
    `ggml_backend_metal_sllm_cascadeq_status()` (sllm-core: `ggml_metal.cascade_mixed`).
  - hooks: the source in `CMakeLists.txt`, the init call after the cascade's in `ggml-metal-device.m`, a branch before
    the cascade's at the top of `ggml_metal_op_flash_attn_ext` (a marked op this path cannot take aborts: ggml's
    kernels cannot read the q8_0 views), and the scratch in `get_alloc_size`.
  - Off with `SLLM_CASCADE=0`, like the cascade; no row floor (any number of query rows).
- `patches/0007-sllm-prefill-attn.patch` (phase 2 prefillattn): the GQA-packed prefill attention (`pfa`).
  - `kernels/prefill.metal` (new kind `prefill`): `kernel_sllm_pfa_q_d{128,256}_rt32_g{2,4}` (Q as scaled half, once
    per op) and `kernel_sllm_pfa_d{128,256}_rt32_c64_sg4_g{2,4}` (32 rows = 32/G tokens x the G query heads of one
    KV head per threadgroup, 64-cell blocks, one online softmax over the shared prefix and the own causal cells).
    Per row it does exactly ggml's `kernel_flash_attn_ext_tensor` arithmetic (lazy rescale, one simdgroup per row,
    2 cells per lane, simd_sum, the same matmul2d inner sizes), so its output is **bitwise ggml's**: only the K/V
    traffic changes (each K/V tile feeds the G heads), no token changes.
  - `ggml-metal-pfa.{h,m}`: the hint (`op_params[13] = "SLPF"`), the plan, the encoder (ggml's own
    `kernel_flash_attn_ext_blk` with nqptg = 32/G and ncpsg = 64, the Q stage, a barrier, the attention), a self-test
    (4 cases: head size 128 and 256, G = 2 and 4, partial row tiles, hidden cells in the view, window layers; every
    output bitwise equal to ggml's own blk + tensor kernel on the same inputs and to a second run, a double reference
    on sampled tokens, every output finite) and the
    status export `ggml_backend_metal_sllm_prefill_status()` (sllm-core: `ggml_metal.prefill`).
  - hooks: the source and kernel in `CMakeLists.txt`; the `PREFILL` library kind, its compile-failure fallback and the
    init call in `ggml-metal-device.m` (three lines of the 0002/0004 hooks now also name it); a branch after the
    cascade's in `ggml_metal_op_flash_attn_ext`; the scratch (block classes, half Q) in `get_alloc_size`.
- `patches/0008-sllm-fa-kv-f16-scratch.patch` (phase 2 fakv, `ggml-metal-ops.cpp` only): flash attention's F16
  dequant scratch is reserved only when the K/V type is one the dequant pass converts (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0).
  Stock ggml reserves it for every type ("always reserve the temp buffer to avoid graph reallocations"): a full F16
  copy of the op's whole K and V view, which F16/BF16/F32 K/V never read (the kernels read the cache in place; the
  dequant pass and the reads of the scratch are behind `use_kv_f16`, which is false for those types). Over a
  206,848-cell f16 pool that was 808 MiB per attention op in the compute buffer. The test is on the type only, so the
  size still does not change with the query rows (no new graph reallocations); quantized K/V keep the stock size. The
  cascade's scratch (0004, 0006) sits right after the op's output and is added separately, so it is unchanged.
  0008 touches none of 0007's hunks (they apply in numeric order without offsets).
- `patches/0009-sllm-argmax.patch` (phase2-decodeprof): greedy sampling of many rows. ggml's `kernel_argmax_f32`
  gives a row 64 threads at 32 rows (its rule: nth doubles from 32 while nth x rows < 256), so a 32-row decode step's
  argmax over the vocabulary took 1.3 ms (Qwen3-0.6B, 151,936 tokens) and 3.6 ms (Gemma-3-4B, 262,208), at about
  3% of the bandwidth.
  - `kernels/qmm.metal`: `kernel_sllm_argmax`, appended: one threadgroup of 1,024 threads per row, float4 reads.
    Measured 79 us and 129 us for those steps (84% and 89% of the bandwidth roofline).
  - `ggml-metal-argmax.{h,m}`: the conditions (f32 rows of at least 2,048 elements, 16-byte aligned rows, 2-D, I32
    dst), the encoder, a self-test at device init (70 rows against a CPU reference: ties, a row of -inf, NaNs,
    scalar tails, padded strides) and the status export `ggml_backend_metal_sllm_argmax_status()` (sllm-core:
    `ggml_metal.argmax`, `ggml_metal.argmax_active`).
  - hooks: the source in `CMakeLists.txt`, the init call after the cascades' in `ggml-metal-device.m`, the branch at
    the top of `ggml_metal_op_argmax`. No scratch: ggml-base allows no extra allocation for ARGMAX.
  - Result: the first index of the maximum, NaN ignored, -1 when nothing is above -inf. Equal to ggml's whenever the
    maximum is unique; on an exact tie ggml keeps whichever index its thread layout (which depends on the row count)
    keeps, this kernel the lowest (as llama.cpp's CPU greedy sampler).
  - `SLLM_ARGMAX=0` sends every argmax to ggml (the A/B switch).
- `patches/0010-sllm-qmm32.patch` (phase2-qmm32, amended in phase2-qmm32b): qmm with one accumulation order for every
  row count, changing only `kernels/qmm.metal` and `ggml-metal-qmm.{h,m}` (no new hooks).
  - Table (SLLM_QMM32 on, the default): Q4_0 / Q4_1 for every N >= 2 (prefill chunks included), Q6_K for
    2 <= N <= 128. N = 1 stays on ggml's mat-vec (batch-1 byte identity, llama.cpp parity) unless `SLLM_QMM32_N1=1`.
    A token's matmul results do not depend on how many rows its step has (decode rows next to a prefill chunk, chunk
    sizes, joins and leaves).
  - `kernel_sllm_qmm_ks` (Q4_0 / Q4_1): one threadgroup per R x 32 weight rows x 16 or 32 token rows; its G x R
    simdgroups split K inside the threadgroup (G = 8 slices when M <= 1024, else 4; interleaved 32-K chunks) and sum
    their partials in threadgroup memory in a fixed order. The weights are matmul2d's LEFT operand, whose register
    layout gives each lane four consecutive bytes of a row per block. R = 2 row tiles share a threadgroup above 64
    rows when K >= 2048 (the activation chunk is read once per two row tiles; prefill). The token tile size, R and
    the row count change no per-row arithmetic (bitwise, checked by the self-test). No grid split-K, no scratch.
  - `kernel_sllm_qmm_q6p` (Q6_K, the LM heads): inside every 32-K chunk the K order is permuted (tile column 8e + t
    holds k = 4t + e) so a lane's four weights of a row are four consecutive bytes of ql and qh; the threadgroup
    (4 simdgroups x 32 rows) stages the activation chunk as half (round to nearest, saturated at half's range) in the
    same order. 16- and 32-token tiles. Needs rows whose byte stride is a multiple of 4 and 4-byte aligned weights.
  - Numerics: matmul2d rounds an f32 operand's mantissa toward zero but keeps its f32 exponent, so a half copy of the
    activations is not bitwise the same (values outside half's normal range); the Q4 path reads f32. The Q6_K path
    rounds to nearest (relative RMS against a double reference 3.4e-4 instead of 4.7e-4).
  - Self-test: 46 cases, including a bitwise row-count check per type (N = 512, 256, 130, 40, 32, 20, 12, 3 on the
    same weights and rows, both row-tile forms, activations with half-denormal, beyond-half and zero values).
    Status: `qmm.qmm32` in `ggml_metal`.
  - `SLLM_QMM32=0` restores 0001's kernels and table (the A/B switch); `SLLM_QMM32_G=4|8`, `SLLM_QMM32_IL=0|1`,
    `SLLM_QMM32_RT=1|2` force the slice count / order / row tiles (`_RT` changes no bits); `SLLM_QMM32_Q6=0` keeps
    Q6_K on 0001's kernel; `SLLM_QMM32_MAXN=N` sends Q4 mat-muls above N rows to ggml; `SLLM_QMM32_N1=1` (above).
- `patches/0011-sllm-pfa-pair.patch` (phase2-prefill2): `kernel_sllm_pfa2` in `kernels/prefill.metal` and its
  dispatch in `ggml-metal-pfa.m` (no new hooks).
  - The per-row arithmetic of 0007's kernel (bitwise ggml's `kernel_flash_attn_ext_tensor`), with two consecutive
    visible 64-cell blocks sharing one Q K^T `matmul2d` run (and one P V run when the second block rescales no row;
    `matmul2d`'s result does not depend on how its K extent is split into runs) and O rescaled only when a row's
    running max moved. 1.15 to 1.66x the 0007 kernel on the server's prefill shapes (Qwen3-0.6B 8 x 256 / 1 x 512).
  - Streams: a hinted op with ne3 = S > 1 (q, k, v, dst; K/V may share one view with nb3 = 0; the mask per stream)
    runs as one dispatch; sllm-core puts the equal prefill chunks of a step in one such op, so their outputs need no
    concat. Args field `pad0` is now `ns`, five stream strides appended (0007's kernel takes S = 1 only).
  - Self-test: a fifth 0007-style case (long shared prefix, a hidden block that shifts the pairing) and a stream
    case (3 streams sharing K/V, each bitwise the one-stream result). Status field `prefill.kernel`.
  - `SLLM_PFA_PAIR=0` runs 0007's kernel (the A/B switch; then streams go to ggml's kernels).
- `patches/0013-sllm-fuse.patch` (phase2-fuse; 0012 is left to the qmm work): bitwise fusions of the small ops.
  - Five entries in ggml-metal's fusion table (`ggml-metal-fusion.{h,cpp}`: ids, checks; the graph optimizer packs
    them like ggml's own), encoders in `ggml-metal-ops.cpp` (hooks at the top of `ggml_metal_op_norm` and in
    `ggml_metal_op_bin`), kernels appended to `kernels/prefill.metal`:
    RMS_NORM+MUL+ROPE (NEOX) (q heads), +SCALE (Gemma-3 q), +VIEW+SET_ROWS f16 (k heads into the cache),
    ADD+RMS_NORM+MUL (Qwen3 residual + next norm, the sum kept), RMS_NORM+MUL+ADD+RMS_NORM+MUL (Gemma-3 post-norm +
    residual + next norm, the sum kept).
  - Each kernel is ggml's code (`kernel_rms_norm_fuse_impl`, `kernel_rope_neox`, the scale unary, `kernel_set_rows`)
    with ggml's loops and `ggml_metal_op_norm`'s thread count; the two rope products are written as the fmas ggml's
    expressions compile to (the plain expressions contract the other way here: `bench/phase2-fuse/fuse_bench`). When
    the thread count does not fit, the ops run unfused.
  - `SLLM_FUSE=0` turns them off, `SLLM_FUSE_MASK` (bits 0..4) per pattern; status
    `ggml_backend_metal_sllm_fuse_status` (sllm-core `ggml_metal.fuse`, dispatch count). No init self-test: the
    bitwise property is checked end to end (results/phase2-fuse/exact) and by `fuse_bench`.
- `patches/0014-sllm-cascade-exact.patch` (phase2-cascadeexact; 0012 is left to the qmm work): a decode row's bits
  no longer depend on the rest of its step. Before, one agent's logits changed with the size of its shared-prefix
  group (`sllm-core` test `decode_rows_are_independent_of_the_group`: 4 to 7 bit classes per probe on Qwen3-0.6B), and
  the server's `burst_flags_keep_the_greedy_tokens` failed about 1 run in 4.
  - `kernels/cascade.metal`: the prefix pass's softmax uses 4 threads x 16 cells per row with both row tiles (the
    16-row tile had 8 x 8: a different sum order); the 16-row tile's upper two simdgroups sit out the softmax.
  - `ggml-metal-cascade.{h,m}`: the prefix splits depend on the prefix length only (`ggml_metal_sllm_cascade_splits`,
    default 2; before: 4 up to 32 rows, else 2); auto mode takes any number of rows (no 8-row floor); an init
    self-test that every row of a 20-agent op gets the same bits with 1, 2, 8, 9 and 17 agents (three shapes; it fails
    on the old softmax layout); status fields `exact_rows`, `splits`. `SLLM_CASCADE_S=N` forces the splits (A/Bs).
  - `ggml-metal-cascadeq.m`: the same splits rule.
  - `ggml-metal-qmm.{h,m}`: a MUL_MAT marked `op_params[13] = "SLQ1"` takes qmm for one row too (status
    `qmm32.row_mark`). sllm-core marks the mat-muls of a one-row step whose row decodes through the cascade; unmarked
    single rows keep ggml's mat-vec (batch-1 parity).
  - sllm-core side (`batch.rs`, `CascadePath`): an agent's decode-attention path is fixed at its first decode step.
    An agent whose leading pages are shared then always takes the cascade with those pages as its prefix range (alone
    or in a group of any size, compact span or not). One that is not never takes the cascade. With qmm32 on,
    `split_mm` is not applied (it changed Gemma-3-4B's bits at 17..32 rows).
- `patches-profile/` (applied only with `SLLM_METAL_PROFILE=1 build.sh`; the default build is unchanged):
  - `0101-sllm-gpu-timing.patch`: per-step GPU time of the command buffers (`ggml_backend_metal_sllm_gpu_times`).
  - `0102-sllm-node-timing.patch` (phase2-decodeprof): with `SLLM_METAL_NODEPROF=<file.jsonl>` at run time every
    encoded op gets its own compute encoder with stage-boundary GPU timestamps (`MTLCounterSampleBuffer`), and every
    synchronize appends one JSON line per op (start/end ns, fused ops, dst/src names, types, shapes, op_params). Ops
    lose their overlap and pay one encoder boundary each, so read the per-op times, not their sum.
    `results/phase2-decodeprof/nodes.py` turns a file into the per-op-class table.
- `MANIFEST.json`: source commit and tarball sha256, patch sha256s, flags, the output sha256. Committed.
- `lib/`, `build/`: build output, not committed.

Run time:
- `SLLM_QMM=0` sends every mat-mul to ggml (the A/B switch); unset or `1` is the default (auto);
  `SLLM_QMM=force` (tests only) drops the N table.
- At device init the patch runs a self-test (Q4_0, Q4_1, Q6_K; partial row tile, both token tiles, split-K)
  against a CPU reference. On a mismatch it logs why and leaves every mat-mul to ggml.
- `ggml_backend_metal_sllm_qmm_status()` (exported) returns a JSON status. sllm-core reports it as
  `ggml_metal` in `sllm-core caps`, `/caps` and its run records.
- Build time: `SLLM_GGML_METAL=official cargo build ...` links the official set instead.

Cascade (shared-prefix decode attention):
- The hint: sllm-core's `batch.rs` writes `op_params[13] = 0x534C4341` ("SLCA") and the prefix range `[p0, p1)` of the
  op's K view (cells) into `op_params[14..15]` of a shared-prefix decode group's `flash_attn_ext`. Official ggml uses
  slots 0..4 only and ignores them. The kernel is exact for any range; the range only decides the speed.
- Conditions: the hint; tensor API; f16 K/V, head size 128 or 256, G = H / HKV in {2, 4}; an f16 mask; no sinks,
  ALiBi or softcap. Any number of query rows (patch 0014; it was at least 8). Otherwise ggml's own kernels run the op.
- A row's result is bitwise independent of the other rows of the op, their number and the K/V view around its cells
  (patch 0014, checked by the self-test); it depends on the prefix range, which sllm-core fixes per agent.
- `SLLM_CASCADE=0` sends every flash attention to ggml (the A/B switch); `SLLM_CASCADE=force` is kept (no effect since
  patch 0014); `SLLM_CASCADE_TRACE=1` prints each dispatch, and why a hinted op was not taken, to stderr.
- Self-test at device init: 4 cases (head size 128 and 256, G = 2 and 4, padded row tiles, a prefix in the middle of
  the view, window-hidden and partly hidden blocks, constant V and a fully masked row) against a double reference,
  plus patch 0014's 3 row-independence checks (bitwise). On a mismatch it logs why and every
  flash attention stays on ggml. `ggml_backend_metal_sllm_cascade_status()` (exported) returns a JSON status; sllm-core
  reports it as `ggml_metal.cascade` and `ggml_metal.cascade_active`.

Prefill attention (pfa):
- The hint: sllm-core's `batch.rs` writes `op_params[13] = 0x534C5046` ("SLPF") on a prefill chunk's
  `flash_attn_ext` in a step with more than one item (never in a batch-1 step, which stays byte-identical to the
  single-stream path; with the bitwise-equal kernel this is a precaution, not a need). Same slot as the cascade's hint, which a prefill chunk never carries; a mixed-precision op
  (`op_params[8]`) never carries it and takes the 0006 path first.
- Conditions: the hint; tensor API; f16 K/V, head size 128 or 256, G in {2, 4}, a view of a multiple of 64 cells, dense
  (no n_kv_max); an f16 mask as wide as the view, one for all heads; no sinks, ALiBi or softcap; and ggml's own
  tensor-path floor (at least 20 rows and ceil(rows / 32) x heads x head size >= 8192), so pfa takes exactly the ops
  ggml would run on `kernel_flash_attn_ext_tensor`. Otherwise ggml's own kernels.
- `SLLM_PREFILL_CASCADE=0` sends every prefill attention to ggml (the A/B switch); `SLLM_PREFILL_CASCADE=force` drops
  the floor (tests only; below it the bits differ from ggml's, which would use its vector or half8x8 kernels);
  `SLLM_CASCADE_TRACE=1` also prints each pfa dispatch and each refusal.

qmm dispatch table (cold measurements of the spike, the faster path per type and N): Q4_0 and Q4_1 take qmm for
9 ≤ N ≤ 128, Q6_K for 9 ≤ N ≤ 64. N is the number of src1 rows. N = 1 (`mul_mv`), N ≤ 8 and N > 128 stay on ggml.

When the pin moves: re-extract, re-apply the patches in order (they touch ggml-metal only), rebuild, rerun the checks.
