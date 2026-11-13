// sllm-next engine library: Qwen3 and Gemma 4 GGUFs on Metal with SLLM's own kernels. One step runs a set of rows (a
// token, its sequence slot and position) through the model and gives the greedy next token of the rows asked for. A row's
// result does not depend on the other rows of the step (the same answer at any agent count).
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sn_engine sn_engine;

// Load a GGUF with room for `slots` sequences of up to `ctx` cells each, a cache of `pages` pages of 64 cells (0: every
// sequence its whole ctx) and steps of up to `max_rows` rows. kernels: the kernels.metal source (NUL-terminated). NULL on
// failure, the reason in err. A sequence takes pages as its cells are written; sequences may share pages (sn_copy).
sn_engine * sn_open(const char * gguf_path, const char * kernels, int slots, int ctx, int max_rows, int pages, char * err, int err_len);

// Whether sn_open can run the GGUF (its header and tensor table only, nothing loaded): 0, or -1 with the reason in err.
// rows (if not NULL): the prompt rows a step should take, 512 or 1024: 512 (llama.cpp's ubatch) when the model has
// sliding-window layers (they hold a window plus a step) or wide layers (a step's buffers above 144 KB a row), so one
// agent's memory stays at or below ggml's.
int sn_check(const char * gguf_path, int * rows, char * err, int err_len);
// The prompt rows a step for this model while no chat is about (bigger steps read a MoE faster), 0: its usual rows.
int sn_quiet_rows(const char * gguf_path);

int sn_n_vocab(const sn_engine * e);
int sn_n_ctx_train(const sn_engine * e);
// "qwen3" or "gemma4"
const char * sn_arch(const sn_engine * e);

// One step of R rows. Rows [0, n_write) are write rows (one per writing sequence; the first n_fast of them belong to
// interactive agents and take the one-row kernels, at most 4), rows [n_write, R) prompt rows (consecutive positions of a
// sequence stay consecutive rows). out_rows: the n_out rows whose next token is wanted; out_tok receives them.
// n_fast < 0 (a wide draft check of -n_fast <= 8 write rows of one sequence): those rows' attention takes the one-row kernels,
// their matmuls the block matmuls (not the bits of one-row steps).
// chain: the caller expects to run the same rows next, each fed its token from this step: a write step (every row sampled,
// in order) then has that next step queued behind it (on the GPU before this call returns; a later call that is not it, or
// any cache call, drops it and zeroes the cells it wrote). The tokens are the same either way.
// Returns 0, -2 when the cache is out of pages (nothing ran), or -1 with the reason in err.
int sn_step(sn_engine * e, int R, const int32_t * tok, const int32_t * seq, const int32_t * pos, int n_write, int n_fast,
            int n_out, const int32_t * out_rows, int32_t * out_tok, int chain, char * err, int err_len);

// GPU milliseconds of the last step.
double sn_last_gpu_ms(const sn_engine * e);

// The scores of output row k of the last step (n_vocab floats) into dst: for a row the caller samples itself (a grammar
// refused the argmax, or the argmax is an end-of-generation token the request may not end on).
int sn_logits(sn_engine * e, int k, float * dst, char * err, int err_len);
// The scores of n output rows ks of the last step into dst (n x V floats), in one copy. 0 or -1 (err).
int sn_logits_rows(sn_engine * e, int n, const int * ks, float * dst, char * err, int err_len);

// Token probabilities: while on, the steps encoded (a queued step too, at the time it is queued) take the argmax_p kernel in
// place of argmax: the same chosen tokens, and for each output row its two best scores (over the whole vocabulary, ids
// ascending among equal scores) and its log-sum-exp. Off (the default): the steps are the same code as before. A queued step
// encoded with the other setting than the call it is asked for with is dropped like any queued step that is not taken.
void sn_want_probs(sn_engine * e, int on);
// The records of the last returned step's first n_out output rows into dst (n_out x 8 words, 4 bytes each): chosen id (int),
// chosen score (float), best id (int), best score, second id, second score, log-sum-exp (float), 0. The chosen token is the
// step's output (the best other token for a row that may not end); its log-probability is score - log-sum-exp. 0 or -1 (err:
// the step was encoded without sn_want_probs on).
int sn_probs(sn_engine * e, int n_out, float * dst, char * err, int err_len);

// Latent rows (thinking without text, LatentMAS): the NEXT sn_step only. Its rows rows[j] (n of them) take vec row j
// (n_embd floats each) in place of their tokens' embeddings (their tokens are still passed, and ignored); hid, when not
// NULL, receives that step's output rows' hidden states after the final norm (n_out x n_embd floats, in out_rows order)
// when it returns. Such a step never takes a queued step nor queues one. Not for models with per-layer token embeddings.
int sn_latent(sn_engine * e, int n, const int32_t * rows, const float * vec, float * hid, char * err, int err_len);
int sn_n_embd(const sn_engine * e);

// Soft words on the GPU (silent thinking whose next input is the mix of the K best words' input rows weighted by
// softmax(score / temp), end-of-generation ids left out): a step's output rows that are soft make that mix on the GPU from
// their scores (nothing goes to the CPU) into a vector of their sequence, and the next step's soft input rows read it.
// sn_soft_config turns it on: temp > 0, k = 1 .. 64, eog: the ids that are not words (at most 32). -1 (err): not for this model
// (per-layer token embeddings, a staged or q4_1 token table, recurrent states).
int sn_soft_config(sn_engine * e, float temp, int k, const int32_t * eog, int n_eog, char * err, int err_len);
// The NEXT sn_step only: its rows in_rows[0 .. n_in) (row numbers of the step) take their sequence's vector in place of their
// tokens' embeddings (their tokens are passed and ignored), and its output rows out_ks[0 .. n_out) (indices into out_rows,
// at most one a sequence) make their sequence's new vector. A prompt's last row is a soft output with no input row (its
// words start the silent turn). A write step whose rows are all of both kinds (every row soft, out_rows 0 .. R-1) with
// chain set has the next such step queued behind it like any write step (the queued step ignores the tokens the caller
// passes for these rows); a queued step that is not taken is dropped as always, and the vectors go back to what the step
// before made. Not with sn_latent's rows in the same step.
int sn_soft_rows(sn_engine * e, int n_in, const int32_t * in_rows, int n_out, const int32_t * out_ks, char * err, int err_len);
// The mean L2 norm of a token's input row as the first layer reads it (measured over the whole table once, then cached):
// a latent vector is scaled to it before it is fed.
// The input rows (as the embedding stage makes them) of n <= 1024 token ids into dst (n x n_embd floats). 0 or -1 (err).
int sn_embed_rows(sn_engine * e, int n, const int32_t * ids, float * dst, char * err, int err_len);
float sn_embd_norm(sn_engine * e);

// The model's end-of-generation tokens (at most 8; -1 when there are more), and whether a slot's output rows skip them:
// its request may not end (ignore_eos). Its argmax is then the best other token, the lowest id among equal scores (what
// the caller would take from sn_logits), so a queued step is fed the token the caller takes. Steps encoded after the call.
int sn_set_eog(sn_engine * e, const int32_t * ids, int n);
void sn_no_end(sn_engine * e, int slot, int on);

// Give slot `to` the cells [0, n) of slot `from` (what `to` held goes): the full pages are shared, the last partly filled
// one copied. A cell depends only on the tokens up to it, so a shared prefix is what a prefill of the same tokens would
// write, bit for bit. -2: out of pages, -3: `from`'s window layers no longer hold the cells [0, n) needs (sn_cut_ok).
int sn_copy(sn_engine * e, int from, int to, int n, char * err, int err_len);

// Cut a slot to its first `from` cells (the cells [from, to) were its last ones): the pages past the cut go back to the
// pool, the page of the cut keeps its first cells (a private copy first when another slot shares it). -2: out of pages,
// -3: the window layers no longer hold the cells before the cut (sn_cut_ok); nothing is changed then.
int sn_clear(sn_engine * e, int slot, int from, int to, char * err, int err_len);

// Whether a slot can be cut to (or copied as) its first n cells: a model's sliding-window layers keep only a sequence's
// last window of cells, so a cut far behind its end cannot keep them (the caller reads those tokens again instead).
int sn_cut_ok(const sn_engine * e, int slot, int n);

// Whether the model keeps recurrent states (Qwen3.5's Gated DeltaNet layers): a sequence can then be cut only to 0, its end
// or its checkpoint (sn_mark), and goes on from its last cell
// only (no queued steps; drafts to take back only with sn_draft_rows).
int sn_recurrent(const sn_engine * e);

// Whether the GGUF carries the makers' guess head (an extra block after the model's own, nextn_predict_layers 1) in a form
// sllm-next runs. The model's own answers never depend on it.
int sn_guess_head(const sn_engine * e);
// Guessing on or off (off at open; returns what it is now, 0 without a guess head). While on, every step also writes the
// head's cache cells of its rows (a few small matmuls a step), so sn_guess can run at any time. The model's own answers do
// not change.
int sn_guess_on(sn_engine * e, int on);
// k guesses (1 .. 4) in one go: the head's row for token `tok` at position `pos` of sequence `seq` (the token the caller is
// about to run there) gives the best next token; then a row for that guess at pos + 1, and so on, each fed the head's own
// output state (the first pairs with the model's hidden state at pos - 1: the sequence's last step, or after sn_clear the
// last row kept). out_tok receives the k guesses. A guess is only a draft for sn_step to check. 0, -2 out of pages, -1 (err).
int sn_guess(sn_engine * e, int seq, int32_t tok, int pos, int k, int32_t * out_tok, char * err, int err_len);

// Draft rows on a recurrent model: make sure sequence `seq` holds `want` (0 .. 3) spare state entries and return how many it
// holds (fewer when the pool cannot give them); want 0 gives them back. A later step whose first rows are an interactive run
// (n_fast > 0) of that sequence, of at most 1 + that many rows, also keeps the state after each row but its last, and
// until the sequence's next step sn_cut_ok / sn_clear / sn_cut_best take a cut back to just after any of those rows (the
// entries swap, nothing is copied). Other models need none of this: `want` comes back as it is.
int sn_draft_rows(sn_engine * e, int seq, int want);

// SN_SMALL: steps of at most this many background write rows take the few-row kernel for them (0: off). Their tokens are
// the same whenever at most this many write, and whenever more do, not between the two: a caller keeps a small step small
// (a background writer's drafts within it).
int sn_small_rows(const sn_engine * e);

// Whether an interactive row's tokens are the same with and without background writers in its steps (SN_CHATBATCH=0); by
// default its matmuls join theirs when there are any (one weight read), not the bits of its one-row kernels.
int sn_chat_exact(const sn_engine * e);

// A recurrent model: keep the slot's state where it is now as its checkpoint (a cut or copy may then go back to it), when the
// state pool has a free entry; 1 kept, 0 not (or not a recurrent model).
int sn_mark(sn_engine * e, int slot);
int sn_state_take(sn_engine * e, int seq);  // a recurrent state entry for seq now; -1: none free

// The most cells of a slot's first n it can be cut back to (sn_cut_ok's best at or below n; a recurrent model: 0, its end, or
// its checkpoint where its last prompt ended); the caller reads the rest again.
int sn_cut_best(const sn_engine * e, int slot, int n);

// The pool: pages in all, free now, bytes of one page (with the window layers' share); a slot's pages, `own` of them
// held by it alone. The window layers' own pool follows it (it grows as sequences need, to the same page count at most).
int sn_pages_total(const sn_engine * e);
int sn_pages_free(const sn_engine * e);
size_t sn_page_bytes(const sn_engine * e);
int sn_seq_pages(const sn_engine * e, int seq, int * own);

// The window layers' pool (0 for models without sliding windows): pages a sequence of n cells may hold there (with a full
// step of rows past it), pages it can still give (free ones, and what it may grow by within the GPU's working set; set at
// open), and the pages a slot holds there. A caller reserves both pools' pages for what it admits.
int sn_win_need(const sn_engine * e, int n);
int sn_win_room(const sn_engine * e);
int sn_win_pages(const sn_engine * e, int seq);

// Bytes of the cache of n cells; copy a slot's first n cells to host memory and back (suspend / resume).
size_t sn_kv_bytes(const sn_engine * e, int n);
int sn_kv_save(sn_engine * e, int slot, int n, void * dst, char * err, int err_len);
int sn_kv_load(sn_engine * e, int slot, int n, const void * src, char * err, int err_len);

// Give memory back after a busy moment (placement sparse pools): the pages (and Qwen3.5 state entries) in use above
// what is held, plus `spare` chunks, move down to free ones and the chunks above are unmapped. Synchronous; waits for the
// GPU (a queued step is dropped, as by any cache call). The bytes given back.
size_t sn_trim(sn_engine * e, int spare);
// Memory sn_trim gave back that the GPU is still unmapping (its heaps are let go here when done): how many batches wait.
int sn_trim_pending(sn_engine * e);
// Bytes the cache holds now: pages in use (both pools) and Qwen3.5's state entries in use.
size_t sn_mem_used(const sn_engine * e);
// A recurrent model's state memory: the bytes of one state entry (0: the model is not recurrent), and how many entries a
// sequence holds now (its state, and its checkpoint when it has one). The server caps what ended sequences keep with these.
size_t sn_state_bytes(const sn_engine * e);
int sn_state_entries(const sn_engine * e, int seq);

void sn_close(sn_engine * e);

#ifdef __cplusplus
}
#endif
