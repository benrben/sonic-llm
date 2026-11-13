//! The library's C interface (engine/sllm-next/proto/sn.h) and Accelerate's mat-vec.

use std::ffi::{c_char, c_int, CStr};

#[repr(C)]
pub(super) struct SnEngine {
    _p: [u8; 0],
}

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    pub(super) fn cblas_sgemv(order: c_int, trans: c_int, m: c_int, n: c_int, alpha: f32, a: *const f32, lda: c_int, x: *const f32, incx: c_int,
                   beta: f32, y: *mut f32, incy: c_int);
}

extern "C" {
    pub(super) fn sn_open(path: *const c_char, kernels: *const c_char, slots: c_int, ctx: c_int, max_rows: c_int, pages: c_int, err: *mut c_char, err_len: c_int) -> *mut SnEngine;
    pub(super) fn sn_pages_total(e: *const SnEngine) -> c_int;
    pub(super) fn sn_pages_free(e: *const SnEngine) -> c_int;
    pub(super) fn sn_page_bytes(e: *const SnEngine) -> usize;
    pub(super) fn sn_seq_pages(e: *const SnEngine, seq: c_int, own: *mut c_int) -> c_int;
    pub(super) fn sn_win_need(e: *const SnEngine, n: c_int) -> c_int;
    pub(super) fn sn_win_room(e: *const SnEngine) -> c_int;
    pub(super) fn sn_win_pages(e: *const SnEngine, seq: c_int) -> c_int;
    pub(super) fn sn_recurrent(e: *const SnEngine) -> c_int;
    pub(super) fn sn_draft_rows(e: *mut SnEngine, seq: c_int, want: c_int) -> c_int;
    pub(super) fn sn_guess_head(e: *const SnEngine) -> c_int;
    pub(super) fn sn_guess_on(e: *mut SnEngine, on: c_int) -> c_int;
    pub(super) fn sn_guess(e: *mut SnEngine, seq: c_int, tok: i32, pos: c_int, k: c_int, out_tok: *mut i32, err: *mut c_char, err_len: c_int) -> c_int;
    pub(super) fn sn_small_rows(e: *const SnEngine) -> c_int;
    pub(super) fn sn_cut_best(e: *const SnEngine, slot: c_int, n: c_int) -> c_int;
    pub(super) fn sn_mark(e: *mut SnEngine, slot: c_int) -> c_int;
    pub(super) fn sn_state_take(e: *mut SnEngine, seq: c_int) -> c_int;
    pub(super) fn sn_check(path: *const c_char, rows: *mut c_int, err: *mut c_char, err_len: c_int) -> c_int;
    pub(super) fn sn_quiet_rows(path: *const c_char) -> c_int;
    pub(super) fn sn_n_vocab(e: *const SnEngine) -> c_int;
    pub(super) fn sn_step(e: *mut SnEngine, r: c_int, tok: *const i32, seq: *const i32, pos: *const i32, n_write: c_int, n_fast: c_int, n_out: c_int,
               out_rows: *const i32, out_tok: *mut i32, chain: c_int, err: *mut c_char, err_len: c_int) -> c_int;
    pub(super) fn sn_last_gpu_ms(e: *const SnEngine) -> f64;
    pub(super) fn sn_logits(e: *mut SnEngine, k: c_int, dst: *mut f32, err: *mut c_char, err_len: c_int) -> c_int;
    pub(super) fn sn_want_probs(e: *mut SnEngine, on: c_int);
    pub(super) fn sn_probs(e: *mut SnEngine, n_out: c_int, dst: *mut f32, err: *mut c_char, err_len: c_int) -> c_int;
    pub(super) fn sn_logits_rows(e: *mut SnEngine, n: c_int, ks: *const c_int, dst: *mut f32, err: *mut c_char, err_len: c_int) -> c_int;
    pub(super) fn sn_set_eog(e: *mut SnEngine, ids: *const i32, n: c_int) -> c_int;
    pub(super) fn sn_no_end(e: *mut SnEngine, slot: c_int, on: c_int);
    pub(super) fn sn_copy(e: *mut SnEngine, from: c_int, to: c_int, n: c_int, err: *mut c_char, err_len: c_int) -> c_int;
    pub(super) fn sn_clear(e: *mut SnEngine, slot: c_int, from: c_int, to: c_int, err: *mut c_char, err_len: c_int) -> c_int;
    pub(super) fn sn_kv_bytes(e: *const SnEngine, n: c_int) -> usize;
    pub(super) fn sn_kv_save(e: *mut SnEngine, slot: c_int, n: c_int, dst: *mut u8, err: *mut c_char, err_len: c_int) -> c_int;
    pub(super) fn sn_kv_load(e: *mut SnEngine, slot: c_int, n: c_int, src: *const u8, err: *mut c_char, err_len: c_int) -> c_int;
    pub(super) fn sn_close(e: *mut SnEngine);
    pub(super) fn sn_trim(e: *mut SnEngine, spare: c_int) -> usize;
    pub(super) fn sn_mem_used(e: *const SnEngine) -> usize;
    pub(super) fn sn_state_bytes(e: *const SnEngine) -> usize;
    pub(super) fn sn_state_entries(e: *const SnEngine, seq: c_int) -> c_int;
    pub(super) fn sn_trim_pending(e: *mut SnEngine) -> c_int;
    pub(super) fn sn_latent(e: *mut SnEngine, n: c_int, rows: *const i32, vec: *const f32, hid: *mut f32, err: *mut c_char, err_len: c_int) -> c_int;
    pub(super) fn sn_n_embd(e: *const SnEngine) -> c_int;
    pub(super) fn sn_soft_config(e: *mut SnEngine, temp: f32, k: c_int, eog: *const i32, n_eog: c_int, err: *mut c_char, err_len: c_int) -> c_int;
    pub(super) fn sn_soft_rows(e: *mut SnEngine, n_in: c_int, in_rows: *const i32, n_out: c_int, out_ks: *const i32, err: *mut c_char, err_len: c_int) -> c_int;
    pub(super) fn sn_embd_norm(e: *mut SnEngine) -> f32;
    pub(super) fn sn_embed_rows(e: *mut SnEngine, n: c_int, ids: *const i32, dst: *mut f32, err: *mut c_char, err_len: c_int) -> c_int;
}

/// A library call with an error buffer: Ok, or the library's reason.
pub(super) fn call(f: impl FnOnce(*mut c_char, c_int) -> c_int) -> Result<(), String> {
    let mut buf = vec![0 as c_char; 512];
    if f(buf.as_mut_ptr(), buf.len() as c_int) == 0 {
        Ok(())
    } else {
        Err(unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned())
    }
}
