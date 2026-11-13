use super::*;
unsafe extern "C" {
    pub fn llama_memory_seq_rm(mem: sys::llama_memory_t, seq: sys::llama_seq_id, p0: sys::llama_pos, p1: sys::llama_pos) -> bool;
    pub fn llama_memory_seq_cp(mem: sys::llama_memory_t, src: sys::llama_seq_id, dst: sys::llama_seq_id, p0: sys::llama_pos, p1: sys::llama_pos);
    pub fn llama_memory_seq_pos_max(mem: sys::llama_memory_t, seq: sys::llama_seq_id) -> sys::llama_pos;
    pub fn llama_state_seq_get_size(ctx: *mut sys::llama_context, seq: sys::llama_seq_id) -> usize;
    pub fn llama_state_seq_get_data(ctx: *mut sys::llama_context, dst: *mut u8, size: usize, seq: sys::llama_seq_id) -> usize;
    pub fn llama_state_seq_set_data(ctx: *mut sys::llama_context, src: *const u8, size: usize, seq: sys::llama_seq_id) -> usize;
    pub fn llama_model_n_layer(model: *const sys::llama_model) -> i32;
    pub fn llama_model_n_head(model: *const sys::llama_model) -> i32;
    pub fn llama_model_n_swa(model: *const sys::llama_model) -> i32;
    pub fn llama_model_n_ctx_train(model: *const sys::llama_model) -> i32;
    pub fn llama_model_size(model: *const sys::llama_model) -> u64;
    pub fn llama_model_is_recurrent(model: *const sys::llama_model) -> bool;
    pub fn llama_model_is_hybrid(model: *const sys::llama_model) -> bool;
    // the grammar sampler (llama.h: llama_sampler_init_grammar, llama_sampler_apply, llama_sampler_accept, llama_sampler_free)
    pub fn llama_sampler_init_grammar(vocab: *const sys::llama_vocab, grammar_str: *const c_char, grammar_root: *const c_char) -> *mut Sampler;
    pub fn llama_sampler_apply(smpl: *mut Sampler, cur_p: *mut TokenDataArray);
    pub fn llama_sampler_accept(smpl: *mut Sampler, token: i32);
    pub fn llama_sampler_free(smpl: *mut Sampler);
    pub fn dlsym(handle: *mut std::ffi::c_void, symbol: *const c_char) -> *mut std::ffi::c_void;
}

/// `struct llama_sampler` (opaque).
pub enum Sampler {}

/// `llama_token_data`: a candidate token.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TokenData {
    pub id: i32,
    pub logit: f32,
    pub p: f32,
}

/// `llama_token_data_array`.
#[repr(C)]
pub struct TokenDataArray {
    pub data: *mut TokenData,
    pub size: usize,
    pub selected: i64,
    pub sorted: bool,
}

/// Set ggml-metal's row-exact switch (`ggml_backend_metal_sllm_set_row_exact`, engine/vendor/ggml-rowexact-b11669); false
/// when the linked ggml does not export it (the official one).
pub fn set_row_exact(on: bool) -> bool {
    // RTLD_DEFAULT on macOS
    let h = -2isize as *mut std::ffi::c_void;
    let p = unsafe { dlsym(h, c"ggml_backend_metal_sllm_set_row_exact".as_ptr()) };
    if p.is_null() {
        return false;
    }
    let f: extern "C" fn(i32) = unsafe { std::mem::transmute(p) };
    f(on as i32);
    true
}

/// `ggml_backend_metal_sllm_set_copy_rows` (engine/vendor/ggml-rowexact-b11669/patches/0011-sllm-copy-rows.patch), looked up
/// once; None when the linked ggml does not export it (the official one).
fn copy_rows_fn() -> Option<extern "C" fn(i64)> {
    static F: std::sync::OnceLock<Option<extern "C" fn(i64)>> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let h = -2isize as *mut std::ffi::c_void; // RTLD_DEFAULT on macOS
        let p = unsafe { dlsym(h, c"ggml_backend_metal_sllm_set_copy_rows".as_ptr()) };
        (!p.is_null()).then(|| unsafe { std::mem::transmute::<*mut std::ffi::c_void, extern "C" fn(i64)>(p) })
    })
}

/// Whether the linked ggml can stop a stream copy after some rows.
pub fn has_copy_rows() -> bool {
    copy_rows_fn().is_some()
}

/// (copies ggml-metal cut after the hinted rows, bytes they did not move) since the process started; zeros without the hook.
pub fn copy_stats() -> (i64, i64) {
    let h = -2isize as *mut std::ffi::c_void; // RTLD_DEFAULT on macOS
    let p = unsafe { dlsym(h, c"ggml_backend_metal_sllm_copy_stats".as_ptr()) };
    if p.is_null() {
        return (0, 0);
    }
    let f: extern "C" fn(*mut i64, *mut i64) = unsafe { std::mem::transmute(p) };
    let (mut cuts, mut saved) = (0i64, 0i64);
    f(&mut cuts, &mut saved);
    (cuts, saved)
}

/// While it lives, a KV stream copy that libllama applies inside a `llama_decode` on this thread stops after `rows` rows
/// (`rows` 0: no hint, whole streams). ggml keeps the hint per thread and cuts only a copy between two streams of one cache
/// tensor, so it reaches no other context's copy, and it ends with the guard (also on a panic).
pub struct CopyHint(bool, std::marker::PhantomData<*const ()>);

impl CopyHint {
    pub fn set(rows: i64) -> Self {
        let f = copy_rows_fn().filter(|_| rows > 0);
        if let Some(f) = f {
            f(rows);
        }
        CopyHint(f.is_some(), std::marker::PhantomData)
    }
}

impl Drop for CopyHint {
    fn drop(&mut self) {
        if let Some(f) = copy_rows_fn().filter(|_| self.0) {
            f(0);
        }
    }
}
