//! ggml's GGUF reader (libggml-base): the header and tensor table of a file, read without loading it.

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;

// ---------------------------------------------------------------------------------------------------------------------
// ggml's GGUF reader (libggml-base). Declared here so that nothing depends on sllm-core.
// ---------------------------------------------------------------------------------------------------------------------

pub(super) mod ffi {
    use super::*;

    #[repr(C)]
    pub struct GgufInitParams {
        pub no_alloc: bool,
        pub ctx: *mut *mut c_void,
    }

    #[link(name = "ggml-base")]
    unsafe extern "C" {
        pub fn gguf_init_from_file(fname: *const c_char, params: GgufInitParams) -> *mut c_void;
        pub fn gguf_free(ctx: *mut c_void);
        pub fn gguf_find_key(ctx: *const c_void, key: *const c_char) -> i64;
        pub fn gguf_get_kv_type(ctx: *const c_void, id: i64) -> i32;
        pub fn gguf_get_arr_type(ctx: *const c_void, id: i64) -> i32;
        pub fn gguf_get_val_data(ctx: *const c_void, id: i64) -> *const c_void;
        pub fn gguf_get_val_str(ctx: *const c_void, id: i64) -> *const c_char;
        pub fn gguf_get_arr_n(ctx: *const c_void, id: i64) -> usize;
        pub fn gguf_get_arr_data(ctx: *const c_void, id: i64) -> *const c_void;
        pub fn gguf_get_n_tensors(ctx: *const c_void) -> i64;
        pub fn gguf_get_tensor_name(ctx: *const c_void, id: i64) -> *const c_char;
        pub fn gguf_get_tensor_ne(ctx: *const c_void, id: i64) -> *const i64;
        pub fn gguf_get_tensor_type(ctx: *const c_void, id: i64) -> i32;
        pub fn gguf_get_tensor_size(ctx: *const c_void, id: i64) -> usize;
        pub fn ggml_row_size(ty: i32, ne: i64) -> usize;
    }

    // libllama's model properties, declared once with the engine's own
    pub use crate::llama_engine::ffi::{
        llama_model_is_hybrid, llama_model_is_recurrent, llama_model_n_ctx_train, llama_model_n_head, llama_model_n_layer, llama_model_n_swa,
        llama_model_size,
    };
}

// GGUF value types (gguf.h).
const T_U8: i32 = 0;
const T_I8: i32 = 1;
const T_U16: i32 = 2;
const T_I16: i32 = 3;
const T_U32: i32 = 4;
const T_I32: i32 = 5;
const T_F32: i32 = 6;
const T_BOOL: i32 = 7;
const T_STR: i32 = 8;
const T_ARR: i32 = 9;
const T_U64: i32 = 10;
const T_I64: i32 = 11;
const T_F64: i32 = 12;


#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub ne: [i64; 4],
    pub ty: i32,
    pub bytes: u64,
}

/// An open GGUF header (metadata and tensor table; no tensor data is read).
pub struct Gguf {
    ctx: *mut c_void,
}

// SAFETY: the context is read-only after init and owned by this value.
unsafe impl Send for Gguf {}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf, String> {
        let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| "path contains NUL".to_string())?;
        let ctx = unsafe { ffi::gguf_init_from_file(c.as_ptr(), ffi::GgufInitParams { no_alloc: true, ctx: std::ptr::null_mut() }) };
        if ctx.is_null() {
            return Err(format!("{}: not a readable GGUF file", path.display()));
        }
        Ok(Gguf { ctx })
    }

    fn find(&self, key: &str) -> Option<i64> {
        sllm_core::gguf::find_key_id(key, |c| unsafe { ffi::gguf_find_key(self.ctx, c.as_ptr()) })
    }

    /// The id of key `key`, when its value is of type `ty`.
    fn find_typed(&self, key: &str, ty: i32) -> Option<i64> {
        let id = self.find(key)?;
        (unsafe { ffi::gguf_get_kv_type(self.ctx, id) } == ty).then_some(id)
    }

    /// A scalar number (or bool) as f64.
    pub fn num(&self, key: &str) -> Option<f64> {
        let id = self.find(key)?;
        let ty = unsafe { ffi::gguf_get_kv_type(self.ctx, id) };
        let p = unsafe { ffi::gguf_get_val_data(self.ctx, id) };
        if p.is_null() || ty == T_STR || ty == T_ARR {
            return None;
        }
        Some(unsafe { read_num(ty, p) })
    }

    pub fn str(&self, key: &str) -> Option<String> {
        let id = self.find_typed(key, T_STR)?;
        let p = unsafe { ffi::gguf_get_val_str(self.ctx, id) };
        (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }

    /// A numeric (or bool) array as f64.
    pub fn nums(&self, key: &str) -> Option<Vec<f64>> {
        let id = self.find_typed(key, T_ARR)?;
        let ty = unsafe { ffi::gguf_get_arr_type(self.ctx, id) };
        if ty == T_STR || ty == T_ARR {
            return None;
        }
        let n = unsafe { ffi::gguf_get_arr_n(self.ctx, id) };
        let p = unsafe { ffi::gguf_get_arr_data(self.ctx, id) } as *const u8;
        let sz = num_size(ty);
        Some((0..n).map(|i| unsafe { read_num(ty, p.add(i * sz) as *const c_void) }).collect())
    }

    /// Length of an array of any element type.
    pub fn arr_len(&self, key: &str) -> Option<usize> {
        let id = self.find(key)?;
        (unsafe { ffi::gguf_get_kv_type(self.ctx, id) } == T_ARR).then(|| unsafe { ffi::gguf_get_arr_n(self.ctx, id) })
    }

    pub fn has(&self, key: &str) -> bool {
        self.find(key).is_some()
    }

    pub fn tensors(&self) -> Vec<TensorInfo> {
        let n = unsafe { ffi::gguf_get_n_tensors(self.ctx) };
        (0..n)
            .map(|i| unsafe {
                let ne = ffi::gguf_get_tensor_ne(self.ctx, i);
                TensorInfo {
                    name: CStr::from_ptr(ffi::gguf_get_tensor_name(self.ctx, i)).to_string_lossy().into_owned(),
                    ne: [*ne, *ne.add(1), *ne.add(2), *ne.add(3)],
                    ty: ffi::gguf_get_tensor_type(self.ctx, i),
                    bytes: ffi::gguf_get_tensor_size(self.ctx, i) as u64,
                }
            })
            .collect()
    }
}

impl Drop for Gguf {
    fn drop(&mut self) {
        unsafe { ffi::gguf_free(self.ctx) }
    }
}

fn num_size(ty: i32) -> usize {
    match ty {
        T_U8 | T_I8 | T_BOOL => 1,
        T_U16 | T_I16 => 2,
        T_U32 | T_I32 | T_F32 => 4,
        _ => 8,
    }
}

/// SAFETY: `p` points at a value of GGUF type `ty` (not a string or array).
unsafe fn read_num(ty: i32, p: *const c_void) -> f64 {
    unsafe {
        match ty {
            T_U8 | T_BOOL => *(p as *const u8) as f64,
            T_I8 => *(p as *const i8) as f64,
            T_U16 => (p as *const u16).read_unaligned() as f64,
            T_I16 => (p as *const i16).read_unaligned() as f64,
            _ => read_wide_num(ty, p),
        }
    }
}

/// [`read_num`] for the types of 4 and 8 bytes (NaN for any other type).
/// SAFETY: as [`read_num`].
unsafe fn read_wide_num(ty: i32, p: *const c_void) -> f64 {
    unsafe {
        match ty {
            T_U32 => (p as *const u32).read_unaligned() as f64,
            T_I32 => (p as *const i32).read_unaligned() as f64,
            T_F32 => (p as *const f32).read_unaligned() as f64,
            T_U64 => (p as *const u64).read_unaligned() as f64,
            T_I64 => (p as *const i64).read_unaligned() as f64,
            T_F64 => (p as *const f64).read_unaligned(),
            _ => f64::NAN,
        }
    }
}
