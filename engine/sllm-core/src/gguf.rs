//! GGUF metadata and tensor infos, read with ggml's own `gguf_init_from_file` (`no_alloc = true`).
//!
//! Nothing here maps or reads tensor data. The file handle holds two ggml objects:
//! the `gguf_context` (metadata) and a `ggml_context` that carries one data-less `ggml_tensor`
//! per tensor (name, type, shape).

use std::ffi::{c_char, CStr, CString};
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::error::{Error, Result};
use crate::{cstr, sys};

/// The index of metadata key `key` as `lookup` finds it (`gguf_find_key` of whichever binding the caller has, which answers a
/// negative number for a key that is not there).
pub fn find_key_id(key: &str, lookup: impl FnOnce(&CStr) -> i64) -> Option<i64> {
    let c = CString::new(key).ok()?;
    let id = lookup(&c);
    (id >= 0).then_some(id)
}

/// One metadata value. Arrays are summarised (token tables have 150k+ entries); use the `arr_*` accessors for contents.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array { elem_type: String, len: usize },
}

impl GgufValue {
    /// Any integer variant as i128 (None for floats, strings, bools, arrays).
    pub fn as_int(&self) -> Option<i128> {
        Some(match *self {
            GgufValue::U8(v) => v as i128,
            GgufValue::I8(v) => v as i128,
            GgufValue::U16(v) => v as i128,
            GgufValue::I16(v) => v as i128,
            GgufValue::U32(v) => v as i128,
            GgufValue::I32(v) => v as i128,
            GgufValue::U64(v) => v as i128,
            GgufValue::I64(v) => v as i128,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            GgufValue::F32(v) => Some(v as f64),
            GgufValue::F64(v) => Some(v),
            _ => self.as_int().map(|i| i as f64),
        }
    }
}

/// Name, type, shape and location of one tensor.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TensorInfo {
    pub name: String,
    /// `enum ggml_type` value.
    pub ggml_type: u32,
    /// ggml's name for the type, e.g. "q4_0" (gguf-py spells it "Q4_0").
    pub type_name: String,
    /// ggml `ne`: ne[0] is the contiguous dimension; unused dimensions are 1.
    pub ne: [i64; 4],
    /// Byte offset relative to the start of the tensor data section.
    pub offset: u64,
    pub nbytes: u64,
}

pub struct GgufFile {
    path: PathBuf,
    file_size: u64,
    ctx: *mut sys::gguf_context,
    tensor_ctx: *mut sys::ggml_context,
    version: u32,
    alignment: usize,
    data_offset: usize,
    tensors: Vec<TensorInfo>,
}

impl GgufFile {
    /// Parse metadata and tensor infos. Validates that every tensor lies inside the file.
    pub fn open(path: impl AsRef<Path>) -> Result<GgufFile> {
        let path = path.as_ref().to_path_buf();
        let file_size = std::fs::metadata(&path)?.len();
        let cpath = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| Error::Gguf(format!("path contains NUL: {}", path.display())))?;

        let mut tensor_ctx: *mut sys::ggml_context = std::ptr::null_mut();
        let params = sys::gguf_init_params { no_alloc: true, ctx: &mut tensor_ctx };
        // SAFETY: cpath is a valid C string; params.ctx points at a live local that ggml fills in.
        let ctx = unsafe { sys::gguf_init_from_file(cpath.as_ptr(), params) };
        if ctx.is_null() {
            return Err(Error::Gguf(format!("gguf_init_from_file failed for {}", path.display())));
        }
        // From here `f` owns both objects and frees them on any early return.
        let mut f = GgufFile {
            path,
            file_size,
            ctx,
            tensor_ctx,
            version: 0,
            alignment: 0,
            data_offset: 0,
            tensors: Vec::new(),
        };
        // SAFETY: ctx is a valid gguf_context for the lifetime of `f`.
        unsafe {
            f.version = sys::gguf_get_version(ctx);
            f.alignment = sys::gguf_get_alignment(ctx);
            f.data_offset = sys::gguf_get_data_offset(ctx);
            let n = sys::gguf_get_n_tensors(ctx);
            for i in 0..n {
                let ne_p = sys::gguf_get_tensor_ne(ctx, i);
                let mut ne = [1i64; 4];
                for (d, slot) in ne.iter_mut().enumerate() {
                    *slot = *ne_p.add(d);
                }
                let ty = sys::gguf_get_tensor_type(ctx, i);
                f.tensors.push(TensorInfo {
                    name: cstr(sys::gguf_get_tensor_name(ctx, i)),
                    ggml_type: ty,
                    type_name: cstr(sys::ggml_type_name(ty)),
                    ne,
                    offset: sys::gguf_get_tensor_offset(ctx, i) as u64,
                    nbytes: sys::gguf_get_tensor_size(ctx, i) as u64,
                });
            }
        }
        let data_len = (file_size as usize)
            .checked_sub(f.data_offset)
            .ok_or_else(|| Error::Gguf(format!("data offset {} is past the end of the file ({file_size} bytes)", f.data_offset)))?;
        for t in &f.tensors {
            if t.offset.checked_add(t.nbytes).map_or(true, |end| end > data_len as u64) {
                return Err(Error::Gguf(format!(
                    "tensor {} (offset {}, {} bytes) extends past the end of the file; truncated download?",
                    t.name, t.offset, t.nbytes
                )));
            }
        }
        Ok(f)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn file_size(&self) -> u64 {
        self.file_size
    }
    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn alignment(&self) -> usize {
        self.alignment
    }
    /// Absolute file offset of the tensor data section.
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor(name).is_some()
    }

    // ---- metadata -------------------------------------------------------------------------

    pub fn n_kv(&self) -> usize {
        // SAFETY: valid ctx.
        unsafe { sys::gguf_get_n_kv(self.ctx) as usize }
    }

    fn find(&self, key: &str) -> Option<i64> {
        // SAFETY: valid ctx and C string.
        find_key_id(key, |c| unsafe { sys::gguf_find_key(self.ctx, c.as_ptr()) })
    }

    fn value_at(&self, id: i64) -> GgufValue {
        // SAFETY: id is a valid key index (from gguf_find_key or 0..n_kv); each getter is used with the type reported by gguf_get_kv_type.
        unsafe {
            let c = self.ctx;
            match sys::gguf_get_kv_type(c, id) {
                sys::GGUF_TYPE_FLOAT32 => GgufValue::F32(sys::gguf_get_val_f32(c, id)),
                sys::GGUF_TYPE_FLOAT64 => GgufValue::F64(sys::gguf_get_val_f64(c, id)),
                sys::GGUF_TYPE_BOOL => GgufValue::Bool(sys::gguf_get_val_bool(c, id)),
                sys::GGUF_TYPE_STRING => GgufValue::Str(cstr(sys::gguf_get_val_str(c, id))),
                sys::GGUF_TYPE_ARRAY => GgufValue::Array {
                    elem_type: cstr(sys::gguf_type_name(sys::gguf_get_arr_type(c, id))),
                    len: sys::gguf_get_arr_n(c, id),
                },
                ty => self.int_at(id, ty),
            }
        }
    }

    /// [`Self::value_at`] for the integer types (and the text for a type gguf.h does not have).
    ///
    /// # Safety
    /// As `value_at`: `id` is a valid key index and `ty` its type.
    unsafe fn int_at(&self, id: i64, ty: sys::gguf_type) -> GgufValue {
        let c = self.ctx;
        unsafe {
            match ty {
                sys::GGUF_TYPE_UINT8 => GgufValue::U8(sys::gguf_get_val_u8(c, id)),
                sys::GGUF_TYPE_INT8 => GgufValue::I8(sys::gguf_get_val_i8(c, id)),
                sys::GGUF_TYPE_UINT16 => GgufValue::U16(sys::gguf_get_val_u16(c, id)),
                sys::GGUF_TYPE_INT16 => GgufValue::I16(sys::gguf_get_val_i16(c, id)),
                sys::GGUF_TYPE_UINT32 => GgufValue::U32(sys::gguf_get_val_u32(c, id)),
                sys::GGUF_TYPE_INT32 => GgufValue::I32(sys::gguf_get_val_i32(c, id)),
                sys::GGUF_TYPE_UINT64 => GgufValue::U64(sys::gguf_get_val_u64(c, id)),
                sys::GGUF_TYPE_INT64 => GgufValue::I64(sys::gguf_get_val_i64(c, id)),
                other => GgufValue::Str(format!("<unknown gguf type {other}>")),
            }
        }
    }

    pub fn get(&self, key: &str) -> Option<GgufValue> {
        self.find(key).map(|id| self.value_at(id))
    }

    /// Every key and value, in file order (arrays summarised).
    pub fn metadata(&self) -> Vec<(String, GgufValue)> {
        (0..self.n_kv() as i64)
            // SAFETY: id in range.
            .map(|id| (cstr(unsafe { sys::gguf_get_key(self.ctx, id) }), self.value_at(id)))
            .collect()
    }

    pub fn get_str(&self, key: &str) -> Option<String> {
        match self.get(key)? {
            GgufValue::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Integer value of any integer type, as u32 (None if absent, not an integer, or out of range).
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get(key)?.as_int().and_then(|v| u32::try_from(v).ok())
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_int().and_then(|v| u64::try_from(v).ok())
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key)?.as_f64().map(|v| v as f32)
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key)? {
            GgufValue::Bool(b) => Some(b),
            _ => None,
        }
    }

    /// Length of an array key.
    pub fn arr_len(&self, key: &str) -> Option<usize> {
        match self.get(key)? {
            GgufValue::Array { len, .. } => Some(len),
            _ => None,
        }
    }

    /// String element `i` of a string-array key.
    pub fn arr_str(&self, key: &str, i: usize) -> Option<String> {
        let id = self.find(key)?;
        // SAFETY: bounds-checked against gguf_get_arr_n; element type checked first.
        unsafe {
            if sys::gguf_get_kv_type(self.ctx, id) != sys::GGUF_TYPE_ARRAY
                || sys::gguf_get_arr_type(self.ctx, id) != sys::GGUF_TYPE_STRING
                || i >= sys::gguf_get_arr_n(self.ctx, id)
            {
                return None;
            }
            Some(cstr(sys::gguf_get_arr_str(self.ctx, id, i) as *const c_char))
        }
    }

    /// JSON with the same shape as `engine/scripts/gguf_dump_py.py`, used for the cross-check against gguf-py.
    pub fn dump_json(&self) -> serde_json::Value {
        use serde_json::{json, Map, Value};
        let mut kv = Map::new();
        let mut arrays = Map::new();
        for (k, v) in self.metadata() {
            match v {
                GgufValue::Array { len, .. } => {
                    arrays.insert(k, json!(len));
                }
                // gguf-py reports f32 values as the f64 that holds them exactly.
                GgufValue::F32(x) => {
                    kv.insert(k, json!(x as f64));
                }
                other => {
                    kv.insert(k, serde_json::to_value(other).expect("serialise gguf value"));
                }
            }
        }
        let tensors: Vec<Value> = self
            .tensors
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "type": t.type_name.to_uppercase(),
                    "ne": t.ne,
                    "offset": t.offset,
                    "nbytes": t.nbytes,
                })
            })
            .collect();
        json!({
            "version": self.version,
            "alignment": self.alignment,
            "data_offset": self.data_offset,
            "n_tensors": self.tensors.len(),
            "kv": kv,
            "arrays": arrays,
            "tensors": tensors,
        })
    }
}

impl Drop for GgufFile {
    fn drop(&mut self) {
        // SAFETY: both pointers were returned by ggml for this file and are freed exactly once.
        // The tensor structs live in tensor_ctx, so nothing may use them after this.
        unsafe {
            sys::gguf_free(self.ctx);
            if !self.tensor_ctx.is_null() {
                sys::ggml_free(self.tensor_ctx);
            }
        }
    }
}

#[cfg(test)]
mod tests;
