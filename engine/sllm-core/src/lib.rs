//! sllm-core: what sllm-server shares with the vendored llama.cpp b11669 ggml dylibs besides libllama itself.
//!
//! The model math runs in libllama (see `llama-ref` and `sllm-server`'s `LlamaEngine`); this crate is the slim rest:
//!
//! * [`gguf`]: GGUF metadata and tensor infos (ggml's `gguf_init_from_file` with `no_alloc`).
//! * [`backend`]: the ggml device registry, initialised once with its log captured.
//! * [`caps`]: the [`caps::Capabilities`] record (device facts and the memory budget).
//! * [`mem`]: the process memory snapshot (`proc_pid_rusage`).
//! * [`sys`]: raw bindings (checked in; regenerate with `--features regen-bindings`).
//!
//! The native, model-specific engine that used to live here (hand-built ggml graphs for Qwen3 and Gemma 3, the paged KV
//! pool, the batched executor) is preserved at the repository tag `native-engine-final`.

pub mod backend;
pub mod caps;
pub mod error;
pub mod gguf;
pub mod mem;
pub mod sys;

#[cfg(test)]
pub(crate) mod testutil;

pub use error::{Error, Result};

use std::ffi::{c_char, CStr};

mod shim {
    //! Hand-written declarations for `shim/sllm_shim.h` (kept in step with the header by the layout test).
    use std::ffi::{c_char, c_int};

    #[repr(C)]
    pub struct DevInfo {
        pub name: [c_char; 64],
        pub description: [c_char; 128],
        pub type_: c_int,
        pub memory_free: u64,
        pub memory_total: u64,
        pub async_: bool,
        pub host_buffer: bool,
        pub buffer_from_host_ptr: bool,
        pub events: bool,
        pub mmap_support: bool,
    }

    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy)]
    pub struct Layout {
        pub sizeof_ggml_tensor: u64,
        pub offsetof_type: u64,
        pub offsetof_buffer: u64,
        pub offsetof_ne: u64,
        pub offsetof_nb: u64,
        pub offsetof_op: u64,
        pub offsetof_src: u64,
        pub offsetof_data: u64,
        pub offsetof_name: u64,
        pub sizeof_dev_props: u64,
        pub sizeof_type_traits: u64,
        pub offsetof_to_float: u64,
        pub sizeof_gguf_init_params: u64,
        pub ggml_max_dims: u64,
        pub ggml_max_name: u64,
        pub ggml_max_src: u64,
    }

    extern "C" {
        pub fn sllm_log_capture_begin(echo: bool) -> bool;
        pub fn sllm_log_capture_end() -> usize;
        pub fn sllm_log_capture_copy(dst: *mut c_char, cap: usize) -> usize;
        pub fn sllm_dev_query(i: usize, info: *mut DevInfo) -> bool;
        pub fn sllm_layout_probe(out: *mut Layout);
    }
}

pub use shim::Layout;
pub(crate) use shim::DevInfo;

/// Copy a NUL-terminated C string into an owned `String` (lossy on invalid UTF-8). NULL gives "".
pub(crate) fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: callers pass pointers returned by ggml, which are NUL-terminated and live for the call.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

pub(crate) fn cstr_array(a: &[c_char]) -> String {
    let bytes: Vec<u8> = a.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Start capturing ggml's log output. Returns false if a capture is already active.
pub(crate) fn log_capture_begin(echo: bool) -> bool {
    // SAFETY: plain C call; the shim serialises access with a mutex.
    unsafe { shim::sllm_log_capture_begin(echo) }
}

/// Stop capturing, restore the default logger, and return everything captured.
pub(crate) fn log_capture_end() -> String {
    // SAFETY: as above; the buffer is sized from the length the shim reports.
    unsafe {
        let n = shim::sllm_log_capture_end();
        let mut buf = vec![0u8; n + 1];
        shim::sllm_log_capture_copy(buf.as_mut_ptr() as *mut c_char, buf.len());
        buf.truncate(n);
        String::from_utf8_lossy(&buf).into_owned()
    }
}

pub(crate) fn dev_query(i: usize) -> Option<DevInfo> {
    // SAFETY: DevInfo is plain data; the shim fully initialises it on success.
    unsafe {
        let mut info = std::mem::MaybeUninit::<DevInfo>::zeroed();
        if shim::sllm_dev_query(i, info.as_mut_ptr()) {
            Some(info.assume_init())
        } else {
            None
        }
    }
}

pub fn layout_probe() -> Layout {
    let mut l = Layout::default();
    // SAFETY: Layout is repr(C) plain data.
    unsafe { shim::sllm_layout_probe(&mut l) };
    l
}

/// ggml version and commit of the linked dylibs, e.g. ("0.27.0", "620d222...").
pub fn ggml_version() -> (String, String) {
    // SAFETY: both return static NUL-terminated strings.
    unsafe { (cstr(sys::ggml_version()), cstr(sys::ggml_commit())) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// The checked-in bindings must describe the structs the linked headers/dylibs use.
    #[test]
    fn bindings_match_c_layout() {
        let l = layout_probe();
        assert_eq!(l.sizeof_ggml_tensor as usize, size_of::<sys::ggml_tensor>());
        assert_eq!(l.offsetof_type as usize, offset_of!(sys::ggml_tensor, type_));
        assert_eq!(l.offsetof_buffer as usize, offset_of!(sys::ggml_tensor, buffer));
        assert_eq!(l.offsetof_ne as usize, offset_of!(sys::ggml_tensor, ne));
        assert_eq!(l.offsetof_nb as usize, offset_of!(sys::ggml_tensor, nb));
        assert_eq!(l.offsetof_op as usize, offset_of!(sys::ggml_tensor, op));
        assert_eq!(l.offsetof_src as usize, offset_of!(sys::ggml_tensor, src));
        assert_eq!(l.offsetof_data as usize, offset_of!(sys::ggml_tensor, data));
        assert_eq!(l.offsetof_name as usize, offset_of!(sys::ggml_tensor, name));
        assert_eq!(l.sizeof_dev_props as usize, size_of::<sys::ggml_backend_dev_props>());
        assert_eq!(l.sizeof_type_traits as usize, size_of::<sys::ggml_type_traits>());
        assert_eq!(l.offsetof_to_float as usize, offset_of!(sys::ggml_type_traits, to_float));
        assert_eq!(l.sizeof_gguf_init_params as usize, size_of::<sys::gguf_init_params>());
        assert_eq!(l.ggml_max_dims, sys::GGML_MAX_DIMS as u64);
        assert_eq!(l.ggml_max_name, sys::GGML_MAX_NAME as u64);
        assert_eq!(l.ggml_max_src, sys::GGML_MAX_SRC as u64);
    }

    #[test]
    fn linked_ggml_is_b11669() {
        let (v, c) = ggml_version();
        assert_eq!(v, "0.27.0");
        assert!(c.starts_with("620d222"), "unexpected ggml commit {c}");
    }
}
