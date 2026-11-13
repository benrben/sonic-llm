//! The ggml device registry, initialised once with its log captured.
//!
//! The registry of the linked `libggml` initialises Metal the first time it is touched, and ggml prints the
//! device facts (`has tensor`, `use residency sets`, the recommended working set ...) only then. So every
//! entry point goes through [`registry`], which installs the log capture before the first touch and keeps the
//! text for [`crate::caps`].

use std::sync::OnceLock;

use crate::{log_capture_begin, log_capture_end, sys};

/// What the first touch of the ggml registry printed, and how many devices it found.
#[derive(Debug)]
pub struct RegistryInfo {
    pub init_log: String,
    pub n_devices: usize,
}

static REGISTRY: OnceLock<RegistryInfo> = OnceLock::new();

/// Initialise the ggml registry (once per process), capturing its log output.
pub fn registry() -> &'static RegistryInfo {
    REGISTRY.get_or_init(|| {
        let capturing = log_capture_begin(false);
        // SAFETY: plain query; this is the call that constructs the registry and initialises Metal.
        let n_devices = unsafe { sys::ggml_backend_dev_count() };
        // Query every device once so lazy per-device init logging also lands in the capture.
        for i in 0..n_devices {
            let _ = crate::dev_query(i);
        }
        let init_log = if capturing { log_capture_end() } else { String::new() };
        RegistryInfo { init_log, n_devices }
    })
}
