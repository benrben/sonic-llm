//! The binary's own command-line handling (not part of the `sllm_server` library): reading the flags and turning them
//! into the server's configuration.

pub(crate) mod flags;
pub(crate) mod llama_flags;
pub(crate) mod setup;
