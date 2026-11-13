// Thin C shim between sllm-core (Rust) and the b11669 ggml dylibs.
//
// It holds only what Rust cannot express cleanly:
//   * capture of ggml's log output (a C varargs-free callback with a lock and a growing buffer);
//   * a flat device-property query;
//   * struct layout probes, so a mismatch between the checked-in bindings and the headers fails a test.
// Everything else is reached through the bindings in src/sys.rs.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- log capture ---------------------------------------------------------------------------
// begin: install a ggml log callback that appends every message to an internal buffer (the buffer is cleared).
//        When echo is true the text is also written to stderr, as ggml does by default.
//        Returns false when a capture is already active.
bool   sllm_log_capture_begin(bool echo);
// end: restore the default ggml logger. Returns the number of captured bytes.
size_t sllm_log_capture_end(void);
// copy: copy up to cap-1 captured bytes into dst and NUL-terminate. Returns the full captured length.
size_t sllm_log_capture_copy(char * dst, size_t cap);

// ---- device query --------------------------------------------------------------------------
struct sllm_dev_info {
    char     name[64];
    char     description[128];
    int      type;          // enum ggml_backend_dev_type
    uint64_t memory_free;   // bytes, from ggml_backend_dev_memory
    uint64_t memory_total;  // bytes, from ggml_backend_dev_memory
    bool     async;
    bool     host_buffer;
    bool     buffer_from_host_ptr;
    bool     events;
    bool     mmap_support;
};
// Fill info for device index i of the global registry. Returns false when i is out of range.
bool sllm_dev_query(size_t i, struct sllm_dev_info * info);

// ---- layout probes -------------------------------------------------------------------------
struct sllm_layout {
    uint64_t sizeof_ggml_tensor;
    uint64_t offsetof_type;
    uint64_t offsetof_buffer;
    uint64_t offsetof_ne;
    uint64_t offsetof_nb;
    uint64_t offsetof_op;
    uint64_t offsetof_src;
    uint64_t offsetof_data;
    uint64_t offsetof_name;
    uint64_t sizeof_dev_props;
    uint64_t sizeof_type_traits;
    uint64_t offsetof_to_float;
    uint64_t sizeof_gguf_init_params;
    uint64_t ggml_max_dims;
    uint64_t ggml_max_name;
    uint64_t ggml_max_src;
};
void sllm_layout_probe(struct sllm_layout * out);

#ifdef __cplusplus
}
#endif
