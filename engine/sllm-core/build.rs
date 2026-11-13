//! Build script for sllm-core.
//!
//! * links the vendored b11669 ggml dylibs and bakes their directory in as an rpath. By default that is the official
//!   set in `engine/vendor/ggml-b11669/lib` (fetched by `fetch.sh`): with the native engine gone (2026-10-28), SLLM's
//!   patched libggml-metal only slowed the generic engine (ITL 14.4 vs 12.4 ms, SmolLM3 batch-8) and flipped a near-tie
//!   token against llama-server. `SLLM_GGML_METAL=sllm` links `engine/vendor/sllm-metal-b11669/lib` (D042) instead. The chosen directory is exported as `DEP_SLLM_GGML_LIBDIR` (the `links` key) for dependants'
//!   rpaths (sllm-server). At run time `SLLM_QMM=0` turns the qmm path off without relinking;
//! * compiles `shim/sllm_shim.c` with the `cc` crate;
//! * with `--features regen-bindings`, regenerates `src/sys.rs` with bindgen (needs libclang).
//!   A normal build never runs bindgen, so it needs no libclang.

use std::env;
use std::path::PathBuf;

/// Functions the crate calls. Anything not listed here is not bound.
#[cfg(feature = "regen-bindings")]
const FUNCTIONS: &[&str] = &[
    // ggml core
    "ggml_version", "ggml_commit", "ggml_log_set", "ggml_init", "ggml_free",
    "ggml_tensor_overhead", "ggml_graph_overhead", "ggml_new_tensor_1d", "ggml_new_tensor_2d",
    "ggml_mul_mat", "ggml_new_graph", "ggml_build_forward_expand", "ggml_graph_node",
    "ggml_graph_n_nodes", "ggml_set_input", "ggml_set_output", "ggml_set_name", "ggml_get_name",
    "ggml_nbytes", "ggml_nelements", "ggml_nrows", "ggml_row_size", "ggml_blck_size",
    "ggml_type_size", "ggml_type_name", "ggml_is_quantized", "ggml_get_first_tensor",
    "ggml_get_next_tensor", "ggml_get_tensor", "ggml_get_type_traits", "ggml_get_max_tensor_size",
    // ggml core: graph building (task 2, the dense forward pass)
    "ggml_new_tensor_3d", "ggml_new_tensor_4d", "ggml_new_graph_custom", "ggml_graph_overhead_custom",
    "ggml_graph_size", "ggml_get_rows", "ggml_set_rows", "ggml_rms_norm", "ggml_mul", "ggml_add",
    "ggml_scale", "ggml_reshape_2d", "ggml_reshape_3d", "ggml_view_2d", "ggml_view_3d", "ggml_view_4d",
    "ggml_permute", "ggml_cont_2d", "ggml_rope_ext", "ggml_flash_attn_ext",
    "ggml_flash_attn_ext_set_n_kv_max", "ggml_flash_attn_ext_add_sinks", "ggml_prec_set_acc",
    "ggml_swiglu_split", "ggml_geglu_split", "ggml_tanh", "ggml_argmax", "ggml_build_forward_expand",
    "ggml_build_forward_order", "ggml_op_name", "ggml_op_desc", "ggml_fp32_to_fp16", "ggml_fp16_to_fp32",
    "ggml_element_size", "ggml_is_contiguous", "ggml_tensor_overhead",
    // attention rotation (D034)
    "ggml_reshape_4d", "ggml_mul_mat_set_hint",
    // kvpool spike (phase 1 task 3a)
    "ggml_new_tensor", "ggml_cpy", "ggml_cast",
    // ggml-alloc
    "ggml_backend_alloc_ctx_tensors",
    // ggml-backend: scheduler, buffers
    "ggml_backend_sched_new", "ggml_backend_sched_free", "ggml_backend_sched_reserve",
    "ggml_backend_sched_alloc_graph", "ggml_backend_sched_graph_compute",
    "ggml_backend_sched_graph_compute_async", "ggml_backend_sched_synchronize", "ggml_backend_sched_reset",
    "ggml_backend_sched_get_n_splits", "ggml_backend_sched_get_tensor_backend",
    "ggml_backend_sched_get_buffer_size", "ggml_backend_sched_set_eval_callback",
    "ggml_backend_get_default_buffer_type", "ggml_backend_dev_host_buffer_type", "ggml_backend_buffer_clear",
    "ggml_backend_tensor_get_async", "ggml_backend_tensor_set_async",
    // ggml-backend: devices, registry, buffers, tensors, graph compute
    "ggml_backend_dev_count", "ggml_backend_dev_get", "ggml_backend_dev_by_name",
    "ggml_backend_dev_by_type", "ggml_backend_dev_name", "ggml_backend_dev_description",
    "ggml_backend_dev_memory", "ggml_backend_dev_type", "ggml_backend_dev_get_props",
    "ggml_backend_dev_backend_reg", "ggml_backend_dev_init", "ggml_backend_dev_buffer_type",
    "ggml_backend_dev_buffer_from_host_ptr", "ggml_backend_dev_supports_op",
    "ggml_backend_reg_count", "ggml_backend_reg_get", "ggml_backend_reg_name",
    "ggml_backend_buffer_name", "ggml_backend_buffer_free", "ggml_backend_buffer_get_base",
    "ggml_backend_buffer_get_size", "ggml_backend_buffer_is_host", "ggml_backend_buffer_set_usage",
    "ggml_backend_tensor_alloc", "ggml_backend_tensor_get", "ggml_backend_tensor_set",
    "ggml_backend_name", "ggml_backend_free", "ggml_backend_synchronize",
    "ggml_backend_graph_compute", "ggml_backend_get_device", "ggml_backend_buft_name",
    // ggml-cpu and ggml-metal
    "ggml_backend_cpu_init", "ggml_backend_cpu_set_n_threads", "ggml_backend_is_metal",
    "ggml_backend_metal_supports_family",
    // gguf (read side only)
    "gguf_init_from_file", "gguf_free", "gguf_type_name", "gguf_get_version", "gguf_get_alignment",
    "gguf_get_data_offset", "gguf_get_n_kv", "gguf_find_key", "gguf_get_key", "gguf_get_kv_type",
    "gguf_get_arr_type", "gguf_get_val_u8", "gguf_get_val_i8", "gguf_get_val_u16",
    "gguf_get_val_i16", "gguf_get_val_u32", "gguf_get_val_i32", "gguf_get_val_f32",
    "gguf_get_val_u64", "gguf_get_val_i64", "gguf_get_val_f64", "gguf_get_val_bool",
    "gguf_get_val_str", "gguf_get_arr_n", "gguf_get_arr_data", "gguf_get_arr_str",
    "gguf_get_n_tensors", "gguf_find_tensor", "gguf_get_tensor_offset", "gguf_get_tensor_name",
    "gguf_get_tensor_ne", "gguf_get_tensor_type", "gguf_get_tensor_size",
];

#[cfg(feature = "regen-bindings")]
fn regen_bindings(include: &std::path::Path, out: &std::path::Path) {
    let mut b = bindgen::Builder::default()
        .header(PathBuf::from("shim/wrapper.h").to_str().unwrap())
        .clang_arg(format!("-I{}", include.display()))
        .use_core()
        .layout_tests(false)
        .derive_default(true)
        .derive_debug(true)
        .prepend_enum_name(false)
        .default_enum_style(bindgen::EnumVariation::Consts)
        .generate_comments(false)
        .formatter(bindgen::Formatter::None)
        .size_t_is_usize(true)
        .raw_line("//! Raw ggml/gguf bindings for the vendored b11669 headers.")
        .raw_line("//! Generated by `cargo build -p sllm-core --features regen-bindings`; do not edit by hand.")
        .raw_line("#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]")
        // Constants the crate relies on.
        .allowlist_var("GGML_MAX_DIMS|GGML_MAX_SRC|GGML_MAX_NAME|GGML_DEFAULT_GRAPH_SIZE|GGUF_DEFAULT_ALIGNMENT")
        // Types are pulled in transitively by the functions; name the ones we touch directly.
        .allowlist_type("ggml_prec|ggml_op_hint|ggml_tensor|ggml_type|gguf_type|ggml_status|ggml_op|ggml_init_params|gguf_init_params|ggml_backend_dev_props|ggml_backend_dev_caps|ggml_backend_dev_type|ggml_backend_buffer_usage|ggml_log_level|ggml_type_traits");
    for f in FUNCTIONS {
        b = b.allowlist_function(format!("^{f}$"));
    }
    b.generate()
        .expect("bindgen failed (is libclang available? try DYLD_FALLBACK_LIBRARY_PATH or LIBCLANG_PATH)")
        .write_to_file(out)
        .expect("write src/sys.rs");
    // No rustfmt/prettyplease output is usable here (bindgen emits one long line), so break lines ourselves.
    let raw = std::fs::read_to_string(out).expect("read src/sys.rs");
    std::fs::write(out, break_lines(&raw)).expect("write src/sys.rs");
}

/// Put one item per line, and one field / parameter-list entry per line inside braces.
#[cfg(feature = "regen-bindings")]
fn break_lines(src: &str) -> String {
    let (mut out, mut brace, mut paren, mut bracket) = (String::new(), 0i32, 0i32, 0i32);
    let mut prev = ' ';
    for line in src.lines() {
        if line.starts_with("/*") || line.starts_with("//") || line.starts_with("#!") || line.is_empty() {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        for c in line.chars() {
            out.push(c);
            match c {
                '{' => brace += 1,
                '}' => {
                    brace -= 1;
                    if brace == 0 && paren == 0 {
                        out.push('\n');
                    }
                }
                '(' => paren += 1,
                ')' => paren -= 1,
                '[' => bracket += 1,
                ']' => bracket -= 1,
                ';' if paren == 0 && bracket == 0 => out.push('\n'),
                ',' if paren == 0 && bracket == 0 && brace >= 1 => out.push('\n'),
                _ => {}
            }
            prev = c;
        }
    }
    let _ = prev;
    out.lines().map(str::trim_start).collect::<Vec<_>>().join("\n") + "\n"
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = manifest.join("../vendor/ggml-b11669").canonicalize().expect("engine/vendor/ggml-b11669 missing");
    let include = vendor.join("include");
    println!("cargo:rerun-if-env-changed=SLLM_GGML_METAL");
    // default: the official b11669 Metal backend plus the row-exact patch (D061: exact decode for ~2%); `official` links the
    // untouched official set (llama.cpp's own kernels, bit for bit)
    let which = env::var("SLLM_GGML_METAL").unwrap_or_else(|_| "rowexact".into());
    let lib = match which.as_str() {
        "official" => vendor.join("lib"),
        "sllm" => {
            let d = manifest.join("../vendor/sllm-metal-b11669/lib");
            if !d.join("libggml-metal.0.dylib").exists() {
                panic!(
                    "{} not found. Run engine/vendor/sllm-metal-b11669/build.sh (after ggml-b11669/fetch.sh), \
                     or build with SLLM_GGML_METAL=official.",
                    d.display()
                );
            }
            d.canonicalize().expect("canonicalize sllm-metal lib dir")
        }
        "rowexact" => {
            let d = manifest.join("../vendor/ggml-rowexact-b11669/lib");
            if !d.join("libggml-metal.0.dylib").exists() {
                panic!(
                    "{} not found. Run engine/vendor/ggml-rowexact-b11669/build.sh (after ggml-b11669/fetch.sh), \
                     or build with SLLM_GGML_METAL=official.",
                    d.display()
                );
            }
            d.canonicalize().expect("canonicalize ggml-rowexact lib dir")
        }
        other => panic!("SLLM_GGML_METAL={other}: expected rowexact (default), official or sllm"),
    };
    println!("cargo:rustc-env=SLLM_GGML_METAL_LINKED={which}");
    println!("cargo:libdir={}", lib.display());

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shim/sllm_shim.c");
    println!("cargo:rerun-if-changed=shim/sllm_shim.h");
    println!("cargo:rerun-if-changed=shim/wrapper.h");
    println!("cargo:rerun-if-changed={}", include.display());

    for name in ["libggml.0.dylib", "libggml-base.0.dylib", "libggml-cpu.0.dylib", "libggml-metal.0.dylib", "libggml-rpc.0.dylib"] {
        let p = lib.join(name);
        if !p.exists() {
            panic!(
                "{} not found. Run engine/vendor/ggml-b11669/fetch.sh to copy and verify the ggml dylibs.",
                p.display()
            );
        }
        println!("cargo:rerun-if-changed={}", p.display());
    }

    #[cfg(feature = "regen-bindings")]
    regen_bindings(&include, &manifest.join("src/sys.rs"));

    cc::Build::new()
        .file("shim/sllm_shim.c")
        .include("shim")
        .include(&include)
        .flag_if_supported("-std=c11")
        .warnings(true)
        .compile("sllm_shim");

    println!("cargo:rustc-link-search=native={}", lib.display());
    // libggml holds the backend registry; ggml-base the core, buffers and gguf; the others the backends.
    for l in ["ggml", "ggml-base", "ggml-cpu", "ggml-metal"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
}
