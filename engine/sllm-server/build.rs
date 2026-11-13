//! Build script for sllm-server.
//!
//! * bakes the git commit (`SLLM_GIT_COMMIT`) and the vendored ggml build (`SLLM_GGML_BUILD`, from the directory name
//!   `engine/vendor/ggml-bNNNNN`) into the binary for `--version`;
//! * adds the rpath of the vendored dylibs (`engine/vendor/ggml-b11669/lib`, fetched by `fetch.sh`): the dependencies'
//!   `rustc-link-arg` does not reach this crate's own binary and test executables.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SLLM_GIT_COMMIT");

    let commit = env::var("SLLM_GIT_COMMIT").ok().or_else(|| git(&["rev-parse", "--short=12", "HEAD"])).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=SLLM_GIT_COMMIT={commit}");
    // Re-run when HEAD moves: the HEAD file itself, and the branch ref it points at.
    if let Some(head) = git(&["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={head}");
    }
    if let Some(r) = git(&["symbolic-ref", "-q", "HEAD"]) {
        if let Some(p) = git(&["rev-parse", "--git-path", &r]) {
            println!("cargo:rerun-if-changed={p}");
        }
    }

    let vendor_root = manifest.join("../vendor");
    let dir = std::fs::read_dir(&vendor_root)
        .expect("engine/vendor missing")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .find(|n| n.starts_with("ggml-b"))
        .expect("engine/vendor/ggml-bNNNNN missing");
    let build = dir.trim_start_matches("ggml-").to_string();
    println!("cargo:rustc-env=SLLM_GGML_BUILD={build}");

    // The ggml dylib directory sllm-core linked (D042: the patched libggml-metal by default, the official set with
    // SLLM_GGML_METAL=official), so the server and its llama-ref load one consistent set.
    println!("cargo:rerun-if-env-changed=DEP_SLLM_GGML_LIBDIR");
    let lib = match env::var("DEP_SLLM_GGML_LIBDIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => vendor_root.join(&dir).join("lib").canonicalize().expect("run engine/vendor/ggml-b11669/fetch.sh first"),
    };
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());

    // --engine next: the sllm-next library (Objective-C + Metal), compiled with the system clang, no extra crate
    let proto = manifest.join("../sllm-next/proto");
    for f in ["sn.m", "sn.h", "core.h", "core_mm.inc.h", "sn_check.inc.h", "sn_memory.inc.h", "sn_load.inc.h", "sn_open.inc.h", "sn_attn.inc.h",
              "sn_layers.inc.h", "sn_state.inc.h", "sn_encode.inc.h", "sn_step.inc.h", "sn_cache.inc.h", "sn_seq.inc.h"] {
        println!("cargo:rerun-if-changed={}", proto.join(f).display());
    }
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj = out.join("sn.o");
    let ok = Command::new("clang")
        .args(["-c", "-O2", "-fobjc-arc", "-Wno-unused-function", "-Wno-unused-variable"])
        .arg(proto.join("sn.m"))
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("clang");
    assert!(ok.success(), "clang failed on sn.m");
    let ok = Command::new("ar").arg("rcs").arg(out.join("libsn.a")).arg(&obj).status().expect("ar");
    assert!(ok.success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=sn");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=objc");
}
