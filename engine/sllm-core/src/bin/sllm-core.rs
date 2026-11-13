//! `sllm-core` command line: device capabilities and the GGUF header dump (no model math: that is libllama's).
//!
//! JSON goes to stdout; ggml's own log lines go to stderr. `caps` touches Metal, so wrap it in the lock protocol of
//! `playground/src/sllm_playground/gpu.py` (rule 094); `dump` reads only the file header.

use std::process::ExitCode;

use serde_json::json;
use sllm_core::caps::Capabilities;
use sllm_core::gguf::GgufFile;

const USAGE: &str = "usage:
  sllm-core caps [--with-log]
  sllm-core dump <model.gguf>                       # GGUF header only (no GPU): metadata, tensor infos";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("sllm-core: {e}");
            ExitCode::from(1)
        }
    }
}

fn run(args: &[String]) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let Some(cmd) = args.first() else {
        eprintln!("{USAGE}");
        return Ok(ExitCode::from(64));
    };
    let rest = &args[1..];
    match cmd.as_str() {
        "caps" => caps(rest),
        "dump" => dump(rest),
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        other => Err(format!("unknown command {other:?}\n{USAGE}").into()),
    }
}

/// `caps [--with-log]`: the device capability record.
fn caps(rest: &[String]) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let mut v = serde_json::to_value(Capabilities::probe()?)?;
    if rest.iter().any(|a| a == "--with-log") {
        v["init_log"] = json!(sllm_core::backend::registry().init_log);
    }
    println!("{}", serde_json::to_string_pretty(&v)?);
    Ok(ExitCode::SUCCESS)
}

/// `dump <model.gguf>`: the GGUF header.
fn dump(rest: &[String]) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let path = rest.first().ok_or(USAGE)?;
    let g = GgufFile::open(path)?;
    println!("{}", serde_json::to_string_pretty(&g.dump_json())?);
    Ok(ExitCode::SUCCESS)
}
