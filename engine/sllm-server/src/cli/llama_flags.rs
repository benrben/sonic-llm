//! llama-server's command-line spellings, so a launch script written for `llama-server` starts this server unchanged.
//!
//! * Aliases of flags this server has: `-m/--model`, `-c/--ctx-size` (`--ctx`), `-t/--threads`, `-np/--parallel` (`--agents`),
//!   `-ub/--ubatch-size` (`--ubatch`), `-a/--alias`, `--api-key`, `--api-key-file`, `--reasoning-format`.
//! * Flags with no meaning here, because the behaviour is always on or automatic: accepted, one line on stderr says so, and
//!   nothing else changes ([`IGNORED`]).
//!
//! Any other flag stays an error.

use super::flags::{Arg, Flags};
use sllm_server::api::llama_compat::{auth::parse_keys, ReasoningFormat};

/// The flag a llama-server spelling stands for (the name is returned unchanged when it has no alias).
pub(crate) fn long_name(a: &str) -> &str {
    match a {
        "-m" => "--model",
        "-c" | "--ctx-size" => "--ctx",
        "-t" => "--threads",
        "-np" | "--parallel" => "--parallel",
        "-ub" | "--ubatch-size" => "--ubatch",
        "-a" => "--alias",
        _ => a,
    }
}

/// How an ignored flag reads its value.
#[derive(Clone, Copy, PartialEq)]
enum Takes {
    Nothing,
    One,
    /// `-fa [on|off|auto]`: a value only when the next word is one of these.
    OnOffAuto,
}

/// (spellings, value, why it does nothing here).
const IGNORED: &[(&[&str], Takes, &str)] = &[
    (&["-ngl", "--n-gpu-layers", "--gpu-layers"], Takes::One, "every layer runs on the GPU"),
    (&["--jinja", "--no-jinja"], Takes::Nothing, "the model's chat template always runs as a Jinja template"),
    (&["-fa", "--flash-attn"], Takes::OnOffAuto, "attention kernels are chosen automatically"),
    (&["-b", "--batch-size"], Takes::One, "the scheduler sizes steps itself (see --step-tokens, --chunk)"),
    (&["--mlock", "--no-mmap", "--mmap"], Takes::Nothing, "the weights are always memory-mapped"),
    (&["-kvu", "--kv-unified", "-no-kvu", "--no-kv-unified"], Takes::Nothing, "the KV cache layout is chosen by the engine (see --kv-streams)"),
    (&["--slots", "--no-slots"], Takes::Nothing, "GET /slots is always on"),
    (&["--metrics"], Takes::Nothing, "GET /metrics is always on (JSON, not Prometheus text)"),
    (&["-cb", "--cont-batching", "-nocb", "--no-cont-batching"], Takes::Nothing, "continuous batching is always on"),
];

/// Reads `a` when it is one of the ignored flags (and its value): a note on stderr, nothing else. `false`: not one of them.
pub(crate) fn read_ignored(a: &str, rest: &mut std::slice::Iter<String>) -> bool {
    let Some((_, takes, why)) = IGNORED.iter().find(|(names, _, _)| names.contains(&a)) else { return false };
    match takes {
        Takes::Nothing => {}
        Takes::One => {
            if rest.next().is_none() {
                crate::die(format!("{a} needs a value"));
            }
        }
        Takes::OnOffAuto => {
            if rest.as_slice().first().is_some_and(|v| matches!(v.as_str(), "on" | "off" | "auto")) {
                rest.next();
            }
        }
    }
    eprintln!("Sonic LLM: note: {a} is accepted and ignored: {why}");
    true
}

/// `-a` / `--alias NAME[,MORE]`: the first name is the model's name, the others are listed as its aliases.
pub(crate) fn read_alias(a: &mut Arg, f: &mut Flags) {
    let v = a.val();
    let mut names = v.split(',').map(str::trim).filter(|n| !n.is_empty()).map(str::to_string);
    f.cfg.compat.alias = names.next();
    f.cfg.compat.aliases = names.collect();
}

/// `--api-key KEY[,KEY...]` (also given more than once).
pub(crate) fn read_api_key(a: &mut Arg, f: &mut Flags) {
    let v = a.val();
    f.cfg.compat.api_keys.extend(parse_keys(Some(&v), None));
}

/// `--api-key-file FILE`: one key per line, `#` lines are comments.
pub(crate) fn read_api_key_file(a: &mut Arg, f: &mut Flags) {
    let path = a.path();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| crate::die(format!("--api-key-file {}: {e}", path.display())));
    f.cfg.compat.api_keys.extend(parse_keys(None, Some(&text)));
}

/// `--reasoning-format auto|deepseek|none`.
pub(crate) fn read_reasoning_format(a: &mut Arg, f: &mut Flags) {
    f.cfg.compat.reasoning = a.parse_with_flag(ReasoningFormat::parse);
}

/// `-np N` / `--parallel N`: llama-server's number of slots.
pub(crate) fn read_parallel(a: &mut Arg, f: &mut Flags) {
    f.parallel = Some(a.num());
}

/// After the whole command line: `-np N` is `--agents N` (the number of agents the server is sized for), and that sizing exists only
/// under `--auto-mode`. Without it nothing is sized by `-np`: the lanes' own limits apply, and a note says so (`--agents` itself
/// still needs `--auto-mode`). An explicit `--agents` wins.
pub(crate) fn resolve_parallel(f: &mut Flags) {
    let Some(n) = f.parallel else { return };
    if f.auto_mode {
        f.agents = f.agents.or(Some(n));
    } else {
        eprintln!("Sonic LLM: note: -np/--parallel {n} is --agents {n}, which sizes the cache only with --auto-mode; without it the lanes' own limits apply (--interactive-cap, --background-cap)");
    }
}
