//! What the machine offers: its memory and memory bandwidth (a config value, not a probe).

use std::path::{Path, PathBuf};

/// What the machine offers. The bandwidth is a config value, not a probe.
#[derive(Debug, Clone)]
pub struct Machine {
    pub brand: String,
    pub mem_bytes: u64,
    pub bandwidth_gbs: Option<f64>,
    /// Where the bandwidth came from.
    pub bandwidth_source: String,
}

pub(crate) fn sysctl(name: &str) -> Option<String> {
    let out = std::process::Command::new("sysctl").args(["-n", name]).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string()).filter(|s| !s.is_empty())
}

pub fn default_machine_config_path() -> PathBuf {
    std::env::var_os("SLLM_MACHINE_CONFIG").map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("config").join("machine.toml"))
}

/// `bandwidth_gbs` of the table `[machine."<brand>"]` in a machine.toml (the small TOML subset the file uses).
pub fn parse_machine_bandwidth(text: &str, brand: &str) -> Option<f64> {
    let mut in_table = false;
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if let Some(rest) = line.strip_prefix("[machine.") {
            let name = rest.trim_end_matches(']').trim().trim_matches('"');
            in_table = name == brand;
        } else if in_table {
            if let Some((k, v)) = line.split_once('=') {
                if k.trim() == "bandwidth_gbs" {
                    return v.trim().parse().ok();
                }
            }
        }
    }
    None
}

impl Machine {
    pub fn detect() -> Machine {
        let brand = sysctl("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".into());
        let mem_bytes = sysctl("hw.memsize").and_then(|s| s.parse().ok()).unwrap_or(0);
        let (bandwidth_gbs, bandwidth_source) = if let Some(v) = std::env::var("SLLM_BANDWIDTH_GBS").ok().and_then(|s| s.parse::<f64>().ok()) {
            (Some(v), "env SLLM_BANDWIDTH_GBS".to_string())
        } else {
            let path = default_machine_config_path();
            match std::fs::read_to_string(&path) {
                Ok(t) => match parse_machine_bandwidth(&t, &brand) {
                    Some(v) => (Some(v), format!("{} [machine.\"{brand}\"]", path.display())),
                    None => (None, format!("no [machine.\"{brand}\"] in {}", path.display())),
                },
                Err(e) => (None, format!("{}: {e}", path.display())),
            }
        };
        Machine { brand, mem_bytes, bandwidth_gbs, bandwidth_source }
    }

    /// Cache key of this machine for stored calibrations.
    pub fn key(&self) -> String {
        format!("{}/{}GiB", self.brand, self.mem_bytes >> 30)
    }
}
