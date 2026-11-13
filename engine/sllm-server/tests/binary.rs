//! The `sllm-server` binary: --version, argument errors, --fake end to end, SIGTERM exits cleanly.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_sllm-server");

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

#[test]
fn version_prints_commit_ggml_and_crate_version() {
    let out = Command::new(BIN).arg("--version").output().unwrap();
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(s.starts_with(&format!("Sonic LLM server (sllm-server) {} (commit ", env!("CARGO_PKG_VERSION"))), "{s}");
    assert!(s.trim_end().ends_with("ggml b11669"), "{s}");
    let commit = s.split("(commit ").nth(1).unwrap().split(')').next().unwrap();
    assert!(commit.len() >= 7 && commit.chars().all(|c| c.is_ascii_hexdigit()) || commit == "unknown", "{commit}");
}

#[test]
fn argument_errors_exit_2_with_a_message() {
    for (args, needle) in [
        (vec![], "--fake"),
        (vec!["--fake", "--http-threads", "0"], "at least 1"),
        (vec!["--fake", "--cpu-threads", "0"], "--cpu-threads must be at least 1"),
        (vec!["--fake", "--threads", "1"], "--threads must be at least 2"),
        (vec!["--fake", "--threads", "many"], "bad value"),
        (vec!["--fake", "--port", "abc"], "bad value"),
        (vec!["--fake", "--model", "/nonexistent.gguf"], "GGUF not found"),
        (vec!["--bogus"], "unknown argument"),
        (vec!["--fake", "--port"], "needs a value"),
    ] {
        let out = Command::new(BIN).args(&args).output().unwrap();
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        assert!(String::from_utf8_lossy(&out.stderr).contains(needle), "{args:?}: {}", String::from_utf8_lossy(&out.stderr));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn fake_server_serves_and_stops_on_sigterm() {
    let port = free_port();
    let mut child = Command::new(BIN)
        .args(["--fake", "--port", &port.to_string(), "--http-threads", "3", "--fake-token-ms", "0", "--fake-prefill-ms", "0", "--fake-seed", "42"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let mut first = String::new();
    out.read_line(&mut first).unwrap();
    assert!(first.starts_with("Sonic LLM server (sllm-server) "), "{first}");
    let mut second = String::new();
    out.read_line(&mut second).unwrap();
    assert!(second.contains(&format!("listening on http://127.0.0.1:{port}")) && second.contains("http threads 3"), "{second}");

    let c = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let t0 = Instant::now();
    while c.get(format!("{base}/health")).send().await.map(|r| r.status() != 200).unwrap_or(true) {
        assert!(t0.elapsed() < Duration::from_secs(5), "server never became healthy");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let v: Value = c.post(format!("{base}/completion")).json(&json!({"prompt": [1, 2, 3], "n_predict": 4, "return_tokens": true})).send().await.unwrap().json().await.unwrap();
    assert_eq!(v["tokens_predicted"], 4);
    let caps: Value = c.get(format!("{base}/caps")).send().await.unwrap().json().await.unwrap();
    assert_eq!(caps["engine"]["fake"]["seed"], 42);
    assert_eq!(c.get(format!("{base}/metrics")).send().await.unwrap().json::<Value>().await.unwrap()["http_threads"], 3);

    Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap();
    let t0 = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s;
        }
        assert!(t0.elapsed() < Duration::from_secs(5), "SIGTERM did not stop the server");
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert!(status.success(), "exit {status:?}");
    let mut rest = String::new();
    out.read_to_string(&mut rest).unwrap();
    assert!(rest.contains("stopped"), "{rest}");
}

#[test]
fn port_in_use_is_an_error_not_a_hang() {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let out = Command::new(BIN).args(["--fake", "--port", &port.to_string()]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot start"));
}

/// Start a fake server with `args`, wait until it is healthy, return `(child, startup line, /caps)`; the caller kills it.
async fn fake_caps(args: &[&str]) -> (std::process::Child, String, Value) {
    let port = free_port();
    let mut child = Command::new(BIN)
        .args(["--fake", "--port", &port.to_string(), "--fake-token-ms", "0", "--fake-prefill-ms", "0"])
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let (mut version, mut line) = (String::new(), String::new());
    out.read_line(&mut version).unwrap();
    out.read_line(&mut line).unwrap();
    let c = reqwest::Client::builder().no_proxy().build().unwrap();
    let caps = c.get(format!("http://127.0.0.1:{port}/caps")).send().await.unwrap().json::<Value>().await.unwrap();
    (child, line, caps)
}

fn stop(mut child: std::process::Child) {
    child.kill().unwrap();
    child.wait().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn threads_flag_splits_the_budget_and_overrides_are_recorded_in_caps() {
    let (child, line, caps) = fake_caps(&["--threads", "8", "--threads-config", "/nonexistent/threads.toml"]).await;
    let t = &caps["threads"];
    assert_eq!((t["budget"].as_u64(), t["effective"]["http"].as_u64(), t["effective"]["ggml_cpu"].as_u64()), (Some(8), Some(2), Some(6)), "{t}");
    assert_eq!(t["source"]["kind"], "flag");
    assert_eq!(t["over_budget"], false);
    assert!(t["split_rule"].as_str().unwrap().contains("budget / 4"));
    assert!(line.contains("http threads 2, ggml cpu threads 6, thread budget 8"), "{line}");
    stop(child);

    let (child, line, caps) = fake_caps(&["--threads", "8", "--cpu-threads", "2", "--http-threads", "3"]).await;
    let t = &caps["threads"];
    assert_eq!((t["effective"]["http"].as_u64(), t["effective"]["ggml_cpu"].as_u64()), (Some(3), Some(2)), "{t}");
    assert_eq!((t["overrides"]["cpu_threads"].as_u64(), t["overrides"]["http_threads"].as_u64()), (Some(2), Some(3)));
    assert_eq!((t["split"]["http"].as_u64(), t["split"]["ggml_cpu"].as_u64()), (Some(2), Some(6)), "the split itself is unchanged");
    assert!(line.contains("http threads 3, ggml cpu threads 2"), "{line}");
    stop(child);
}

#[tokio::test(flavor = "multi_thread")]
async fn default_budget_is_the_stored_value_for_this_machine_class() {
    // The file is keyed by the real chip, so the test writes one entry for the machine it runs on (sysctl, as the server does).
    let sysctl = |n: &str| String::from_utf8(Command::new("sysctl").args(["-n", n]).output().unwrap().stdout).unwrap().trim().to_string();
    let (brand, p, e) = (sysctl("machdep.cpu.brand_string"), sysctl("hw.perflevel0.physicalcpu"), sysctl("hw.perflevel1.physicalcpu"));
    if brand.is_empty() || p.is_empty() || e.is_empty() {
        eprintln!("no perflevel sysctls on this machine; skipped");
        return;
    }
    let dir = std::env::temp_dir().join(format!("sllm-threads-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("threads.toml");
    std::fs::write(
        &path,
        format!(
            "schema = 1\n[machine.\"{brand}/perf0={p}/perf1={e}\"]\nbrand = \"{brand}\"\nperf_cores = {p}\nefficiency_cores = {e}\nthreads = 6\nevidence_run_ids = [\"run-a\", \"run-b\"]\n"
        ),
    )
    .unwrap();
    let (child, line, caps) = fake_caps(&["--threads-config", path.to_str().unwrap()]).await;
    let t = &caps["threads"];
    assert_eq!(t["budget"], 6, "{t}");
    assert_eq!(t["source"]["kind"], "stored");
    assert_eq!(t["source"]["evidence_run_ids"], json!(["run-a", "run-b"]));
    assert_eq!((t["effective"]["http"].as_u64(), t["effective"]["ggml_cpu"].as_u64()), (Some(1), Some(5)));
    assert_eq!(t["machine"]["key"], format!("{brand}/perf0={p}/perf1={e}"));
    assert!(line.contains("thread budget 6"), "{line}");
    stop(child);

    // A machine class with no entry gets the fixed fallback, and says so.
    std::fs::write(&path, "schema = 1\n").unwrap();
    let (child, _, caps) = fake_caps(&["--threads-config", path.to_str().unwrap()]).await;
    assert_eq!(caps["threads"]["source"]["kind"], "fallback");
    assert_eq!(caps["threads"]["budget"], 4);
    stop(child);

    // A config that exists but is broken refuses to start rather than silently falling back.
    std::fs::write(&path, "threads = 4\n").unwrap();
    let out = Command::new(BIN).args(["--fake", "--threads-config", path.to_str().unwrap()]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("outside"), "{}", String::from_utf8_lossy(&out.stderr));
    let _ = std::fs::remove_dir_all(&dir);
}
