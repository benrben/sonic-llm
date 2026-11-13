//! The fake engine's settings.

use super::{FakeHooks, StepTrace};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct FakeConfig {
    pub seed: u64,
    pub prefill_ms: f64,
    pub prefill_ms_per_token: f64,
    pub token_ms: f64,
    pub n_vocab: usize,
    pub n_ctx: usize,
    pub eos_id: i32,
    /// Tokens after which a request with `ignore_eos` false stops; `None` derives 8..64 from the prompt.
    pub natural_len: Option<usize>,
    /// The `capabilities` member of `/caps` (the task 1 record when probed, otherwise null).
    pub capabilities: Option<Value>,
    /// What the `model` field of responses says.
    pub model: String,
    /// Model a page pool of this many 256-token pages (admission tests); `None` = no pool (the metrics stay null).
    pub pool_pages: Option<usize>,
    /// Record the items of every step.
    pub trace: Option<StepTrace>,
    /// Model prefix sharing between `/completion` requests, in pages of this many tokens (`None` = no sharing).
    pub prefix_page: Option<usize>,
    /// Agent lifecycle (phase 3): a restore takes this long (it completes at the first poll after it).
    pub restore_ms: f64,
    /// Every restore fails (the snapshot is "lost"): the agent becomes empty and is re-prefilled from its history.
    pub restore_fails: bool,
    /// Scripted output: token `i` of every request has the text `script[i]` (the natural length is the script's, so the
    /// request ends with EOS after it); past the script, or without one, the text is `" t{tok}"`.
    pub script: Option<Vec<String>>,
    /// Token probabilities (`n_probs`): every token's probability and its runner-up's. `None`: the engine serves none.
    pub token_probs: Option<(f64, f64)>,
    /// Test hooks (all off by default): see [`FakeHooks`].
    pub hooks: FakeHooks,
}

impl Default for FakeConfig {
    fn default() -> Self {
        FakeConfig {
            seed: 0x5111_4d00,
            prefill_ms: 1.0,
            prefill_ms_per_token: 0.0,
            token_ms: 1.0,
            n_vocab: 151_936,
            n_ctx: 4096,
            eos_id: 151_645, // Qwen3 <|im_end|>
            natural_len: None,
            capabilities: None,
            model: "fake".into(),
            pool_pages: None,
            trace: None,
            prefix_page: None,
            restore_ms: 0.0,
            restore_fails: false,
            script: None,
            token_probs: None,
            hooks: FakeHooks::default(),
        }
    }
}
