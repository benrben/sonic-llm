//! The settings of the llama-server compatibility layer (`--alias`, `--api-key`, `--reasoning-format`).

/// Where `<think>...</think>` text goes in a chat reply (llama-server's `--reasoning-format`, and the request field of the same name).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningFormat {
    /// llama-server's default: the same as `deepseek` (the reasoning comes out of the content).
    #[default]
    Auto,
    /// The reasoning is `message.reasoning_content` (streamed as `delta.reasoning_content`); the content is the rest.
    Deepseek,
    /// The reply is left as the model wrote it, tags included.
    None,
}

variant_names!(ReasoningFormat { Auto => "auto", Deepseek => "deepseek", None => "none" },
    "expected auto, deepseek or none, got {:?} (deepseek-legacy is not implemented)");

impl ReasoningFormat {
    /// Whether the reasoning is split off the content.
    pub fn splits(self) -> bool {
        self != ReasoningFormat::None
    }
}

#[derive(Debug, Clone, Default)]
pub struct CompatConfig {
    /// `-a` / `--alias`: the model name `/v1/models` lists and the replies echo when the request names none.
    pub alias: Option<String>,
    /// The other names after the first in `--alias A,B`, listed as `aliases` of the model.
    pub aliases: Vec<String>,
    /// `--api-key` / `--api-key-file`: when non-empty every route except `/health` and `/v1/health` needs one of these.
    pub api_keys: Vec<String>,
    /// `--reasoning-format`: the default of a chat request that does not say.
    pub reasoning: ReasoningFormat,
}
