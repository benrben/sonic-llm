//! Text to token ids for the text endpoints (`/v1/completions`, `/tokenize`, string prompts).
//!
//! With a GGUF this is llama.cpp's own tokenizer through llama-ref's vocab-only path (no weights, no GPU), so the ids are
//! exactly what llama-server produces. The vocabulary is loaded ONCE ([`llama_ref::Vocab`]) and every call reuses that
//! handle. Without a GGUF (`--fake` and no `--model`) a byte tokenizer stands in: every UTF-8 byte is a token id 0..=255.
//! Token-id endpoints never touch it.

use llama_ref::Vocab;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub enum Tokenizer {
    Gguf { path: PathBuf, vocab: Arc<Vocab> },
    /// Test stand-in: one token per UTF-8 byte.
    Bytes,
}

impl Tokenizer {
    /// Load the GGUF's vocabulary once. Panics when the file cannot be read; use [`Tokenizer::try_gguf`] to handle that.
    pub fn gguf(path: PathBuf) -> Tokenizer {
        Tokenizer::try_gguf(path).unwrap_or_else(|e| panic!("cannot load the vocabulary: {e}"))
    }

    pub fn try_gguf(path: PathBuf) -> Result<Tokenizer, String> {
        let vocab = Arc::new(Vocab::load(&path).map_err(|e| format!("{}: {e}", path.display()))?);
        Ok(Tokenizer::Gguf { path, vocab })
    }

    /// The cached vocabulary, for the real engine (token text, end-of-generation ids).
    pub fn vocab(&self) -> Option<Arc<Vocab>> {
        match self {
            Tokenizer::Gguf { vocab, .. } => Some(vocab.clone()),
            Tokenizer::Bytes => None,
        }
    }

    pub fn gguf_path(&self) -> Option<&Path> {
        match self {
            Tokenizer::Gguf { path, .. } => Some(path),
            Tokenizer::Bytes => None,
        }
    }

    pub fn name(&self) -> String {
        match self {
            Tokenizer::Gguf { path, .. } => format!("gguf:{}", path.display()),
            Tokenizer::Bytes => "bytes (no --model: one token per UTF-8 byte)".into(),
        }
    }

    /// The bytes of one token. `special`: control tokens print their text (llama-server's `/detokenize` and the pieces of
    /// `/tokenize`), otherwise they print as nothing. The bytes are not always valid UTF-8 on their own. The byte tokenizer's
    /// ids above 255 are no tokens: they print as nothing.
    pub fn piece(&self, id: i32, special: bool) -> Vec<u8> {
        match self {
            Tokenizer::Bytes => u8::try_from(id).map(|b| vec![b]).unwrap_or_default(),
            Tokenizer::Gguf { vocab, .. } if special => vocab.piece_special(id),
            Tokenizer::Gguf { vocab, .. } => vocab.piece(id),
        }
    }

    /// Blocking: call from `spawn_blocking`.
    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>, String> {
        match self {
            Tokenizer::Bytes => Ok(text.bytes().map(|b| b as i32).collect()),
            Tokenizer::Gguf { vocab, .. } => vocab.tokenize(text, add_special, parse_special).map_err(|e| e.to_string()),
        }
    }
}
