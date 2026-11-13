use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// The GGUF file is unreadable, truncated or inconsistent.
    Gguf(String),
    /// A model key is missing or has an unexpected type or value.
    Model(String),
    /// A ggml backend call failed.
    Backend(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Gguf(s) => write!(f, "gguf error: {s}"),
            Error::Model(s) => write!(f, "model error: {s}"),
            Error::Backend(s) => write!(f, "backend error: {s}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
