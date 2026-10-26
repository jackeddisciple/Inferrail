use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum TokenizeError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot parse {path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("{0}")]
    NotFound(String),
    #[error("tokenizer failed: {0}")]
    Tokenizer(String),
    #[error("chat template failed: {0}")]
    Template(String),
    /// The engine would refuse the request as well.
    #[error("the engine would reject the request: {0}")]
    Invalid(String),
    /// The request is fine, but its tokens cannot be known ahead of the engine.
    #[error("cannot tokenize {0} ahead of the engine")]
    Unsupported(&'static str),
}

impl TokenizeError {
    /// A short label for metrics: `invalid`, `unsupported`, `template` or `failed`.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Invalid(_) => "invalid",
            Self::Unsupported(_) => "unsupported",
            Self::Template(_) => "template",
            Self::Io { .. } | Self::Parse { .. } | Self::NotFound(_) | Self::Tokenizer(_) => {
                "failed"
            }
        }
    }
}
