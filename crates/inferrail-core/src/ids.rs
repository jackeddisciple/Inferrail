use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IdError {
    #[error("{kind} must not be empty")]
    Empty { kind: &'static str },
    #[error("{kind} is longer than {max} bytes")]
    TooLong { kind: &'static str, max: usize },
    #[error("{kind} {value:?} contains {found:?}; allowed: {allowed}")]
    InvalidChar {
        kind: &'static str,
        value: String,
        found: char,
        allowed: &'static str,
    },
}

macro_rules! shared_id {
    ($(#[$meta:meta])* $name:ident, $kind:literal, $max:expr, $allowed:literal, $valid:expr) => {
        $(#[$meta])*
        #[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(try_from = "String", into = "String")]
        pub struct $name(Arc<str>);

        impl $name {
            pub const MAX_LEN: usize = $max;

            pub fn new(value: impl AsRef<str>) -> Result<Self, IdError> {
                let value = value.as_ref();
                if value.is_empty() {
                    return Err(IdError::Empty { kind: $kind });
                }
                if value.len() > Self::MAX_LEN {
                    return Err(IdError::TooLong { kind: $kind, max: Self::MAX_LEN });
                }
                let valid: fn(char) -> bool = $valid;
                if let Some(found) = value.chars().find(|&c| !valid(c)) {
                    return Err(IdError::InvalidChar {
                        kind: $kind,
                        value: value.to_owned(),
                        found,
                        allowed: $allowed,
                    });
                }
                Ok(Self(Arc::from(value)))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl TryFrom<String> for $name {
            type Error = IdError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl From<$name> for String {
            fn from(id: $name) -> Self {
                id.0.as_ref().to_owned()
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({:?})", stringify!($name), &*self.0)
            }
        }
    };
}

shared_id!(
    /// Names one engine replica; used in logs, metrics labels and the admin API.
    ReplicaId,
    "replica id",
    128,
    "ASCII letters, digits, '.', '_' and '-'",
    |c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')
);

shared_id!(
    /// A served model name such as `meta-llama/Llama-3.1-8B-Instruct`.
    ModelId,
    "model id",
    256,
    "printable characters without whitespace",
    |c| !c.is_whitespace() && !c.is_control()
);

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn accepts_typical_names() {
        assert_eq!(
            ReplicaId::new("vllm-0.us_east").unwrap().as_str(),
            "vllm-0.us_east"
        );
        assert_eq!(
            ModelId::new("meta-llama/Llama-3.1-8B-Instruct")
                .unwrap()
                .to_string(),
            "meta-llama/Llama-3.1-8B-Instruct"
        );
        assert!(ModelId::new("ft:gpt-4o-mini:acme::abc").is_ok());
    }

    #[test]
    fn rejects_bad_names_with_reasons() {
        assert_eq!(
            ReplicaId::new(""),
            Err(IdError::Empty { kind: "replica id" })
        );
        assert!(matches!(
            ReplicaId::new("a b"),
            Err(IdError::InvalidChar { found: ' ', .. })
        ));
        assert!(matches!(
            ReplicaId::new("r/1"),
            Err(IdError::InvalidChar { found: '/', .. })
        ));
        assert!(matches!(
            ModelId::new("m\u{7}"),
            Err(IdError::InvalidChar { .. })
        ));
        assert!(matches!(
            ReplicaId::new("x".repeat(129)),
            Err(IdError::TooLong { max: 128, .. })
        ));
    }

    #[test]
    fn serde_validates_and_round_trips() {
        let id: ReplicaId = serde_json::from_str("\"r1\"").unwrap();
        assert_eq!(serde_json::to_string(&id).unwrap(), "\"r1\"");
        assert!(serde_json::from_str::<ReplicaId>("\"r 1\"").is_err());
        assert_eq!(format!("{id:?}"), "ReplicaId(\"r1\")");
    }

    #[test]
    fn clones_share_storage() {
        let id = ModelId::new("m").unwrap();
        let copy = id.clone();
        assert!(std::ptr::eq(id.as_str(), copy.as_str()));
    }

    proptest! {
        #[test]
        fn valid_replica_ids_round_trip(raw in "[A-Za-z0-9._-]{1,128}") {
            let id = ReplicaId::new(&raw).unwrap();
            let json = serde_json::to_string(&id).unwrap();
            prop_assert_eq!(serde_json::from_str::<ReplicaId>(&json).unwrap(), id);
        }

        #[test]
        fn construction_never_panics(raw in ".{0,300}") {
            let _ = ReplicaId::new(&raw);
            let _ = ModelId::new(&raw);
        }
    }
}
