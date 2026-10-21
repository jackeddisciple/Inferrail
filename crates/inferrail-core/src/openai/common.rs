use std::borrow::Cow;
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use super::Nullable;

/// Fields a type does not model, kept verbatim so payloads round trip.
pub type Extra = Map<String, Value>;

macro_rules! open_string_enum {
    ($(#[$meta:meta])* $name:ident { $($variant:ident => $text:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum $name {
            $($variant,)+
            Other(String),
        }

        impl $name {
            pub fn as_str(&self) -> &str {
                match self {
                    $(Self::$variant => $text,)+
                    Self::Other(other) => other,
                }
            }
        }

        impl From<&str> for $name {
            fn from(text: &str) -> Self {
                match text {
                    $($text => Self::$variant,)+
                    other => Self::Other(other.to_owned()),
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let text = Cow::<'de, str>::deserialize(deserializer)?;
                Ok(Self::from(text.as_ref()))
            }
        }
    };
}

open_string_enum! {
    Role {
        System => "system",
        Developer => "developer",
        User => "user",
        Assistant => "assistant",
        Tool => "tool",
        Function => "function",
    }
}

open_string_enum! {
    FinishReason {
        Stop => "stop",
        Length => "length",
        ToolCalls => "tool_calls",
        ContentFilter => "content_filter",
        FunctionCall => "function_call",
        Abort => "abort",
        Error => "error",
        Repetition => "repetition",
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub include_usage: Nullable<bool>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub continuous_usage_stats: Nullable<bool>,
    #[serde(flatten)]
    pub extra: Extra,
}

impl StreamOptions {
    pub fn include_usage(options: &Nullable<Self>) -> bool {
        options
            .value()
            .and_then(|o| o.include_usage.value().copied())
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub function: Nullable<FunctionDefinition>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub description: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub parameters: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub strict: Nullable<bool>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub id: Nullable<String>,
    #[serde(rename = "type", default, skip_serializing_if = "Nullable::is_absent")]
    pub kind: Nullable<String>,
    pub function: FunctionCall,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
    #[serde(flatten)]
    pub extra: Extra,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn known_roles_parse_to_variants() {
        for (text, role) in [
            ("system", Role::System),
            ("developer", Role::Developer),
            ("user", Role::User),
            ("assistant", Role::Assistant),
            ("tool", Role::Tool),
            ("function", Role::Function),
        ] {
            assert_eq!(serde_json::from_value::<Role>(json!(text)).unwrap(), role);
            assert_eq!(serde_json::to_value(&role).unwrap(), json!(text));
        }
    }

    #[test]
    fn unknown_strings_are_kept_verbatim() {
        let role: Role = serde_json::from_value(json!("critic")).unwrap();
        assert_eq!(role, Role::Other("critic".into()));
        assert_eq!(role.to_string(), "critic");
        let reason: FinishReason = serde_json::from_value(json!("eos_or_whatever")).unwrap();
        assert_eq!(
            serde_json::to_value(&reason).unwrap(),
            json!("eos_or_whatever")
        );
    }

    #[test]
    fn stop_accepts_string_or_list() {
        assert_eq!(
            serde_json::from_value::<Stop>(json!("\n")).unwrap(),
            Stop::One("\n".into())
        );
        assert_eq!(
            serde_json::from_value::<Stop>(json!(["a", "b"])).unwrap(),
            Stop::Many(vec!["a".into(), "b".into()])
        );
        assert!(serde_json::from_value::<Stop>(json!(3)).is_err());
    }

    #[test]
    fn include_usage_defaults_to_false() {
        assert!(!StreamOptions::include_usage(&Nullable::Absent));
        assert!(!StreamOptions::include_usage(&Nullable::Null));
        let on: Nullable<StreamOptions> =
            Nullable::Value(serde_json::from_value(json!({"include_usage": true})).unwrap());
        assert!(StreamOptions::include_usage(&on));
    }

    #[test]
    fn tool_keeps_unknown_fields() {
        let raw = json!({
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}, "x-cost": 2},
            "cache_control": {"type": "ephemeral"}
        });
        let tool: Tool = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(tool.extra["cache_control"], json!({"type": "ephemeral"}));
        assert_eq!(serde_json::to_value(&tool).unwrap(), raw);
    }
}
