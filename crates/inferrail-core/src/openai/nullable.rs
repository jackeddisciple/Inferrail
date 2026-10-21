use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A JSON field that can be missing, explicitly `null`, or set.
///
/// OpenAI compatible servers distinguish all three: vLLM serializes stream
/// chunks with `exclude_unset`, so a field set to `None` is written as `null`
/// while an unset one is omitted. Use with
/// `#[serde(default, skip_serializing_if = "Nullable::is_absent")]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Nullable<T> {
    #[default]
    Absent,
    Null,
    Value(T),
}

impl<T> Nullable<T> {
    pub const fn is_absent(&self) -> bool {
        matches!(self, Self::Absent)
    }

    pub const fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    pub const fn is_value(&self) -> bool {
        matches!(self, Self::Value(_))
    }

    pub const fn value(&self) -> Option<&T> {
        match self {
            Self::Value(v) => Some(v),
            Self::Absent | Self::Null => None,
        }
    }

    pub fn value_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Value(v) => Some(v),
            Self::Absent | Self::Null => None,
        }
    }

    pub fn into_value(self) -> Option<T> {
        match self {
            Self::Value(v) => Some(v),
            Self::Absent | Self::Null => None,
        }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Nullable<U> {
        match self {
            Self::Absent => Nullable::Absent,
            Self::Null => Nullable::Null,
            Self::Value(v) => Nullable::Value(f(v)),
        }
    }

    /// `Some(v)` becomes a value and `None` becomes an explicit `null`.
    pub fn null_or(value: Option<T>) -> Self {
        value.map_or(Self::Null, Self::Value)
    }

    /// `Some(v)` becomes a value and `None` leaves the field out.
    pub fn absent_or(value: Option<T>) -> Self {
        value.map_or(Self::Absent, Self::Value)
    }
}

impl<T> From<T> for Nullable<T> {
    fn from(value: T) -> Self {
        Self::Value(value)
    }
}

impl<T: Serialize> Serialize for Nullable<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Value(v) => v.serialize(serializer),
            Self::Absent | Self::Null => serializer.serialize_none(),
        }
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Nullable<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Option::<T>::deserialize(deserializer).map(Self::null_or)
    }
}

#[cfg(test)]
mod tests {
    use super::Nullable;
    use proptest::prelude::*;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Probe {
        #[serde(default, skip_serializing_if = "Nullable::is_absent")]
        field: Nullable<i64>,
    }

    #[test]
    fn missing_field_is_absent_and_stays_missing() {
        let probe: Probe = serde_json::from_value(json!({})).unwrap();
        assert_eq!(probe.field, Nullable::Absent);
        assert_eq!(serde_json::to_string(&probe).unwrap(), "{}");
    }

    #[test]
    fn explicit_null_survives_round_trip() {
        let probe: Probe = serde_json::from_str(r#"{"field":null}"#).unwrap();
        assert_eq!(probe.field, Nullable::Null);
        assert_eq!(serde_json::to_string(&probe).unwrap(), r#"{"field":null}"#);
    }

    #[test]
    fn value_survives_round_trip() {
        let probe: Probe = serde_json::from_str(r#"{"field":-7}"#).unwrap();
        assert_eq!(probe.field, Nullable::Value(-7));
        assert_eq!(serde_json::to_string(&probe).unwrap(), r#"{"field":-7}"#);
    }

    #[test]
    fn wrong_type_is_rejected() {
        assert!(serde_json::from_str::<Probe>(r#"{"field":"seven"}"#).is_err());
    }

    #[test]
    fn helpers_map_options() {
        assert_eq!(Nullable::null_or(None::<u8>), Nullable::Null);
        assert_eq!(Nullable::absent_or(None::<u8>), Nullable::Absent);
        assert_eq!(
            Nullable::absent_or(Some(3)).map(|v| v * 2),
            Nullable::Value(6)
        );
        assert_eq!(Nullable::Value(4).into_value(), Some(4));
        assert!(Nullable::<u8>::Null.value().is_none());
    }

    fn any_nullable() -> impl Strategy<Value = Nullable<i64>> {
        prop_oneof![
            Just(Nullable::Absent),
            Just(Nullable::Null),
            any::<i64>().prop_map(Nullable::Value),
        ]
    }

    proptest! {
        #[test]
        fn every_state_round_trips(field in any_nullable()) {
            let probe = Probe { field };
            let text = serde_json::to_string(&probe).unwrap();
            let back: Probe = serde_json::from_str(&text).unwrap();
            prop_assert_eq!(back, probe);
        }
    }
}
