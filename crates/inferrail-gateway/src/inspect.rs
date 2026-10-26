//! Reads just enough of a request body to route it, without re-encoding it.

use std::borrow::Cow;
use std::ops::Range;

use serde::de::{IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::value::RawValue;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InspectError {
    #[error("Invalid request: {0}")]
    InvalidJson(String),
    #[error("Invalid request: the body must be a JSON object")]
    NotAnObject,
    #[error("`model` must be a string")]
    ModelNotString,
}

/// The fields routing reads. A repeated `model` is refused, as before; a
/// repeated `user` keeps its last value, as Python's JSON reader does.
#[derive(Default)]
struct Peek<'a> {
    model: Option<&'a RawValue>,
    user: Option<&'a RawValue>,
}

impl<'de> Deserialize<'de> for Peek<'de> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(PeekVisitor)
    }
}

struct PeekVisitor;

impl<'de> Visitor<'de> for PeekVisitor {
    type Value = Peek<'de>;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a JSON object")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Peek<'de>, M::Error> {
        let mut peek = Peek::default();
        while let Some(key) = map.next_key::<Cow<'de, str>>()? {
            match key.as_ref() {
                "model" if peek.model.is_some() => {
                    return Err(serde::de::Error::duplicate_field("model"));
                }
                "model" => peek.model = Some(map.next_value()?),
                "user" => peek.user = Some(map.next_value()?),
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(peek)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inspected<'a> {
    /// The requested model; `None` when the field is absent or null.
    pub model: Option<Cow<'a, str>>,
    /// The `user` field when it is a string, which names a conversation for
    /// session affinity.
    pub user: Option<Cow<'a, str>>,
    span: Option<Range<usize>>,
    open: usize,
}

fn string(raw: &RawValue) -> Option<Cow<'_, str>> {
    let text = raw.get();
    if !text.starts_with('"') {
        return None;
    }
    match serde_json::from_str::<&str>(text) {
        Ok(value) => Some(Cow::Borrowed(value)),
        Err(_) => serde_json::from_str::<String>(text).ok().map(Cow::Owned),
    }
}

const BOM: &[u8] = b"\xef\xbb\xbf";

/// A leading byte order mark is skipped, as Python engines decoding JSON
/// bytes do, and kept in the forwarded body.
pub fn inspect(body: &[u8]) -> Result<Inspected<'_>, InspectError> {
    let skip = if body.starts_with(BOM) { BOM.len() } else { 0 };
    let json = &body[skip..];
    let open = json
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .filter(|&i| json[i] == b'{')
        .map(|i| skip + i)
        .ok_or(InspectError::NotAnObject)?;
    let peek: Peek<'_> =
        serde_json::from_slice(json).map_err(|e| InspectError::InvalidJson(e.to_string()))?;
    let user = peek.user.and_then(string);
    let Some(raw) = peek.model else {
        return Ok(Inspected {
            model: None,
            user,
            span: None,
            open,
        });
    };
    let text = raw.get();
    let start = (text.as_ptr() as usize).wrapping_sub(body.as_ptr() as usize);
    let span = start..start + text.len();
    debug_assert_eq!(body.get(span.clone()), Some(text.as_bytes()));
    let model = match text.as_bytes().first() {
        Some(b'n') => None,
        Some(b'"') => Some(match serde_json::from_str::<&str>(text) {
            Ok(name) => Cow::Borrowed(name),
            Err(_) => Cow::Owned(
                serde_json::from_str::<String>(text)
                    .map_err(|e| InspectError::InvalidJson(e.to_string()))?,
            ),
        }),
        _ => return Err(InspectError::ModelNotString),
    };
    Ok(Inspected {
        model,
        user,
        span: body.get(span.clone()).map(|_| span),
        open,
    })
}

impl Inspected<'_> {
    /// The body with `model` set to `name`. Only the model value changes (or
    /// the field is inserted first); every other byte is kept as sent.
    pub fn with_model(&self, body: &[u8], name: &str) -> Vec<u8> {
        let quoted = serde_json::Value::from(name).to_string();
        match &self.span {
            Some(span) => splice(body, span.clone(), quoted.as_bytes()),
            None => {
                let after = self.open + 1;
                let empty = body[after..]
                    .iter()
                    .find(|b| !b.is_ascii_whitespace())
                    .is_some_and(|&b| b == b'}');
                let mut field = Vec::with_capacity(quoted.len() + 10);
                field.extend_from_slice(b"\"model\":");
                field.extend_from_slice(quoted.as_bytes());
                if !empty {
                    field.push(b',');
                }
                splice(body, after..after, &field)
            }
        }
    }
}

fn splice(body: &[u8], span: Range<usize>, with: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() - span.len() + with.len());
    out.extend_from_slice(&body[..span.start]);
    out.extend_from_slice(with);
    out.extend_from_slice(&body[span.end..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::{Value, json};

    fn model_of(body: &str) -> Result<Option<String>, InspectError> {
        inspect(body.as_bytes()).map(|i| i.model.map(Cow::into_owned))
    }

    #[test]
    fn reads_the_model_and_borrows_plain_names() {
        let body = br#"{"messages":[],"model":"llama","stream":true}"#;
        let inspected = inspect(body).unwrap();
        assert!(matches!(inspected.model, Some(Cow::Borrowed("llama"))));
        assert_eq!(
            model_of(r#" { "model" : "ll\u0061ma\n" } "#)
                .unwrap()
                .as_deref(),
            Some("llama\n")
        );
    }

    #[test]
    fn a_leading_byte_order_mark_is_skipped_and_kept() {
        let body = "\u{feff}{\"model\":\"alias\",\"n\":1}".as_bytes();
        let inspected = inspect(body).unwrap();
        assert_eq!(inspected.model.as_deref(), Some("alias"));
        assert_eq!(
            inspected.with_model(body, "real"),
            "\u{feff}{\"model\":\"real\",\"n\":1}".as_bytes()
        );
        let bare = "\u{feff} {}".as_bytes();
        assert_eq!(
            inspect(bare).unwrap().with_model(bare, "m"),
            "\u{feff} {\"model\":\"m\"}".as_bytes()
        );
        for bad in ["\u{feff}\u{feff}{}", "\u{feff}[]", " \u{feff}{}"] {
            assert!(inspect(bad.as_bytes()).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_string_user_is_read_and_a_repeated_one_keeps_the_last() {
        let user = |body: &str| {
            inspect(body.as_bytes())
                .map(|i| i.user.map(Cow::into_owned))
                .unwrap()
        };
        assert_eq!(
            user(r#"{"model":"m","user":"alice"}"#).as_deref(),
            Some("alice")
        );
        assert_eq!(user(r#"{"user":"a\u0062c"}"#).as_deref(), Some("abc"));
        assert_eq!(user(r#"{"user":"a","user":"b"}"#).as_deref(), Some("b"));
        for body in [
            r#"{"user":7}"#,
            r#"{"user":null}"#,
            r#"{"user":["a"]}"#,
            "{}",
        ] {
            assert_eq!(user(body), None, "{body}");
        }
        assert!(matches!(
            inspect(br#"{"user":"a"}"#).unwrap().user,
            Some(Cow::Borrowed("a"))
        ));
    }

    #[test]
    fn absent_and_null_models_are_none() {
        assert_eq!(model_of(r#"{"messages":[]}"#).unwrap(), None);
        assert_eq!(model_of(r#"{"model":null}"#).unwrap(), None);
        assert_eq!(model_of("{}").unwrap(), None);
    }

    #[test]
    fn rejects_non_objects_bad_json_and_non_string_models() {
        for body in ["", "  ", "[]", r#"["llama"]"#, "42", "null", "\"x\""] {
            assert_eq!(model_of(body), Err(InspectError::NotAnObject), "{body:?}");
        }
        for body in ["{", r#"{"model":"a","model":"b"}"#, r#"{"model":"a"} x"#] {
            assert!(
                matches!(model_of(body), Err(InspectError::InvalidJson(_))),
                "{body:?}"
            );
        }
        for body in [r#"{"model":42}"#, r#"{"model":["a"]}"#, r#"{"model":{}}"#] {
            assert_eq!(model_of(body), Err(InspectError::ModelNotString), "{body}");
        }
        assert!(matches!(
            model_of("{\"model\":\"\\ud800\"}"),
            Err(InspectError::InvalidJson(_))
        ));
    }

    #[test]
    fn alias_rewrite_touches_only_the_model_value() {
        let body = br#"{ "model" :  "alias" , "messages":[{"role":"user","content":"hi"}], "temperature": 1.0 }"#;
        let out = inspect(body).unwrap().with_model(body, "org/model-8B");
        assert_eq!(
            String::from_utf8(out).unwrap(),
            r#"{ "model" :  "org/model-8B" , "messages":[{"role":"user","content":"hi"}], "temperature": 1.0 }"#
        );
    }

    #[test]
    fn missing_or_null_model_is_filled_in() {
        for (body, want) in [
            (r#"{"messages":[]}"#, r#"{"model":"m","messages":[]}"#),
            ("{}", r#"{"model":"m"}"#),
            (" { \n } ", " {\"model\":\"m\" \n } "),
            (r#"{"model":null,"n":1}"#, r#"{"model":"m","n":1}"#),
        ] {
            let out = inspect(body.as_bytes())
                .unwrap()
                .with_model(body.as_bytes(), "m");
            let out = String::from_utf8(out).unwrap();
            assert_eq!(out, want, "{body:?}");
            assert_eq!(model_of(&out).unwrap().as_deref(), Some("m"));
        }
    }

    fn json_value() -> impl Strategy<Value = Value> {
        let leaf = prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map(Value::from),
            any::<i64>().prop_map(Value::from),
            (-1e9f64..1e9).prop_map(Value::from),
            "\\PC{0,24}".prop_map(Value::from),
        ];
        leaf.prop_recursive(3, 24, 6, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..5).prop_map(Value::Array),
                prop::collection::btree_map("[a-z_]{1,8}", inner, 0..5)
                    .prop_map(|m| Value::Object(m.into_iter().collect())),
            ]
        })
    }

    proptest! {
        #[test]
        fn splice_keeps_every_other_byte(
            fields in prop::collection::btree_map("[a-z_]{1,10}", json_value(), 0..6),
            model in prop::option::of("\\PC{0,30}"),
            target in "\\PC{1,40}",
            pretty: bool,
            bom: bool,
        ) {
            let mut object: serde_json::Map<String, Value> =
                fields.into_iter().filter(|(k, _)| k != "model").collect();
            if let Some(model) = &model {
                object.insert("model".to_owned(), Value::from(model.as_str()));
            }
            let original = Value::Object(object);
            let mut body = if bom { BOM.to_vec() } else { Vec::new() };
            if pretty {
                body.extend(serde_json::to_vec_pretty(&original).unwrap());
            } else {
                body.extend(serde_json::to_vec(&original).unwrap());
            }
            let inspected = inspect(&body).unwrap();
            prop_assert_eq!(inspected.model.as_deref(), model.as_deref());
            let out = inspected.with_model(&body, &target);
            let mut want = original.clone();
            want["model"] = Value::from(target.as_str());
            prop_assert_eq!(out.starts_with(BOM), bom);
            let got: Value = serde_json::from_slice(&out[if bom { 3 } else { 0 }..]).unwrap();
            prop_assert_eq!(got, want);
            if let Some(span) = &inspected.span {
                prop_assert_eq!(&out[..span.start], &body[..span.start]);
                prop_assert_eq!(&out[out.len() - (body.len() - span.end)..], &body[span.end..]);
            } else {
                prop_assert_eq!(&out[..=inspected.open], &body[..=inspected.open]);
                prop_assert!(out.ends_with(&body[inspected.open + 1..]));
            }
        }

        #[test]
        fn arbitrary_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..256)) {
            if let Ok(inspected) = inspect(&bytes) {
                let out = inspected.with_model(&bytes, "m");
                let model = model_of(std::str::from_utf8(&out).unwrap()).unwrap();
                prop_assert_eq!(model.as_deref(), Some("m"));
            }
        }

        #[test]
        fn objects_with_a_model_prefix_never_panic(tail in "\\PC{0,64}") {
            let body = format!("{{\"model\":{tail}");
            let _ = inspect(body.as_bytes());
        }
    }

    #[test]
    fn large_chat_bodies_are_inspected() {
        let messages: Vec<Value> = (0..2_000)
            .map(|i| json!({"role": "user", "content": format!("turn {i} {}", "x".repeat(100))}))
            .collect();
        let body = serde_json::to_vec(&json!({"messages": messages, "model": "big"})).unwrap();
        assert_eq!(inspect(&body).unwrap().model.as_deref(), Some("big"));
    }
}
