//! Value formatting as CPython does it, so templates render what
//! transformers renders: `str()`, `repr()` and `json.dumps()`.

use std::fmt::Write as _;

use minijinja::value::{Value, ValueKind};
use minijinja::{Error, ErrorKind};

use crate::printable::is_printable;

/// `str(value)`; Jinja2's `Undefined` prints as nothing.
pub(crate) fn str_of(value: &Value, out: &mut String) {
    match value.kind() {
        ValueKind::Undefined => {}
        ValueKind::String => out.push_str(value.as_str().unwrap_or_default()),
        _ => repr_of(value, out),
    }
}

pub(crate) fn repr_of(value: &Value, out: &mut String) {
    match value.kind() {
        ValueKind::Undefined => {}
        ValueKind::None => out.push_str("None"),
        ValueKind::Bool => out.push_str(if value.is_true() { "True" } else { "False" }),
        ValueKind::Number => number(value, out),
        ValueKind::String => string_repr(value.as_str().unwrap_or_default(), out),
        ValueKind::Bytes => {
            out.push_str("b'");
            for &b in value.as_bytes().unwrap_or_default() {
                match b {
                    b'\\' => out.push_str("\\\\"),
                    b'\'' => out.push_str("\\'"),
                    b'\n' => out.push_str("\\n"),
                    b'\r' => out.push_str("\\r"),
                    b'\t' => out.push_str("\\t"),
                    0x20..=0x7e => out.push(b as char),
                    _ => {
                        let _ = write!(out, "\\x{b:02x}");
                    }
                }
            }
            out.push('\'');
        }
        ValueKind::Map => {
            out.push('{');
            let mut first = true;
            if let Ok(keys) = value.try_iter() {
                for key in keys {
                    if !first {
                        out.push_str(", ");
                    }
                    first = false;
                    repr_of(&key, out);
                    out.push_str(": ");
                    repr_of(&value.get_item(&key).unwrap_or_default(), out);
                }
            }
            out.push('}');
        }
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            if let Ok(items) = value.try_iter() {
                for (i, item) in items.enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    repr_of(&item, out);
                }
            }
            out.push(']');
        }
        _ => {
            let _ = write!(out, "{value}");
        }
    }
}

fn number(value: &Value, out: &mut String) {
    if let Some(f) = float_of(value) {
        float_repr(f, out);
    } else {
        let _ = write!(out, "{value}");
    }
}

fn float_of(value: &Value) -> Option<f64> {
    if value.is_integer() {
        return None;
    }
    f64::try_from(value.clone()).ok()
}

/// `repr(float)`: the shortest digits that round trip, in fixed notation
/// for exponents from -4 to 15 and scientific notation otherwise.
pub(crate) fn float_repr(f: f64, out: &mut String) {
    if f.is_nan() {
        out.push_str("nan");
        return;
    }
    if f.is_infinite() {
        out.push_str(if f > 0.0 { "inf" } else { "-inf" });
        return;
    }
    let sci = format!("{f:e}");
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if negative {
        out.push('-');
    }
    if (-4..16).contains(&exponent) {
        if exponent < 0 {
            out.push_str("0.");
            for _ in 0..(-exponent - 1) {
                out.push('0');
            }
            out.push_str(&digits);
        } else {
            let point = exponent as usize + 1;
            if digits.len() <= point {
                out.push_str(&digits);
                for _ in digits.len()..point {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&digits[..point]);
                out.push('.');
                out.push_str(&digits[point..]);
            }
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        );
    }
}

fn string_repr(s: &str, out: &mut String) {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if is_printable(c) => out.push(c),
            c if (c as u32) < 0x100 => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c if (c as u32) < 0x10000 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => {
                let _ = write!(out, "\\U{:08x}", c as u32);
            }
        }
    }
    out.push(quote);
}

/// The options of transformers' `tojson` filter, which calls `json.dumps`.
#[derive(Debug, Clone)]
pub(crate) struct JsonStyle {
    pub(crate) indent: Option<String>,
    pub(crate) item_separator: String,
    pub(crate) key_separator: String,
    pub(crate) sort_keys: bool,
    pub(crate) ensure_ascii: bool,
}

impl JsonStyle {
    pub(crate) fn new(indent: Option<String>) -> Self {
        let item_separator = if indent.is_some() { "," } else { ", " };
        Self {
            indent,
            item_separator: item_separator.to_owned(),
            key_separator: ": ".to_owned(),
            sort_keys: false,
            ensure_ascii: false,
        }
    }
}

pub(crate) fn json_dumps(value: &Value, style: &JsonStyle) -> Result<String, Error> {
    let mut out = String::new();
    write_json(value, style, 0, &mut out)?;
    Ok(out)
}

fn newline(style: &JsonStyle, level: usize, out: &mut String) {
    if let Some(indent) = &style.indent {
        out.push('\n');
        for _ in 0..level {
            out.push_str(indent);
        }
    }
}

fn not_serializable(value: &Value) -> Error {
    let name = match value.kind() {
        ValueKind::Undefined => "Undefined".to_owned(),
        kind => kind.to_string(),
    };
    Error::new(
        ErrorKind::InvalidOperation,
        format!("Object of type {name} is not JSON serializable"),
    )
}

fn write_json(
    value: &Value,
    style: &JsonStyle,
    level: usize,
    out: &mut String,
) -> Result<(), Error> {
    match value.kind() {
        ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number => match float_of(value) {
            Some(f) if f.is_nan() => out.push_str("NaN"),
            Some(f) if f.is_infinite() => {
                out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" })
            }
            Some(f) => float_repr(f, out),
            None => {
                let _ = write!(out, "{value}");
            }
        },
        ValueKind::String => json_string(value.as_str().unwrap_or_default(), style, out),
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<Value> = value.try_iter()?.collect();
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&style.item_separator);
                }
                newline(style, level + 1, out);
                write_json(item, style, level + 1, out)?;
            }
            newline(style, level, out);
            out.push(']');
        }
        ValueKind::Map => {
            let mut entries = Vec::new();
            for key in value.try_iter()? {
                let item = value.get_item(&key)?;
                entries.push((json_key(&key)?, item));
            }
            if entries.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            if style.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(&b.0));
            }
            out.push('{');
            for (i, (key, item)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(&style.item_separator);
                }
                newline(style, level + 1, out);
                json_string(key, style, out);
                out.push_str(&style.key_separator);
                write_json(item, style, level + 1, out)?;
            }
            newline(style, level, out);
            out.push('}');
        }
        _ => return Err(not_serializable(value)),
    }
    Ok(())
}

fn json_key(key: &Value) -> Result<String, Error> {
    let mut out = String::new();
    match key.kind() {
        ValueKind::String => out.push_str(key.as_str().unwrap_or_default()),
        ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if key.is_true() { "true" } else { "false" }),
        ValueKind::Number => number(key, &mut out),
        _ => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!(
                    "keys must be str, int, float, bool or None, not {}",
                    key.kind()
                ),
            ));
        }
    }
    Ok(out)
}

fn json_string(s: &str, style: &JsonStyle, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c if style.ensure_ascii && !c.is_ascii() => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repr(value: Value) -> String {
        let mut out = String::new();
        repr_of(&value, &mut out);
        out
    }

    fn float(f: f64) -> String {
        let mut out = String::new();
        float_repr(f, &mut out);
        out
    }

    #[test]
    fn floats_print_like_cpython() {
        for (value, want) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (-2.5, "-2.5"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (123456789.125, "123456789.125"),
            (1e-7, "1e-07"),
            (1e20, "1e+20"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (f64::INFINITY, "inf"),
            (2.0f64.powi(70), "1.1805916207174113e+21"),
        ] {
            assert_eq!(float(value), want, "{value}");
        }
    }

    #[test]
    fn reprs_match_python() {
        assert_eq!(repr(Value::from(())), "None");
        assert_eq!(repr(Value::from(true)), "True");
        assert_eq!(repr(Value::from("it's")), "\"it's\"");
        assert_eq!(repr(Value::from("say \"hi\" it's")), "'say \"hi\" it\\'s'");
        assert_eq!(repr(Value::from("a\nb\u{1}\u{a0}é")), "'a\\nb\\x01\\xa0é'");
        assert_eq!(
            repr(Value::from(
                "\u{e000}\u{600}\u{378}\u{e0041}😀\u{feff}\u{3000} é日\u{ad}\u{2028}\u{10fff}\u{7f}\u{85}"
            )),
            "'\\ue000\\u0600\\u0378\\U000e0041😀\\ufeff\\u3000 é日\\xad\\u2028\\U00010fff\\x7f\\x85'"
        );
        let list = Value::from(vec![Value::from(1), Value::from("x"), Value::from(())]);
        assert_eq!(repr(list), "[1, 'x', None]");
        let map: Value = serde_json::from_str::<Value>(r#"{"b": 1, "a": [true, 2.0]}"#).unwrap();
        assert_eq!(repr(map), "{'b': 1, 'a': [True, 2.0]}");
    }

    #[test]
    fn json_matches_python_dumps() {
        let value: Value =
            serde_json::from_str(r#"{"z": "é\n\u0001\"", "a": [1, 2.50, null, {}], "e": []}"#)
                .unwrap();
        assert_eq!(
            json_dumps(&value, &JsonStyle::new(None)).unwrap(),
            r#"{"z": "é\n\u0001\"", "a": [1, 2.5, null, {}], "e": []}"#
        );
        assert_eq!(
            json_dumps(&value, &JsonStyle::new(Some("  ".to_owned()))).unwrap(),
            "{\n  \"z\": \"é\\n\\u0001\\\"\",\n  \"a\": [\n    1,\n    2.5,\n    null,\n    {}\n  ],\n  \"e\": []\n}"
        );
        let mut sorted = JsonStyle::new(None);
        sorted.sort_keys = true;
        sorted.ensure_ascii = true;
        assert_eq!(
            json_dumps(&value, &sorted).unwrap(),
            r#"{"a": [1, 2.5, null, {}], "e": [], "z": "\u00e9\n\u0001\""}"#
        );
        let astral = Value::from("😀");
        assert_eq!(json_dumps(&astral, &sorted).unwrap(), r#""\ud83d\ude00""#);
        assert!(json_dumps(&Value::UNDEFINED, &JsonStyle::new(None)).is_err());
    }
}
