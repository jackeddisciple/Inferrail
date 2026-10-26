//! JSON read the way Python's `json.loads` reads it, since vLLM parses
//! requests and tool call arguments with it: integers keep every digit,
//! `-0` is the integer zero, floats too large become infinities, and `NaN`,
//! `Infinity` and `-Infinity` are accepted.

use indexmap::IndexMap;
use minijinja::Value;

const MAX_DEPTH: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Failure {
    /// `json.loads` rejects it too.
    Malformed,
    /// Python reads it, but it holds an integer wider than 128 bits, a
    /// lone surrogate or nesting too deep to follow.
    Unsupported,
}

pub(crate) fn loads(text: &str) -> Result<Value, Failure> {
    let mut parser = Parser {
        text,
        bytes: text.as_bytes(),
        at: 0,
    };
    parser.space();
    let value = parser.value(0)?;
    parser.space();
    if parser.at == parser.bytes.len() {
        Ok(value)
    } else {
        Err(Failure::Malformed)
    }
}

struct Parser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn space(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.at += 1;
        }
    }

    fn eat(&mut self, literal: &str) -> bool {
        let found = self.text[self.at..].starts_with(literal);
        if found {
            self.at += literal.len();
        }
        found
    }

    fn expect(&mut self, byte: u8) -> Result<(), Failure> {
        self.space();
        if self.peek() == Some(byte) {
            self.at += 1;
            Ok(())
        } else {
            Err(Failure::Malformed)
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, Failure> {
        if depth > MAX_DEPTH {
            return Err(Failure::Unsupported);
        }
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => self.string().map(Value::from),
            Some(b'-') if self.eat("-Infinity") => Ok(Value::from(f64::NEG_INFINITY)),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ if self.eat("null") => Ok(Value::from(())),
            _ if self.eat("true") => Ok(Value::from(true)),
            _ if self.eat("false") => Ok(Value::from(false)),
            _ if self.eat("NaN") => Ok(Value::from(f64::NAN)),
            _ if self.eat("Infinity") => Ok(Value::from(f64::INFINITY)),
            _ => Err(Failure::Malformed),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, Failure> {
        self.at += 1;
        let mut map = IndexMap::new();
        self.space();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Value::from(map));
        }
        loop {
            self.space();
            if self.peek() != Some(b'"') {
                return Err(Failure::Malformed);
            }
            let key = self.string()?;
            self.expect(b':')?;
            self.space();
            let value = self.value(depth + 1)?;
            map.insert(Value::from(key), value);
            self.space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Value::from(map));
                }
                _ => return Err(Failure::Malformed),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value, Failure> {
        self.at += 1;
        let mut items = Vec::new();
        self.space();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Value::from(items));
        }
        loop {
            self.space();
            items.push(self.value(depth + 1)?);
            self.space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Value::from(items));
                }
                _ => return Err(Failure::Malformed),
            }
        }
    }

    fn digits(&mut self) {
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.at += 1;
        }
    }

    fn number(&mut self) -> Result<Value, Failure> {
        let start = self.at;
        let negative = self.peek() == Some(b'-');
        if negative {
            self.at += 1;
        }
        match self.peek() {
            Some(b'0') => self.at += 1,
            Some(b'1'..=b'9') => self.digits(),
            _ => return Err(Failure::Malformed),
        }
        let mut float = false;
        if self.peek() == Some(b'.')
            && self
                .bytes
                .get(self.at + 1)
                .is_some_and(|b| b.is_ascii_digit())
        {
            self.at += 1;
            self.digits();
            float = true;
        }
        if let Some(b'e' | b'E') = self.peek() {
            let mut end = self.at + 1;
            if let Some(b'+' | b'-') = self.bytes.get(end) {
                end += 1;
            }
            if self.bytes.get(end).is_some_and(|b| b.is_ascii_digit()) {
                self.at = end;
                self.digits();
                float = true;
            }
        }
        let literal = &self.text[start..self.at];
        if float {
            return literal
                .parse::<f64>()
                .map(Value::from)
                .map_err(|_| Failure::Malformed);
        }
        let integer = if negative {
            literal
                .parse::<i64>()
                .map(Value::from)
                .or_else(|_| literal.parse::<i128>().map(Value::from))
        } else {
            literal
                .parse::<u64>()
                .map(Value::from)
                .or_else(|_| literal.parse::<u128>().map(Value::from))
        };
        integer.map_err(|_| Failure::Unsupported)
    }

    fn hex(&mut self) -> Result<u32, Failure> {
        let digits = self
            .text
            .get(self.at..self.at + 4)
            .filter(|d| d.bytes().all(|b| b.is_ascii_hexdigit()))
            .ok_or(Failure::Malformed)?;
        self.at += 4;
        u32::from_str_radix(digits, 16).map_err(|_| Failure::Malformed)
    }

    fn string(&mut self) -> Result<String, Failure> {
        self.at += 1;
        let mut out = String::new();
        loop {
            let run = self.bytes[self.at..]
                .iter()
                .position(|&b| b == b'"' || b == b'\\' || b < 0x20)
                .ok_or(Failure::Malformed)?;
            out.push_str(&self.text[self.at..self.at + run]);
            self.at += run;
            match self.bytes[self.at] {
                b'"' => {
                    self.at += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.at += 1;
                    let escape = self.peek().ok_or(Failure::Malformed)?;
                    self.at += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.code_point()?),
                        _ => return Err(Failure::Malformed),
                    }
                }
                _ => return Err(Failure::Malformed),
            }
        }
    }

    fn code_point(&mut self) -> Result<char, Failure> {
        let first = self.hex()?;
        let unit = if (0xd800..0xdc00).contains(&first) && self.text[self.at..].starts_with("\\u") {
            let back = self.at;
            self.at += 2;
            let second = self.hex()?;
            if (0xdc00..0xe000).contains(&second) {
                0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00)
            } else {
                self.at = back;
                first
            }
        } else {
            first
        };
        char::from_u32(unit).ok_or(Failure::Unsupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::python::{JsonStyle, json_dumps};

    fn dumps(text: &str) -> String {
        json_dumps(&loads(text).unwrap(), &JsonStyle::new(None)).unwrap()
    }

    /// Each pair is `json.dumps(json.loads(text))` in CPython 3.12.
    #[test]
    fn numbers_read_as_python_reads_them() {
        for (text, want) in [
            ("-0", "0"),
            ("-0.0", "-0.0"),
            ("0", "0"),
            (
                "123456789012345678901234567890",
                "123456789012345678901234567890",
            ),
            ("-9223372036854775809", "-9223372036854775809"),
            ("18446744073709551615", "18446744073709551615"),
            (
                "340282366920938463463374607431768211455",
                "340282366920938463463374607431768211455",
            ),
            (
                "-170141183460469231731687303715884105728",
                "-170141183460469231731687303715884105728",
            ),
            ("1e400", "Infinity"),
            ("-1e400", "-Infinity"),
            ("1E5", "100000.0"),
            ("2.50", "2.5"),
            ("5e-324", "5e-324"),
            ("1e-400", "0.0"),
            ("NaN", "NaN"),
            ("Infinity", "Infinity"),
            ("-Infinity", "-Infinity"),
            (" [1, -2, 3.0e0] ", "[1, -2, 3.0]"),
        ] {
            assert_eq!(dumps(text), want, "{text}");
        }
        assert!(loads("-0").unwrap().is_integer());
    }

    #[test]
    fn objects_keep_the_first_position_and_the_last_value() {
        assert_eq!(
            dumps(r#"{"b": 1, "a": {"x": []}, "b": 2, "": "\u00e9\ud83d\ude00\n\/"}"#),
            "{\"b\": 2, \"a\": {\"x\": []}, \"\": \"é😀\\n/\"}"
        );
    }

    #[test]
    fn malformed_documents_are_refused_like_python_refuses_them() {
        for text in [
            "",
            "01",
            "1.",
            "1e",
            "-",
            "+1",
            ".5",
            "[1,]",
            "{\"a\": 1,}",
            "{a: 1}",
            "tru",
            "nan",
            "\"\\x\"",
            "\"a\nb\"",
            "\"open",
            "[1 2]",
            "{\"a\" 1}",
            "1 2",
            "\"\\u12\"",
            "-infinity",
        ] {
            assert_eq!(loads(text), Err(Failure::Malformed), "{text:?}");
        }
    }

    #[test]
    fn what_python_reads_but_cannot_be_kept_is_unsupported() {
        for text in [
            "340282366920938463463374607431768211456",
            "-170141183460469231731687303715884105729",
            "\"\\ud800\"",
            "\"\\udc00x\"",
        ] {
            assert_eq!(loads(text), Err(Failure::Unsupported), "{text}");
        }
        let deep = format!("{}{}", "[".repeat(MAX_DEPTH + 2), "]".repeat(MAX_DEPTH + 2));
        assert_eq!(loads(&deep), Err(Failure::Unsupported));
        let fine = format!("{}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        assert!(loads(&fine).is_ok());
    }
}
