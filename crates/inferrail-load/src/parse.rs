//! A scanner for the Prometheus text exposition format that borrows from
//! the page and skips anything it cannot read instead of failing.

use std::borrow::Cow;

/// One sample line: `name{labels} value [timestamp]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample<'a> {
    pub name: &'a str,
    labels: &'a str,
    pub value: f64,
}

impl<'a> Sample<'a> {
    /// The value of label `key`, unescaped.
    pub fn label(&self, key: &str) -> Option<Cow<'a, str>> {
        Labels(self.labels).find_map(|(k, v)| (k == key).then(|| unescape(v)))
    }

    pub fn labels(&self) -> impl Iterator<Item = (&'a str, Cow<'a, str>)> {
        Labels(self.labels).map(|(k, v)| (k, unescape(v)))
    }
}

/// Every well formed sample on the page, in order. Comments, blank lines
/// and malformed lines are skipped.
pub fn samples(text: &str) -> impl Iterator<Item = Sample<'_>> {
    text.lines().filter_map(sample)
}

/// The metric name a sample line starts with, without parsing the rest.
pub fn name(line: &str) -> Option<&str> {
    let line = line.trim_start();
    let end = line
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == ':'))
        .unwrap_or(line.len());
    let name = &line[..end];
    name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_' || c == ':')
        .then_some(name)
}

/// Parses one line; `None` for comments and anything malformed.
pub fn sample(line: &str) -> Option<Sample<'_>> {
    let line = line.trim_start();
    let name = name(line)?;
    let mut rest = &line[name.len()..];
    let mut labels = "";
    if let Some(inner) = rest.strip_prefix('{') {
        let close = closing_brace(inner)?;
        labels = &inner[..close];
        Labels(labels).validate()?;
        rest = &inner[close + 1..];
    }
    if !rest.starts_with([' ', '\t']) {
        return None;
    }
    let token = rest.split_ascii_whitespace().next()?;
    Some(Sample {
        name,
        labels,
        value: parse_value(token)?,
    })
}

fn parse_value(token: &str) -> Option<f64> {
    match token {
        "+Inf" | "Inf" => Some(f64::INFINITY),
        "-Inf" => Some(f64::NEG_INFINITY),
        "NaN" => Some(f64::NAN),
        _ if token
            .bytes()
            .all(|b| b.is_ascii_digit() || b"+-.eE".contains(&b)) =>
        {
            token.parse().ok()
        }
        _ => None,
    }
}

/// Index of the `}` that closes a label block, skipping quoted values.
fn closing_brace(inner: &str) -> Option<usize> {
    let bytes = inner.as_bytes();
    let mut quoted = false;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if quoted => i += 1,
            b'"' => quoted = !quoted,
            b'}' if !quoted => return Some(i),
            _ => {}
        }
        i += 1;
    }
    None
}

/// Iterates `key="value"` pairs of a label block, values still escaped.
struct Labels<'a>(&'a str);

impl Labels<'_> {
    fn validate(mut self) -> Option<()> {
        for _ in self.by_ref() {}
        self.0.trim_matches([' ', ',']).is_empty().then_some(())
    }
}

impl<'a> Iterator for Labels<'a> {
    type Item = (&'a str, &'a str);

    fn next(&mut self) -> Option<Self::Item> {
        let rest = self.0.trim_start_matches([' ', ',']);
        let eq = rest.find('=')?;
        let key = rest[..eq].trim();
        let value_start = rest[eq + 1..].trim_start();
        let body = value_start.strip_prefix('"')?;
        let bytes = body.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'\\' => i += 2,
                b'"' => break,
                _ => i += 1,
            }
        }
        if i >= bytes.len() || key.is_empty() {
            return None;
        }
        self.0 = &body[i + 1..];
        Some((key, &body[..i]))
    }
}

fn unescape(raw: &str) -> Cow<'_, str> {
    if !raw.contains('\\') {
        return Cow::Borrowed(raw);
    }
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some(other) => out.push(other),
            None => out.push('\\'),
        }
    }
    Cow::Owned(out)
}

/// The inverse of [`unescape`], for renderers and tests.
pub fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn plain_and_labelled_samples_parse() {
        let s = sample("up 1").unwrap();
        assert_eq!((s.name, s.value), ("up", 1.0));
        assert_eq!(s.label("x"), None);

        let s = sample(r#"vllm:num_requests_running{engine="0",model_name="m"} 2.0"#).unwrap();
        assert_eq!(s.name, "vllm:num_requests_running");
        assert_eq!(s.value, 2.0);
        assert_eq!(s.label("engine").as_deref(), Some("0"));
        assert_eq!(s.label("model_name").as_deref(), Some("m"));
        assert_eq!(s.labels().count(), 2);
    }

    #[test]
    fn special_values_timestamps_and_exemplars() {
        assert_eq!(sample("a +Inf").unwrap().value, f64::INFINITY);
        assert_eq!(sample("a -Inf").unwrap().value, f64::NEG_INFINITY);
        assert!(sample("a NaN").unwrap().value.is_nan());
        assert_eq!(sample("a 1e+06 1712345678000").unwrap().value, 1e6);
        assert_eq!(sample("a{b=\"c\"}\t3").unwrap().value, 3.0);
        assert_eq!(
            sample(r#"a_bucket{le="0.5"} 4 # {trace_id="x"} 0.4"#)
                .unwrap()
                .value,
            4.0
        );
        assert_eq!(sample("  a{} 7").unwrap().value, 7.0);
        assert_eq!(sample("a{b=\"c\",} 8").unwrap().value, 8.0);
    }

    #[test]
    fn escaped_label_values_survive_braces_and_quotes() {
        let line = r#"m{model_name="a}b\"c\\d\ne",x="y"} 5"#;
        let s = sample(line).unwrap();
        assert_eq!(s.label("model_name").as_deref(), Some("a}b\"c\\d\ne"));
        assert_eq!(s.label("x").as_deref(), Some("y"));
        assert_eq!(s.value, 5.0);
    }

    #[test]
    fn malformed_lines_are_skipped() {
        for line in [
            "",
            "# HELP a b",
            "# TYPE a gauge",
            "a",
            "a{",
            "a{b=\"c\" 1",
            "a{b=c} 1",
            "a{=\"c\"} 1",
            "a{b=\"c\"}1",
            "a one",
            "a 1.2.3x",
            "9a 1",
            "{b=\"c\"} 1",
            "a{b=\"c\" d=\"e\"} 1junk",
        ] {
            assert!(sample(line).is_none(), "{line:?}");
        }
        let page = "# HELP a x\na 1\ngarbage here\n\nb{c=\"d\"} 2\n";
        let names: Vec<_> = samples(page).map(|s| s.name).collect();
        assert_eq!(names, ["a", "b"]);
    }

    proptest! {
        #[test]
        fn arbitrary_text_never_panics(text in "(?s).{0,600}") {
            for s in samples(&text) {
                let _ = s.labels().count();
                let _ = s.label("engine");
            }
        }

        #[test]
        fn arbitrary_label_values_round_trip(
            values in prop::collection::vec("(?s).{0,40}", 1..5),
            value in prop::num::f64::ANY,
        ) {
            let pairs: Vec<String> = values
                .iter()
                .enumerate()
                .map(|(i, v)| format!("k{i}=\"{}\"", escape(v)))
                .collect();
            let rendered = match value {
                v if v.is_nan() => "NaN".to_owned(),
                v if v == f64::INFINITY => "+Inf".to_owned(),
                v if v == f64::NEG_INFINITY => "-Inf".to_owned(),
                v => format!("{v:e}"),
            };
            let line = format!("metric{{{}}} {rendered}", pairs.join(","));
            let s = sample(&line).expect("well formed");
            for (i, v) in values.iter().enumerate() {
                let key = format!("k{i}");
                let got = s.label(&key);
                prop_assert_eq!(got.as_deref(), Some(v.as_str()));
            }
            if value.is_nan() {
                prop_assert!(s.value.is_nan());
            } else {
                prop_assert_eq!(s.value, value);
            }
        }
    }
}
