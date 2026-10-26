//! Case changes as CPython 3.12 makes them. `str.title()` starts a word
//! after any character without case and `str.capitalize()` starts one at
//! the front; both title case that first letter, which for ligatures and
//! digraphs is not its upper case, and lower case the rest with `Σ` becoming
//! `ς` at the end of a word. Jinja2's `title` and `capitalize` filters are
//! built on them.

pub(crate) fn title(s: &str) -> String {
    let lowered = s.to_lowercase();
    let mut out = String::with_capacity(s.len());
    let mut at = 0;
    let mut previous_cased = false;
    for c in s.chars() {
        let len = lower_len(c);
        if previous_cased {
            out.push_str(&lowered[at..at + len]);
        } else {
            push_title(c, &mut out);
        }
        at += len;
        previous_cased = is_cased(c);
    }
    out
}

pub(crate) fn capitalize(s: &str) -> String {
    let Some(first) = s.chars().next() else {
        return String::new();
    };
    let mut out = String::with_capacity(s.len());
    push_title(first, &mut out);
    out.push_str(&s.to_lowercase()[lower_len(first)..]);
    out
}

/// Jinja2's `title` filter: each piece between runs of whitespace, dashes
/// and opening brackets gets its first character upper cased and the rest
/// lower cased.
pub(crate) fn title_filter(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while !rest.is_empty() {
        let split = |c: char| is_space(c) || matches!(c, '-' | '(' | '{' | '[' | '<');
        let end = rest.find(|c: char| !split(c)).unwrap_or(rest.len());
        out.push_str(&rest[..end]);
        rest = &rest[end..];
        let end = rest.find(split).unwrap_or(rest.len());
        let mut word = rest[..end].chars();
        if let Some(first) = word.next() {
            out.extend(first.to_uppercase());
            out.push_str(&word.as_str().to_lowercase());
        }
        rest = &rest[end..];
    }
    out
}

/// `str.isspace()`, which unlike Unicode's White_Space includes the
/// information separators.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn lower_len(c: char) -> usize {
    c.to_lowercase().map(char::len_utf8).sum()
}

fn is_cased(c: char) -> bool {
    c.is_lowercase()
        || c.is_uppercase()
        || matches!(
            c,
            '\u{1c5}'
                | '\u{1c8}'
                | '\u{1cb}'
                | '\u{1f2}'
                | '\u{1f88}'..='\u{1f8f}'
                | '\u{1f98}'..='\u{1f9f}'
                | '\u{1fa8}'..='\u{1faf}'
                | '\u{1fbc}'
                | '\u{1fcc}'
                | '\u{1ffc}'
        )
}

fn push_title(c: char, out: &mut String) {
    match TITLE.binary_search_by_key(&c, |&(from, _)| from) {
        Ok(i) => out.push_str(TITLE[i].1),
        Err(_) => out.extend(c.to_uppercase()),
    }
}

/// Characters whose title case is not their upper case, from CPython 3.12.
#[rustfmt::skip]
const TITLE: [(char, &str); 135] = [
    ('\u{df}', "Ss"), ('\u{1c4}', "\u{1c5}"), ('\u{1c5}', "\u{1c5}"), ('\u{1c6}', "\u{1c5}"),
    ('\u{1c7}', "\u{1c8}"), ('\u{1c8}', "\u{1c8}"), ('\u{1c9}', "\u{1c8}"), ('\u{1ca}', "\u{1cb}"),
    ('\u{1cb}', "\u{1cb}"), ('\u{1cc}', "\u{1cb}"), ('\u{1f1}', "\u{1f2}"), ('\u{1f2}', "\u{1f2}"),
    ('\u{1f3}', "\u{1f2}"), ('\u{587}', "\u{535}\u{582}"), ('\u{10d0}', "\u{10d0}"),
    ('\u{10d1}', "\u{10d1}"), ('\u{10d2}', "\u{10d2}"), ('\u{10d3}', "\u{10d3}"),
    ('\u{10d4}', "\u{10d4}"), ('\u{10d5}', "\u{10d5}"), ('\u{10d6}', "\u{10d6}"),
    ('\u{10d7}', "\u{10d7}"), ('\u{10d8}', "\u{10d8}"), ('\u{10d9}', "\u{10d9}"),
    ('\u{10da}', "\u{10da}"), ('\u{10db}', "\u{10db}"), ('\u{10dc}', "\u{10dc}"),
    ('\u{10dd}', "\u{10dd}"), ('\u{10de}', "\u{10de}"), ('\u{10df}', "\u{10df}"),
    ('\u{10e0}', "\u{10e0}"), ('\u{10e1}', "\u{10e1}"), ('\u{10e2}', "\u{10e2}"),
    ('\u{10e3}', "\u{10e3}"), ('\u{10e4}', "\u{10e4}"), ('\u{10e5}', "\u{10e5}"),
    ('\u{10e6}', "\u{10e6}"), ('\u{10e7}', "\u{10e7}"), ('\u{10e8}', "\u{10e8}"),
    ('\u{10e9}', "\u{10e9}"), ('\u{10ea}', "\u{10ea}"), ('\u{10eb}', "\u{10eb}"),
    ('\u{10ec}', "\u{10ec}"), ('\u{10ed}', "\u{10ed}"), ('\u{10ee}', "\u{10ee}"),
    ('\u{10ef}', "\u{10ef}"), ('\u{10f0}', "\u{10f0}"), ('\u{10f1}', "\u{10f1}"),
    ('\u{10f2}', "\u{10f2}"), ('\u{10f3}', "\u{10f3}"), ('\u{10f4}', "\u{10f4}"),
    ('\u{10f5}', "\u{10f5}"), ('\u{10f6}', "\u{10f6}"), ('\u{10f7}', "\u{10f7}"),
    ('\u{10f8}', "\u{10f8}"), ('\u{10f9}', "\u{10f9}"), ('\u{10fa}', "\u{10fa}"),
    ('\u{10fd}', "\u{10fd}"), ('\u{10fe}', "\u{10fe}"), ('\u{10ff}', "\u{10ff}"),
    ('\u{1f80}', "\u{1f88}"), ('\u{1f81}', "\u{1f89}"), ('\u{1f82}', "\u{1f8a}"),
    ('\u{1f83}', "\u{1f8b}"), ('\u{1f84}', "\u{1f8c}"), ('\u{1f85}', "\u{1f8d}"),
    ('\u{1f86}', "\u{1f8e}"), ('\u{1f87}', "\u{1f8f}"), ('\u{1f88}', "\u{1f88}"),
    ('\u{1f89}', "\u{1f89}"), ('\u{1f8a}', "\u{1f8a}"), ('\u{1f8b}', "\u{1f8b}"),
    ('\u{1f8c}', "\u{1f8c}"), ('\u{1f8d}', "\u{1f8d}"), ('\u{1f8e}', "\u{1f8e}"),
    ('\u{1f8f}', "\u{1f8f}"), ('\u{1f90}', "\u{1f98}"), ('\u{1f91}', "\u{1f99}"),
    ('\u{1f92}', "\u{1f9a}"), ('\u{1f93}', "\u{1f9b}"), ('\u{1f94}', "\u{1f9c}"),
    ('\u{1f95}', "\u{1f9d}"), ('\u{1f96}', "\u{1f9e}"), ('\u{1f97}', "\u{1f9f}"),
    ('\u{1f98}', "\u{1f98}"), ('\u{1f99}', "\u{1f99}"), ('\u{1f9a}', "\u{1f9a}"),
    ('\u{1f9b}', "\u{1f9b}"), ('\u{1f9c}', "\u{1f9c}"), ('\u{1f9d}', "\u{1f9d}"),
    ('\u{1f9e}', "\u{1f9e}"), ('\u{1f9f}', "\u{1f9f}"), ('\u{1fa0}', "\u{1fa8}"),
    ('\u{1fa1}', "\u{1fa9}"), ('\u{1fa2}', "\u{1faa}"), ('\u{1fa3}', "\u{1fab}"),
    ('\u{1fa4}', "\u{1fac}"), ('\u{1fa5}', "\u{1fad}"), ('\u{1fa6}', "\u{1fae}"),
    ('\u{1fa7}', "\u{1faf}"), ('\u{1fa8}', "\u{1fa8}"), ('\u{1fa9}', "\u{1fa9}"),
    ('\u{1faa}', "\u{1faa}"), ('\u{1fab}', "\u{1fab}"), ('\u{1fac}', "\u{1fac}"),
    ('\u{1fad}', "\u{1fad}"), ('\u{1fae}', "\u{1fae}"), ('\u{1faf}', "\u{1faf}"),
    ('\u{1fb2}', "\u{1fba}\u{345}"), ('\u{1fb3}', "\u{1fbc}"), ('\u{1fb4}', "\u{386}\u{345}"),
    ('\u{1fb7}', "\u{391}\u{342}\u{345}"), ('\u{1fbc}', "\u{1fbc}"),
    ('\u{1fc2}', "\u{1fca}\u{345}"), ('\u{1fc3}', "\u{1fcc}"), ('\u{1fc4}', "\u{389}\u{345}"),
    ('\u{1fc7}', "\u{397}\u{342}\u{345}"), ('\u{1fcc}', "\u{1fcc}"),
    ('\u{1ff2}', "\u{1ffa}\u{345}"), ('\u{1ff3}', "\u{1ffc}"), ('\u{1ff4}', "\u{38f}\u{345}"),
    ('\u{1ff7}', "\u{3a9}\u{342}\u{345}"), ('\u{1ffc}', "\u{1ffc}"), ('\u{fb00}', "Ff"),
    ('\u{fb01}', "Fi"), ('\u{fb02}', "Fl"), ('\u{fb03}', "Ffi"), ('\u{fb04}', "Ffl"),
    ('\u{fb05}', "St"), ('\u{fb06}', "St"), ('\u{fb13}', "\u{544}\u{576}"),
    ('\u{fb14}', "\u{544}\u{565}"), ('\u{fb15}', "\u{544}\u{56b}"), ('\u{fb16}', "\u{54e}\u{576}"),
    ('\u{fb17}', "\u{544}\u{56d}"),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Each expectation is CPython 3.12's, or Jinja2's for the filter.
    #[test]
    fn case_changes_match_python() {
        for (text, titled, capitalized, filtered) in [
            ("hello world", "Hello World", "Hello world", "Hello World"),
            ("HELLO wORLD", "Hello World", "Hello world", "Hello World"),
            (
                "ﬁx ßa 1st a_b they're",
                "Fix Ssa 1St A_B They'Re",
                "Fix ßa 1st a_b they're",
                "FIx SSa 1st A_b They're",
            ),
            (
                "ΣΑΣ ΟΔΟΣ ΣΑ.Σ",
                "Σας Οδος Σα.Σ",
                "Σας οδος σα.ς",
                "Σας Οδος Σα.ς",
            ),
            ("ǆungla ǈ", "ǅungla ǈ", "ǅungla ǉ", "Ǆungla Ǉ"),
            ("ა ᾳ ŉ ǰ", "ა ᾼ ʼN J\u{30c}", "ა ᾳ ŉ ǰ", "Ა ΑΙ ʼN J\u{30c}"),
            (
                "\u{1f}padded\u{1c}x",
                "\u{1f}Padded\u{1c}X",
                "\u{1f}padded\u{1c}x",
                "\u{1f}Padded\u{1c}X",
            ),
            ("a-b(c[d<e{f", "A-B(C[D<E{F", "A-b(c[d<e{f", "A-B(C[D<E{F"),
            ("", "", "", ""),
            ("İstanbul", "İstanbul", "İstanbul", "İstanbul"),
        ] {
            assert_eq!(title(text), titled, "title {text:?}");
            assert_eq!(capitalize(text), capitalized, "capitalize {text:?}");
            assert_eq!(title_filter(text), filtered, "filter {text:?}");
        }
        for pair in TITLE.windows(2) {
            assert!(pair[0].0 < pair[1].0);
        }
    }
}
