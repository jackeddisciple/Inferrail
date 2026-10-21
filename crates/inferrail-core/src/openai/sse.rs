//! Server sent event framing as used by OpenAI compatible streams.

pub const DONE_DATA: &str = "[DONE]";
pub const DONE_FRAME: &[u8] = b"data: [DONE]\n\n";

const DEFAULT_MAX_LINE: usize = 16 * 1024 * 1024;

/// Frames one JSON payload. Compact JSON never contains a raw newline.
pub fn data_frame(payload: &[u8]) -> Vec<u8> {
    debug_assert!(!payload.contains(&b'\n'));
    let mut frame = Vec::with_capacity(payload.len() + 8);
    frame.extend_from_slice(b"data: ");
    frame.extend_from_slice(payload);
    frame.extend_from_slice(b"\n\n");
    frame
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

impl SseEvent {
    pub fn is_done(&self) -> bool {
        self.data == DONE_DATA
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SseError {
    #[error("event stream line is not valid UTF-8")]
    InvalidUtf8,
    #[error("event stream line exceeds {0} bytes")]
    LineTooLong(usize),
}

/// Incremental decoder that accepts the stream in arbitrary byte chunks.
#[derive(Debug)]
pub struct SseDecoder {
    buf: Vec<u8>,
    skip_lf: bool,
    data: String,
    has_data: bool,
    event: Option<String>,
    max_line: usize,
}

impl Default for SseDecoder {
    fn default() -> Self {
        Self::with_max_line(DEFAULT_MAX_LINE)
    }
}

impl SseDecoder {
    pub fn with_max_line(max_line: usize) -> Self {
        Self {
            buf: Vec::new(),
            skip_lf: false,
            data: String::new(),
            has_data: false,
            event: None,
            max_line,
        }
    }

    pub fn push(&mut self, chunk: &[u8], out: &mut Vec<SseEvent>) -> Result<(), SseError> {
        self.buf.extend_from_slice(chunk);
        let mut start = 0;
        while start < self.buf.len() {
            if self.skip_lf {
                self.skip_lf = false;
                if self.buf[start] == b'\n' {
                    start += 1;
                    continue;
                }
            }
            let Some(offset) = self.buf[start..]
                .iter()
                .position(|&b| b == b'\n' || b == b'\r')
            else {
                break;
            };
            let end = start + offset;
            self.skip_lf = self.buf[end] == b'\r';
            let line = std::str::from_utf8(&self.buf[start..end])
                .map_err(|_| SseError::InvalidUtf8)?
                .to_owned();
            self.process_line(&line, out);
            start = end + 1;
        }
        self.buf.drain(..start);
        if self.buf.len() > self.max_line {
            return Err(SseError::LineTooLong(self.max_line));
        }
        Ok(())
    }

    /// True when every byte pushed so far belongs to a completed event.
    pub fn is_idle(&self) -> bool {
        self.buf.is_empty() && !self.has_data && self.event.is_none()
    }

    fn process_line(&mut self, line: &str, out: &mut Vec<SseEvent>) {
        if line.is_empty() {
            if self.has_data {
                if self.data.ends_with('\n') {
                    self.data.pop();
                }
                out.push(SseEvent {
                    event: self.event.take(),
                    data: std::mem::take(&mut self.data),
                });
            }
            self.data.clear();
            self.has_data = false;
            self.event = None;
            return;
        }
        if line.starts_with(':') {
            return;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "data" => {
                self.data.push_str(value);
                self.data.push('\n');
                self.has_data = true;
            }
            "event" => self.event = Some(value.to_owned()),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn decode_all(chunks: &[&[u8]]) -> Vec<SseEvent> {
        let mut decoder = SseDecoder::default();
        let mut out = Vec::new();
        for chunk in chunks {
            decoder.push(chunk, &mut out).unwrap();
        }
        out
    }

    fn data(text: &str) -> SseEvent {
        SseEvent {
            event: None,
            data: text.to_owned(),
        }
    }

    #[test]
    fn decodes_openai_stream() {
        let mut stream = data_frame(br#"{"a":1}"#);
        stream.extend_from_slice(DONE_FRAME);
        let events = decode_all(&[&stream]);
        assert_eq!(events, vec![data(r#"{"a":1}"#), data("[DONE]")]);
        assert!(events[1].is_done());
    }

    #[test]
    fn comments_unknown_fields_and_empty_events_are_ignored() {
        let events = decode_all(&[b": keepalive\n\nid: 7\nretry: 10\n\nevent: ping\ndata\n\n"]);
        assert_eq!(
            events,
            vec![SseEvent {
                event: Some("ping".into()),
                data: String::new()
            }]
        );
    }

    #[test]
    fn multi_line_data_joins_with_newline() {
        assert_eq!(
            decode_all(&[b"data: a\ndata:b\ndata:  c\n\n"]),
            vec![data("a\nb\n c")]
        );
    }

    #[test]
    fn crlf_split_across_chunks_is_one_terminator() {
        let events = decode_all(&[b"data: x\r", b"\n\r", b"\ndata: y\r\r"]);
        assert_eq!(events, vec![data("x"), data("y")]);
    }

    #[test]
    fn incomplete_event_is_held_back() {
        let mut decoder = SseDecoder::default();
        let mut out = Vec::new();
        decoder.push(b"data: partial", &mut out).unwrap();
        assert!(out.is_empty());
        assert!(!decoder.is_idle());
        decoder.push(b"\n\n", &mut out).unwrap();
        assert_eq!(out, vec![data("partial")]);
        assert!(decoder.is_idle());
    }

    #[test]
    fn utf8_split_across_chunks_is_fine_but_invalid_bytes_fail() {
        let text = "data: héllo\n\n".as_bytes();
        let (a, b) = text.split_at(8);
        assert_eq!(decode_all(&[a, b]), vec![data("héllo")]);
        let mut out = Vec::new();
        assert_eq!(
            SseDecoder::default().push(b"data: \xff\n\n", &mut out),
            Err(SseError::InvalidUtf8)
        );
    }

    #[test]
    fn oversized_line_is_rejected() {
        let mut decoder = SseDecoder::with_max_line(8);
        let mut out = Vec::new();
        assert_eq!(
            decoder.push(b"data: 0123456789", &mut out),
            Err(SseError::LineTooLong(8))
        );
    }

    fn arb_event() -> impl Strategy<Value = SseEvent> {
        (
            proptest::option::of("[a-z_]{1,8}"),
            proptest::collection::vec("[^\r\n]{0,12}", 1..4),
        )
            .prop_map(|(event, lines)| SseEvent {
                event,
                data: lines.join("\n"),
            })
    }

    fn encode(events: &[SseEvent], newline: &str) -> Vec<u8> {
        let mut out = String::new();
        for event in events {
            if let Some(name) = &event.event {
                out.push_str(&format!("event: {name}{newline}"));
            }
            for line in event.data.split('\n') {
                out.push_str(&format!("data: {line}{newline}"));
            }
            out.push_str(newline);
        }
        out.into_bytes()
    }

    proptest! {
        #[test]
        fn any_chunking_decodes_to_the_same_events(
            events in proptest::collection::vec(arb_event(), 0..8),
            newline in prop_oneof![Just("\n"), Just("\r\n"), Just("\r")],
            cuts in proptest::collection::vec(any::<prop::sample::Index>(), 0..12),
        ) {
            let bytes = encode(&events, newline);
            let mut points: Vec<usize> = cuts.iter().map(|c| c.index(bytes.len() + 1)).collect();
            points.sort_unstable();
            let mut chunks = Vec::new();
            let mut last = 0;
            for p in points {
                chunks.push(&bytes[last..p]);
                last = p;
            }
            chunks.push(&bytes[last..]);
            prop_assert_eq!(decode_all(&chunks), events);
        }
    }
}
