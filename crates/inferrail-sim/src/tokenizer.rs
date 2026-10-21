//! A deterministic stand in for a real tokenizer and chat template.
//!
//! Token boundaries depend only on the text to their left, so a shared text
//! prefix always yields a shared token prefix, which is what prefix caching
//! needs. Pieces average about four characters, close to real BPE on English.

use inferrail_core::openai::{ChatMessage, Tool, ToolCall};
use std::fmt::Write as _;

use xxhash_rust::xxh3::{xxh3_64, xxh3_64_with_seed};

pub const BOS: u32 = 128_000;
pub const START_HEADER: u32 = 128_006;
pub const END_HEADER: u32 = 128_007;
pub const EOT: u32 = 128_009;
pub const TEXT_VOCAB: u32 = 128_000;

const MAX_PIECE_CHARS: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Space,
    OtherSpace,
    Word,
    Punct,
}

fn class(c: char) -> Class {
    if c == ' ' {
        Class::Space
    } else if c.is_whitespace() {
        Class::OtherSpace
    } else if c.is_alphanumeric() {
        Class::Word
    } else {
        Class::Punct
    }
}

fn continues(piece: &str, piece_chars: usize, prev: char, next: char) -> bool {
    if piece_chars >= MAX_PIECE_CHARS {
        return false;
    }
    match (class(prev), class(next)) {
        (Class::Word, Class::Word) => true,
        (Class::Space, Class::Word) => piece == " ",
        (a, b) => a == b && a != Class::Word,
    }
}

/// Splits text into pieces whose concatenation is the original text.
pub fn pieces(text: &str) -> Vec<&str> {
    let mut out = Vec::with_capacity(text.len() / 3 + 1);
    let mut start = 0;
    let mut chars_in_piece = 0;
    let mut prev = None;
    for (i, c) in text.char_indices() {
        if let Some(p) = prev
            && !continues(&text[start..i], chars_in_piece, p, c)
        {
            out.push(&text[start..i]);
            start = i;
            chars_in_piece = 0;
        }
        chars_in_piece += 1;
        prev = Some(c);
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

pub fn piece_id(piece: &str) -> u32 {
    (xxh3_64(piece.as_bytes()) % u64::from(TEXT_VOCAB)) as u32
}

pub fn tokenize(text: &str) -> Vec<u32> {
    pieces(text).into_iter().map(piece_id).collect()
}

pub fn encode_into(text: &str, out: &mut Vec<u32>) {
    out.extend(pieces(text).into_iter().map(piece_id));
}

/// Completions prompts get a leading BOS, as vLLM adds for Llama models.
pub fn encode_prompt(text: &str) -> Vec<u32> {
    let mut ids = Vec::with_capacity(text.len() / 3 + 2);
    ids.push(BOS);
    encode_into(text, &mut ids);
    ids
}

/// Renders a Llama 3 style chat template straight to token ids.
pub fn render_chat(
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    add_generation_prompt: bool,
) -> Vec<u32> {
    let mut ids = vec![BOS];
    if let Some(tools) = tools.filter(|t| !t.is_empty()) {
        let json = serde_json::to_string(tools).unwrap_or_default();
        push_block(&mut ids, "system", &format!("Tools: {json}"));
    }
    for message in messages {
        let mut content = message.text().map(|t| t.into_owned()).unwrap_or_default();
        if let Some(calls) = message.tool_calls.value() {
            append_tool_calls(&mut content, calls);
        }
        push_block(&mut ids, message.role.as_str(), &content);
    }
    if add_generation_prompt {
        push_header(&mut ids, "assistant");
    }
    ids
}

fn push_header(ids: &mut Vec<u32>, role: &str) {
    ids.push(START_HEADER);
    encode_into(role, ids);
    ids.push(END_HEADER);
    encode_into("\n\n", ids);
}

fn push_block(ids: &mut Vec<u32>, role: &str, content: &str) {
    push_header(ids, role);
    encode_into(content, ids);
    ids.push(EOT);
}

fn append_tool_calls(content: &mut String, calls: &[ToolCall]) {
    for call in calls {
        if !content.is_empty() {
            content.push('\n');
        }
        let _ = write!(
            content,
            "{{\"name\":{},\"parameters\":{}}}",
            serde_json::Value::from(call.function.name.as_str()),
            call.function.arguments
        );
    }
}

const WORDS: [&str; 32] = [
    "the", "cache", "router", "prefix", "token", "replica", "latency", "block", "engine", "queue",
    "decode", "prefill", "memory", "warm", "request", "stream", "model", "batch", "first", "load",
    "hit", "miss", "fleet", "graph", "signal", "budget", "tail", "steady", "quick", "bright",
    "river", "stone",
];

/// Deterministic model output: exactly `count` pieces that retokenize to the
/// same ids, so a later turn that resends this text reuses its blocks.
pub fn generate(seed: u64, count: usize) -> Vec<(u32, String)> {
    let mut text = String::with_capacity(count * 6);
    for i in 0..count {
        if i > 0 {
            text.push(' ');
        }
        let pick = xxh3_64_with_seed(&(i as u64).to_le_bytes(), seed);
        text.push_str(WORDS[(pick % WORDS.len() as u64) as usize]);
    }
    pieces(&text)
        .into_iter()
        .take(count)
        .map(|p| (piece_id(p), p.to_owned()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferrail_core::openai::Role;
    use proptest::prelude::*;

    #[test]
    fn pieces_look_like_bpe() {
        assert_eq!(
            pieces("The quick brownfox, jumps!!\n\nok"),
            [
                "The", " quic", "k", " brow", "nfox", ",", " jump", "s", "!!", "\n\n", "ok"
            ]
        );
        assert!(pieces("").is_empty());
        assert_eq!(pieces("  a"), ["  ", "a"]);
    }

    #[test]
    fn text_ids_never_collide_with_special_tokens() {
        for piece in ["a", " the", "<|eot_id|>", "\n\n"] {
            assert!(piece_id(piece) < TEXT_VOCAB);
        }
    }

    #[test]
    fn prompt_encoding_starts_with_bos() {
        let ids = encode_prompt("hello world");
        assert_eq!(ids[0], BOS);
        assert_eq!(&ids[1..], tokenize("hello world").as_slice());
    }

    #[test]
    fn chat_template_frames_every_message() {
        let messages = [
            ChatMessage::new(Role::System, "be brief"),
            ChatMessage::new(Role::User, "hi"),
        ];
        let ids = render_chat(&messages, None, true);
        assert_eq!(ids[0], BOS);
        assert_eq!(ids.iter().filter(|&&t| t == START_HEADER).count(), 3);
        assert_eq!(ids.iter().filter(|&&t| t == EOT).count(), 2);
        assert_eq!(*ids.last().unwrap(), tokenize("\n\n")[0]);
    }

    #[test]
    fn tools_and_tool_calls_add_tokens() {
        let plain = render_chat(&[ChatMessage::new(Role::User, "x")], None, true);
        let tool: Tool = serde_json::from_value(serde_json::json!({
            "type": "function", "function": {"name": "f", "parameters": {}}
        }))
        .unwrap();
        let with_tools = render_chat(&[ChatMessage::new(Role::User, "x")], Some(&[tool]), true);
        assert!(with_tools.len() > plain.len());

        let call: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant", "content": null, "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]
        }))
        .unwrap();
        let rendered = render_chat(&[call], None, false);
        assert!(rendered.len() > 6);
    }

    #[test]
    fn generation_is_deterministic_and_exact_length() {
        let a = generate(7, 50);
        assert_eq!(a.len(), 50);
        assert_eq!(a, generate(7, 50));
        assert_ne!(a, generate(8, 50));
        assert!(generate(1, 0).is_empty());
    }

    fn messages() -> impl Strategy<Value = Vec<(bool, String)>> {
        proptest::collection::vec((any::<bool>(), "[ -~\n]{0,40}"), 1..6)
    }

    proptest! {
        #[test]
        fn pieces_concatenate_to_input(text in "\\PC{0,200}") {
            prop_assert_eq!(pieces(&text).concat(), text);
        }

        #[test]
        fn shared_text_prefix_means_shared_token_prefix(a in "\\PC{0,120}", b in "\\PC{0,120}") {
            let short = tokenize(&a);
            let long = tokenize(&format!("{a}{b}"));
            let stable = short.len().saturating_sub(1);
            prop_assert_eq!(&short[..stable], &long[..stable]);
        }

        #[test]
        fn generated_text_retokenizes_identically(seed in any::<u64>(), count in 0usize..300) {
            let out = generate(seed, count);
            let text: String = out.iter().map(|(_, p)| p.as_str()).collect();
            let ids: Vec<u32> = out.iter().map(|(id, _)| *id).collect();
            prop_assert_eq!(tokenize(&text), ids);
        }

        #[test]
        fn earlier_turns_are_a_token_prefix_of_later_turns(turns in messages()) {
            let msgs: Vec<ChatMessage> = turns
                .iter()
                .map(|(user, text)| {
                    ChatMessage::new(if *user { Role::User } else { Role::Assistant }, text.as_str())
                })
                .collect();
            let full = render_chat(&msgs, None, false);
            for k in 0..msgs.len() {
                let head = render_chat(&msgs[..k], None, false);
                prop_assert!(full.starts_with(&head));
            }
        }

        #[test]
        fn generation_prompt_is_a_prefix_of_the_answered_turn(question in "[ -~]{0,40}", seed in any::<u64>()) {
            let ask = [ChatMessage::new(Role::User, question.as_str())];
            let prompt = render_chat(&ask, None, true);
            let answer: String = generate(seed, 12).into_iter().map(|(_, p)| p).collect();
            let answered = [ask[0].clone(), ChatMessage::new(Role::Assistant, answer)];
            prop_assert!(render_chat(&answered, None, false).starts_with(&prompt));
        }
    }
}
