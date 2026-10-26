//! Hugging Face tokenizer files, loaded the way transformers 5.17 loads
//! them, since that is what vLLM v0.31.0 tokenizes with.
//!
//! transformers keeps `tokenizer.json` as written only for generic tokenizer
//! classes. Model specific classes rebuild the pipeline from the vocabulary
//! and merges, so for `LlamaTokenizer` the file's normalizer is replaced by
//! a Metaspace pre tokenizer, which changes how leading spaces tokenize.
//! The classes rebuilt here are the ones the golden fixtures verify.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value as Json;
use serde_json::value::RawValue;
use tokenizers::Tokenizer;

use crate::TokenizeError;
use crate::template::{ChatTemplate, Clock};

const SPECIAL_ATTRIBUTES: [&str; 7] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
];

const QWEN2_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Whether the tokenizer class is one whose transformers pipeline Inferrail
/// rebuilds and the golden fixtures check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parity {
    Verified,
    Unverified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Generic,
    Llama,
    Qwen2,
    Gpt2,
    Other,
}

impl Class {
    fn of(name: Option<&str>) -> Self {
        let Some(name) = name else {
            return Self::Generic;
        };
        match name.strip_suffix("Fast").unwrap_or(name) {
            "PreTrainedTokenizer" | "TokenizersBackend" => Self::Generic,
            "LlamaTokenizer" => Self::Llama,
            "Qwen2Tokenizer" => Self::Qwen2,
            "GPT2Tokenizer" => Self::Gpt2,
            _ => Self::Other,
        }
    }

    fn defaults(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Llama => &[
                ("unk_token", "<unk>"),
                ("bos_token", "<s>"),
                ("eos_token", "</s>"),
            ],
            Self::Qwen2 => &[
                ("unk_token", "<|endoftext|>"),
                ("eos_token", "<|endoftext|>"),
                ("pad_token", "<|endoftext|>"),
            ],
            Self::Gpt2 => &[
                ("unk_token", "<|endoftext|>"),
                ("bos_token", "<|endoftext|>"),
                ("eos_token", "<|endoftext|>"),
            ],
            Self::Generic | Self::Other => &[],
        }
    }
}

/// Chat templates by name, chosen as `get_chat_template` chooses them.
#[derive(Debug, Default)]
pub(crate) struct Templates {
    single: Option<ChatTemplate>,
    named: Vec<(String, ChatTemplate)>,
}

impl Templates {
    pub(crate) fn select(&self, tools: bool) -> Result<&ChatTemplate, TokenizeError> {
        if let Some(single) = &self.single {
            return Ok(single);
        }
        if self.named.is_empty() {
            return Err(TokenizeError::Unsupported(
                "a model without a chat template",
            ));
        }
        let find = |name: &str| self.named.iter().find(|(n, _)| n == name).map(|(_, t)| t);
        tools
            .then(|| find("tool_use"))
            .flatten()
            .or_else(|| find("default"))
            .ok_or_else(|| {
                TokenizeError::Invalid("the model has several chat templates and no default".into())
            })
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.single.is_none() && self.named.is_empty()
    }
}

#[derive(Debug)]
pub(crate) struct HfFiles {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) templates: Templates,
    pub(crate) special: Vec<(&'static str, String)>,
    pub(crate) parity: Parity,
}

fn read(path: &Path) -> Result<String, TokenizeError> {
    std::fs::read_to_string(path).map_err(|source| TokenizeError::Io {
        path: path.to_owned(),
        source,
    })
}

fn read_json(path: &Path) -> Result<Option<Json>, TokenizeError> {
    if !path.is_file() {
        return Ok(None);
    }
    serde_json::from_str(&read(path)?)
        .map(Some)
        .map_err(|e| TokenizeError::Parse {
            path: path.to_owned(),
            message: e.to_string(),
        })
}

fn token_text(value: &Json) -> Option<Option<String>> {
    match value {
        Json::Null => Some(None),
        Json::String(s) => Some(Some(s.clone())),
        Json::Object(o) => o
            .get("content")
            .and_then(Json::as_str)
            .map(|s| Some(s.to_owned())),
        _ => None,
    }
}

pub(crate) fn load(
    dir: &Path,
    template_override: Option<&Path>,
    clock: &Clock,
) -> Result<HfFiles, TokenizeError> {
    let tokenizer_path = dir.join("tokenizer.json");
    if !tokenizer_path.is_file() {
        return Err(TokenizeError::NotFound(format!(
            "{} has no tokenizer.json",
            dir.display()
        )));
    }
    let config = read_json(&dir.join("tokenizer_config.json"))?.unwrap_or(Json::Null);
    let class = Class::of(config.get("tokenizer_class").and_then(Json::as_str));
    let text = read(&tokenizer_path)?;
    let parse_error = |message: String| TokenizeError::Parse {
        path: tokenizer_path.clone(),
        message,
    };
    let rebuilt = rebuild(&text, class, &config).map_err(parse_error)?;
    let mut tokenizer =
        Tokenizer::from_bytes(rebuilt.as_bytes()).map_err(|e| TokenizeError::Parse {
            path: tokenizer_path.clone(),
            message: e.to_string(),
        })?;
    tokenizer
        .with_truncation(None)
        .map_err(|e| TokenizeError::Tokenizer(e.to_string()))?;
    tokenizer.with_padding(None);
    let special = special_tokens(dir, &config, class)?;
    add_missing_tokens(&mut tokenizer, &config, &special)?;
    let split = config
        .get("split_special_tokens")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    tokenizer.set_encode_special_tokens(split);
    let templates = templates(dir, &config, template_override, clock)?;
    let parity = match class {
        Class::Generic | Class::Llama | Class::Qwen2 | Class::Gpt2 => Parity::Verified,
        Class::Other => Parity::Unverified,
    };
    Ok(HfFiles {
        tokenizer,
        templates,
        special,
        parity,
    })
}

/// Applies what the transformers class constructor builds over the file.
fn rebuild(text: &str, class: Class, config: &Json) -> Result<String, String> {
    let mut top: BTreeMap<String, Box<RawValue>> =
        serde_json::from_str(text).map_err(|e| e.to_string())?;
    let flag = |key: &str| config.get(key).and_then(Json::as_bool);
    let raw = |value: Json| -> Result<Box<RawValue>, String> {
        RawValue::from_string(value.to_string()).map_err(|e| e.to_string())
    };
    let model_flags: Option<Vec<(&str, Json)>> = match class {
        Class::Llama => Some(vec![
            ("unk_token", Json::Null),
            ("continuing_subword_prefix", Json::Null),
            ("end_of_word_suffix", Json::Null),
            ("fuse_unk", Json::Bool(true)),
            ("byte_fallback", Json::Bool(true)),
            ("ignore_merges", Json::Bool(false)),
            ("dropout", Json::Null),
        ]),
        Class::Qwen2 => Some(vec![
            ("unk_token", Json::Null),
            ("continuing_subword_prefix", Json::from("")),
            ("end_of_word_suffix", Json::from("")),
            ("fuse_unk", Json::Bool(false)),
            ("byte_fallback", Json::Bool(false)),
            ("ignore_merges", Json::Bool(false)),
            ("dropout", Json::Null),
        ]),
        Class::Gpt2 => Some(vec![
            ("unk_token", Json::Null),
            ("continuing_subword_prefix", Json::from("")),
            ("end_of_word_suffix", Json::from("")),
            ("fuse_unk", Json::Bool(false)),
            ("byte_fallback", Json::Bool(false)),
            ("ignore_merges", Json::Bool(false)),
            ("dropout", Json::Null),
        ]),
        Class::Generic | Class::Other => None,
    };
    if let Some(flags) = model_flags {
        let model = top.get("model").ok_or("tokenizer.json has no model")?;
        let mut model: BTreeMap<String, Box<RawValue>> =
            serde_json::from_str(model.get()).map_err(|e| e.to_string())?;
        if model.get("type").map(|t| t.get()) != Some("\"BPE\"") {
            return Err(format!("{class:?} tokenizers must have a BPE model"));
        }
        for (key, value) in flags {
            model.insert(key.to_owned(), raw(value)?);
        }
        let model = serde_json::to_string(&model).map_err(|e| e.to_string())?;
        top.insert("model".to_owned(), raw_str(model)?);
    }
    match class {
        Class::Llama => {
            let add_prefix_space = flag("add_prefix_space").unwrap_or(true);
            let legacy = flag("legacy").unwrap_or(false);
            let scheme = match (add_prefix_space, legacy) {
                (false, _) => "never",
                (true, true) => "always",
                (true, false) => "first",
            };
            top.insert("normalizer".to_owned(), raw(Json::Null)?);
            top.insert(
                "pre_tokenizer".to_owned(),
                raw(serde_json::json!({
                    "type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": scheme, "split": false
                }))?,
            );
        }
        Class::Qwen2 => {
            top.insert(
                "normalizer".to_owned(),
                raw(serde_json::json!({"type": "NFC"}))?,
            );
            top.insert(
                "pre_tokenizer".to_owned(),
                raw(serde_json::json!({"type": "Sequence", "pretokenizers": [
                    {"type": "Split", "pattern": {"Regex": QWEN2_REGEX}, "behavior": "Isolated", "invert": false},
                    {"type": "ByteLevel", "add_prefix_space": flag("add_prefix_space").unwrap_or(false), "trim_offsets": true, "use_regex": false},
                ]}))?,
            );
        }
        Class::Gpt2 => {
            top.insert("normalizer".to_owned(), raw(Json::Null)?);
            top.insert(
                "pre_tokenizer".to_owned(),
                raw(serde_json::json!({
                    "type": "ByteLevel", "add_prefix_space": flag("add_prefix_space").unwrap_or(false), "trim_offsets": true, "use_regex": true
                }))?,
            );
        }
        Class::Generic | Class::Other => {}
    }
    if top.get("post_processor").is_none_or(|p| p.get() == "null") {
        top.insert(
            "post_processor".to_owned(),
            raw(serde_json::json!({
                "type": "TemplateProcessing",
                "single": [{"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
                "special_tokens": {},
            }))?,
        );
    }
    serde_json::to_string(&top).map_err(|e| e.to_string())
}

fn raw_str(text: String) -> Result<Box<RawValue>, String> {
    RawValue::from_string(text).map_err(|e| e.to_string())
}

/// `special_tokens_map`: the named special tokens a template sees.
fn special_tokens(
    dir: &Path,
    config: &Json,
    class: Class,
) -> Result<Vec<(&'static str, String)>, TokenizeError> {
    let mut values: BTreeMap<&'static str, Option<String>> = BTreeMap::new();
    for name in SPECIAL_ATTRIBUTES {
        if let Some(token) = config.get(name).and_then(token_text) {
            values.insert(name, token);
        }
    }
    if config.get("added_tokens_decoder").is_none()
        && let Some(map) = read_json(&dir.join("special_tokens_map.json"))?
    {
        for name in SPECIAL_ATTRIBUTES {
            if let Some(token) = map.get(name).and_then(token_text) {
                values.insert(name, token);
            }
        }
    }
    for (name, default) in class.defaults() {
        values
            .entry(name)
            .or_insert_with(|| Some((*default).to_owned()));
    }
    Ok(SPECIAL_ATTRIBUTES
        .iter()
        .filter_map(|name| values.get(name).cloned().flatten().map(|t| (*name, t)))
        .collect())
}

/// Tokens transformers adds after building the backend: entries of
/// `added_tokens_decoder` the file lacks, and named special tokens.
fn add_missing_tokens(
    tokenizer: &mut Tokenizer,
    config: &Json,
    special: &[(&'static str, String)],
) -> Result<(), TokenizeError> {
    let present = |tokenizer: &Tokenizer, content: &str| {
        tokenizer
            .get_added_tokens_decoder()
            .values()
            .any(|t| t.content == content)
    };
    let mut decoded: Vec<(u64, tokenizers::AddedToken)> = Vec::new();
    if let Some(entries) = config.get("added_tokens_decoder").and_then(Json::as_object) {
        for (id, entry) in entries {
            let (Ok(id), Some(content)) = (
                id.parse::<u64>(),
                entry.get("content").and_then(Json::as_str),
            ) else {
                continue;
            };
            let flag =
                |k: &str, default: bool| entry.get(k).and_then(Json::as_bool).unwrap_or(default);
            let special = flag("special", false);
            let token = tokenizers::AddedToken::from(content.to_owned(), special)
                .lstrip(flag("lstrip", false))
                .rstrip(flag("rstrip", false))
                .single_word(flag("single_word", false))
                .normalized(flag("normalized", !special));
            decoded.push((id, token));
        }
    }
    decoded.sort_by_key(|(id, _)| *id);
    let missing: Vec<tokenizers::AddedToken> = decoded
        .into_iter()
        .map(|(_, t)| t)
        .filter(|t| !present(tokenizer, &t.content))
        .collect();
    let failed = |e: tokenizers::Error| TokenizeError::Tokenizer(e.to_string());
    if !missing.is_empty() {
        tokenizer.add_tokens(missing).map_err(failed)?;
    }
    let named: Vec<tokenizers::AddedToken> = special
        .iter()
        .filter(|(_, content)| {
            !present(tokenizer, content) && tokenizer.token_to_id(content).is_none()
        })
        .map(|(_, content)| tokenizers::AddedToken::from(content.clone(), true))
        .collect();
    if !named.is_empty() {
        tokenizer.add_special_tokens(named).map_err(failed)?;
    }
    Ok(())
}

fn templates(
    dir: &Path,
    config: &Json,
    template_override: Option<&Path>,
    clock: &Clock,
) -> Result<Templates, TokenizeError> {
    let compile = |source: &str| ChatTemplate::compile(source, clock);
    if let Some(path) = template_override {
        return Ok(Templates {
            single: Some(compile(&read(path)?)?),
            named: Vec::new(),
        });
    }
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    let main = dir.join("chat_template.jinja");
    if main.is_file() {
        files.push(("default".to_owned(), main));
    }
    if let Ok(entries) = std::fs::read_dir(dir.join("additional_chat_templates")) {
        let mut extra: Vec<(String, PathBuf)> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter_map(|p| {
                let name = p.file_name()?.to_str()?.strip_suffix(".jinja")?.to_owned();
                Some((name, p))
            })
            .collect();
        extra.sort();
        files.extend(extra);
    }
    if !files.is_empty() {
        if let [(name, path)] = files.as_slice()
            && name == "default"
        {
            return Ok(Templates {
                single: Some(compile(&read(path)?)?),
                named: Vec::new(),
            });
        }
        let mut named = Vec::with_capacity(files.len());
        for (name, path) in files {
            named.push((name, compile(&read(&path)?)?));
        }
        return Ok(Templates {
            single: None,
            named,
        });
    }
    match config.get("chat_template") {
        Some(Json::String(source)) => Ok(Templates {
            single: Some(compile(source)?),
            named: Vec::new(),
        }),
        Some(Json::Array(list)) => {
            let mut named = Vec::with_capacity(list.len());
            for entry in list {
                let (Some(name), Some(source)) = (
                    entry.get("name").and_then(Json::as_str),
                    entry.get("template").and_then(Json::as_str),
                ) else {
                    continue;
                };
                named.push((name.to_owned(), compile(source)?));
            }
            Ok(Templates {
                single: None,
                named,
            })
        }
        _ => Ok(Templates::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word_level(dir: &Path, config: Json) {
        std::fs::create_dir_all(dir).unwrap();
        let tokenizer = serde_json::json!({
            "version": "1.0",
            "truncation": {"direction": "Right", "max_length": 2, "strategy": "LongestFirst", "stride": 0},
            "padding": null,
            "added_tokens": [{"id": 0, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"<s>": 0, "<unk>": 1, "hello": 2, "world": 3, "again": 4}, "unk_token": "<unk>"},
        });
        std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("inferrail-hf-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn generic_tokenizers_load_as_written_without_truncation() {
        let dir = scratch("generic");
        word_level(
            &dir,
            serde_json::json!({
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": {"content": "<s>"},
                "chat_template": "{{ bos_token }}{% for m in messages %}{{ m.content }} {% endfor %}",
                "added_tokens_decoder": {"5": {"content": "<new>", "special": true}},
            }),
        );
        let files = load(&dir, None, &Clock::default()).unwrap();
        assert_eq!(files.parity, Parity::Verified);
        assert_eq!(files.special, [("bos_token", "<s>".to_owned())]);
        let ids = files
            .tokenizer
            .encode_fast("hello world again <new>", false)
            .unwrap();
        assert_eq!(ids.get_ids(), [2, 3, 4, 5]);
        assert!(files.templates.select(false).is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn template_files_take_priority_and_named_templates_are_chosen_by_tools() {
        let dir = scratch("named");
        word_level(
            &dir,
            serde_json::json!({
                "tokenizer_class": "SomethingNew",
                "chat_template": [{"name": "default", "template": "a"}, {"name": "tool_use", "template": "b"}],
            }),
        );
        let files = load(&dir, None, &Clock::default()).unwrap();
        assert_eq!(files.parity, Parity::Unverified);
        let pick = |tools| {
            files
                .templates
                .select(tools)
                .unwrap()
                .render(minijinja::Value::from(()))
                .unwrap()
        };
        assert_eq!((pick(false), pick(true)), ("a".to_owned(), "b".to_owned()));
        std::fs::write(dir.join("chat_template.jinja"), "file").unwrap();
        let files = load(&dir, None, &Clock::default()).unwrap();
        assert_eq!(
            files
                .templates
                .select(true)
                .unwrap()
                .render(minijinja::Value::from(()))
                .unwrap(),
            "file"
        );
        let custom = dir.join("custom.jinja");
        std::fs::write(&custom, "override").unwrap();
        let files = load(&dir, Some(&custom), &Clock::default()).unwrap();
        assert_eq!(
            files
                .templates
                .select(false)
                .unwrap()
                .render(minijinja::Value::from(()))
                .unwrap(),
            "override"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_or_broken_files_are_reported() {
        let dir = scratch("broken");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(matches!(
            load(&dir, None, &Clock::default()),
            Err(TokenizeError::NotFound(_))
        ));
        std::fs::write(dir.join("tokenizer.json"), "{not json").unwrap();
        assert!(matches!(
            load(&dir, None, &Clock::default()),
            Err(TokenizeError::Parse { .. })
        ));
        word_level(
            &dir,
            serde_json::json!({"tokenizer_class": "LlamaTokenizer"}),
        );
        let err = load(&dir, None, &Clock::default()).unwrap_err();
        assert!(err.to_string().contains("BPE"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn class_defaults_fill_missing_special_tokens() {
        let dir = scratch("defaults");
        word_level(
            &dir,
            serde_json::json!({"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": null}),
        );
        std::fs::write(
            dir.join("special_tokens_map.json"),
            r#"{"eos_token": "</s>", "pad_token": {"content": "<pad>"}}"#,
        )
        .unwrap();
        let files = load(&dir, None, &Clock::default()).unwrap();
        assert_eq!(
            files.special,
            [
                ("eos_token", "</s>".to_owned()),
                ("pad_token", "<pad>".to_owned())
            ]
        );
        assert!(files.templates.is_empty());
        assert_eq!(Class::Llama.defaults()[1], ("bos_token", "<s>"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
