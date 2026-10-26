use std::path::PathBuf;

use inferrail_core::openai::{ChatCompletionRequest, CompletionRequest, PromptRef};
use minijinja::Value;

use crate::chat::{self, Chat};
use crate::hf::{self, HfFiles, Parity};
use crate::source::{self, Source};
use crate::template::{Clock, ContentFormat};
use crate::{TokenizeError, sim};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    Chat,
    Completion,
}

/// A request's prompt as the engine will see it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoded {
    pub tokens: Vec<u32>,
    /// vLLM's per request `cache_salt`, which keeps prefixes apart.
    pub cache_salt: Option<String>,
}

/// How a model's tokenizer is found and configured, mirroring the vLLM
/// server flags that change what a request tokenizes to.
#[derive(Debug, Clone, Default)]
pub struct TokenizerOptions {
    /// A model directory, a cached Hugging Face repository id
    /// (`org/name` or `org/name@revision`), or `builtin:sim`.
    pub source: String,
    /// `--chat-template`: a template file used instead of the model's.
    pub chat_template: Option<PathBuf>,
    /// `--chat-template-content-format`; `None` detects it as vLLM does.
    pub content_format: Option<ContentFormat>,
    /// `--default-chat-template-kwargs`.
    pub template_kwargs: Vec<(String, serde_json::Value)>,
    pub clock: Clock,
}

#[derive(Debug)]
pub struct ModelTokenizer {
    backend: Backend,
}

#[derive(Debug)]
enum Backend {
    Hf(Box<HfModel>),
    Sim,
}

#[derive(Debug)]
struct HfModel {
    files: HfFiles,
    format: Option<ContentFormat>,
    defaults: Vec<(String, Value)>,
}

impl ModelTokenizer {
    pub fn load(options: &TokenizerOptions) -> Result<Self, TokenizeError> {
        let backend = match source::resolve(&options.source)? {
            Source::Sim => Backend::Sim,
            Source::Dir(dir) => {
                let files = hf::load(&dir, options.chat_template.as_deref(), &options.clock)?;
                let defaults = options
                    .template_kwargs
                    .iter()
                    .map(|(k, v)| {
                        serde_json::from_value::<Value>(v.clone())
                            .map(|v| (k.clone(), v))
                            .map_err(|e| {
                                TokenizeError::Invalid(format!("chat template keyword {k}: {e}"))
                            })
                    })
                    .collect::<Result<_, _>>()?;
                Backend::Hf(Box::new(HfModel {
                    files,
                    format: options.content_format,
                    defaults,
                }))
            }
        };
        Ok(Self { backend })
    }

    /// The simulator's tokenizer, which needs no files.
    pub fn sim() -> Self {
        Self {
            backend: Backend::Sim,
        }
    }

    pub fn parity(&self) -> Parity {
        match &self.backend {
            Backend::Hf(model) => model.files.parity,
            Backend::Sim => Parity::Verified,
        }
    }

    /// Whether chat requests can be tokenized: the model has a template.
    pub fn has_chat_template(&self) -> bool {
        match &self.backend {
            Backend::Hf(model) => !model.files.templates.is_empty(),
            Backend::Sim => true,
        }
    }

    /// The token ids the engine will compute for a request body.
    pub fn encode(&self, endpoint: Endpoint, body: &[u8]) -> Result<Encoded, TokenizeError> {
        match (&self.backend, endpoint) {
            (Backend::Hf(model), Endpoint::Chat) => model.encode_chat(body),
            (Backend::Hf(model), Endpoint::Completion) => model.encode_completion(body),
            (Backend::Sim, Endpoint::Chat) => sim_chat(body),
            (Backend::Sim, Endpoint::Completion) => sim_completion(body),
        }
    }

    /// The prompt text a chat request renders to, for inspection.
    pub fn render_chat(&self, body: &[u8]) -> Result<String, TokenizeError> {
        match &self.backend {
            Backend::Hf(model) => model.render(&Chat::parse(body, &model.defaults)?),
            Backend::Sim => Err(TokenizeError::Unsupported(
                "rendering the simulator template",
            )),
        }
    }
}

impl HfModel {
    fn render(&self, chat: &Chat) -> Result<String, TokenizeError> {
        let template = self.files.templates.select(chat.has_tools())?;
        chat.render(template, self.format, &self.files.special)
    }

    fn tokens(&self, text: &str, special: bool) -> Result<Vec<u32>, TokenizeError> {
        self.files
            .tokenizer
            .encode_fast(text, special)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| TokenizeError::Tokenizer(e.to_string()))
    }

    fn encode_chat(&self, body: &[u8]) -> Result<Encoded, TokenizeError> {
        let chat = Chat::parse(body, &self.defaults)?;
        let text = self.render(&chat)?;
        Ok(Encoded {
            tokens: self.tokens(&text, chat.add_special_tokens)?,
            cache_salt: chat.cache_salt,
        })
    }

    fn encode_completion(&self, body: &[u8]) -> Result<Encoded, TokenizeError> {
        let request = completion(chat::strip_bom(body))?;
        let special = request
            .extra
            .get("add_special_tokens")
            .map(|v| {
                v.as_bool().ok_or_else(|| {
                    TokenizeError::Invalid("add_special_tokens must be a boolean".into())
                })
            })
            .transpose()?
            .unwrap_or(true);
        let tokens = match single_prompt(&request)? {
            PromptRef::Text(text) => self.tokens(text, special)?,
            PromptRef::Tokens(tokens) => tokens.to_vec(),
        };
        Ok(Encoded {
            tokens,
            cache_salt: cache_salt(&request.extra)?,
        })
    }
}

fn completion(body: &[u8]) -> Result<CompletionRequest, TokenizeError> {
    serde_json::from_slice(body)
        .map_err(|e| TokenizeError::Invalid(format!("cannot read the request: {e}")))
}

fn single_prompt(request: &CompletionRequest) -> Result<PromptRef<'_>, TokenizeError> {
    for unsupported in ["prompt_embeds", "truncate_prompt_tokens"] {
        if request.extra.get(unsupported).is_some_and(|v| !v.is_null()) {
            return Err(TokenizeError::Unsupported(
                "prompt embeddings or truncation",
            ));
        }
    }
    let prompt = request
        .prompt
        .value()
        .ok_or_else(|| TokenizeError::Invalid("a prompt is required".into()))?;
    prompt
        .single()
        .ok_or(TokenizeError::Unsupported("a batch of prompts"))
}

fn cache_salt(
    extra: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<String>, TokenizeError> {
    match extra.get("cache_salt") {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(salt)) => Ok(Some(salt.clone())),
        Some(_) => Err(TokenizeError::Invalid("cache_salt must be a string".into())),
    }
}

fn sim_chat(body: &[u8]) -> Result<Encoded, TokenizeError> {
    let request: ChatCompletionRequest = serde_json::from_slice(body)
        .map_err(|e| TokenizeError::Invalid(format!("cannot read the request: {e}")))?;
    let add_generation_prompt = request
        .extra
        .get("add_generation_prompt")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    Ok(Encoded {
        tokens: sim::render_chat(
            &request.messages,
            request.tools.value().map(Vec::as_slice),
            add_generation_prompt,
        ),
        cache_salt: None,
    })
}

fn sim_completion(body: &[u8]) -> Result<Encoded, TokenizeError> {
    let request = completion(body)?;
    let tokens = match single_prompt(&request)? {
        PromptRef::Text(text) => sim::encode_prompt(text),
        PromptRef::Tokens(tokens) => tokens.to_vec(),
    };
    Ok(Encoded {
        tokens,
        cache_salt: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sim_backend_tokenizes_like_the_simulator() {
        let tokenizer = ModelTokenizer::sim();
        let body = br#"{"model": "m", "messages": [{"role": "user", "content": "hello there"}]}"#;
        let chat = tokenizer.encode(Endpoint::Chat, body).unwrap();
        let request: ChatCompletionRequest = serde_json::from_slice(body).unwrap();
        assert_eq!(chat.tokens, sim::render_chat(&request.messages, None, true));
        let no_prompt =
            br#"{"messages": [{"role": "user", "content": "x"}], "add_generation_prompt": false}"#;
        assert!(
            tokenizer
                .encode(Endpoint::Chat, no_prompt)
                .unwrap()
                .tokens
                .len()
                < chat.tokens.len()
        );
        let text = tokenizer
            .encode(Endpoint::Completion, br#"{"prompt": "hello there"}"#)
            .unwrap();
        assert_eq!(text.tokens, sim::encode_prompt("hello there"));
        let ids = tokenizer
            .encode(Endpoint::Completion, br#"{"prompt": [5, 6, 7]}"#)
            .unwrap();
        assert_eq!(ids.tokens, [5, 6, 7]);
        assert!(matches!(
            tokenizer.encode(Endpoint::Completion, br#"{"prompt": ["a", "b"]}"#),
            Err(TokenizeError::Unsupported(_))
        ));
        assert!(matches!(
            tokenizer.encode(Endpoint::Chat, b"{"),
            Err(TokenizeError::Invalid(_))
        ));
        assert!(tokenizer.has_chat_template());
        assert_eq!(tokenizer.parity(), Parity::Verified);
    }

    #[test]
    fn missing_sources_fail_to_load() {
        let options = TokenizerOptions {
            source: "/definitely/not/a/model".to_owned(),
            ..TokenizerOptions::default()
        };
        assert!(matches!(
            ModelTokenizer::load(&options),
            Err(TokenizeError::NotFound(_))
        ));
    }
}
