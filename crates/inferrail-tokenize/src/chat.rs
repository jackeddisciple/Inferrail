//! Chat requests turned into template inputs the way vLLM v0.31.0 turns
//! them: messages rebuilt from the fields vLLM reads, text parts joined or
//! wrapped, tool call arguments parsed into objects, tools dumped from their
//! schema, and template keywords merged and filtered.

use std::sync::Arc;

use minijinja::value::{Enumerator, Object, ObjectRepr, Value, ValueKind};
use serde::Deserialize;

use crate::TokenizeError;
use crate::template::{ChatTemplate, ContentFormat};

/// A map that keeps insertion order, as Python dicts do.
#[derive(Debug, Clone, Default)]
pub(crate) struct Dict(Vec<(Value, Value)>);

impl Dict {
    pub(crate) fn insert(&mut self, key: &str, value: Value) {
        match self.0.iter_mut().find(|(k, _)| k.as_str() == Some(key)) {
            Some(slot) => slot.1 = value,
            None => self.0.push((Value::from(key), value)),
        }
    }

    fn get(&self, key: &str) -> Option<&Value> {
        self.0
            .iter()
            .find(|(k, _)| k.as_str() == Some(key))
            .map(|(_, v)| v)
    }

    fn remove(&mut self, key: &str) -> Option<Value> {
        let at = self.0.iter().position(|(k, _)| k.as_str() == Some(key))?;
        Some(self.0.remove(at).1)
    }

    fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub(crate) fn into_value(self) -> Value {
        Value::from_object(self)
    }

    fn from_map(value: &Value) -> Self {
        let mut out = Self::default();
        if let Ok(keys) = value.try_iter() {
            for key in keys {
                let item = value.get_item(&key).unwrap_or_default();
                out.0.push((key, item));
            }
        }
        out
    }
}

impl Object for Dict {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Map
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        Enumerator::Values(self.0.iter().map(|(k, _)| k.clone()).collect())
    }

    fn enumerator_len(self: &Arc<Self>) -> Option<usize> {
        Some(self.0.len())
    }
}

#[derive(Deserialize)]
struct Body {
    messages: Vec<Value>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    add_generation_prompt: Option<bool>,
    #[serde(default)]
    continue_final_message: Option<bool>,
    #[serde(default)]
    add_special_tokens: Option<bool>,
    #[serde(default)]
    documents: Option<Value>,
    #[serde(default)]
    chat_template: Option<Value>,
    #[serde(default)]
    chat_template_kwargs: Option<Value>,
    #[serde(default)]
    reasoning_effort: Option<Value>,
    #[serde(default)]
    cache_salt: Option<String>,
    #[serde(default)]
    truncate_prompt_tokens: Option<Value>,
}

const REASONING_EFFORTS: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// The parameters of transformers' `apply_chat_template`: vLLM always
/// accepts them as keywords, and they never reach the template itself.
const NAMED_PARAMS: [&str; 14] = [
    "conversation",
    "tools",
    "documents",
    "chat_template",
    "add_generation_prompt",
    "continue_final_message",
    "tokenize",
    "padding",
    "truncation",
    "max_length",
    "return_tensors",
    "return_dict",
    "return_assistant_tokens_mask",
    "tokenizer_kwargs",
];

const CONTINUE_TAG: &str = "CONTINUE_FINAL_MESSAGE_TAG ";

const KNOWN_PART_FIELDS: [&str; 19] = [
    "text",
    "type",
    "image_url",
    "input_audio",
    "file",
    "refusal",
    "audio_url",
    "video_url",
    "image_pil",
    "image_embeds",
    "audio_embeds",
    "video_embeds",
    "data",
    "uuid",
    "thinking",
    "closed",
    "name",
    "detail",
    "prompt_cache_breakpoint",
];

/// A chat request with everything that does not depend on the template.
#[derive(Debug)]
pub(crate) struct Chat {
    messages: Vec<Value>,
    tools: Option<Value>,
    kwargs: Dict,
    pub(crate) add_special_tokens: bool,
    pub(crate) cache_salt: Option<String>,
}

fn invalid(message: impl Into<String>) -> TokenizeError {
    TokenizeError::Invalid(message.into())
}

fn field(map: &Value, key: &str) -> Option<Value> {
    map.get_attr(key).ok().filter(|v| !v.is_undefined())
}

fn truthy(value: &Value) -> bool {
    value.is_true()
}

pub(crate) fn strip_bom(body: &[u8]) -> &[u8] {
    body.strip_prefix(b"\xef\xbb\xbf").unwrap_or(body)
}

impl Chat {
    pub(crate) fn parse(body: &[u8], defaults: &[(String, Value)]) -> Result<Self, TokenizeError> {
        let body: Body = serde_json::from_slice(strip_bom(body))
            .map_err(|e| invalid(format!("cannot read the request: {e}")))?;
        if body.chat_template.is_some() {
            return Err(TokenizeError::Unsupported("a chat template in the request"));
        }
        if body.truncate_prompt_tokens.is_some() {
            return Err(TokenizeError::Unsupported("truncate_prompt_tokens"));
        }
        let add_generation_prompt = body.add_generation_prompt.unwrap_or(true);
        let continue_final = body.continue_final_message.unwrap_or(false);
        if continue_final && add_generation_prompt {
            return Err(invalid(
                "continue_final_message needs add_generation_prompt to be false",
            ));
        }
        if body.messages.is_empty() {
            return Err(invalid("messages must not be empty"));
        }
        let tools = match body.tools {
            None => None,
            Some(tools) => Some(
                tools
                    .iter()
                    .map(dump_tool)
                    .collect::<Result<Vec<Value>, _>>()?
                    .into_iter()
                    .collect::<Value>(),
            ),
        };
        let reasoning = match body.reasoning_effort.filter(|v| !v.is_none()) {
            None => None,
            Some(v) => match v.as_str() {
                Some(effort) if REASONING_EFFORTS.contains(&effort) => Some(v),
                _ => return Err(invalid("unknown reasoning_effort")),
            },
        };
        let user = match body.chat_template_kwargs.filter(|v| !v.is_none()) {
            None => Dict::default(),
            Some(v) if v.kind() == ValueKind::Map => Dict::from_map(&v),
            Some(_) => return Err(invalid("chat_template_kwargs must be an object")),
        };
        let mut extra = Dict::default();
        extra.insert("add_generation_prompt", Value::from(add_generation_prompt));
        extra.insert("continue_final_message", Value::from(continue_final));
        extra.insert("documents", body.documents.unwrap_or_default_none());
        extra.insert(
            "reasoning_effort",
            reasoning.clone().unwrap_or_default_none(),
        );
        if let Some(effort) = &reasoning
            && !user.contains("enable_thinking")
        {
            extra.insert(
                "enable_thinking",
                Value::from(effort.as_str() != Some("none")),
            );
        }
        let mut request = user;
        merge(&mut request, &extra);
        let mut kwargs = Dict::default();
        for (key, value) in defaults {
            kwargs.insert(key, value.clone());
        }
        if let Some(tools) = &tools {
            kwargs.insert("tools", tools.clone());
        }
        kwargs.insert("tokenize", Value::from(false));
        merge(&mut kwargs, &request);
        kwargs.insert("return_dict", Value::from(false));
        if kwargs.get("tokenize").is_some_and(truthy) {
            return Err(TokenizeError::Unsupported(
                "tokenize in chat_template_kwargs",
            ));
        }
        let tools = kwargs.remove("tools").filter(|v| !v.is_none());
        kwargs.remove("tokenize");
        kwargs.remove("chat_template");
        Ok(Self {
            messages: body.messages,
            tools,
            kwargs,
            add_special_tokens: body.add_special_tokens.unwrap_or(false),
            cache_salt: body.cache_salt,
        })
    }

    pub(crate) fn has_tools(&self) -> bool {
        self.tools.is_some()
    }

    /// Renders the prompt text, as `apply_chat_template(tokenize=False)`.
    pub(crate) fn render(
        &self,
        template: &ChatTemplate,
        format: Option<ContentFormat>,
        special: &[(&'static str, String)],
    ) -> Result<String, TokenizeError> {
        let format = format.unwrap_or_else(|| template.format());
        let mut conversation = self
            .messages
            .iter()
            .map(|m| convert(m, format))
            .collect::<Result<Vec<Dict>, _>>()?;
        postprocess(&mut conversation)?;
        if !template.mentions_developer()
            && conversation
                .iter()
                .any(|m| m.get("role").and_then(Value::as_str) == Some("developer"))
        {
            conversation = consolidate(
                conversation
                    .into_iter()
                    .map(|mut m| {
                        if m.get("role").and_then(Value::as_str) == Some("developer") {
                            m.insert("role", Value::from("system"));
                            m.remove("tools");
                        }
                        m
                    })
                    .collect(),
            );
        }
        let resolved: Vec<(&Value, &Value)> = self
            .kwargs
            .0
            .iter()
            .filter(|(k, _)| {
                k.as_str().is_some_and(|k| {
                    k != "chat_template"
                        && k != "tokenize"
                        && (template.uses(k) || k == "self" || NAMED_PARAMS.contains(&k))
                })
            })
            .map(|(k, v)| (k, v))
            .collect();
        let named = |name: &str| {
            resolved
                .iter()
                .find(|(k, _)| k.as_str() == Some(name))
                .map(|(_, v)| (*v).clone())
        };
        if named("conversation").is_some() || named("self").is_some() {
            return Err(invalid("chat_template_kwargs repeats an argument"));
        }
        let continue_final = named("continue_final_message").is_some_and(|v| truthy(&v));
        let add_generation_prompt = named("add_generation_prompt").unwrap_or(Value::from(false));
        if continue_final && truthy(&add_generation_prompt) {
            return Err(invalid(
                "continue_final_message and add_generation_prompt are not compatible",
            ));
        }
        let final_text = if continue_final {
            if !template.mentions_content() {
                return Err(invalid("the template has no content to continue"));
            }
            Some(tag_final_message(&mut conversation)?)
        } else {
            None
        };
        let mut context = Dict::default();
        for (name, token) in special {
            context.insert(name, Value::from(token.as_str()));
        }
        for (key, value) in &resolved {
            if let Some(key) = key.as_str()
                && !NAMED_PARAMS.contains(&key)
            {
                context.insert(key, (*value).clone());
            }
        }
        context.insert(
            "messages",
            conversation.into_iter().map(Dict::into_value).collect(),
        );
        context.insert("tools", self.tools.clone().unwrap_or_default_none());
        context.insert("documents", named("documents").unwrap_or_default_none());
        context.insert("add_generation_prompt", add_generation_prompt);
        let rendered = template
            .render(context.into_value())
            .map_err(|e| TokenizeError::Template(e.to_string()))?;
        match final_text {
            None => Ok(rendered),
            Some(text) => cut_at_tag(rendered, &text),
        }
    }
}

trait NoneDefault {
    fn unwrap_or_default_none(self) -> Value;
}

impl NoneDefault for Option<Value> {
    fn unwrap_or_default_none(self) -> Value {
        self.unwrap_or_else(|| Value::from(()))
    }
}

/// `merge_kwargs`: overrides win unless they are `None` or `"auto"`.
fn merge(into: &mut Dict, overrides: &Dict) {
    for (key, value) in &overrides.0 {
        if value.is_none() || value.as_str() == Some("auto") {
            continue;
        }
        if let Some(key) = key.as_str() {
            into.insert(key, value.clone());
        }
    }
}

/// `ChatCompletionToolsParam.model_dump()`: only the schema's fields, with
/// `description` and `parameters` always present.
fn dump_tool(tool: &Value) -> Result<Value, TokenizeError> {
    if tool.kind() != ValueKind::Map {
        return Err(invalid("each tool must be an object"));
    }
    if field(tool, "type").is_some_and(|t| t.as_str() != Some("function")) {
        return Err(invalid("tools must be functions"));
    }
    let function = field(tool, "function")
        .filter(|f| f.kind() == ValueKind::Map)
        .ok_or_else(|| invalid("each tool needs a function"))?;
    let name = field(&function, "name")
        .filter(|n| n.kind() == ValueKind::String)
        .ok_or_else(|| invalid("each tool function needs a name"))?;
    let defer = field(tool, "defer_loading").filter(|v| !v.is_none());
    let mut dumped = Dict::default();
    dumped.insert("name", name);
    let optional = |key: &str, kind: ValueKind| -> Result<Value, TokenizeError> {
        match field(&function, key).filter(|v| !v.is_none()) {
            None => Ok(Value::from(())),
            Some(v) if v.kind() == kind => Ok(v),
            Some(_) => Err(invalid(format!("tool function {key} has the wrong type"))),
        }
    };
    dumped.insert("description", optional("description", ValueKind::String)?);
    dumped.insert("parameters", optional("parameters", ValueKind::Map)?);
    if let Some(strict) = field(&function, "strict").filter(|v| !v.is_none()) {
        dumped.insert("strict", strict);
    }
    if let Some(own) = field(&function, "defer_loading")
        .filter(|v| !v.is_none())
        .or_else(|| defer.clone())
    {
        dumped.insert("defer_loading", own);
    }
    let mut out = Dict::default();
    out.insert("type", Value::from("function"));
    out.insert("function", dumped.into_value());
    if let Some(defer) = defer {
        out.insert("defer_loading", defer);
    }
    Ok(out.into_value())
}

fn reorder(value: &Value, first: &[&str]) -> Value {
    if value.kind() != ValueKind::Map {
        return value.clone();
    }
    let all = Dict::from_map(value);
    let mut out = Dict::default();
    for key in first {
        if let Some(v) = all.get(key) {
            out.insert(key, v.clone());
        }
    }
    for (key, v) in all.0 {
        if let Some(name) = key.as_str()
            && !out.contains(name)
        {
            out.insert(name, v);
        }
    }
    out.into_value()
}

/// Tool calls as pydantic validates them: the schema's fields first, in
/// schema order, then any extra fields as sent.
fn validated_calls(calls: &Value) -> Result<Value, TokenizeError> {
    if calls.kind() != ValueKind::Seq {
        return Err(invalid("tool_calls must be a list"));
    }
    Ok(calls
        .try_iter()
        .map_err(|e| invalid(e.to_string()))?
        .map(|call| {
            let call = reorder(&call, &["id", "function", "type"]);
            match field(&call, "function") {
                Some(function) if function.kind() == ValueKind::Map => {
                    let mut out = Dict::from_map(&call);
                    out.insert("function", reorder(&function, &["arguments", "name"]));
                    out.into_value()
                }
                _ => call,
            }
        })
        .collect())
}

fn text_of(part: &Value) -> Result<Option<Value>, TokenizeError> {
    if part.kind() == ValueKind::String {
        return Ok(Some(part.clone()));
    }
    if part.kind() != ValueKind::Map {
        return Err(invalid("content parts must be strings or objects"));
    }
    let kind = field(part, "type").ok_or_else(|| invalid("content part without a type"))?;
    let key = match kind.as_str() {
        Some("text" | "input_text" | "output_text") => "text",
        Some("refusal") => "refusal",
        Some("thinking") => "thinking",
        Some(_) => return Err(TokenizeError::Unsupported("non text content")),
        None => return Err(invalid("content part type must be a string")),
    };
    match field(part, key).filter(|v| !v.is_none()) {
        None => Ok(None),
        Some(text) if text.kind() == ValueKind::String => Ok(Some(text)),
        Some(_) => Err(invalid("content part text must be a string")),
    }
}

fn parse_content(content: Option<Value>, format: ContentFormat) -> Result<Value, TokenizeError> {
    let parts: Vec<Value> = match content.filter(|c| !c.is_none()) {
        None => Vec::new(),
        Some(c) if c.kind() == ValueKind::String => vec![c],
        Some(c) if c.kind() == ValueKind::Seq => {
            c.try_iter().map_err(|e| invalid(e.to_string()))?.collect()
        }
        Some(_) => return Err(invalid("message content must be a string or a list")),
    };
    match format {
        ContentFormat::String => {
            let mut texts = Vec::with_capacity(parts.len());
            for part in &parts {
                if let Some(text) = text_of(part)? {
                    texts.push(text);
                }
            }
            let joined = texts
                .iter()
                .map(|t| t.as_str().unwrap_or_default())
                .collect::<Vec<_>>()
                .join("\n");
            Ok(Value::from(joined))
        }
        ContentFormat::OpenAi => {
            let mut out = Vec::with_capacity(parts.len());
            for part in &parts {
                let Some(text) = text_of(part)? else { continue };
                let mut wrapped = Dict::default();
                wrapped.insert("type", Value::from("text"));
                wrapped.insert("text", text);
                if part.kind() == ValueKind::Map {
                    for (key, value) in Dict::from_map(part).0 {
                        if let Some(name) = key.as_str()
                            && !KNOWN_PART_FIELDS.contains(&name)
                        {
                            wrapped.insert(name, value);
                        }
                    }
                }
                out.push(wrapped.into_value());
            }
            Ok(Value::from(out))
        }
    }
}

fn convert(message: &Value, format: ContentFormat) -> Result<Dict, TokenizeError> {
    if message.kind() != ValueKind::Map {
        return Err(invalid("each message must be an object"));
    }
    let role = field(message, "role")
        .filter(|r| r.kind() == ValueKind::String)
        .ok_or_else(|| invalid("each message needs a role"))?;
    let role_name = role.as_str().unwrap_or_default().to_owned();
    let mut out = Dict::default();
    out.insert("role", role);
    out.insert("content", parse_content(field(message, "content"), format)?);
    match role_name.as_str() {
        "assistant" => {
            if let Some(calls) = field(message, "tool_calls").filter(|c| !c.is_none()) {
                out.insert("tool_calls", validated_calls(&calls)?);
            }
            if let Some(reasoning) = field(message, "reasoning").filter(|r| !r.is_none()) {
                out.insert("reasoning", reasoning.clone());
                out.insert("reasoning_content", reasoning);
            }
        }
        "tool" => {
            if let Some(id) = field(message, "tool_call_id") {
                out.insert("tool_call_id", id);
            }
            if let Some(content) = out.get("content").filter(|c| c.kind() == ValueKind::Seq) {
                let items: Vec<Value> = content
                    .try_iter()
                    .map_err(|e| invalid(e.to_string()))?
                    .collect();
                let all_text = items.iter().all(|i| {
                    i.kind() != ValueKind::Map
                        || field(i, "type")
                            .and_then(|t| t.as_str().map(str::to_owned))
                            .as_deref()
                            == Some("text")
                });
                if all_text {
                    let texts: Vec<String> = items
                        .iter()
                        .filter(|i| i.kind() == ValueKind::Map)
                        .map(|i| {
                            field(i, "text")
                                .map(|t| {
                                    let mut s = String::new();
                                    crate::python::str_of(&t, &mut s);
                                    s
                                })
                                .unwrap_or_default()
                        })
                        .collect();
                    out.insert("content", Value::from(texts.join("\n")));
                }
            }
        }
        _ => {}
    }
    for key in ["name", "task"] {
        if let Some(v) = field(message, key).filter(|v| v.kind() == ValueKind::String) {
            out.insert(key, v);
        }
    }
    if role_name == "developer" {
        out.insert("tools", field(message, "tools").unwrap_or_default_none());
    }
    Ok(out)
}

/// `_postprocess_messages`: arguments become objects, malformed ones `{}`.
fn postprocess(conversation: &mut [Dict]) -> Result<(), TokenizeError> {
    for message in conversation.iter_mut() {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(calls) = message.get("tool_calls").cloned() else {
            continue;
        };
        if calls.kind() != ValueKind::Seq {
            continue;
        }
        let calls: Vec<Value> = calls
            .try_iter()
            .map_err(|e| invalid(e.to_string()))?
            .collect();
        if calls.is_empty() {
            message.remove("tool_calls");
            continue;
        }
        let mut fixed = Vec::with_capacity(calls.len());
        for call in calls {
            if call.kind() != ValueKind::Map {
                return Err(invalid("assistant tool_calls entries must be objects"));
            }
            let kind = field(&call, "type").unwrap_or_else(|| Value::from("function"));
            let function = field(&call, "function").filter(|f| f.kind() == ValueKind::Map);
            let (Some(function), Some("function")) = (function, kind.as_str()) else {
                return Err(invalid("tool calls must be of type function"));
            };
            let arguments = match field(&function, "arguments").filter(truthy) {
                None => Value::from_object(Dict::default()),
                Some(args) if args.kind() == ValueKind::Map => args,
                Some(args) => match args.as_str() {
                    Some(text) => match serde_json::from_str::<Value>(text) {
                        Ok(parsed) if parsed.kind() == ValueKind::Map => parsed,
                        _ => Value::from_object(Dict::default()),
                    },
                    None => Value::from_object(Dict::default()),
                },
            };
            let mut function = Dict::from_map(&function);
            function.insert("arguments", arguments);
            let mut call = Dict::from_map(&call);
            call.insert("function", function.into_value());
            fixed.push(call.into_value());
        }
        message.insert("tool_calls", Value::from(fixed));
    }
    Ok(())
}

fn consolidate(conversation: Vec<Dict>) -> Vec<Dict> {
    let mut contents: Vec<String> = Vec::new();
    let mut rest = Vec::new();
    let mut needed = false;
    for (i, message) in conversation.iter().enumerate() {
        if message.get("role").and_then(Value::as_str) != Some("system") {
            rest.push(message.clone());
            continue;
        }
        if i > 0 || !contents.is_empty() {
            needed = true;
        }
        let content = match message.get("content") {
            Some(c) if c.kind() == ValueKind::Seq => c
                .try_iter()
                .map(|parts| {
                    parts
                        .filter_map(|p| match p.kind() {
                            ValueKind::String => Some(p.as_str().unwrap_or_default().to_owned()),
                            ValueKind::Map => field(&p, "text").map(|t| {
                                let mut s = String::new();
                                crate::python::str_of(&t, &mut s);
                                s
                            }),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default(),
            Some(c) => {
                let mut s = String::new();
                crate::python::str_of(c, &mut s);
                s
            }
            None => String::new(),
        };
        if !content.is_empty() {
            contents.push(content);
        }
    }
    if !needed {
        return conversation;
    }
    let mut merged = Dict::default();
    merged.insert("role", Value::from("system"));
    merged.insert("content", Value::from(contents.join("\n\n")));
    std::iter::once(merged).chain(rest).collect()
}

/// transformers marks the end of the message to continue with a tag, renders
/// and cuts there, so the template's own end of turn tokens fall away.
fn tag_final_message(conversation: &mut [Dict]) -> Result<String, TokenizeError> {
    let last = conversation
        .last_mut()
        .ok_or_else(|| invalid("nothing to continue"))?;
    let content = last
        .get("content")
        .cloned()
        .filter(|c| !c.is_none())
        .ok_or_else(|| invalid("the final message has no content to continue"))?;
    match content.kind() {
        ValueKind::String => {
            let text = content.as_str().unwrap_or_default().to_owned();
            last.insert("content", Value::from(format!("{text}{CONTINUE_TAG}")));
            Ok(text)
        }
        ValueKind::Seq => {
            let mut blocks: Vec<Value> = content
                .try_iter()
                .map_err(|e| invalid(e.to_string()))?
                .collect();
            let at = blocks
                .iter()
                .rposition(|b| field(b, "text").is_some())
                .ok_or_else(|| invalid("no text to continue in the final message"))?;
            let mut block = Dict::from_map(&blocks[at]);
            let text = block
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            block.insert("text", Value::from(format!("{text}{CONTINUE_TAG}")));
            blocks[at] = block.into_value();
            last.insert("content", Value::from(blocks));
            Ok(text)
        }
        _ => Err(invalid("the final message content cannot be continued")),
    }
}

fn cut_at_tag(rendered: String, final_text: &str) -> Result<String, TokenizeError> {
    let tag = CONTINUE_TAG.trim();
    if !rendered.contains(final_text.trim()) || !rendered.contains(tag) {
        return Err(invalid("the template dropped the final message"));
    }
    let at = rendered.rfind(tag).unwrap_or_default();
    if rendered[at..].starts_with(CONTINUE_TAG) {
        Ok(rendered[..at].to_owned())
    } else {
        Ok(rendered[..at].trim_end().to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::Clock;

    fn chat(body: &str) -> Chat {
        Chat::parse(body.as_bytes(), &[]).unwrap()
    }

    fn render(source: &str, body: &str) -> String {
        let template = ChatTemplate::compile(source, &Clock::default()).unwrap();
        chat(body)
            .render(&template, None, &[("bos_token", "<s>".to_owned())])
            .unwrap()
    }

    const ECHO: &str = "{{ bos_token }}{% for m in messages %}{{ m|tojson }}\n{% endfor %}{{ tools|tojson }}|{{ documents }}|{{ add_generation_prompt }}";

    #[test]
    fn messages_are_rebuilt_from_the_fields_vllm_reads() {
        let out = render(
            ECHO,
            r#"{"messages": [
                {"content": ["a", {"type": "text", "text": "b"}], "role": "user", "extra": 1, "name": "ann"},
                {"role": "assistant", "content": null, "reasoning": "hm",
                 "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "{\"z\": 1, \"a\": [2]}"}, "id": "c1", "index": 0}]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}
            ]}"#,
        );
        assert_eq!(
            out,
            "<s>{\"role\": \"user\", \"content\": \"a\\nb\", \"name\": \"ann\"}\n\
             {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"id\": \"c1\", \"function\": {\"arguments\": {\"z\": 1, \"a\": [2]}, \"name\": \"f\"}, \"type\": \"function\", \"index\": 0}], \"reasoning\": \"hm\", \"reasoning_content\": \"hm\"}\n\
             {\"role\": \"tool\", \"content\": \"ok\", \"tool_call_id\": \"c1\"}\n\
             null|None|True"
        );
    }

    #[test]
    fn malformed_arguments_become_empty_objects() {
        for arguments in ["{not json", "[1]", "", "7"] {
            let out = render(
                "{% for c in messages[0].tool_calls %}{{ c.function.arguments|tojson }}{% endfor %}",
                &serde_json::json!({"messages": [{"role": "assistant", "tool_calls": [
                    {"id": "x", "type": "function", "function": {"name": "f", "arguments": arguments}}
                ]}]})
                .to_string(),
            );
            assert_eq!(out, "{}", "{arguments:?}");
        }
        let out = render(
            "{{ messages[0].tool_calls is defined }}",
            r#"{"messages": [{"role": "assistant", "content": "x", "tool_calls": []}]}"#,
        );
        assert_eq!(out, "False");
    }

    #[test]
    fn tools_are_dumped_with_their_schema_fields() {
        let out = render(
            "{{ tools|tojson }}",
            r#"{"messages": [{"role": "user", "content": "x"}], "tools": [
                {"type": "function", "x": 1, "function": {"name": "f", "parameters": {"b": 1, "a": 2}, "other": 3}},
                {"function": {"name": "g", "description": "d", "strict": true}, "defer_loading": true}
            ]}"#,
        );
        assert_eq!(
            out,
            r#"[{"type": "function", "function": {"name": "f", "description": null, "parameters": {"b": 1, "a": 2}}}, {"type": "function", "function": {"name": "g", "description": "d", "parameters": null, "strict": true, "defer_loading": true}, "defer_loading": true}]"#
        );
    }

    #[test]
    fn keywords_merge_and_filter_like_vllm() {
        let source =
            "{{ enable_thinking }}|{{ other }}|{{ unused is defined }}|{{ documents|tojson }}";
        let body = r#"{
            "messages": [{"role": "user", "content": "x"}],
            "chat_template_kwargs": {"other": "kept", "enable_thinking": null},
            "reasoning_effort": "none",
            "documents": [{"title": "t", "text": "x"}]
        }"#;
        assert_eq!(
            render(source, body),
            "|kept|False|[{\"title\": \"t\", \"text\": \"x\"}]"
        );
        let effort = r#"{"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "none",
            "chat_template_kwargs": {"unused": 1}}"#;
        assert_eq!(render(source, effort), "False||True|null");
        let auto = r#"{
            "messages": [{"role": "user", "content": "x"}],
            "chat_template_kwargs": {"other": "auto", "enable_thinking": true},
            "reasoning_effort": "high"
        }"#;
        assert_eq!(render(source, auto), "True||False|null");
        let filtered = ChatTemplate::compile("{{ hidden is defined }}", &Clock::default()).unwrap();
        let hidden = chat(
            r#"{"messages": [{"role": "user", "content": "x"}], "chat_template_kwargs": {"other": 1}}"#,
        );
        assert_eq!(hidden.render(&filtered, None, &[]).unwrap(), "False");
    }

    #[test]
    fn developer_messages_become_one_system_message() {
        let out = render(
            "{% for m in messages %}[{{ m.role }}:{{ m.content }}]{% endfor %}",
            r#"{"messages": [
                {"role": "system", "content": "s"},
                {"role": "developer", "content": "d", "tools": []},
                {"role": "user", "content": "u"}
            ]}"#,
        );
        assert_eq!(out, "[system:s\n\nd][user:u]");
        let kept = render(
            "{% for m in messages %}[{{ m.role }}]{% endfor %}{# 'developer' #}",
            r#"{"messages": [{"role": "developer", "content": "d"}]}"#,
        );
        assert_eq!(kept, "[developer]");
    }

    #[test]
    fn openai_format_wraps_text_parts() {
        let template = ChatTemplate::compile(
            "{% for m in messages %}{% for p in m.content %}{{ p|tojson }}{% endfor %}{% endfor %}",
            &Clock::default(),
        )
        .unwrap();
        assert_eq!(template.format(), ContentFormat::OpenAi);
        let out = chat(
            r#"{"messages": [
            {"role": "user", "content": ["plain", {"type": "text", "text": "t", "cache": 1}]}
        ]}"#,
        )
        .render(&template, None, &[])
        .unwrap();
        assert_eq!(
            out,
            r#"{"type": "text", "text": "plain"}{"type": "text", "text": "t", "cache": 1}"#
        );
    }

    #[test]
    fn continuing_the_final_message_cuts_after_it() {
        let source = "{% for m in messages %}<{{ m.role }}>{{ m.content }}</s>{% endfor %}";
        let body = r#"{
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "The answer is"}],
            "add_generation_prompt": false,
            "continue_final_message": true
        }"#;
        assert_eq!(render(source, body), "<user>q</s><assistant>The answer is");
        let trimmed = "{% for m in messages %}<{{ m.role }}>{{ m.content|trim }}</s>{% endfor %}";
        let body = r#"{
            "messages": [{"role": "assistant", "content": "  spaced  "}],
            "add_generation_prompt": false,
            "continue_final_message": true
        }"#;
        assert_eq!(render(trimmed, body), "<assistant>spaced");
    }

    #[test]
    fn requests_the_engine_rejects_or_we_cannot_follow_are_errors() {
        for (body, unsupported) in [
            (serde_json::json!({"messages": []}), false),
            (
                serde_json::json!({"messages": [{"role": "user", "content": "x"}], "continue_final_message": true}),
                false,
            ),
            (
                serde_json::json!({"messages": [{"role": "user", "content": "x"}], "chat_template": "{{ x }}"}),
                true,
            ),
            (
                serde_json::json!({"messages": [{"role": "user", "content": "x"}], "truncate_prompt_tokens": 5}),
                true,
            ),
            (
                serde_json::json!({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "turbo"}),
                false,
            ),
            (
                serde_json::json!({"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "custom"}]}),
                false,
            ),
            (serde_json::json!({"messages": "x"}), false),
        ] {
            match Chat::parse(body.to_string().as_bytes(), &[]) {
                Err(TokenizeError::Unsupported(_)) => assert!(unsupported, "{body}"),
                Err(TokenizeError::Invalid(_)) => assert!(!unsupported, "{body}"),
                other => panic!("{body}: {other:?}"),
            }
        }
        let template = ChatTemplate::compile("{{ messages }}", &Clock::default()).unwrap();
        for body in [
            r#"{"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}"#,
            r#"{"messages": [{"role": "assistant", "tool_calls": ["x"]}]}"#,
            r#"{"messages": [{"content": "x"}]}"#,
        ] {
            assert!(chat(body).render(&template, None, &[]).is_err(), "{body}");
        }
    }

    #[test]
    fn a_byte_order_mark_and_defaults_are_honored() {
        let mut body = b"\xef\xbb\xbf".to_vec();
        body.extend_from_slice(br#"{"messages": [{"role": "user", "content": "x"}], "chat_template_kwargs": {"b": 2}}"#);
        let defaults = vec![
            ("a".to_owned(), Value::from(1)),
            ("b".to_owned(), Value::from(1)),
        ];
        let template = ChatTemplate::compile("{{ a }}{{ b }}", &Clock::default()).unwrap();
        let parsed = Chat::parse(&body, &defaults).unwrap();
        assert_eq!(parsed.render(&template, None, &[]).unwrap(), "12");
    }
}
