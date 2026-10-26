//! Chat templates rendered the way transformers renders them: Jinja2 in a
//! sandbox with `trim_blocks` and `lstrip_blocks`, loop controls, Python
//! string methods, its own `tojson`, `raise_exception` and `strftime_now`.

use std::collections::HashSet;
use std::sync::Arc;

use minijinja::machinery::ast::{CallArg, Expr, Stmt};
use minijinja::syntax::SyntaxConfig;
use minijinja::value::{Kwargs, Rest, Value, ValueKind};
use minijinja::{AutoEscape, Environment, Error, ErrorKind, UndefinedBehavior};

use crate::python::{self, JsonStyle};
use crate::{TokenizeError, case};

const NAME: &str = "chat";
const FUEL: u64 = 50_000_000;

/// Whether message content reaches the template as one string or as a list
/// of typed parts, as vLLM's `--chat-template-content-format` decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentFormat {
    String,
    OpenAi,
}

/// Where `strftime_now` reads the time; fixed for reproducible tests.
#[derive(Debug, Clone, Default)]
pub enum Clock {
    #[default]
    Local,
    Fixed(jiff::civil::DateTime),
}

impl Clock {
    fn now(&self) -> jiff::civil::DateTime {
        match self {
            Self::Local => jiff::Zoned::now().datetime(),
            Self::Fixed(at) => *at,
        }
    }
}

#[derive(Debug)]
pub(crate) struct ChatTemplate {
    env: Environment<'static>,
    variables: HashSet<String>,
    format: ContentFormat,
    developer: bool,
    mentions_content: bool,
}

impl ChatTemplate {
    pub(crate) fn compile(source: &str, clock: &Clock) -> Result<Self, TokenizeError> {
        let source = without_generation_tags(source);
        let syntax = SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .map_err(|e| TokenizeError::Template(e.to_string()))?;
        let (format, variables) = match minijinja::machinery::parse(&source, NAME, syntax.clone()) {
            Ok(root) => (content_format(&root), crate::variables::undeclared(&root)),
            Err(_) => (ContentFormat::String, HashSet::new()),
        };
        let mut env = Environment::new();
        env.set_syntax(syntax);
        env.set_undefined_behavior(UndefinedBehavior::Chainable);
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.set_unknown_method_callback(|state, value, method, args| {
            match (value.as_str(), method, args) {
                (Some(text), "title", []) => Ok(Value::from(case::title(text))),
                (Some(text), "capitalize", []) => Ok(Value::from(case::capitalize(text))),
                _ => {
                    minijinja_contrib::pycompat::unknown_method_callback(state, value, method, args)
                }
            }
        });
        env.set_fuel(Some(FUEL));
        env.set_formatter(|out, _state, value| {
            if let Some(text) = value.as_str() {
                return out.write_str(text).map_err(Error::from);
            }
            let mut text = String::new();
            python::str_of(value, &mut text);
            out.write_str(&text).map_err(Error::from)
        });
        env.add_filter("tojson", tojson);
        env.add_filter("items", items);
        env.add_filter("length", length);
        env.add_filter("count", length);
        env.add_filter("title", |value: Value| case::title_filter(&text_of(&value)));
        env.add_filter("capitalize", |value: Value| {
            case::capitalize(&text_of(&value))
        });
        env.add_filter("string", |value: Value| text_of(&value));
        env.add_function(
            "raise_exception",
            |message: Value| -> Result<Value, Error> {
                let mut text = String::new();
                python::str_of(&message, &mut text);
                Err(Error::new(ErrorKind::InvalidOperation, text))
            },
        );
        let clock = Arc::new(clock.clone());
        env.add_function(
            "strftime_now",
            move |format: &str| -> Result<String, Error> {
                jiff::fmt::strtime::format(format, clock.now())
                    .map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))
            },
        );
        let developer = source.contains("\"developer\"") || source.contains("'developer'");
        let mentions_content = source.contains("content");
        env.add_template_owned(NAME, source)
            .map_err(|e| TokenizeError::Template(e.to_string()))?;
        Ok(Self {
            env,
            variables,
            format,
            developer,
            mentions_content,
        })
    }

    pub(crate) fn render(&self, context: Value) -> Result<String, Error> {
        self.env.get_template(NAME)?.render(context)
    }

    pub(crate) fn format(&self) -> ContentFormat {
        self.format
    }

    pub(crate) fn uses(&self, variable: &str) -> bool {
        self.variables.contains(variable)
    }

    pub(crate) fn mentions_developer(&self) -> bool {
        self.developer
    }

    pub(crate) fn mentions_content(&self) -> bool {
        self.mentions_content
    }
}

/// transformers' `{% generation %}` marks assistant spans for training masks
/// and renders its body unchanged; an always true `if` keeps the same
/// whitespace behavior.
fn without_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(start) = rest.find("{%") {
        out.push_str(&rest[..start]);
        let tag = &rest[start..];
        let Some(end) = tag.find("%}") else {
            out.push_str(tag);
            return out;
        };
        let inner = &tag[2..end];
        let (open, body, close) = trim_markers(inner);
        out.push_str("{%");
        out.push_str(open);
        match body {
            "generation" => out.push_str(" if true "),
            "endgeneration" => out.push_str(" endif "),
            _ => out.push_str(&inner[open.len()..inner.len() - close.len()]),
        }
        out.push_str(close);
        out.push_str("%}");
        rest = &tag[end + 2..];
    }
    out.push_str(rest);
    out
}

fn trim_markers(inner: &str) -> (&str, &str, &str) {
    let open = if inner.starts_with(['-', '+']) {
        &inner[..1]
    } else {
        ""
    };
    let close = if inner.len() > open.len() && inner.ends_with(['-', '+']) {
        &inner[inner.len() - 1..]
    } else {
        ""
    };
    (
        open,
        inner[open.len()..inner.len() - close.len()].trim(),
        close,
    )
}

fn text_of(value: &Value) -> String {
    let mut text = String::new();
    python::str_of(value, &mut text);
    text
}

/// Jinja2's `items`, which yields nothing for an undefined value.
fn items(value: Value) -> Result<Value, Error> {
    if value.is_undefined() {
        return Ok(Value::from(Vec::<Value>::new()));
    }
    if value.kind() != ValueKind::Map {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "Can only get item pairs from a mapping",
        ));
    }
    let mut pairs = Vec::new();
    for key in value.try_iter()? {
        let item = value.get_item(&key)?;
        pairs.push(Value::from(vec![key, item]));
    }
    Ok(Value::from(pairs))
}

/// Jinja2's `length`, which is zero for an undefined value.
fn length(value: Value) -> Result<usize, Error> {
    if value.is_undefined() {
        return Ok(0);
    }
    value.len().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidOperation,
            format!("object of type {} has no len()", value.kind()),
        )
    })
}

/// transformers' `tojson(x, ensure_ascii=False, indent=None,
/// separators=None, sort_keys=False)`, which is `json.dumps`.
fn tojson(value: Value, args: Rest<Value>, kwargs: Kwargs) -> Result<Value, Error> {
    let positional = |i: usize, name: &str| -> Result<Option<Value>, Error> {
        match (args.get(i), kwargs.get::<Option<Value>>(name)?) {
            (Some(_), Some(_)) => Err(Error::new(
                ErrorKind::TooManyArguments,
                format!("tojson got multiple values for {name}"),
            )),
            (Some(v), None) => Ok(Some(v.clone())),
            (None, v) => Ok(v),
        }
    };
    let ensure_ascii = positional(0, "ensure_ascii")?;
    let indent = positional(1, "indent")?;
    let separators = positional(2, "separators")?;
    let sort_keys = positional(3, "sort_keys")?;
    if args.len() > 4 {
        return Err(Error::from(ErrorKind::TooManyArguments));
    }
    kwargs.assert_all_used()?;
    let indent = match indent {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) if v.kind() == ValueKind::String => Some(v.as_str().unwrap_or_default().to_owned()),
        Some(v) => {
            let n = v
                .as_i64()
                .or_else(|| (v.kind() == ValueKind::Bool).then(|| i64::from(v.is_true())))
                .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "indent must be an int"))?;
            Some(" ".repeat(usize::try_from(n).unwrap_or(0)))
        }
    };
    let mut style = JsonStyle::new(indent);
    if let Some(separators) = separators.filter(|v| !v.is_none()) {
        let pair: Vec<Value> = separators.try_iter()?.collect();
        let [item, key] = pair.as_slice() else {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "separators must be a pair",
            ));
        };
        style.item_separator = item.as_str().unwrap_or_default().to_owned();
        style.key_separator = key.as_str().unwrap_or_default().to_owned();
    }
    style.sort_keys = sort_keys.is_some_and(|v| v.is_true());
    style.ensure_ascii = ensure_ascii.is_some_and(|v| v.is_true());
    python::json_dumps(&value, &style).map(Value::from)
}

/// vLLM's detection: the template takes OpenAI style content parts when it
/// loops over the content of a message it got by looping over `messages`
/// (or a variable assigned from it), over a macro parameter that receives
/// such content, or over a bare `content` outside macros. Anything the walk
/// cannot follow means plain strings, as vLLM falls back to.
fn content_format(root: &Stmt<'_>) -> ContentFormat {
    let mut walk = Walk::default();
    walk.stmt(root, None);
    walk.verdict().unwrap_or(ContentFormat::String)
}

#[derive(Default)]
struct Walk<'a> {
    sets: Vec<(&'a Expr<'a>, &'a Expr<'a>)>,
    loops: Vec<Loop<'a>>,
    macros: Vec<MacroDecl<'a>>,
    calls: Vec<(&'a Expr<'a>, &'a [CallArg<'a>])>,
}

struct Loop<'a> {
    target: &'a Expr<'a>,
    iter: &'a Expr<'a>,
    macros: Vec<usize>,
}

struct MacroDecl<'a> {
    name: &'a str,
    args: Vec<&'a str>,
}

impl<'a> Walk<'a> {
    fn stmt(&mut self, stmt: &'a Stmt<'a>, inside: Option<&[usize]>) {
        let inside = inside.unwrap_or(&[]);
        match stmt {
            Stmt::Template(t) => self.body(&t.children, inside),
            Stmt::EmitExpr(e) => self.expr(&e.expr),
            Stmt::EmitRaw(_) | Stmt::Continue(_) | Stmt::Break(_) => {}
            Stmt::ForLoop(f) => {
                self.loops.push(Loop {
                    target: &f.target,
                    iter: &f.iter,
                    macros: inside.to_vec(),
                });
                self.expr(&f.iter);
                if let Some(filter) = &f.filter_expr {
                    self.expr(filter);
                }
                self.body(&f.body, inside);
                self.body(&f.else_body, inside);
            }
            Stmt::IfCond(i) => {
                self.expr(&i.expr);
                self.body(&i.true_body, inside);
                self.body(&i.false_body, inside);
            }
            Stmt::WithBlock(w) => {
                for (_, value) in &w.assignments {
                    self.expr(value);
                }
                self.body(&w.body, inside);
            }
            Stmt::Set(s) => {
                self.sets.push((&s.target, &s.expr));
                self.expr(&s.expr);
            }
            Stmt::SetBlock(s) => self.body(&s.body, inside),
            Stmt::AutoEscape(a) => self.body(&a.body, inside),
            Stmt::FilterBlock(f) => {
                self.expr(&f.filter);
                self.body(&f.body, inside);
            }
            Stmt::Block(b) => self.body(&b.body, inside),
            Stmt::Macro(m) => {
                let index = self.macros.len();
                self.macros.push(MacroDecl {
                    name: m.name,
                    args: m
                        .args
                        .iter()
                        .filter_map(|a| match a {
                            Expr::Var(v) => Some(v.id),
                            _ => None,
                        })
                        .collect(),
                });
                for default in &m.defaults {
                    self.expr(default);
                }
                let mut nested = inside.to_vec();
                nested.push(index);
                self.body(&m.body, &nested);
            }
            Stmt::CallBlock(c) => {
                self.calls.push((&c.call.expr, &c.call.args));
                self.expr(&c.call.expr);
                self.args(&c.call.args);
                self.body(&c.macro_decl.body, inside);
            }
            Stmt::Do(d) => {
                self.calls.push((&d.call.expr, &d.call.args));
                self.expr(&d.call.expr);
                self.args(&d.call.args);
            }
            Stmt::Import(_) | Stmt::FromImport(_) | Stmt::Extends(_) | Stmt::Include(_) => {}
        }
    }

    fn body(&mut self, body: &'a [Stmt<'a>], inside: &[usize]) {
        for stmt in body {
            self.stmt(stmt, Some(inside));
        }
    }

    fn args(&mut self, args: &'a [CallArg<'a>]) {
        for arg in args {
            match arg {
                CallArg::Pos(e)
                | CallArg::Kwarg(_, e)
                | CallArg::PosSplat(e)
                | CallArg::KwargSplat(e) => {
                    self.expr(e);
                }
            }
        }
    }

    fn expr(&mut self, expr: &'a Expr<'a>) {
        match expr {
            Expr::Var(_) | Expr::Const(_) => {}
            Expr::Slice(s) => {
                self.expr(&s.expr);
                for part in [&s.start, &s.stop, &s.step].into_iter().flatten() {
                    self.expr(part);
                }
            }
            Expr::UnaryOp(u) => self.expr(&u.expr),
            Expr::BinOp(b) => {
                self.expr(&b.left);
                self.expr(&b.right);
            }
            Expr::Compare(c) => {
                self.expr(&c.expr);
                for op in &c.ops {
                    self.expr(&op.expr);
                }
            }
            Expr::IfExpr(i) => {
                self.expr(&i.test_expr);
                self.expr(&i.true_expr);
                if let Some(e) = &i.false_expr {
                    self.expr(e);
                }
            }
            Expr::Filter(f) => {
                if let Some(e) = &f.expr {
                    self.expr(e);
                }
                self.args(&f.args);
            }
            Expr::Test(t) => {
                self.expr(&t.expr);
                self.args(&t.args);
            }
            Expr::GetAttr(g) => self.expr(&g.expr),
            Expr::GetItem(g) => {
                self.expr(&g.expr);
                self.expr(&g.subscript_expr);
            }
            Expr::Call(c) => {
                self.calls.push((&c.expr, &c.args));
                self.expr(&c.expr);
                self.args(&c.args);
            }
            Expr::List(l) => {
                for item in &l.items {
                    self.expr(item);
                }
            }
            Expr::Tuple(t) => {
                for item in &t.items {
                    self.expr(item);
                }
            }
            Expr::Map(m) => {
                for (k, v) in m.keys.iter().zip(&m.values) {
                    self.expr(k);
                    self.expr(v);
                }
            }
        }
    }

    fn assigned_from(&self, name: &'a str) -> Option<Vec<&'a str>> {
        let mut names = vec![name];
        let mut queue = std::collections::VecDeque::from([name]);
        while let Some(current) = queue.pop_front() {
            for (target, value) in &self.sets {
                if elems_of(value, current, None) {
                    let Expr::Var(target) = target else {
                        return None;
                    };
                    names.push(target.id);
                    if target.id != current
                        && names.iter().filter(|n| **n == target.id).count() < 64
                    {
                        queue.push_back(target.id);
                    }
                }
            }
        }
        Some(names)
    }

    fn message_names(&self) -> Option<Vec<&'a str>> {
        let sources = self.assigned_from("messages")?;
        let mut names = Vec::new();
        for lp in &self.loops {
            if sources.iter().any(|s| elems_of(lp.iter, s, None)) {
                let Expr::Var(target) = lp.target else {
                    return None;
                };
                names.push(target.id);
            }
        }
        Some(names)
    }

    fn verdict(&self) -> Option<ContentFormat> {
        let messages = self.message_names()?;
        let reads_content = |e: &Expr<'_>| messages.iter().any(|m| elems_of(e, m, Some("content")));
        let mut params: Vec<HashSet<&str>> = Vec::with_capacity(self.macros.len());
        for decl in &self.macros {
            let mut found = HashSet::new();
            for (callee, args) in &self.calls {
                if !matches!(callee, Expr::Var(v) if v.id == decl.name) {
                    continue;
                }
                let mut position = 0;
                for arg in *args {
                    match arg {
                        CallArg::Pos(e) => {
                            if let Some(name) = decl.args.get(position)
                                && reads_content(e)
                            {
                                found.insert(*name);
                            }
                            position += 1;
                        }
                        CallArg::Kwarg(key, e) => {
                            if decl.args.contains(key) && reads_content(e) {
                                found.insert(*key);
                            }
                        }
                        CallArg::PosSplat(_) | CallArg::KwargSplat(_) => {}
                    }
                }
            }
            params.push(found);
        }
        for lp in &self.loops {
            if reads_content(lp.iter) {
                return Some(ContentFormat::OpenAi);
            }
            let Expr::Var(iter) = lp.iter else { continue };
            let in_param = lp
                .macros
                .iter()
                .rev()
                .find(|&&m| !params[m].is_empty())
                .is_some_and(|&m| params[m].contains(iter.id));
            if in_param || (lp.macros.is_empty() && iter.id == "content") {
                return Some(ContentFormat::OpenAi);
            }
        }
        Some(ContentFormat::String)
    }
}

fn is_var(expr: &Expr<'_>, name: &str) -> bool {
    matches!(expr, Expr::Var(v) if v.id == name)
}

fn is_attr(expr: &Expr<'_>, name: &str, key: &str) -> bool {
    match expr {
        Expr::GetItem(g) => {
            is_var(&g.expr, name)
                && matches!(&g.subscript_expr, Expr::Const(c) if c.value.as_str() == Some(key))
        }
        Expr::GetAttr(g) => is_var(&g.expr, name) && g.name == key,
        _ => false,
    }
}

fn elems_of(expr: &Expr<'_>, name: &str, key: Option<&str>) -> bool {
    match expr {
        Expr::Filter(f) => f.expr.as_ref().is_some_and(|e| elems_of(e, name, key)),
        Expr::Test(t) => elems_of(&t.expr, name, key),
        Expr::Slice(s) => elems_of(&s.expr, name, key),
        _ => match key {
            Some(key) => is_attr(expr, name, key),
            None => is_var(expr, name),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(source: &str, context: serde_json::Value) -> Result<String, String> {
        let template =
            ChatTemplate::compile(source, &Clock::default()).map_err(|e| e.to_string())?;
        let context: Value = serde_json::from_value(context).unwrap();
        template.render(context).map_err(|e| e.to_string())
    }

    fn ok(source: &str, context: serde_json::Value) -> String {
        render(source, context).unwrap()
    }

    fn format_of(source: &str) -> ContentFormat {
        ChatTemplate::compile(source, &Clock::default())
            .unwrap()
            .format()
    }

    #[test]
    fn whitespace_follows_trim_and_lstrip_blocks() {
        let source = "a\n  {% if true %}\n  b\n  {% endif %}\nc\n";
        assert_eq!(ok(source, serde_json::json!({})), "a\n  b\nc");
    }

    #[test]
    fn values_print_as_python_prints_them() {
        let out = ok(
            "{{ n }} {{ t }} {{ f }} {{ x }} {{ l }} {{ d }} {{ missing }}|{{ s|string }}",
            serde_json::json!({"n": null, "t": true, "f": 1.0, "x": 2, "l": ["a", 1], "d": {"k": false}, "s": null}),
        );
        assert_eq!(out, "None True 1.0 2 ['a', 1] {'k': False} |None");
    }

    #[test]
    fn tojson_is_python_json_dumps() {
        let template = |source: &str| ChatTemplate::compile(source, &Clock::default()).unwrap();
        let context: Value = serde_json::from_str(r#"{"v": {"b": [1, "<&>"], "a": "é"}}"#).unwrap();
        let out = |source: &str| template(source).render(context.clone()).unwrap();
        assert_eq!(out("{{ v|tojson }}"), r#"{"b": [1, "<&>"], "a": "é"}"#);
        assert_eq!(
            out("{{ v|tojson(indent=2) }}"),
            "{\n  \"b\": [\n    1,\n    \"<&>\"\n  ],\n  \"a\": \"é\"\n}"
        );
        assert_eq!(
            out("{{ v|tojson(separators=(',', ':'), sort_keys=true) }}"),
            r#"{"a":"é","b":[1,"<&>"]}"#
        );
        assert_eq!(
            out("{{ v|tojson(true) }}"),
            "{\"b\": [1, \"<&>\"], \"a\": \"\\u00e9\"}"
        );
    }

    #[test]
    fn python_methods_and_namespaces_work() {
        let out = ok(
            "{% set ns = namespace(n=0) %}{% for m in xs %}{% set ns.n = ns.n + 1 %}{% endfor %}\
             {{ ns.n }} {{ s.split('</think>')[-1].lstrip('\\n') }} {{ s.startswith('<think>') }} \
             {% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %} {{ d.get('z', 'none') }}",
            serde_json::json!({"xs": [1, 2, 3], "s": "<think>x</think>\n\nanswer", "d": {"a": 1, "b": 2}}),
        );
        assert_eq!(out, "3 answer True a=1;b=2; none");
    }

    #[test]
    fn loop_controls_and_adjacent_items_work() {
        let out = ok(
            "{% for x in xs %}{% if x == 2 %}{% continue %}{% endif %}{% if x == 4 %}{% break %}{% endif %}\
             {{ x }}{{ loop.previtem or '-' }}{% endfor %}",
            serde_json::json!({"xs": [1, 2, 3, 4, 5]}),
        );
        assert_eq!(out, "1-32");
    }

    #[test]
    fn undefined_values_degrade_like_jinja2() {
        assert_eq!(
            ok(
                "[{{ a }}{{ a.b.c }}]{% for x in a %}x{% endfor %}",
                serde_json::json!({})
            ),
            "[]"
        );
        assert_eq!(
            ok(
                "{{ a is defined }} {{ n is none }}",
                serde_json::json!({"n": null})
            ),
            "False True"
        );
    }

    #[test]
    fn items_and_length_of_undefined_are_empty() {
        let out = ok(
            "{% for k, v in t.p.props|items %}x{% endfor %}{{ t.p.props|length }}{{ d|items|list|length }}{{ s|length }}",
            serde_json::json!({"t": {"p": null}, "d": {"a": 1, "b": 2}, "s": "abc"}),
        );
        assert_eq!(out, "023");
        assert!(render("{{ 5|items }}", serde_json::json!({})).is_err());
        assert!(render("{{ 5|length }}", serde_json::json!({})).is_err());
    }

    #[test]
    fn raise_exception_fails_the_render() {
        let err = render(
            "{{ raise_exception('roles must alternate') }}",
            serde_json::json!({}),
        )
        .unwrap_err();
        assert!(err.contains("roles must alternate"), "{err}");
    }

    #[test]
    fn strftime_now_reads_the_clock() {
        let at = jiff::civil::date(2026, 1, 15).at(9, 30, 0, 0);
        let template =
            ChatTemplate::compile("{{ strftime_now('%d %b %Y, %B %-d') }}", &Clock::Fixed(at))
                .unwrap();
        assert_eq!(
            template.render(Value::from(())).unwrap(),
            "15 Jan 2026, January 15"
        );
    }

    #[test]
    fn generation_tags_render_their_body() {
        let source = "{% for m in messages %}\n{%- generation -%}\n{{ m }}\n{%- endgeneration %}\n{% endfor %}";
        assert_eq!(
            without_generation_tags(source),
            "{% for m in messages %}\n{%- if true -%}\n{{ m }}\n{%- endif %}\n{% endfor %}"
        );
        assert_eq!(
            ok(source, serde_json::json!({"messages": ["a", "b"]})),
            "ab"
        );
    }

    #[test]
    fn undeclared_variables_are_known() {
        let template = ChatTemplate::compile(
            "{% set x = 1 %}{{ x }}{{ enable_thinking }}{% for m in messages %}{{ m.content }}{% endfor %}",
            &Clock::default(),
        )
        .unwrap();
        assert!(template.uses("enable_thinking") && template.uses("messages"));
        assert!(!template.uses("x") && !template.uses("m"));
        let reread = ChatTemplate::compile(
            "{% set enable_thinking = enable_thinking | default(true) %}{{ enable_thinking }}",
            &Clock::default(),
        )
        .unwrap();
        assert!(reread.uses("enable_thinking"), "read before it is assigned");
    }

    #[test]
    fn content_format_follows_vllm_detection() {
        for (source, want) in [
            (
                "{% for m in messages %}{{ m['content'] }}{% endfor %}",
                ContentFormat::String,
            ),
            (
                "{% for m in messages %}{% for c in m['content'] %}{{ c.text }}{% endfor %}{% endfor %}",
                ContentFormat::OpenAi,
            ),
            (
                "{% for m in messages %}{% for c in m.content | selectattr('type', 'equalto', 'text') %}{% endfor %}{% endfor %}",
                ContentFormat::OpenAi,
            ),
            (
                "{% set loop_messages = messages[1:] %}{% for m in loop_messages %}{% for c in m.content %}{% endfor %}{% endfor %}",
                ContentFormat::OpenAi,
            ),
            (
                "{% for content in contents %}{% endfor %}{% for c in content %}{% endfor %}",
                ContentFormat::OpenAi,
            ),
            (
                "{% macro r(c) %}{% for p in c %}{{ p }}{% endfor %}{% endmacro %}{% for m in messages %}{{ r(m.content) }}{% endfor %}",
                ContentFormat::OpenAi,
            ),
            (
                "{% macro r(content) %}{% for p in content %}{% endfor %}{% endmacro %}{{ r(x) }}",
                ContentFormat::String,
            ),
            (
                "{% for m in others %}{% for c in m.content %}{% endfor %}{% endfor %}",
                ContentFormat::String,
            ),
            (
                "{% for i, m in messages %}{% endfor %}",
                ContentFormat::String,
            ),
            (
                "{% for m in messages %}{{ m.content }}",
                ContentFormat::String,
            ),
        ] {
            let format = minijinja::machinery::parse(source, NAME, SyntaxConfig::default())
                .map_or(ContentFormat::String, |root| content_format(&root));
            assert_eq!(format, want, "{source}");
        }
        assert_eq!(
            format_of("{% for m in messages %}{% for c in m.content %}{% endfor %}{% endfor %}"),
            ContentFormat::OpenAi
        );
    }
}
