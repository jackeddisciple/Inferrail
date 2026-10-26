//! The variables a template reads without assigning them first, found the
//! way Jinja2's `meta.find_undeclared_variables` finds them, since vLLM
//! passes a request's template keyword only when the template reads it.
//!
//! Every loop, macro, call, filter, `with` and `set` block body is a frame
//! of its own, analyzed once the frame around it is complete, so a body may
//! read a name its parent assigns further down. Inside a frame names are
//! followed in order: `{% set x = x %}` reads `x` before assigning it. A
//! name first assigned inside an `if` counts as read, whichever branches
//! assign it, as it does in Jinja2.

use std::collections::{HashMap, HashSet};

use minijinja::machinery::ast::{CallArg, Expr, Macro, Stmt};

/// Jinja2's default globals, never reported.
const GLOBALS: [&str; 6] = ["range", "dict", "lipsum", "cycler", "joiner", "namespace"];

pub(crate) fn undeclared(root: &Stmt<'_>) -> HashSet<String> {
    let mut frames = Frames::default();
    let body = match root {
        Stmt::Template(t) => t.children.as_slice(),
        other => std::slice::from_ref(other),
    };
    frames.run(Frame {
        params: vec!["self"],
        body,
        ..Frame::default()
    });
    frames.out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Load {
    Resolve,
    Known,
}

#[derive(Debug, Clone, Default)]
struct Symbols<'a> {
    refs: HashMap<&'a str, Load>,
    stores: HashSet<&'a str>,
}

/// A body analyzed as a frame of its own: its parameters, the targets it
/// assigns on entry, the expressions it reads first, and its statements.
#[derive(Default)]
struct Frame<'a> {
    params: Vec<&'a str>,
    targets: Vec<&'a Expr<'a>>,
    reads: Vec<&'a Expr<'a>>,
    body: &'a [Stmt<'a>],
}

#[derive(Default)]
struct Frames<'a> {
    stack: Vec<Symbols<'a>>,
    out: HashSet<String>,
}

fn names<'a>(target: &'a Expr<'a>, out: &mut Vec<&'a str>) {
    match target {
        Expr::Var(v) => out.push(v.id),
        Expr::Tuple(t) => t.items.iter().for_each(|item| names(item, out)),
        Expr::List(l) => l.items.iter().for_each(|item| names(item, out)),
        _ => {}
    }
}

fn macro_frame<'a>(m: &'a Macro<'a>) -> Frame<'a> {
    let mut params = vec!["caller", "varargs", "kwargs"];
    m.args.iter().for_each(|arg| names(arg, &mut params));
    Frame {
        params,
        reads: m.defaults.iter().collect(),
        body: &m.body,
        ..Frame::default()
    }
}

impl<'a> Frames<'a> {
    fn top(&mut self) -> &mut Symbols<'a> {
        self.stack.last_mut().expect("a frame is open")
    }

    fn known(&self, name: &str) -> bool {
        self.stack.iter().any(|f| f.refs.contains_key(name))
    }

    fn known_outside(&self, name: &str) -> bool {
        self.stack[..self.stack.len() - 1]
            .iter()
            .any(|f| f.refs.contains_key(name))
    }

    fn load(&mut self, name: &'a str) {
        if !self.known(name) {
            self.top().refs.insert(name, Load::Resolve);
        }
    }

    fn store(&mut self, name: &'a str) {
        let top = self.top();
        top.stores.insert(name);
        top.refs.entry(name).or_insert(Load::Known);
    }

    fn run(&mut self, frame: Frame<'a>) {
        let mut symbols = Symbols::default();
        for param in &frame.params {
            symbols.stores.insert(param);
            symbols.refs.insert(param, Load::Known);
        }
        self.stack.push(symbols);
        let mut children = Vec::new();
        for &read in &frame.reads {
            self.expr(read);
        }
        for &target in &frame.targets {
            self.assign(target);
        }
        for stmt in frame.body {
            self.stmt(stmt, &mut children);
        }
        let top = self.top().clone();
        self.out.extend(
            top.refs
                .iter()
                .filter(|(name, load)| **load == Load::Resolve && !GLOBALS.contains(name))
                .map(|(name, _)| (*name).to_owned()),
        );
        for child in children {
            self.run(child);
        }
        self.stack.pop();
    }

    fn assign(&mut self, target: &'a Expr<'a>) {
        match target {
            Expr::Var(v) => self.store(v.id),
            Expr::Tuple(t) => t.items.iter().for_each(|item| self.assign(item)),
            Expr::List(l) => l.items.iter().for_each(|item| self.assign(item)),
            Expr::GetAttr(g) => self.expr(&g.expr),
            other => self.expr(other),
        }
    }

    fn branches(&mut self, bodies: [&'a [Stmt<'a>]; 2], children: &mut Vec<Frame<'a>>) {
        let base = self.top().clone();
        let mut refs = base.refs.clone();
        let mut fresh = HashSet::new();
        for body in bodies {
            *self.top() = base.clone();
            for stmt in body {
                self.stmt(stmt, children);
            }
            let branch = std::mem::take(self.top());
            fresh.extend(branch.stores.difference(&base.stores).copied());
            for (name, load) in branch.refs {
                refs.entry(name).or_insert(load);
            }
        }
        for name in &fresh {
            let load = if self.known_outside(name) {
                Load::Known
            } else {
                Load::Resolve
            };
            refs.insert(name, load);
        }
        let mut stores = base.stores;
        stores.extend(fresh);
        *self.top() = Symbols { refs, stores };
    }

    fn stmt(&mut self, stmt: &'a Stmt<'a>, children: &mut Vec<Frame<'a>>) {
        match stmt {
            Stmt::Template(t) => t.children.iter().for_each(|s| self.stmt(s, children)),
            Stmt::EmitExpr(e) => self.expr(&e.expr),
            Stmt::EmitRaw(_) | Stmt::Continue(_) | Stmt::Break(_) => {}
            Stmt::ForLoop(f) => {
                self.expr(&f.iter);
                let mut params = vec!["loop"];
                names(&f.target, &mut params);
                if let Some(test) = &f.filter_expr {
                    children.push(Frame {
                        params: params.clone(),
                        reads: vec![test],
                        ..Frame::default()
                    });
                }
                children.push(Frame {
                    params,
                    body: &f.body,
                    ..Frame::default()
                });
                children.push(Frame {
                    body: &f.else_body,
                    ..Frame::default()
                });
            }
            Stmt::IfCond(i) => {
                self.expr(&i.expr);
                self.branches([&i.true_body, &i.false_body], children);
            }
            Stmt::WithBlock(w) => {
                w.assignments.iter().for_each(|(_, value)| self.expr(value));
                children.push(Frame {
                    targets: w.assignments.iter().map(|(target, _)| target).collect(),
                    body: &w.body,
                    ..Frame::default()
                });
            }
            Stmt::Set(s) => {
                self.expr(&s.expr);
                self.assign(&s.target);
            }
            Stmt::SetBlock(s) => {
                self.assign(&s.target);
                children.push(Frame {
                    reads: s.filter.iter().collect(),
                    body: &s.body,
                    ..Frame::default()
                });
            }
            Stmt::AutoEscape(a) => {
                self.expr(&a.enabled);
                a.body.iter().for_each(|s| self.stmt(s, children));
            }
            Stmt::FilterBlock(f) => {
                self.expr(&f.filter);
                children.push(Frame {
                    body: &f.body,
                    ..Frame::default()
                });
            }
            Stmt::Block(b) => children.push(Frame {
                params: vec!["super"],
                body: &b.body,
                ..Frame::default()
            }),
            Stmt::Macro(m) => {
                self.store(m.name);
                children.push(macro_frame(m));
            }
            Stmt::CallBlock(c) => {
                self.expr(&c.call.expr);
                self.args(&c.call.args);
                children.push(macro_frame(&c.macro_decl));
            }
            Stmt::Do(d) => {
                self.expr(&d.call.expr);
                self.args(&d.call.args);
            }
            Stmt::Import(i) => {
                self.expr(&i.expr);
                self.assign(&i.name);
            }
            Stmt::FromImport(i) => {
                self.expr(&i.expr);
                for (name, alias) in &i.names {
                    self.assign(alias.as_ref().unwrap_or(name));
                }
            }
            Stmt::Extends(e) => self.expr(&e.name),
            Stmt::Include(i) => self.expr(&i.name),
        }
    }

    fn args(&mut self, args: &'a [CallArg<'a>]) {
        for arg in args {
            match arg {
                CallArg::Pos(e)
                | CallArg::Kwarg(_, e)
                | CallArg::PosSplat(e)
                | CallArg::KwargSplat(e) => self.expr(e),
            }
        }
    }

    fn expr(&mut self, expr: &'a Expr<'a>) {
        match expr {
            Expr::Var(v) => self.load(v.id),
            Expr::Const(_) => {}
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
                c.ops.iter().for_each(|op| self.expr(&op.expr));
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
                self.expr(&c.expr);
                self.args(&c.args);
            }
            Expr::List(l) => l.items.iter().for_each(|item| self.expr(item)),
            Expr::Tuple(t) => t.items.iter().for_each(|item| self.expr(item)),
            Expr::Map(m) => {
                for (k, v) in m.keys.iter().zip(&m.values) {
                    self.expr(k);
                    self.expr(v);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::syntax::SyntaxConfig;

    fn found(source: &str) -> Vec<String> {
        let syntax = SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .unwrap();
        let root = minijinja::machinery::parse(source, "t", syntax).unwrap();
        let mut names: Vec<String> = undeclared(&root).into_iter().collect();
        names.sort();
        names
    }

    /// Each expectation is what Jinja2 3.1.6 reports for the same source.
    #[test]
    fn variables_are_found_as_jinja2_finds_them() {
        for (source, want) in [
            (
                "{% if c %}{% set x = 1 %}{% else %}{% set x = 2 %}{% endif %}{{ x }}",
                &["c", "x"][..],
            ),
            ("{% if c %}{% set x = 1 %}{% endif %}", &["c", "x"]),
            (
                "{% set x = 0 %}{% if c %}{% set x = 1 %}{% endif %}{{ x }}",
                &["c"],
            ),
            ("{% set x = x | default(1) %}", &["x"]),
            ("{% set x = 1 %}{% set x = x + 1 %}", &[]),
            (
                "{% for m in ms %}{{ y }}{% endfor %}{% set y = 1 %}",
                &["ms"],
            ),
            (
                "{% for m in ms %}{% set y = y ~ 1 %}{% endfor %}",
                &["ms", "y"],
            ),
            (
                "{% for m in ms %}{{ loop.index }}{% else %}{{ loop }}{% endfor %}",
                &["loop", "ms"],
            ),
            (
                "{% macro f(a, b=c) %}{{ a }}{{ d }}{{ caller() }}{{ varargs }}{{ kwargs }}{% endmacro %}",
                &["c", "d"],
            ),
            ("{% macro f() %}{{ z }}{% endmacro %}{% set z = 1 %}", &[]),
            (
                "{% call(u) f(v) %}{{ u }}{{ w }}{% endcall %}",
                &["f", "v", "w"],
            ),
            ("{% filter upper %}{{ q }}{% endfilter %}", &["q"]),
            ("{% with a = b %}{{ a }}{{ c }}{% endwith %}", &["b", "c"]),
            ("{% set blk %}{{ blk }}{{ e }}{% endset %}", &["e"]),
            ("{% set ns = namespace(a=1) %}{% set ns.a = g %}", &["g"]),
            ("{% set a, b = h, i %}{{ a }}", &["h", "i"]),
            (
                "{{ self }}{{ range(3) }}{{ namespace }}{{ lipsum }}{{ dict() }}{{ cycler }}{{ joiner }}{{ raise_exception }}",
                &["raise_exception"],
            ),
            (
                "{% for k, v in items %}{{ k }}{{ v }}{% endfor %}{{ k }}",
                &["items", "k"],
            ),
            ("{% for m in ms if m.ok %}{{ m }}{% endfor %}", &["ms"]),
            (
                "{% for m in ms recursive %}{{ loop(m.children) }}{% endfor %}",
                &["ms"],
            ),
            (
                "{% if a %}{% if b %}{% set x = 1 %}{% endif %}{% endif %}{{ x }}",
                &["a", "b", "x"],
            ),
            (
                "{% for m in ms %}{% if loop.first %}{% set t = 1 %}{% endif %}{{ t }}{% endfor %}",
                &["ms", "t"],
            ),
            (
                "{% set enable_thinking = enable_thinking | default(true) %}{{ enable_thinking }}",
                &["enable_thinking"],
            ),
        ] {
            assert_eq!(found(source), want, "{source}");
        }
    }
}
