use std::path::PathBuf;

use inferrail_load::parse::escape;
use inferrail_load::{Engine, EngineLoad, ParseError, parse};
use proptest::prelude::*;

fn fixture(path: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(path);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn vllm_fixture_reads_load_cache_and_capacity() {
    let load = parse(&fixture("vllm/metrics.txt")).unwrap();
    assert_eq!(
        load,
        EngineLoad {
            engine: Engine::Vllm,
            running: 2,
            waiting: 1,
            kv_usage: 0.0731,
            prefix_queries: 9238,
            prefix_hits: 5040,
            preemptions: 0,
            kv_capacity_tokens: Some(18_240 * 16),
        }
    );
}

#[test]
fn vllm_data_parallel_engines_add_up() {
    let one = fixture("vllm/metrics.txt");
    let other: String = one
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            l.replace("engine=\"0\"", "engine=\"1\"")
                .replace("} 2.0", "} 6.0")
                .replace("} 0.0731", "} 0.5269")
                + "\n"
        })
        .collect();
    let load = parse(&format!("{one}{other}")).unwrap();
    assert_eq!((load.running, load.waiting), (8, 2));
    assert!((load.kv_usage - 0.3).abs() < 1e-9, "{}", load.kv_usage);
    assert_eq!(load.prefix_hits, 10_080);
    assert_eq!(load.kv_capacity_tokens, Some(2 * 18_240 * 16));
}

#[test]
fn sglang_fixture_sums_ranks_and_ignores_priority_breakdowns() {
    let load = parse(&fixture("sglang/metrics.txt")).unwrap();
    assert_eq!(
        load,
        EngineLoad {
            engine: Engine::Sglang,
            running: 8,
            waiting: 2,
            kv_usage: 0.5,
            prefix_queries: 1500,
            prefix_hits: 640,
            preemptions: 3,
            kv_capacity_tokens: Some(400_000),
        }
    );
}

#[test]
fn sglang_tensor_parallel_copies_count_once() {
    let page = "sglang:num_running_reqs{tp_rank=\"0\"} 4\nsglang:num_running_reqs{tp_rank=\"1\"} 4\nsglang:num_queue_reqs{tp_rank=\"0\"} 1\nsglang:num_queue_reqs{tp_rank=\"1\"} 1\nsglang:token_usage{tp_rank=\"0\"} 0.4\nsglang:token_usage{tp_rank=\"1\"} 0.4\n";
    let load = parse(page).unwrap();
    assert_eq!((load.running, load.waiting), (4, 1));
    assert_eq!(load.kv_usage, 0.4);
    assert_eq!(load.kv_capacity_tokens, None);
}

#[test]
fn unknown_and_incomplete_pages_are_errors() {
    assert_eq!(parse(""), Err(ParseError::UnknownSchema));
    assert_eq!(
        parse("# HELP up x\nup 1\nprocess_cpu_seconds_total 3.5\n"),
        Err(ParseError::UnknownSchema)
    );
    let err = parse("vllm:num_requests_running{engine=\"0\"} 1\n").unwrap_err();
    assert_eq!(
        err,
        ParseError::Missing {
            engine: Engine::Vllm,
            metric: "vllm:num_requests_waiting"
        }
    );
    assert_eq!(
        err.to_string(),
        "vllm metrics lack vllm:num_requests_waiting"
    );
    let renamed =
        fixture("vllm/metrics.txt").replace("kv_cache_usage_perc", "gpu_cache_usage_perc");
    assert!(matches!(
        parse(&renamed),
        Err(ParseError::Missing {
            metric: "vllm:kv_cache_usage_perc",
            ..
        })
    ));
}

#[test]
fn nonsense_values_are_ignored_and_usage_is_clamped() {
    let page = "vllm:num_requests_running NaN\nvllm:num_requests_running 3\nvllm:num_requests_waiting -2\nvllm:num_requests_waiting 1\nvllm:kv_cache_usage_perc 1.7\nvllm:prefix_cache_hits_total +Inf\n";
    let load = parse(page).unwrap();
    assert_eq!((load.running, load.waiting), (3, 1));
    assert_eq!(load.kv_usage, 1.0);
    assert_eq!(load.prefix_hits, 0);
}

#[derive(Debug, Clone)]
struct Rank {
    running: u16,
    waiting: u16,
    usage: f64,
    hits: u32,
    queries: u32,
}

fn rank() -> impl Strategy<Value = Rank> {
    (
        any::<u16>(),
        any::<u16>(),
        0.0f64..=1.0,
        any::<u32>(),
        any::<u32>(),
    )
        .prop_map(|(running, waiting, usage, hits, queries)| Rank {
            running,
            waiting,
            usage,
            hits,
            queries,
        })
}

fn render(engine: Engine, ranks: &[Rank], model: &str, junk: &[String]) -> String {
    let model = escape(model);
    let mut lines: Vec<String> = Vec::new();
    for (i, r) in ranks.iter().enumerate() {
        let labels = match engine {
            Engine::Vllm => format!("engine=\"{i}\",model_name=\"{model}\""),
            Engine::Sglang => format!(
                "dp_rank=\"{i}\",engine_type=\"unified\",model_name=\"{model}\",priority=\"\",tp_rank=\"0\""
            ),
        };
        let names: [(&str, f64); 5] = match engine {
            Engine::Vllm => [
                ("vllm:num_requests_running", f64::from(r.running)),
                ("vllm:num_requests_waiting", f64::from(r.waiting)),
                ("vllm:kv_cache_usage_perc", r.usage),
                ("vllm:prefix_cache_hits_total", f64::from(r.hits)),
                ("vllm:prefix_cache_queries_total", f64::from(r.queries)),
            ],
            Engine::Sglang => [
                ("sglang:num_running_reqs", f64::from(r.running)),
                ("sglang:num_queue_reqs", f64::from(r.waiting)),
                ("sglang:token_usage", r.usage),
                ("sglang:cached_tokens_total", f64::from(r.hits)),
                ("sglang:prompt_tokens_total", f64::from(r.queries)),
            ],
        };
        for (name, value) in names {
            lines.push(format!("# TYPE {name} gauge"));
            lines.push(format!("{name}{{{labels}}} {value:?}"));
        }
        if engine == Engine::Sglang {
            lines.push(format!(
                "sglang:num_running_reqs{{{}}} 99999",
                labels.replace("priority=\"\"", "priority=\"7\"")
            ));
        }
    }
    for (i, j) in junk.iter().enumerate() {
        lines.insert((i * 7) % (lines.len() + 1), j.replace('\n', " "));
    }
    lines.join("\n")
}

proptest! {
    #[test]
    fn rendered_pages_parse_to_the_reference_totals(
        sglang: bool,
        ranks in prop::collection::vec(rank(), 1..6),
        model in "(?s).{0,30}",
        junk in prop::collection::vec("[a-z_ #{}=\"0-9.]{0,40}", 0..8),
    ) {
        let engine = if sglang { Engine::Sglang } else { Engine::Vllm };
        let page = render(engine, &ranks, &model, &junk);
        let load = parse(&page).unwrap();
        prop_assert_eq!(load.engine, engine);
        let sum = |f: fn(&Rank) -> u64| ranks.iter().map(f).sum::<u64>();
        prop_assert_eq!(u64::from(load.running), sum(|r| u64::from(r.running)));
        prop_assert_eq!(u64::from(load.waiting), sum(|r| u64::from(r.waiting)));
        prop_assert_eq!(load.prefix_hits, sum(|r| u64::from(r.hits)));
        prop_assert_eq!(load.prefix_queries, sum(|r| u64::from(r.queries)));
        let mean = ranks.iter().map(|r| r.usage).sum::<f64>() / ranks.len() as f64;
        prop_assert!((load.kv_usage - mean).abs() < 1e-9);
    }

    #[test]
    fn mangled_fixtures_never_panic(
        cut in 0usize..60_000,
        insert in "(?s).{0,20}",
        sglang: bool,
    ) {
        let page = fixture(if sglang { "sglang/metrics.txt" } else { "vllm/metrics.txt" });
        let at = page.floor_char_boundary(cut.min(page.len()));
        let mangled = format!("{}{insert}{}", &page[..at], &page[at..]);
        let _ = parse(&mangled);
        let _ = parse(&page[..at]);
    }
}
