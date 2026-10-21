//! Config parsing must fail fast and cleanly on hostile or malformed YAML.

use std::time::{Duration, Instant};

use inferrail_core::config::{Config, ConfigError};

const VALID: &str = "models: [{name: m}]\nreplicas: [{id: r, url: 'http://h:1', models: [m]}]\n";

fn reject(text: &str) -> String {
    let started = Instant::now();
    let err = Config::from_yaml_str(text).expect_err("must be rejected");
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "took {:?} to reject",
        started.elapsed()
    );
    err.to_string()
}

#[test]
fn deep_nesting_fails_fast() {
    let text = format!(
        "models: {}{}\nreplicas: []\n",
        "[".repeat(100_000),
        "]".repeat(100_000)
    );
    assert!(reject(&text).contains("recursion limit"));
}

#[test]
fn alias_expansion_bombs_fail_fast() {
    let mut text = String::from("models:\n  - &a [x, x, x, x, x, x, x, x, x, x]\n");
    let mut prev = 'a';
    for name in 'b'..='j' {
        text.push_str(&format!(
            "  - &{name} [*{prev}, *{prev}, *{prev}, *{prev}, *{prev}, *{prev}, *{prev}, *{prev}, *{prev}, *{prev}]\n"
        ));
        prev = name;
    }
    text.push_str("replicas: []\n");
    reject(&text);
}

#[test]
fn duplicate_keys_tabs_and_multiple_documents_are_rejected() {
    assert!(reject(&format!("{VALID}models: [{{name: n}}]\n")).contains("duplicate"));
    reject("models:\n\t- name: m\n");
    reject(&format!("{VALID}---\nmodels: []\n"));
}

#[test]
fn empty_and_null_documents_are_rejected() {
    for text in ["", "~", "null", "[]", "42"] {
        assert!(matches!(
            Config::from_yaml_str(text),
            Err(ConfigError::Parse(_))
        ));
    }
}

#[test]
fn special_floats_are_caught_by_validation() {
    let err = reject(&format!("{VALID}routing: {{epsilon: .nan, kappa: .inf}}\n"));
    assert!(
        err.contains("routing.epsilon") && err.contains("routing.kappa"),
        "{err}"
    );
}

#[test]
fn out_of_range_numbers_and_durations_are_rejected() {
    reject(
        "models: [{name: m, block_size: -1}]\nreplicas: [{id: r, url: 'http://h:1', models: [m]}]\n",
    );
    reject(
        "models: [{name: m, block_size: 99999999999}]\nreplicas: [{id: r, url: 'http://h:1', models: [m]}]\n",
    );
    reject(&format!(
        "{VALID}health: {{interval: 99999999999999999999999s}}\n"
    ));
    reject(&format!("{VALID}health: {{interval: soon}}\n"));
}

#[test]
fn errors_point_at_the_line() {
    let err = reject(
        "models: [{name: m}]\nreplicas:\n  - id: r\n    url: 'not a url'\n    models: [m]\n",
    );
    assert!(err.contains("line 4"), "{err}");
}

#[test]
fn ipv6_and_paths_in_replica_urls_are_fine() {
    for url in ["http://[::1]:8000", "https://gpu-0.internal:8443/vllm"] {
        let text = VALID.replace("http://h:1", url);
        assert!(Config::from_yaml_str(&text).is_ok(), "{url}");
    }
}
