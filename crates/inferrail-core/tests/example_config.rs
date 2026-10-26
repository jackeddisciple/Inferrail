use std::path::Path;
use std::time::Duration;

use inferrail_core::config::{Config, PolicyKind};

fn example_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../deploy/inferrail.example.yaml")
}

#[test]
fn shipped_example_config_is_valid() {
    let config = Config::load(example_path()).unwrap();
    assert_eq!(config.models.len(), 2);
    assert_eq!(config.replicas.len(), 3);
    assert_eq!(config.discovery.dns.len(), 1);
    assert_eq!(config.replicas[2].max_pending, Some(32));
    assert_eq!(config.server.request_timeout, Duration::from_secs(600));
    assert_eq!(config.health.max_ejection_backoff, Duration::from_secs(60));

    let llama = config.model("llama-8b").unwrap();
    assert_eq!(config.policy_for(llama), PolicyKind::CostModel);
    assert_eq!(config.replicas_serving(&llama.name).count(), 3);

    let qwen = config.model("Qwen/Qwen3-8B").unwrap();
    assert_eq!(config.policy_for(qwen), PolicyKind::LeastRequest);
    assert_eq!(qwen.block_size, 16);
    assert_eq!(config.replicas_serving(&qwen.name).count(), 1);
    assert!(config.is_served(&qwen.name));
    assert_eq!(config.discovery.dns[0].max_concurrency, Some(128));
    assert_eq!(qwen.tokenizer.as_deref(), Some("Qwen/Qwen3-8B"));
    assert_eq!(
        qwen.chat_template_kwargs["enable_thinking"],
        serde_json::json!(false)
    );
}

#[test]
fn example_documents_the_defaults() {
    let config = Config::load(example_path()).unwrap();
    assert_eq!(config.routing, Default::default());
    assert_eq!(config.health, Default::default());
    assert_eq!(config.retry, Default::default());
    assert_eq!(config.logging, Default::default());
    assert_eq!(config.server, Default::default());
    assert_eq!(config.upstream, Default::default());
    assert_eq!(config.load, Default::default());
    assert_eq!(config.observability, Default::default());
    assert_eq!(config.tokenizer, Default::default());
    assert_eq!(config.index, Default::default());
}
