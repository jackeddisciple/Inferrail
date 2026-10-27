//! `--kv-events-config`, read as vLLM v0.31.0 reads its `KVEventsConfig`.

use inferrail_events::PublisherConfig;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct KvEventsConfig {
    pub enable_kv_cache_events: bool,
    /// `zmq` or `null`; unset means `zmq` when events are enabled.
    pub publisher: Option<String>,
    pub endpoint: String,
    pub replay_endpoint: Option<String>,
    pub buffer_steps: usize,
    pub hwm: usize,
    pub max_queue_size: usize,
    pub topic: String,
}

impl Default for KvEventsConfig {
    fn default() -> Self {
        let publisher = PublisherConfig::default();
        Self {
            enable_kv_cache_events: false,
            publisher: None,
            endpoint: publisher.endpoint,
            replay_endpoint: publisher.replay_endpoint,
            buffer_steps: publisher.buffer_steps,
            hwm: publisher.hwm,
            max_queue_size: 100_000,
            topic: publisher.topic,
        }
    }
}

impl KvEventsConfig {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let config: Self = serde_json::from_str(text).map_err(|e| e.to_string())?;
        match config.publisher.as_deref() {
            None | Some("zmq" | "null") => Ok(config),
            Some(other) => Err(format!("unknown event publisher {other:?}")),
        }
    }

    /// The publisher to start, when events are on and go over ZeroMQ.
    pub fn publisher(&self) -> Option<PublisherConfig> {
        let zmq = self.publisher.as_deref().unwrap_or("zmq") == "zmq";
        (self.enable_kv_cache_events && zmq).then(|| PublisherConfig {
            endpoint: self.endpoint.clone(),
            replay_endpoint: self.replay_endpoint.clone(),
            topic: self.topic.clone(),
            buffer_steps: self.buffer_steps,
            hwm: self.hwm,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vllm_style_configs_parse() {
        let on = KvEventsConfig::from_json(r#"{"enable_kv_cache_events": true}"#).unwrap();
        let publisher = on.publisher().unwrap();
        assert_eq!(publisher.endpoint, "tcp://*:5557");
        assert_eq!(publisher.replay_endpoint, None);
        assert_eq!((publisher.buffer_steps, publisher.hwm), (10_000, 100_000));
        let full = KvEventsConfig::from_json(
            r#"{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:6000",
                "replay_endpoint": "tcp://*:6001", "topic": "kv-events", "buffer_steps": 5,
                "hwm": 9, "max_queue_size": 3}"#,
        )
        .unwrap();
        let publisher = full.publisher().unwrap();
        assert_eq!(publisher.replay_endpoint.as_deref(), Some("tcp://*:6001"));
        assert_eq!(publisher.topic, "kv-events");
        assert_eq!((publisher.buffer_steps, publisher.hwm), (5, 9));
    }

    #[test]
    fn events_stay_off_unless_enabled_over_zmq() {
        assert!(KvEventsConfig::default().publisher().is_none());
        let null =
            KvEventsConfig::from_json(r#"{"enable_kv_cache_events": true, "publisher": "null"}"#)
                .unwrap();
        assert!(null.publisher().is_none());
        assert!(KvEventsConfig::from_json(r#"{"publisher": "kafka"}"#).is_err());
        assert!(KvEventsConfig::from_json(r#"{"enable": true}"#).is_err());
        assert!(KvEventsConfig::from_json("not json").is_err());
    }
}
