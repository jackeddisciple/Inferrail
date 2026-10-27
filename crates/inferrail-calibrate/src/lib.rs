//! Profiles an engine and fits the TTFT cost model parameters.
//!
//! Prompts of known cached and uncached lengths go to an idle engine one at
//! a time. A cached prefix is sent alone first, so the probe that follows
//! finds it in the prefix cache. Each probe's time to first token is read
//! from the engine's own histogram where it exports one, so network and
//! HTTP time stay out of the fit, and from the stream otherwise.

use std::hash::{BuildHasher, RandomState};
use std::time::{Duration, Instant};

use inferrail_core::config::CostModelConfig;
use inferrail_load::parse::samples;
use serde::Serialize;
use serde_json::{Value, json};

mod fit;

pub use fit::{Coefficients, Point, fit};

#[derive(Debug, thiserror::Error)]
pub enum CalibrateError {
    #[error("cannot reach {url}: {source}")]
    Http { url: String, source: reqwest::Error },
    #[error("{url} answered {status}: {body}")]
    Status {
        url: String,
        status: u16,
        body: String,
    },
    #[error("the engine lists no model")]
    NoModel,
    #[error("the engine served other requests during a probe; calibrate an idle engine")]
    Busy,
    #[error("{0} probes fit, too few for three parameters")]
    TooFew(usize),
    #[error("the measurements fit no cost model")]
    NoFit,
}

/// What to measure.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub url: String,
    /// The model to send; the engine's first model when `None`.
    pub model: Option<String>,
    /// Tokens per KV block; read from the engine's metrics when `None`.
    pub block_size: Option<u32>,
    /// The longest prompt to send, cached and uncached together.
    pub max_context: u32,
    /// Measurements per point of the grid.
    pub repeats: u32,
    /// Seeds the prompts; drawn afresh for every run when `None`, so a run
    /// never finds an earlier run's prompts in the engine's cache.
    pub seed: Option<u64>,
    pub timeout: Duration,
}

impl Default for Plan {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:8000".to_owned(),
            model: None,
            block_size: None,
            max_context: 16_384,
            repeats: 2,
            seed: None,
            timeout: Duration::from_secs(120),
        }
    }
}

/// One measured prompt.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Probe {
    pub uncached: u32,
    pub cached: u32,
    /// Time to first token as the engine measured it, when it exports it.
    pub engine_ttft_secs: Option<f64>,
    /// Time to first byte of the stream as the client saw it.
    pub client_ttft_secs: f64,
}

impl Probe {
    fn ttft(&self) -> f64 {
        self.engine_ttft_secs.unwrap_or(self.client_ttft_secs)
    }
}

/// The fitted cost model and how well it matches what was measured.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Calibration {
    pub url: String,
    pub model: String,
    pub block_size: u32,
    /// What seeded the prompts, to send the same ones again.
    pub seed: u64,
    /// Whether times came from the engine's histogram or the stream.
    pub engine_timed: bool,
    pub cost_model: CostModelConfig,
    pub r_squared: f64,
    pub median_error: f64,
    pub worst_error: f64,
    /// The median time the client waited beyond the engine's own count,
    /// which the cost model leaves out since every replica pays it.
    pub overhead_secs: Option<f64>,
    pub probes: Vec<Probe>,
}

impl Calibration {
    /// The fitted cost model as a config fragment, for `routing` or a
    /// model's entry, with how it was measured in comments.
    pub fn to_yaml(&self) -> String {
        let c = &self.cost_model;
        let mut out = format!(
            "# inferrail-calibrate {}: {} at {}, seed {}\n# {} probes timed by the {}; R squared {:.5}, median error {:.2}%, worst {:.2}%\n",
            env!("CARGO_PKG_VERSION"),
            self.model,
            self.url,
            self.seed,
            self.probes.len(),
            if self.engine_timed {
                "engine"
            } else {
                "client"
            },
            self.r_squared,
            self.median_error * 100.0,
            self.worst_error * 100.0,
        );
        if let Some(overhead) = self.overhead_secs {
            out.push_str(&format!(
                "# the client waited {:.2} ms more per request than the engine counted\n",
                overhead * 1e3
            ));
        }
        out.push_str(&format!(
            "cost_model:\n  prefill_token_secs: {:.4e}\n  prefill_pair_secs: {:.4e}\n  prefill_fixed_secs: {:.4e}\n  decode_interference_secs: {:.4e}\n",
            c.prefill_token_secs, c.prefill_pair_secs, c.prefill_fixed_secs, c.decode_interference_secs
        ));
        out
    }
}

/// The points measured: every cached length with every uncached length
/// that fits in `max_context`, cached lengths whole blocks.
pub fn grid(max_context: u32, block_size: u32) -> Vec<(u32, u32)> {
    let block = block_size.max(1);
    let cached = [0u32, 1_024, 4_096, 8_192].map(|c| c / block * block);
    let uncached = [32u32, 256, 1_024, 2_048, 4_096];
    let mut out: Vec<(u32, u32)> = Vec::new();
    for c in cached {
        for u in uncached {
            if c + u < max_context && !out.contains(&(u, c)) {
                out.push((u, c));
            }
        }
    }
    out
}

/// Token ids from a seeded stream, in a range every common vocabulary has.
struct Tokens(u64);

impl Tokens {
    fn take(&mut self, n: u32) -> Vec<u32> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                1_000 + ((z ^ (z >> 31)) % 29_000) as u32
            })
            .collect()
    }
}

struct Engine {
    client: reqwest::Client,
    url: String,
    model: String,
    ttft_family: Option<String>,
}

/// A histogram's sum and count on a metrics page.
fn histogram(page: &str, family: &str) -> Option<(f64, f64)> {
    let (mut sum, mut count) = (None, None);
    let (sum_name, count_name) = (format!("{family}_sum"), format!("{family}_count"));
    for sample in samples(page) {
        if sample.name == sum_name {
            *sum.get_or_insert(0.0) += sample.value;
        } else if sample.name == count_name {
            *count.get_or_insert(0.0) += sample.value;
        }
    }
    Some((sum?, count?))
}

/// The engine's TTFT histogram and its block size, from its metrics page.
fn read_metrics(page: &str) -> (Option<String>, Option<u32>) {
    let family = ["vllm:", "sglang:"]
        .iter()
        .map(|prefix| format!("{prefix}time_to_first_token_seconds"))
        .find(|family| histogram(page, family).is_some());
    let block = samples(page)
        .find(|s| s.name == "vllm:cache_config_info")
        .and_then(|s| s.label("block_size").and_then(|b| b.parse().ok()));
    (family, block)
}

impl Engine {
    async fn get(&self, path: &str) -> Result<String, CalibrateError> {
        let url = format!("{}{path}", self.url);
        let response =
            self.client
                .get(&url)
                .send()
                .await
                .map_err(|source| CalibrateError::Http {
                    url: url.clone(),
                    source,
                })?;
        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(|source| CalibrateError::Http {
                url: url.clone(),
                source,
            })?;
        if status != 200 {
            return Err(CalibrateError::Status { url, status, body });
        }
        Ok(body)
    }

    async fn ttft_histogram(&self) -> Result<Option<(f64, f64)>, CalibrateError> {
        let Some(family) = &self.ttft_family else {
            return Ok(None);
        };
        Ok(histogram(&self.get("/metrics").await?, family))
    }

    /// Sends `prompt` for one token and waits for the whole answer; returns
    /// the time to the first byte of the stream.
    async fn complete(&self, prompt: &[u32]) -> Result<Duration, CalibrateError> {
        let url = format!("{}/v1/completions", self.url);
        let body = json!({
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "stream": true,
        });
        let started = Instant::now();
        let mut response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|source| CalibrateError::Http {
                url: url.clone(),
                source,
            })?;
        let status = response.status().as_u16();
        if status != 200 {
            let body = response.text().await.unwrap_or_default();
            return Err(CalibrateError::Status { url, status, body });
        }
        let mut first = None;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|source| CalibrateError::Http {
                url: url.clone(),
                source,
            })?
        {
            if !chunk.is_empty() {
                first.get_or_insert_with(|| started.elapsed());
            }
        }
        Ok(first.unwrap_or_else(|| started.elapsed()))
    }

    async fn probe(
        &self,
        tokens: &mut Tokens,
        uncached: u32,
        cached: u32,
    ) -> Result<Probe, CalibrateError> {
        let mut prompt = tokens.take(cached);
        if cached > 0 {
            self.complete(&prompt).await?;
        }
        prompt.extend(tokens.take(uncached));
        let before = self.ttft_histogram().await?;
        let client = self.complete(&prompt).await?;
        let after = self.ttft_histogram().await?;
        let engine = match (before, after) {
            (Some((sum0, count0)), Some((sum1, count1))) => {
                if (count1 - count0 - 1.0).abs() > 0.5 {
                    return Err(CalibrateError::Busy);
                }
                Some(sum1 - sum0)
            }
            _ => None,
        };
        Ok(Probe {
            uncached,
            cached,
            engine_ttft_secs: engine,
            client_ttft_secs: client.as_secs_f64(),
        })
    }
}

fn first_model(cards: &str) -> Option<String> {
    let list: Value = serde_json::from_str(cards).ok()?;
    list["data"].as_array()?.first()?["id"]
        .as_str()
        .map(str::to_owned)
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

/// Measures the engine at `plan.url` and fits its cost model. `log` hears
/// about every probe as it completes.
pub async fn calibrate(plan: &Plan, log: impl Fn(&Probe)) -> Result<Calibration, CalibrateError> {
    let url = plan.url.trim_end_matches('/').to_owned();
    let client = reqwest::Client::builder()
        .timeout(plan.timeout)
        .build()
        .map_err(|source| CalibrateError::Http {
            url: url.clone(),
            source,
        })?;
    let mut engine = Engine {
        client,
        url,
        model: String::new(),
        ttft_family: None,
    };
    engine.model = match &plan.model {
        Some(model) => model.clone(),
        None => first_model(&engine.get("/v1/models").await?).ok_or(CalibrateError::NoModel)?,
    };
    let (family, block) = read_metrics(&engine.get("/metrics").await.unwrap_or_default());
    engine.ttft_family = family;
    let block_size = plan.block_size.or(block).unwrap_or(16);
    let seed = plan
        .seed
        .unwrap_or_else(|| RandomState::new().hash_one(Instant::now()));
    let mut tokens = Tokens(seed);
    let mut probes = Vec::new();
    for _ in 0..plan.repeats.max(1) {
        for (uncached, cached) in grid(plan.max_context, block_size) {
            let probe = engine.probe(&mut tokens, uncached, cached).await?;
            log(&probe);
            probes.push(probe);
        }
    }
    let points: Vec<Point> = probes
        .iter()
        .map(|p| Point {
            uncached: f64::from(p.uncached),
            cached: f64::from(p.cached),
            ttft: p.ttft(),
        })
        .collect();
    if points.len() < 6 {
        return Err(CalibrateError::TooFew(points.len()));
    }
    let fitted = fit(&points).ok_or(CalibrateError::NoFit)?;
    let errors: Vec<f64> = points
        .iter()
        .map(|p| (fitted.predict(p) - p.ttft).abs() / p.ttft.max(1e-9))
        .collect();
    let engine_timed = probes.iter().all(|p| p.engine_ttft_secs.is_some());
    let overhead = median(
        probes
            .iter()
            .filter_map(|p| p.engine_ttft_secs.map(|e| p.client_ttft_secs - e))
            .collect(),
    );
    Ok(Calibration {
        url: engine.url,
        model: engine.model,
        block_size,
        seed,
        engine_timed,
        cost_model: CostModelConfig {
            prefill_token_secs: fitted.token,
            prefill_pair_secs: fitted.pair,
            prefill_fixed_secs: fitted.fixed,
            decode_interference_secs: 0.0,
        },
        r_squared: fitted.r_squared,
        median_error: median(errors.clone()).unwrap_or(0.0),
        worst_error: errors.into_iter().fold(0.0, f64::max),
        overhead_secs: overhead,
        probes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_stays_inside_the_context_in_whole_blocks() {
        let all = grid(16_384, 16);
        assert_eq!(all.len(), 20);
        assert!(all.iter().all(|&(u, c)| u + c < 16_384 && c % 16 == 0));
        let small = grid(2_048, 16);
        assert!(small.iter().all(|&(u, c)| u + c < 2_048));
        assert!(small.contains(&(1_024, 0)) && small.contains(&(256, 1_024)));
        assert!(grid(4_096, 1_000).iter().all(|&(_, c)| c % 1_000 == 0));
    }

    #[test]
    fn token_streams_are_seeded_and_stay_in_range() {
        let a = Tokens(5).take(1_000);
        assert_eq!(a, Tokens(5).take(1_000));
        assert_ne!(a, Tokens(6).take(1_000));
        assert!(a.iter().all(|&t| (1_000..30_000).contains(&t)));
    }

    #[test]
    fn metrics_pages_give_the_ttft_histogram_and_block_size() {
        let page = "vllm:cache_config_info{block_size=\"32\",engine=\"0\"} 1.0\nvllm:time_to_first_token_seconds_sum{engine=\"0\",model_name=\"m\"} 1.5\nvllm:time_to_first_token_seconds_count{engine=\"0\",model_name=\"m\"} 3.0\n";
        let (family, block) = read_metrics(page);
        assert_eq!(family.as_deref(), Some("vllm:time_to_first_token_seconds"));
        assert_eq!(block, Some(32));
        assert_eq!(
            histogram(page, "vllm:time_to_first_token_seconds"),
            Some((1.5, 3.0))
        );
        assert_eq!(read_metrics("nothing 1\n"), (None, None));
    }

    #[test]
    fn the_first_listed_model_is_used() {
        let cards = r#"{"object":"list","data":[{"id":"a"},{"id":"b"}]}"#;
        assert_eq!(first_model(cards).as_deref(), Some("a"));
        assert_eq!(first_model(r#"{"data":[]}"#), None);
        assert_eq!(first_model("not json"), None);
    }

    #[test]
    fn the_config_fragment_parses_as_a_cost_model() {
        let calibration = Calibration {
            url: "http://e:8000".to_owned(),
            model: "m".to_owned(),
            block_size: 16,
            seed: 7,
            engine_timed: true,
            cost_model: CostModelConfig {
                prefill_token_secs: 1.234_567e-4,
                prefill_pair_secs: 3.3e-9,
                prefill_fixed_secs: 5e-3,
                decode_interference_secs: 0.0,
            },
            r_squared: 0.99999,
            median_error: 0.001,
            worst_error: 0.004,
            overhead_secs: Some(0.0004),
            probes: Vec::new(),
        };
        let fragment: String = calibration
            .to_yaml()
            .lines()
            .map(|line| format!("    {line}\n"))
            .collect();
        let config = inferrail_core::config::Config::from_yaml_str(&format!(
            "models:\n  - name: m\n{fragment}replicas: [{{id: r, url: 'http://h:1', models: [m]}}]\n"
        ))
        .unwrap();
        let parsed = config.models[0].cost_model.expect("a cost model");
        let want = calibration.cost_model;
        for (got, want) in [
            (parsed.prefill_token_secs, want.prefill_token_secs),
            (parsed.prefill_pair_secs, want.prefill_pair_secs),
            (parsed.prefill_fixed_secs, want.prefill_fixed_secs),
            (
                parsed.decode_interference_secs,
                want.decode_interference_secs,
            ),
        ] {
            assert!((got - want).abs() <= want * 1e-4, "{got} {want}");
        }
        assert!(
            calibration
                .to_yaml()
                .contains("prefill_pair_secs: 3.3000e-9")
        );
    }
}
