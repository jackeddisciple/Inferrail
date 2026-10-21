use std::net::{IpAddr, SocketAddr};

use anyhow::{Context, bail};
use clap::Parser;
use inferrail_sim::engine::EngineConfig;
use inferrail_sim::latency::LatencyModel;
use inferrail_sim::server::{Sim, SimConfig};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

/// Mock vLLM engine: same HTTP API, metrics and streaming format, with a
/// calibrated latency model instead of a GPU.
#[derive(Debug, Parser)]
#[command(name = "inferrail-sim", version)]
struct Args {
    #[arg(long, default_value = "0.0.0.0")]
    host: IpAddr,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Model path reported as the root of every served name.
    #[arg(long, default_value = "inferrail/sim-8b")]
    model: String,
    /// Names clients may request; defaults to the model path.
    #[arg(long, num_args = 1..)]
    served_model_name: Vec<String>,
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    #[arg(long, default_value_t = 32_768)]
    max_model_len: usize,
    #[arg(long, default_value_t = 256)]
    max_num_seqs: usize,
    /// KV cache capacity in blocks.
    #[arg(long, alias = "num-gpu-blocks", default_value_t = 4_096)]
    num_gpu_blocks_override: usize,
    /// Report cached prompt tokens in usage, like vLLM's flag of the same name.
    #[arg(long)]
    enable_prompt_tokens_details: bool,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Output length when a request sets no max_tokens; ends with "stop".
    #[arg(long, default_value_t = 256)]
    default_max_tokens: usize,
    /// Prefill cost per uncached token in microseconds (a).
    #[arg(long, default_value_t = 100.0)]
    prefill_token_us: f64,
    /// Attention cost per token pair in nanoseconds (b).
    #[arg(long, default_value_t = 3.3)]
    prefill_pair_ns: f64,
    /// Fixed prefill overhead in milliseconds (d).
    #[arg(long, default_value_t = 5.0)]
    prefill_fixed_ms: f64,
    /// Time per output token in milliseconds.
    #[arg(long, default_value_t = 20.0)]
    decode_token_ms: f64,
    /// Ignore the latency flags and answer as fast as possible.
    #[arg(long)]
    zero_latency: bool,
    #[arg(long, env = "INFERRAIL_LOG", default_value = "info")]
    log: String,
}

impl Args {
    fn sim_config(&self) -> anyhow::Result<SimConfig> {
        if self.block_size == 0 || self.max_model_len == 0 || self.max_num_seqs == 0 {
            bail!("--block-size, --max-model-len and --max-num-seqs must be positive");
        }
        let latency = if self.zero_latency {
            LatencyModel::ZERO
        } else {
            LatencyModel {
                prefill_token_secs: self.prefill_token_us / 1e6,
                prefill_pair_secs: self.prefill_pair_ns / 1e9,
                prefill_fixed_secs: self.prefill_fixed_ms / 1e3,
                decode_token_secs: self.decode_token_ms / 1e3,
            }
        };
        latency.validate().map_err(anyhow::Error::msg)?;
        Ok(SimConfig {
            model: self.model.clone(),
            served_model_names: self.served_model_name.clone(),
            max_model_len: self.max_model_len,
            default_max_tokens: self.default_max_tokens,
            enable_prompt_tokens_details: self.enable_prompt_tokens_details,
            seed: self.seed,
            engine: EngineConfig {
                block_size: self.block_size,
                num_gpu_blocks: self.num_gpu_blocks_override,
                max_num_seqs: self.max_num_seqs,
                latency,
            },
        })
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
    tracing::info!("shutting down");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_new(&args.log).context("invalid --log filter")?)
        .init();
    let config = args.sim_config()?;
    let addr = SocketAddr::new(args.host, args.port);
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("cannot bind {addr}"))?;
    tracing::info!(
        %addr,
        models = ?config.served_names(),
        block_size = config.engine.block_size,
        num_gpu_blocks = config.engine.num_gpu_blocks,
        max_num_seqs = config.engine.max_num_seqs,
        latency = ?config.engine.latency,
        "inferrail-sim listening"
    );
    Sim::new(config).serve(listener, shutdown_signal()).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_build_a_valid_config() {
        let config = Args::parse_from(["inferrail-sim"]).sim_config().unwrap();
        let (got, want) = (config.engine.latency, LatencyModel::GPU_24GB_8B);
        for (a, b) in [
            (got.prefill_token_secs, want.prefill_token_secs),
            (got.prefill_pair_secs, want.prefill_pair_secs),
            (got.prefill_fixed_secs, want.prefill_fixed_secs),
            (got.decode_token_secs, want.decode_token_secs),
        ] {
            assert!((a - b).abs() <= b * 1e-12, "{a} != {b}");
        }
        assert_eq!(config.served_names(), ["inferrail/sim-8b"]);
    }

    #[test]
    fn vllm_flag_names_are_accepted() {
        let args = Args::parse_from([
            "inferrail-sim",
            "--served-model-name",
            "a",
            "b",
            "--num-gpu-blocks-override",
            "99",
            "--enable-prompt-tokens-details",
            "--zero-latency",
        ]);
        let config = args.sim_config().unwrap();
        assert_eq!(config.served_names(), ["a", "b"]);
        assert_eq!(config.engine.num_gpu_blocks, 99);
        assert!(config.enable_prompt_tokens_details);
        assert!(config.engine.latency.is_zero());
    }

    #[test]
    fn invalid_values_are_rejected() {
        assert!(
            Args::parse_from(["inferrail-sim", "--block-size", "0"])
                .sim_config()
                .is_err()
        );
        assert!(
            Args::parse_from(["inferrail-sim", "--decode-token-ms=-1"])
                .sim_config()
                .is_err()
        );
    }
}
