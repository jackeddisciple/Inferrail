use std::net::{IpAddr, SocketAddr};

use anyhow::Context;
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
    /// Tokens one scheduler step may compute; longer prompts prefill in chunks.
    #[arg(long, default_value_t = 2_048)]
    max_num_batched_tokens: usize,
    /// KV cache capacity in blocks.
    #[arg(long, alias = "num-gpu-blocks", default_value_t = 4_096)]
    num_gpu_blocks_override: usize,
    /// Keep finished blocks for later prompts to reuse (the default).
    #[arg(long, overrides_with = "no_enable_prefix_caching")]
    enable_prefix_caching: bool,
    #[arg(long, overrides_with = "enable_prefix_caching")]
    no_enable_prefix_caching: bool,
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
    /// Time of a decode step for one sequence in milliseconds.
    #[arg(long, default_value_t = 20.0)]
    decode_token_ms: f64,
    /// Time each further sequence adds to a decode step in microseconds.
    #[arg(long, default_value_t = 100.0)]
    decode_seq_us: f64,
    /// Ignore the latency flags and answer as fast as possible.
    #[arg(long)]
    zero_latency: bool,
    #[arg(long, env = "INFERRAIL_LOG", default_value = "info")]
    log: String,
}

impl Args {
    fn sim_config(&self) -> SimConfig {
        let latency = if self.zero_latency {
            LatencyModel::ZERO
        } else {
            LatencyModel {
                prefill_token_secs: self.prefill_token_us / 1e6,
                prefill_pair_secs: self.prefill_pair_ns / 1e9,
                prefill_fixed_secs: self.prefill_fixed_ms / 1e3,
                decode_token_secs: self.decode_token_ms / 1e3,
                decode_seq_secs: self.decode_seq_us / 1e6,
            }
        };
        SimConfig {
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
                max_num_batched_tokens: self.max_num_batched_tokens,
                enable_prefix_caching: !self.no_enable_prefix_caching,
                latency,
            },
        }
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
    let config = args.sim_config();
    let sim = Sim::new(config.clone())?;
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
        max_num_batched_tokens = config.engine.max_num_batched_tokens,
        prefix_caching = config.engine.enable_prefix_caching,
        latency = ?config.engine.latency,
        "inferrail-sim listening"
    );
    sim.serve(listener, shutdown_signal()).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_build_a_valid_config() {
        let config = Args::parse_from(["inferrail-sim"]).sim_config();
        assert!(config.validate().is_ok());
        let (got, want) = (config.engine.latency, LatencyModel::GPU_24GB_8B);
        for (a, b) in [
            (got.prefill_token_secs, want.prefill_token_secs),
            (got.prefill_pair_secs, want.prefill_pair_secs),
            (got.prefill_fixed_secs, want.prefill_fixed_secs),
            (got.decode_token_secs, want.decode_token_secs),
            (got.decode_seq_secs, want.decode_seq_secs),
        ] {
            assert!((a - b).abs() <= b * 1e-12, "{a} != {b}");
        }
        assert_eq!(config.served_names(), ["inferrail/sim-8b"]);
        assert_eq!(
            config.engine,
            inferrail_sim::engine::EngineConfig {
                latency: config.engine.latency,
                ..inferrail_sim::engine::EngineConfig::default()
            }
        );
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
            "--max-model-len",
            "1584",
            "--max-num-batched-tokens",
            "512",
            "--enable-prompt-tokens-details",
            "--no-enable-prefix-caching",
            "--zero-latency",
        ]);
        let config = args.sim_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.served_names(), ["a", "b"]);
        assert_eq!(config.engine.num_gpu_blocks, 99);
        assert_eq!(config.engine.max_num_batched_tokens, 512);
        assert!(!config.engine.enable_prefix_caching);
        let on = Args::parse_from([
            "inferrail-sim",
            "--no-enable-prefix-caching",
            "--enable-prefix-caching",
        ]);
        assert!(on.sim_config().engine.enable_prefix_caching);
        assert!(config.enable_prompt_tokens_details);
        assert!(config.engine.latency.is_zero());
    }

    #[test]
    fn invalid_values_are_rejected() {
        for args in [
            &["inferrail-sim", "--block-size", "0"][..],
            &["inferrail-sim", "--max-num-seqs", "0"],
            &["inferrail-sim", "--max-model-len", "0"],
            &["inferrail-sim", "--num-gpu-blocks-override", "99"],
            &["inferrail-sim", "--max-num-batched-tokens", "100"],
            &["inferrail-sim", "--decode-seq-us=-1"],
            &["inferrail-sim", "--decode-token-ms=-1"],
            &["inferrail-sim", "--served-model-name", "has space"],
        ] {
            assert!(
                Args::parse_from(args).sim_config().validate().is_err(),
                "{args:?}"
            );
        }
    }
}
