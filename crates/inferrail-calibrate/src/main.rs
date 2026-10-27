use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use inferrail_calibrate::{Plan, calibrate};

/// Measures an idle engine's prefill and writes the cost model that fits it.
#[derive(Debug, Parser)]
#[command(name = "inferrail-calibrate", version)]
struct Args {
    /// The engine's base URL.
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    url: String,
    /// The model to send; the engine's first model by default.
    #[arg(long)]
    model: Option<String>,
    /// Tokens per KV block; read from the engine's metrics by default.
    #[arg(long)]
    block_size: Option<u32>,
    /// The longest prompt to send, in tokens.
    #[arg(long, default_value_t = 16_384)]
    max_context: u32,
    /// Measurements per point of the grid.
    #[arg(long, default_value_t = 2)]
    repeats: u32,
    /// Seeds the prompts sent; a fresh seed every run by default.
    #[arg(long)]
    seed: Option<u64>,
    /// Seconds to wait for one request.
    #[arg(long, default_value_t = 120)]
    timeout: u64,
    /// Writes the cost model to this file as a config fragment.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Writes every probe and the fit to this file as JSON.
    #[arg(long)]
    json: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let plan = Plan {
        url: args.url,
        model: args.model,
        block_size: args.block_size,
        max_context: args.max_context,
        repeats: args.repeats,
        seed: args.seed,
        timeout: Duration::from_secs(args.timeout),
    };
    let calibration = calibrate(&plan, |probe| {
        let point = probe.point();
        let _ = writeln!(
            std::io::stderr(),
            "{:>6} cached {:>6} new: {:8.2} ms",
            point.cached,
            point.uncached,
            point.ttft * 1e3
        );
    })
    .await?;
    let yaml = calibration.to_yaml();
    match &args.out {
        Some(path) => std::fs::write(path, &yaml)
            .with_context(|| format!("cannot write {}", path.display()))?,
        None => print!("{yaml}"),
    }
    if let Some(path) = &args.json {
        let text = serde_json::to_string_pretty(&calibration)?;
        std::fs::write(path, text).with_context(|| format!("cannot write {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_parse() {
        let args = Args::parse_from([
            "inferrail-calibrate",
            "--url",
            "http://gpu-0:8000",
            "--max-context",
            "4096",
            "--out",
            "cost.yaml",
        ]);
        assert_eq!(args.url, "http://gpu-0:8000");
        assert_eq!(args.max_context, 4_096);
        assert_eq!(args.repeats, 2);
        assert_eq!(args.seed, None);
        assert_eq!(args.out, Some(PathBuf::from("cost.yaml")));
    }
}
