use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use inferrail_core::config::Config;
use inferrail_gateway::Gateway;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

/// KV cache aware router for OpenAI compatible inference engines.
#[derive(Debug, Parser)]
#[command(name = "inferrail-gateway", version)]
struct Args {
    #[arg(
        long,
        short,
        env = "INFERRAIL_CONFIG",
        default_value = "inferrail.yaml"
    )]
    config: PathBuf,
    /// Log filter; defaults to logging.level from the config.
    #[arg(long, env = "INFERRAIL_LOG")]
    log: Option<String>,
    /// Validate the config and exit.
    #[arg(long)]
    check: bool,
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
    let config = Config::load(&args.config)
        .with_context(|| format!("cannot load {}", args.config.display()))?;
    if args.check {
        println!(
            "{}: {} models, {} replicas, {} DNS groups",
            args.config.display(),
            config.models.len(),
            config.replicas.len(),
            config.discovery.dns.len()
        );
        return Ok(());
    }
    let filter = args.log.as_deref().unwrap_or(&config.logging.level);
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_new(filter).context("invalid log filter")?)
        .init();
    let addr = config.server.listen;
    let gateway = Gateway::new(config)?;
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("cannot bind {addr}"))?;
    tracing::info!(
        %addr,
        models = gateway.config().models.len(),
        replicas = gateway.registry().load().replicas().len(),
        "inferrail-gateway listening"
    );
    gateway.serve(listener, shutdown_signal()).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_parse() {
        let args = Args::parse_from(["inferrail-gateway", "-c", "x.yaml", "--check"]);
        assert_eq!(args.config, PathBuf::from("x.yaml"));
        assert!(args.check);
        assert!(args.log.is_none());
    }
}
