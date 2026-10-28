use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use inferrail_core::config::Config;
use inferrail_gateway::{Gateway, telemetry};
use tokio::net::TcpListener;

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

/// Reloads the config file on every SIGHUP.
async fn reload_on_hangup(gateway: Gateway, path: PathBuf) {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let Ok(mut hangups) = signal(SignalKind::hangup()) else {
            return;
        };
        while hangups.recv().await.is_some() {
            match gateway.reload_from(&path) {
                Ok(done) => {
                    tracing::info!(generation = done.generation, changed = ?done.changed, "reloaded on SIGHUP")
                }
                Err(err) => tracing::warn!(error = %err, "config change not applied"),
            }
        }
    }
    #[cfg(not(unix))]
    let _ = (gateway, path);
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = Config::load(&args.config)
        .with_context(|| format!("cannot load {}", args.config.display()))?;
    let filter = args.log.as_deref().unwrap_or(&config.logging.level);
    let _telemetry = telemetry::init(filter, &config.observability)?;
    let addr = config.server.listen;
    if args.check {
        for warning in Gateway::check(&config)? {
            tracing::warn!("{warning}");
        }
        println!(
            "{}: {} models, {} replicas, {} DNS groups",
            args.config.display(),
            config.models.len(),
            config.replicas.len(),
            config.discovery.dns.len()
        );
        return Ok(());
    }
    let gateway = Gateway::new(config)?;
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("cannot bind {addr}"))?;
    let admin_addr = gateway.config().server.admin_listen;
    let admin_listener = TcpListener::bind(admin_addr)
        .await
        .with_context(|| format!("cannot bind {admin_addr}"))?;
    tracing::info!(
        %addr,
        admin = %admin_addr,
        models = gateway.config().models.len(),
        replicas = gateway.registry().load().replicas().len(),
        "inferrail-gateway listening"
    );
    gateway.set_config_path(args.config.clone());
    let interval = gateway.config().server.reload_interval;
    if !interval.is_zero() {
        tokio::spawn(gateway.clone().watch_config(args.config.clone(), interval));
    }
    tokio::spawn(reload_on_hangup(gateway.clone(), args.config.clone()));
    let (stop, mut stopped) = tokio::sync::watch::channel(());
    let admin = tokio::spawn(gateway.clone().serve_admin(admin_listener, async move {
        let _ = stopped.changed().await;
    }));
    gateway.serve(listener, shutdown_signal()).await?;
    drop(stop);
    admin.await??;
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
