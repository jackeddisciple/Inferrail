use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto;
use hyper_util::server::graceful::GracefulShutdown;
use tokio::net::TcpListener;
use tokio::task::JoinSet;

use crate::Shared;
use crate::proxy::handle;

pub(crate) async fn serve(
    shared: Arc<Shared>,
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    let mut builder = auto::Builder::new(TokioExecutor::new());
    builder
        .http1()
        .timer(TokioTimer::new())
        .header_read_timeout(shared.config.server.header_read_timeout);
    builder.http2().timer(TokioTimer::new());
    let graceful = GracefulShutdown::new();
    let drain = shared.config.server.shutdown_timeout;
    let mut connections = JoinSet::new();
    tokio::pin!(shutdown);
    loop {
        let stream = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => stream,
                Err(err) => {
                    tracing::warn!(error = %err, "accept failed");
                    tokio::time::sleep(Duration::from_millis(5)).await;
                    continue;
                }
            },
            Some(_) = connections.join_next(), if !connections.is_empty() => continue,
            () = &mut shutdown => break,
        };
        let _ = stream.set_nodelay(true);
        let shared = Arc::clone(&shared);
        let service = service_fn(move |request| handle(Arc::clone(&shared), request));
        let connection = builder
            .serve_connection(TokioIo::new(stream), service)
            .into_owned();
        let connection = graceful.watch(connection);
        connections.spawn(async move {
            if let Err(err) = connection.await {
                tracing::debug!(error = %err, "client connection ended with an error");
            }
        });
    }
    drop(listener);
    tracing::info!(open = connections.len(), "draining connections");
    if tokio::time::timeout(drain, graceful.shutdown())
        .await
        .is_err()
    {
        tracing::warn!(
            open = connections.len(),
            "shutdown timeout reached; closing remaining connections"
        );
    }
    connections.shutdown().await;
    Ok(())
}
