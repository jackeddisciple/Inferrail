//! A KV event publisher shaped like vLLM's `ZmqEventPublisher`: a PUB
//! socket sends `[topic, seq, payload]` for every batch, numbering batches
//! from 0 in big endian, and an optional ROUTER socket replays the last
//! `buffer_steps` batches from a requested sequence number, ending with
//! sequence -1 and an empty payload. A subscriber that falls more than
//! `hwm` batches behind loses the oldest, as with a ZeroMQ high water mark.

use std::collections::VecDeque;
use std::io;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, PoisonError};

use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc};

use crate::vllm::EventBatch;
use crate::zmtp::{self, Frame};

/// Sequence number of the marker that ends a replay.
pub const END_SEQ: [u8; 8] = (-1i64).to_be_bytes();

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublisherConfig {
    pub endpoint: String,
    pub replay_endpoint: Option<String>,
    pub topic: String,
    pub buffer_steps: usize,
    pub hwm: usize,
}

impl Default for PublisherConfig {
    /// vLLM's defaults.
    fn default() -> Self {
        Self {
            endpoint: "tcp://*:5557".to_owned(),
            replay_endpoint: None,
            topic: String::new(),
            buffer_steps: 10_000,
            hwm: 100_000,
        }
    }
}

/// One published batch as a subscriber receives it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Published {
    pub topic: Vec<u8>,
    pub seq: u64,
    pub payload: Vec<u8>,
}

#[derive(Debug)]
struct State {
    next: u64,
    buffer: VecDeque<Arc<Published>>,
}

#[derive(Debug)]
struct Inner {
    topic: Vec<u8>,
    buffer_steps: usize,
    state: Mutex<State>,
    live: broadcast::Sender<Arc<Published>>,
}

#[derive(Debug, Clone)]
pub struct Publisher {
    inner: Arc<Inner>,
    addr: SocketAddr,
    replay_addr: Option<SocketAddr>,
}

impl Publisher {
    /// Binds the sockets and starts serving them on the current runtime.
    pub fn bind(config: &PublisherConfig) -> io::Result<Self> {
        let listener = listen(&config.endpoint)?;
        let replay = config.replay_endpoint.as_deref().map(listen).transpose()?;
        let (live, _) = broadcast::channel(config.hwm.clamp(1, 1 << 16));
        let inner = Arc::new(Inner {
            topic: config.topic.as_bytes().to_vec(),
            buffer_steps: config.buffer_steps,
            state: Mutex::new(State {
                next: 0,
                buffer: VecDeque::new(),
            }),
            live,
        });
        let addr = listener.local_addr()?;
        tokio::spawn(accept(listener, Arc::clone(&inner), serve_subscriber));
        let replay_addr = match replay {
            Some(listener) => {
                let addr = listener.local_addr()?;
                tokio::spawn(accept(listener, Arc::clone(&inner), serve_replay));
                Some(addr)
            }
            None => None,
        };
        Ok(Self {
            inner,
            addr,
            replay_addr,
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn replay_addr(&self) -> Option<SocketAddr> {
        self.replay_addr
    }

    /// Numbers and sends one batch; returns its sequence number.
    pub fn publish(&self, batch: &EventBatch) -> u64 {
        self.publish_payload(batch.encode())
    }

    pub fn publish_payload(&self, payload: Vec<u8>) -> u64 {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let seq = state.next;
        state.next += 1;
        let message = Arc::new(Published {
            topic: self.inner.topic.clone(),
            seq,
            payload,
        });
        if self.inner.buffer_steps > 0 {
            if state.buffer.len() == self.inner.buffer_steps {
                state.buffer.pop_front();
            }
            state.buffer.push_back(Arc::clone(&message));
        }
        let _ = self.inner.live.send(message);
        seq
    }
}

fn listen(endpoint: &str) -> io::Result<TcpListener> {
    let std = std::net::TcpListener::bind(zmtp::endpoint(endpoint)?)?;
    std.set_nonblocking(true)?;
    TcpListener::from_std(std)
}

async fn accept<F, Fut>(listener: TcpListener, inner: Arc<Inner>, serve: F)
where
    F: Fn(TcpStream, Arc<Inner>) -> Fut,
    Fut: std::future::Future<Output = io::Result<()>> + Send + 'static,
{
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            continue;
        };
        let _ = stream.set_nodelay(true);
        let task = serve(stream, Arc::clone(&inner));
        tokio::spawn(async move {
            if let Err(err) = task.await {
                tracing::debug!(error = %err, "KV event connection ended");
            }
        });
    }
}

#[derive(Debug)]
enum Incoming {
    Subscribe(Vec<u8>),
    Cancel(Vec<u8>),
    Ping(Vec<u8>),
}

async fn serve_subscriber(mut stream: TcpStream, inner: Arc<Inner>) -> io::Result<()> {
    zmtp::handshake(&mut stream, "PUB", &["SUB", "XSUB"]).await?;
    let mut live = inner.live.subscribe();
    let (mut reader, mut writer) = stream.into_split();
    let (tx, mut incoming) = mpsc::unbounded_channel();
    let reading = tokio::spawn(async move {
        loop {
            let next = match zmtp::read_frame(&mut reader).await? {
                Frame::Part { body, .. } => match body.split_first() {
                    Some((1, topic)) => Incoming::Subscribe(topic.to_vec()),
                    Some((0, topic)) => Incoming::Cancel(topic.to_vec()),
                    _ => continue,
                },
                Frame::Command { name, data } => match name.as_str() {
                    "SUBSCRIBE" => Incoming::Subscribe(data),
                    "CANCEL" => Incoming::Cancel(data),
                    "PING" => Incoming::Ping(data.get(2..).unwrap_or_default().to_vec()),
                    _ => continue,
                },
            };
            if tx.send(next).is_err() {
                return Ok::<(), io::Error>(());
            }
        }
    });
    let mut subscriptions: Vec<Vec<u8>> = Vec::new();
    let result = loop {
        tokio::select! {
            got = incoming.recv() => match got {
                Some(Incoming::Subscribe(topic)) => subscriptions.push(topic),
                Some(Incoming::Cancel(topic)) => {
                    if let Some(i) = subscriptions.iter().position(|t| *t == topic) {
                        subscriptions.swap_remove(i);
                    }
                }
                Some(Incoming::Ping(context)) => {
                    if let Err(err) = writer.write_all(&zmtp::command("PONG", &context)).await {
                        break Err(err);
                    }
                }
                None => break Ok(()),
            },
            sent = live.recv() => match sent {
                Ok(message) => {
                    if subscriptions.iter().any(|t| message.topic.starts_with(t)) {
                        let seq = message.seq.to_be_bytes();
                        let bytes = zmtp::message(&[&message.topic, &seq, &message.payload]);
                        if let Err(err) = writer.write_all(&bytes).await {
                            break Err(err);
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => break Ok(()),
            },
        }
    };
    reading.abort();
    result
}

async fn serve_replay(mut stream: TcpStream, inner: Arc<Inner>) -> io::Result<()> {
    zmtp::handshake(&mut stream, "ROUTER", &["REQ", "DEALER"]).await?;
    loop {
        let parts = zmtp::read_message(&mut stream).await?;
        let [delimiter, start] = parts.as_slice() else {
            tracing::debug!(parts = parts.len(), "invalid KV event replay request");
            continue;
        };
        let Ok(start) = <[u8; 8]>::try_from(start.as_slice()) else {
            continue;
        };
        let start = u64::from_be_bytes(start);
        let held: Vec<Arc<Published>> = {
            let state = inner.state.lock().unwrap_or_else(PoisonError::into_inner);
            state
                .buffer
                .iter()
                .filter(|m| m.seq >= start)
                .cloned()
                .collect()
        };
        let mut out = Vec::new();
        for message in held {
            let seq = message.seq.to_be_bytes();
            out.extend(zmtp::message(&[
                delimiter,
                &message.topic,
                &seq,
                &message.payload,
            ]));
        }
        out.extend(zmtp::message(&[delimiter, b"", &END_SEQ, b""]));
        stream.write_all(&out).await?;
    }
}

/// A SUB socket on one publisher.
#[derive(Debug)]
pub struct Subscriber {
    stream: TcpStream,
}

impl Subscriber {
    /// Connects and subscribes to topics starting with `topic`.
    pub async fn connect(addr: SocketAddr, topic: &[u8]) -> io::Result<Self> {
        let mut stream = TcpStream::connect(addr).await?;
        stream.set_nodelay(true)?;
        zmtp::handshake(&mut stream, "SUB", &["PUB", "XPUB"]).await?;
        let mut subscribe = vec![1];
        subscribe.extend_from_slice(topic);
        stream.write_all(&zmtp::message(&[&subscribe])).await?;
        Ok(Self { stream })
    }

    pub async fn recv(&mut self) -> io::Result<Published> {
        loop {
            let parts = zmtp::read_message(&mut self.stream).await?;
            if let Some(published) = published(parts) {
                return Ok(published);
            }
        }
    }
}

fn published(parts: Vec<Vec<u8>>) -> Option<Published> {
    let [topic, seq, payload]: [Vec<u8>; 3] = parts.try_into().ok()?;
    let seq = u64::from_be_bytes(seq.as_slice().try_into().ok()?);
    Some(Published {
        topic,
        seq,
        payload,
    })
}

/// Asks a replay socket, as a REQ peer, for every batch it holds from
/// `start` on.
pub async fn replay(addr: SocketAddr, start: u64) -> io::Result<Vec<Published>> {
    let mut stream = TcpStream::connect(addr).await?;
    zmtp::handshake(&mut stream, "REQ", &["ROUTER", "REP"]).await?;
    stream
        .write_all(&zmtp::message(&[b"", &start.to_be_bytes()]))
        .await?;
    let mut out = Vec::new();
    loop {
        let mut parts = zmtp::read_message(&mut stream).await?;
        if parts.first().is_some_and(Vec::is_empty) {
            parts.remove(0);
        }
        if parts.get(1).is_some_and(|seq| seq.as_slice() == END_SEQ) {
            return Ok(out);
        }
        out.extend(published(parts));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vllm::{BlockRemoved, KvEvent};

    fn config(topic: &str) -> PublisherConfig {
        PublisherConfig {
            endpoint: "tcp://127.0.0.1:0".to_owned(),
            replay_endpoint: Some("tcp://127.0.0.1:0".to_owned()),
            topic: topic.to_owned(),
            buffer_steps: 4,
            hwm: 1_000,
        }
    }

    fn batch(n: u64) -> EventBatch {
        EventBatch {
            ts: n as f64,
            events: vec![KvEvent::BlockRemoved(BlockRemoved {
                block_hashes: vec![n],
                medium: Some("GPU".to_owned()),
                group_idx: Some(0),
            })],
            data_parallel_rank: Some(0),
        }
    }

    async fn subscribed(publisher: &Publisher, topic: &[u8]) -> Subscriber {
        let sub = Subscriber::connect(publisher.local_addr(), topic)
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        sub
    }

    #[tokio::test]
    async fn subscribers_get_numbered_batches_on_their_topic() {
        let publisher = Publisher::bind(&config("kv-events")).unwrap();
        let mut all = subscribed(&publisher, b"").await;
        let mut kv = subscribed(&publisher, b"kv").await;
        let mut other = subscribed(&publisher, b"other").await;
        for n in 0..3 {
            assert_eq!(publisher.publish(&batch(n)), n);
        }
        for sub in [&mut all, &mut kv] {
            for n in 0..3 {
                let got = sub.recv().await.unwrap();
                assert_eq!((got.topic.as_slice(), got.seq), (&b"kv-events"[..], n));
                assert_eq!(EventBatch::decode(&got.payload).unwrap(), batch(n));
            }
        }
        let nothing =
            tokio::time::timeout(std::time::Duration::from_millis(100), other.recv()).await;
        assert!(nothing.is_err(), "a subscriber got another topic");
    }

    #[tokio::test]
    async fn replay_returns_the_buffered_tail_then_the_end_marker() {
        let publisher = Publisher::bind(&config("t")).unwrap();
        for n in 0..6 {
            publisher.publish(&batch(n));
        }
        let addr = publisher.replay_addr().unwrap();
        let seqs = |got: Vec<Published>| got.iter().map(|p| p.seq).collect::<Vec<_>>();
        assert_eq!(seqs(replay(addr, 0).await.unwrap()), [2, 3, 4, 5]);
        assert_eq!(seqs(replay(addr, 4).await.unwrap()), [4, 5]);
        assert!(replay(addr, 9).await.unwrap().is_empty());
        let got = replay(addr, 5).await.unwrap();
        assert_eq!(EventBatch::decode(&got[0].payload).unwrap(), batch(5));
    }

    #[tokio::test]
    async fn a_slow_subscriber_loses_the_oldest_batches_not_the_publisher() {
        let mut small = config("");
        small.hwm = 8;
        let publisher = Publisher::bind(&small).unwrap();
        let mut sub = subscribed(&publisher, b"").await;
        let blob = vec![7u8; 64 * 1024];
        for _ in 0..400 {
            publisher.publish_payload(blob.clone());
        }
        let mut seen = Vec::new();
        while let Ok(Ok(got)) =
            tokio::time::timeout(std::time::Duration::from_millis(200), sub.recv()).await
        {
            seen.push(got.seq);
        }
        assert!(seen.windows(2).all(|w| w[0] < w[1]), "out of order");
        assert_eq!(seen.last(), Some(&399));
        assert!(seen.len() < 400, "nothing was dropped");
    }

    #[tokio::test]
    async fn a_peer_that_is_not_a_subscriber_is_turned_away() {
        let publisher = Publisher::bind(&config("")).unwrap();
        let mut stream = TcpStream::connect(publisher.local_addr()).await.unwrap();
        zmtp::handshake(&mut stream, "PUB", &["PUB"]).await.unwrap();
        assert!(
            zmtp::read_frame(&mut stream).await.is_err(),
            "kept a PUB peer"
        );
        let mut garbage = TcpStream::connect(publisher.local_addr()).await.unwrap();
        garbage.write_all(&[0u8; 64]).await.unwrap();
        publisher.publish(&batch(1));
        assert!(
            Subscriber::connect(publisher.local_addr(), b"")
                .await
                .is_ok()
        );
    }
}
