use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};
use std::time::Instant;

use hyper::StatusCode;
use hyper::body::{Body, Bytes, Frame, Incoming, SizeHint};
use inferrail_control::Lease;

use crate::Shared;
use crate::metrics::{Outcome, Routed};

/// What the gateway sends back: an upstream body passed through frame by
/// frame, or a small body the gateway wrote itself.
#[derive(Debug)]
pub enum ResponseBody {
    Upstream(UpstreamBody),
    Full(Option<Bytes>),
}

impl ResponseBody {
    pub(crate) fn full(bytes: impl Into<Bytes>) -> Self {
        Self::Full(Some(bytes.into()))
    }

    pub(crate) fn empty() -> Self {
        Self::Full(None)
    }
}

/// The metrics a proxied response still owes: time to first byte for a
/// stream, and its outcome and duration when it ends.
#[derive(Debug)]
pub(crate) struct Recording {
    routed: Routed,
    arrival: Instant,
    status: StatusCode,
    stream: bool,
    started: bool,
}

impl Recording {
    pub(crate) fn new(routed: Routed, arrival: Instant) -> Self {
        Self {
            routed,
            arrival,
            status: StatusCode::OK,
            stream: false,
            started: false,
        }
    }

    pub(crate) fn responded(self, status: StatusCode, stream: bool) -> Self {
        Self {
            status,
            stream,
            ..self
        }
    }
}

/// Holds the replica lease until the stream ends, errors or the client
/// goes away; dropping it early closes the upstream connection.
#[derive(Debug)]
pub struct UpstreamBody {
    inner: Incoming,
    lease: Option<Lease>,
    shared: Arc<Shared>,
    recording: Option<Recording>,
}

impl UpstreamBody {
    pub(crate) fn new(
        inner: Incoming,
        lease: Lease,
        shared: Arc<Shared>,
        recording: Recording,
    ) -> Self {
        Self {
            inner,
            lease: Some(lease),
            shared,
            recording: Some(recording),
        }
    }

    fn data(&mut self) {
        if let Some(lease) = self.lease.as_mut() {
            lease.first_byte();
        }
        if let Some(r) = self.recording.as_mut()
            && r.stream
            && !r.started
        {
            r.started = true;
            self.shared
                .metrics
                .first_byte(&r.routed, r.arrival.elapsed());
        }
    }

    fn finish(&mut self, outcome: impl FnOnce(StatusCode) -> Outcome) {
        if let Some(r) = self.recording.take() {
            self.shared
                .metrics
                .finished(&r.routed, outcome(r.status), r.arrival.elapsed());
        }
    }
}

impl Drop for UpstreamBody {
    fn drop(&mut self) {
        let ended = self.inner.is_end_stream();
        self.finish(|status| {
            if ended {
                Outcome::Complete(status)
            } else {
                Outcome::Cancelled
            }
        });
    }
}

impl Body for ResponseBody {
    type Data = Bytes;
    type Error = hyper::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, hyper::Error>>> {
        match self.get_mut() {
            Self::Full(bytes) => Poll::Ready(bytes.take().map(|b| Ok(Frame::data(b)))),
            Self::Upstream(body) => {
                let frame = ready!(Pin::new(&mut body.inner).poll_frame(cx));
                match &frame {
                    None => {
                        drop(body.lease.take());
                        body.finish(Outcome::Complete);
                    }
                    Some(Err(_)) => {
                        if let Some(lease) = body.lease.take() {
                            crate::proxy::record_failure(&body.shared, lease.replica());
                        }
                        body.finish(|_| Outcome::Aborted);
                    }
                    Some(Ok(frame)) => {
                        if frame.is_data() {
                            body.data();
                        }
                    }
                }
                Poll::Ready(frame)
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        match self {
            Self::Full(bytes) => bytes.is_none(),
            Self::Upstream(body) => body.inner.is_end_stream(),
        }
    }

    fn size_hint(&self) -> SizeHint {
        match self {
            Self::Full(bytes) => SizeHint::with_exact(bytes.as_ref().map_or(0, |b| b.len() as u64)),
            Self::Upstream(body) => body.inner.size_hint(),
        }
    }
}
