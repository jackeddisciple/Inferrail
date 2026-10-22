use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use hyper::body::{Body, Bytes, Frame, Incoming, SizeHint};
use inferrail_control::Lease;

use crate::Shared;

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

/// Holds the replica lease until the stream ends, errors or the client
/// goes away; dropping it early closes the upstream connection.
#[derive(Debug)]
pub struct UpstreamBody {
    inner: Incoming,
    lease: Option<Lease>,
    shared: Arc<Shared>,
}

impl UpstreamBody {
    pub(crate) fn new(inner: Incoming, lease: Lease, shared: Arc<Shared>) -> Self {
        Self {
            inner,
            lease: Some(lease),
            shared,
        }
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
                    None => drop(body.lease.take()),
                    Some(Err(_)) => {
                        if let Some(lease) = body.lease.take() {
                            crate::proxy::record_failure(&body.shared, lease.replica());
                        }
                    }
                    Some(Ok(frame)) => {
                        if frame.is_data()
                            && let Some(lease) = body.lease.as_mut()
                        {
                            lease.first_byte();
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
