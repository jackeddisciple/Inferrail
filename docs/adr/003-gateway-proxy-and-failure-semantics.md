# ADR 003: Gateway proxy and failure semantics

Status: accepted

## Context

The gateway sits between OpenAI compatible clients and a fleet of engines.
It has to be invisible when things go well, which means forwarding requests
and streams without changing them and adding well under a millisecond, and
it has to be useful when things go badly: engines crash, hang, restart, fall
out of DNS or answer with errors, and clients disconnect mid stream.

Two properties pull against each other. Retrying a failed request on another
replica hides engine failures from clients, but once any byte of a response
has reached a client, a retry would splice two different answers together.
Buffering responses would make retries always possible, but it would destroy
streaming and add memory proportional to output length.

## Decision

* **hyper directly.** The listener is `hyper_util`'s automatic HTTP/1 and h2c
  server with a header read timeout and graceful shutdown. Four routes are
  matched by hand; everything else is a 404 or a 405 with `Allow`, as OpenAI
  shaped JSON. Upstream calls go through one pooled `hyper_util` client with
  keep alive, `TCP_NODELAY`, a connect timeout and optional h2c.
* **Bodies are forwarded byte for byte.** The request body is read once, up
  to `server.max_body_bytes`, and kept as `Bytes` so retries resend it for
  free. Only the `model` field is parsed, as a borrowed raw JSON value, which
  also validates the body. When a client uses a gateway alias, or leaves the
  model out and the gateway serves one model, only the bytes of that value
  are replaced; every other byte reaches the engine as sent.
* **Responses stream frame by frame.** Each upstream `Bytes` frame is passed
  to the client as it arrives, with trailers and the size hint intact. hyper's
  backpressure bounds what is held per connection, however slowly a client
  reads.
* **Hop by hop headers never cross.** `Connection` and every header it names,
  `Keep-Alive`, `TE`, `Trailer`, `Transfer-Encoding`, `Upgrade` and the proxy
  headers are dropped in both directions; `Host`, `Content-Length` and
  `Expect` are left to the upstream connection. Everything else passes,
  `Authorization` included. Every response carries `x-request-id` (the
  client's own when well formed) and `x-inferrail-replica`.
* **Retry only before the first byte.** An attempt is retryable when the
  connection fails, the replica closes before answering, it answers 502,
  503 or 504, or it fails its health checks while the attempt still waits
  for a response head; a hung engine would otherwise hold those requests
  until the deadline. A 500 or any 4xx is passed through: it describes the request,
  not the replica. Retries prefer replicas the request has not tried, back
  off exponentially from `retry.backoff`, stop at `retry.max_attempts`, and
  all share the `server.request_timeout` deadline. The gateway commits to a
  replica when it hands hyper a response head, so a stream that breaks
  afterwards ends with a broken connection and is never replayed. If every
  attempt fails, the last engine error response is returned as is, or a 502
  when there was none.
* **Health is active and passive.** Every replica gets `GET /health` each
  `health.interval` and leaves rotation after `unhealthy_threshold` failed
  probes, returning after `healthy_threshold` passing ones. Independently,
  `unhealthy_threshold` consecutive request failures eject it for
  `ejection_backoff`, doubling on each repeat up to `max_ejection_backoff`
  and resetting after a clean stretch that long. Because a failed request
  can be the request's fault rather than the replica's, passive ejection
  never takes more than `max_ejection_percent` (half by default) of the
  model's replicas that pass their health checks, and never a sole one. A replica is
  routable only when both agree. There is no panic mode: when nothing is routable the
  client gets a 503 with `Retry-After`, because routing to a replica known to
  be unhealthy breaks the invariant that policies never see one.
* **Breakers fail fast.** Optional per replica caps on requests in flight
  (`max_concurrency`) and on requests still waiting for a response head
  (`max_pending`) are taken by compare and swap, so they hold under any race.
  A full replica is skipped; a full fleet gets an immediate 503. A replica
  that accepts requests but never answers fills its pending cap and stops
  absorbing traffic instead of making every client wait for the deadline.
* **Cancellation is drop.** When a client goes away, hyper drops the
  response body or the pending request future, which closes the pooled
  upstream connection, and the engine aborts the request.
* **Membership is a swapped snapshot.** Static replicas and DNS groups feed
  an `ArcSwap` fleet that the hot path reads without locks. DNS groups are
  resolved before the first request and on every `refresh`; members that stay
  keep their state, members that leave finish their in flight streams, and a
  failed lookup keeps the last known set.

## Consequences

* Clients never see a failure that another replica could have absorbed
  before the response started, and never see two answers glued together.
* A crash during a response still breaks that one response. For non streamed
  requests the window is the instant between the engine's head and its body.
* The gateway holds no per token state and buffers nothing but the request
  body and small error bodies, so memory stays flat under slow readers and
  long streams.
* Generation has no side effects inside the engine, so replaying a request
  that never produced output only costs GPU time. Clients that run tools do
  so after receiving output, which a retry can never duplicate.
* TLS to engines and gateway API keys are not supported yet; https replicas
  are refused at startup rather than silently downgraded.
