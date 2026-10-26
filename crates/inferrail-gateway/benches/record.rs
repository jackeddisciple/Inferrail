use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use hyper::StatusCode;
use inferrail_control::{Clock, Limits, Origin, Registry, ReplicaSpec};
use inferrail_core::config::PolicyKind;
use inferrail_core::ids::{ModelId, ReplicaId};
use inferrail_gateway::metrics::{FleetGauges, Metrics, Outcome, Routed};

fn bench(c: &mut Criterion) {
    let specs = (0..8)
        .map(|i| ReplicaSpec {
            id: ReplicaId::new(format!("r{i}")).expect("id"),
            url: format!("http://10.0.0.{i}:8000").parse().expect("url"),
            models: vec![ModelId::new("m").expect("model")],
            origin: Origin::Static,
            limits: Limits::default(),
            kv_capacity_blocks: None,
        })
        .collect();
    let fleet = Arc::new(Registry::new(specs));
    let metrics = Metrics::new(FleetGauges {
        registry: Arc::clone(&fleet),
        clock: Clock::default(),
        max_age: Duration::from_millis(250),
        policies: Vec::new(),
        index: None,
        tokenizers: Vec::new(),
    });
    let routed: Vec<Routed> = fleet
        .load()
        .replicas()
        .iter()
        .map(|r| Routed {
            model: ModelId::new("m").expect("model"),
            policy: PolicyKind::P2c,
            replica: r.id().clone(),
        })
        .collect();
    let mut i = 0;
    c.bench_function("record/one_streamed_request", |b| {
        b.iter(|| {
            i = (i + 1) % routed.len();
            let r = black_box(&routed[i]);
            metrics.decided(r, Duration::from_micros(2));
            metrics.first_byte(r, Duration::from_millis(30));
            metrics.finished(
                r,
                Outcome::Complete(StatusCode::OK),
                Duration::from_millis(400),
            );
        })
    });
    c.bench_function("record/render_eight_replicas", |b| {
        b.iter(|| black_box(metrics.render().len()))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
