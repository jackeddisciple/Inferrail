use inferrail_calibrate::{Plan, calibrate};
use inferrail_sim::latency::LatencyModel;
use inferrail_sim::server::{Sim, SimConfig};
use tokio::net::TcpListener;

async fn sim(latency: LatencyModel) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("addr"));
    let mut config = SimConfig::default();
    config.engine.latency = latency;
    let sim = Sim::new(config).expect("sim config");
    tokio::spawn(sim.serve(listener, std::future::pending()));
    url
}

fn within(got: f64, want: f64) -> bool {
    (got - want).abs() <= 0.05 * want
}

async fn recovers(latency: LatencyModel, max_context: u32) {
    let plan = Plan {
        url: sim(latency).await,
        max_context,
        repeats: 1,
        ..Plan::default()
    };
    let calibration = calibrate(&plan, |_| {}).await.expect("calibration");
    let fitted = calibration.cost_model;
    assert!(calibration.engine_timed);
    assert_eq!(calibration.model, "inferrail/sim-8b");
    assert_eq!(calibration.block_size, 16);
    assert!(
        within(fitted.prefill_token_secs, latency.prefill_token_secs),
        "{fitted:?}"
    );
    assert!(
        within(fitted.prefill_pair_secs, latency.prefill_pair_secs),
        "{fitted:?}"
    );
    assert!(
        within(fitted.prefill_fixed_secs, latency.prefill_fixed_secs),
        "{fitted:?}"
    );
    assert!(calibration.r_squared > 0.999, "{calibration:?}");
    assert!(calibration.worst_error < 0.05, "{calibration:?}");
    assert!(calibration.overhead_secs.is_some_and(|o| o >= 0.0));
    for probe in &calibration.probes {
        let engine = probe.engine_ttft_secs.expect("engine timing");
        let want = latency.prefill_secs(u64::from(probe.uncached), u64::from(probe.cached));
        assert!((engine - want).abs() < 1e-6, "{probe:?} wanted {want}");
        assert!(probe.client_ttft_secs >= engine, "{probe:?}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_fit_recovers_the_default_simulator_within_five_percent() {
    recovers(LatencyModel::GPU_24GB_8B, 6_000).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_fit_recovers_other_parameters_within_five_percent() {
    recovers(
        LatencyModel {
            prefill_token_secs: 30e-6,
            prefill_pair_secs: 8e-9,
            prefill_fixed_secs: 12e-3,
            decode_token_secs: 5e-3,
        },
        10_000,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_second_run_on_the_same_engine_measures_it_again() {
    let latency = LatencyModel::GPU_24GB_8B;
    let plan = Plan {
        url: sim(latency).await,
        max_context: 6_000,
        repeats: 1,
        ..Plan::default()
    };
    let first = calibrate(&plan, |_| {}).await.expect("a first run");
    let second = calibrate(&plan, |_| {}).await.expect("a second run");
    assert_ne!(first.seed, second.seed);
    for fitted in [first.cost_model, second.cost_model] {
        assert!(
            within(fitted.prefill_token_secs, latency.prefill_token_secs)
                && within(fitted.prefill_pair_secs, latency.prefill_pair_secs)
                && within(fitted.prefill_fixed_secs, latency.prefill_fixed_secs),
            "{fitted:?}"
        );
    }
}

#[tokio::test]
async fn an_engine_without_models_or_answers_is_an_error() {
    let plan = Plan {
        url: "http://127.0.0.1:9".to_owned(),
        timeout: std::time::Duration::from_secs(2),
        ..Plan::default()
    };
    let err = calibrate(&plan, |_| {}).await.unwrap_err();
    assert!(err.to_string().contains("cannot reach"), "{err}");
}
