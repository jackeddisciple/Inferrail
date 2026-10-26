//! Load guards: limits a replica must be within before the cost model may
//! pick it, however cheap its estimate.

use crate::Candidate;

/// Bounded load, after consistent hashing with bounded loads: the most
/// requests any candidate may hold once this one is placed,
/// `ceil((1 + epsilon) * (in_flight + 1) / n)` over all `n` candidates.
/// Some candidate is always below it, since they cannot all be above the
/// average.
pub fn load_cap(candidates: &[Candidate<'_>], epsilon: f64) -> u64 {
    let total: u64 = candidates
        .iter()
        .map(|c| u64::from(c.in_flight))
        .sum::<u64>()
        + 1;
    let share = (1.0 + epsilon) * total as f64 / candidates.len().max(1) as f64;
    share.ceil() as u64
}

pub(crate) fn within_load(candidate: &Candidate<'_>, cap: u64) -> bool {
    u64::from(candidate.in_flight) < cap
}

/// KV pressure: a replica whose engine reports more than `kappa` of its KV
/// cache in use takes no prompt with more than `short` tokens to compute,
/// since admitting one would preempt running requests.
pub(crate) fn under_pressure(
    candidate: &Candidate<'_>,
    kappa: f64,
    uncached: u32,
    short: u32,
) -> bool {
    uncached > short && candidate.kv_usage.is_some_and(|usage| usage > kappa)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferrail_core::ids::ReplicaId;
    use proptest::prelude::*;

    fn ids(n: usize) -> Vec<ReplicaId> {
        (0..n)
            .map(|i| ReplicaId::new(format!("r{i}")).unwrap())
            .collect()
    }

    #[test]
    fn the_cap_is_the_rounded_up_share_after_placing_the_request() {
        let ids = ids(4);
        let fleet: Vec<Candidate<'_>> = ids
            .iter()
            .zip([3, 3, 3, 2])
            .map(|(id, load)| Candidate::new(id, load))
            .collect();
        assert_eq!(load_cap(&fleet, 0.25), 4);
        assert!(fleet.iter().all(|c| within_load(c, 4)));
        assert_eq!(load_cap(&fleet, 0.01), 4);
        assert_eq!(load_cap(&fleet[..1], 0.25), 5);
        assert_eq!(load_cap(&[], 0.25), 2);
    }

    #[test]
    fn pressure_needs_a_full_cache_and_a_long_prompt() {
        let id = ReplicaId::new("r").unwrap();
        let full = Candidate {
            kv_usage: Some(0.95),
            ..Candidate::new(&id, 0)
        };
        assert!(under_pressure(&full, 0.92, 257, 256));
        assert!(!under_pressure(&full, 0.92, 256, 256));
        assert!(!under_pressure(&full, 0.95, 4_000, 256));
        let unknown = Candidate::new(&id, 0);
        assert!(!under_pressure(&unknown, 0.92, 4_000, 256));
    }

    proptest! {
        #[test]
        fn some_candidate_is_always_within_the_cap(
            loads in prop::collection::vec(0u32..1_000, 1..64),
            epsilon in 0.001f64..10.0,
        ) {
            let ids = ids(loads.len());
            let fleet: Vec<Candidate<'_>> = ids
                .iter()
                .zip(&loads)
                .map(|(id, &load)| Candidate::new(id, load))
                .collect();
            let cap = load_cap(&fleet, epsilon);
            prop_assert!(fleet.iter().any(|c| within_load(c, cap)));
        }
    }
}
