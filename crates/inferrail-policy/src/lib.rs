//! Routing policy trait, baseline policies and the TTFT cost model.

use std::fmt;

use inferrail_core::config::PolicyKind;
use inferrail_core::ids::{ModelId, ReplicaId};

mod random;
mod rng;
mod round_robin;

pub use random::Random;
pub use rng::SplitMix;
pub use round_robin::RoundRobin;

/// A replica the gateway already found eligible: it serves the model, is
/// healthy and has room under its caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate<'a> {
    pub id: &'a ReplicaId,
    pub in_flight: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct RouteContext<'a> {
    pub model: &'a ModelId,
    pub candidates: &'a [Candidate<'a>],
}

pub trait RoutingPolicy: Send + Sync + fmt::Debug {
    fn kind(&self) -> PolicyKind;

    /// Picks an index into `ctx.candidates`; `None` only when it is empty.
    fn choose(&self, ctx: &RouteContext<'_>) -> Option<usize>;
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    pub(crate) struct Pool {
        pub(crate) model: ModelId,
        pub(crate) ids: Vec<ReplicaId>,
    }

    impl Pool {
        pub(crate) fn new(n: usize) -> Self {
            Self {
                model: ModelId::new("m").unwrap(),
                ids: (0..n)
                    .map(|i| ReplicaId::new(format!("r{i}")).unwrap())
                    .collect(),
            }
        }

        pub(crate) fn candidates(&self, loads: &[u32]) -> Vec<Candidate<'_>> {
            self.ids
                .iter()
                .zip(loads.iter().copied().chain(std::iter::repeat(0)))
                .map(|(id, in_flight)| Candidate { id, in_flight })
                .collect()
        }

        pub(crate) fn pick(
            &self,
            policy: &dyn RoutingPolicy,
            candidates: &[Candidate<'_>],
        ) -> usize {
            policy
                .choose(&RouteContext {
                    model: &self.model,
                    candidates,
                })
                .expect("a candidate")
        }
    }

    pub(crate) fn chi_square(counts: &[u64]) -> f64 {
        let total: u64 = counts.iter().sum();
        let expected = total as f64 / counts.len() as f64;
        counts
            .iter()
            .map(|&c| (c as f64 - expected).powi(2) / expected)
            .sum()
    }
}
