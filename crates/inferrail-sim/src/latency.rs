use std::time::Duration;

/// Step timing with the same shape as the router's cost model. A prompt of
/// `u` uncached tokens attending to `c` cached ones costs
/// `P(u, c) = a*u + b*u*(c + u/2) + d` however it is cut into chunks, `d`
/// once per prefill. A step that decodes `n` sequences costs
/// `decode_token_secs + decode_seq_secs * (n - 1)` on top of its prefill
/// chunks, so prefills and decodes sharing a step slow each other down.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyModel {
    pub prefill_token_secs: f64,
    pub prefill_pair_secs: f64,
    pub prefill_fixed_secs: f64,
    pub decode_token_secs: f64,
    pub decode_seq_secs: f64,
}

impl LatencyModel {
    pub const ZERO: Self = Self {
        prefill_token_secs: 0.0,
        prefill_pair_secs: 0.0,
        prefill_fixed_secs: 0.0,
        decode_token_secs: 0.0,
        decode_seq_secs: 0.0,
    };

    /// Roughly an 8B model on one 24 GB GPU: 10k prefill tokens/s, 20 ms TPOT
    /// alone and 0.1 ms more for every other sequence in the batch.
    pub const GPU_24GB_8B: Self = Self {
        prefill_token_secs: 100e-6,
        prefill_pair_secs: 3.3e-9,
        prefill_fixed_secs: 5e-3,
        decode_token_secs: 20e-3,
        decode_seq_secs: 100e-6,
    };

    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("prefill_token_secs", self.prefill_token_secs),
            ("prefill_pair_secs", self.prefill_pair_secs),
            ("prefill_fixed_secs", self.prefill_fixed_secs),
            ("decode_token_secs", self.decode_token_secs),
            ("decode_seq_secs", self.decode_seq_secs),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!(
                    "{name} must be finite and non negative, got {value}"
                ));
            }
        }
        Ok(())
    }

    pub fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    pub fn prefill_secs(&self, uncached: u64, cached: u64) -> f64 {
        self.chunk_secs(uncached, cached) + self.prefill_fixed_secs
    }

    pub fn prefill(&self, uncached: u64, cached: u64) -> Duration {
        seconds(self.prefill_secs(uncached, cached))
    }

    /// One chunk of `tokens` prompt tokens after `context` already computed,
    /// without the fixed cost; chunks of one prompt add up to its prefill.
    pub fn chunk_secs(&self, tokens: u64, context: u64) -> f64 {
        let u = tokens as f64;
        let c = context as f64;
        self.prefill_token_secs * u + self.prefill_pair_secs * u * (c + u / 2.0)
    }

    /// One step's decode work for `sequences` decoding together.
    pub fn decode_secs(&self, sequences: u64) -> f64 {
        match sequences {
            0 => 0.0,
            n => self.decode_token_secs + self.decode_seq_secs * (n - 1) as f64,
        }
    }

    pub fn decode_step(&self) -> Duration {
        seconds(self.decode_token_secs)
    }
}

pub(crate) fn duration(secs: f64) -> Duration {
    seconds(secs)
}

impl Default for LatencyModel {
    fn default() -> Self {
        Self::GPU_24GB_8B
    }
}

fn seconds(value: f64) -> Duration {
    Duration::try_from_secs_f64(value).unwrap_or(Duration::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn default_matches_a_24gb_gpu_serving_an_8b_model() {
        let model = LatencyModel::default();
        let cold = model.prefill(6_000, 0).as_secs_f64();
        let warm = model.prefill(2_000, 4_000).as_secs_f64();
        assert!((0.6..0.7).contains(&cold), "{cold}");
        assert!(warm < cold / 2.0, "{warm}");
        assert_eq!(model.decode_step(), Duration::from_millis(20));
        assert_eq!(model.decode_secs(0), 0.0);
        assert!((model.decode_secs(65) - 26.4e-3).abs() < 1e-12);
    }

    #[test]
    fn chunks_add_up_to_the_whole_prefill() {
        let model = LatencyModel::default();
        let (u, c) = (7_000u64, 1_500u64);
        let whole = model.prefill_secs(u, c);
        let mut done = 0;
        let mut chunked = model.prefill_fixed_secs;
        while done < u {
            let n = (u - done).min(2_048);
            chunked += model.chunk_secs(n, c + done);
            done += n;
        }
        assert!((whole - chunked).abs() < 1e-12, "{whole} vs {chunked}");
    }

    #[test]
    fn zero_model_is_free() {
        assert!(LatencyModel::ZERO.is_zero());
        assert_eq!(LatencyModel::ZERO.prefill(100_000, 100_000), Duration::ZERO);
        assert_eq!(LatencyModel::ZERO.decode_step(), Duration::ZERO);
    }

    #[test]
    fn rejects_negative_or_nan_parameters() {
        let mut model = LatencyModel {
            prefill_pair_secs: -1.0,
            ..LatencyModel::default()
        };
        assert!(model.validate().is_err());
        model.prefill_pair_secs = f64::NAN;
        assert!(model.validate().is_err());
        model.prefill_pair_secs = 0.0;
        model.decode_seq_secs = -1e-6;
        assert!(model.validate().is_err());
        assert!(LatencyModel::default().validate().is_ok());
    }

    #[test]
    fn absurd_values_saturate_instead_of_panicking() {
        let model = LatencyModel {
            prefill_token_secs: 1e300,
            ..LatencyModel::ZERO
        };
        assert_eq!(model.prefill(u64::MAX, 0), Duration::MAX);
    }

    proptest! {
        #[test]
        fn more_uncached_tokens_never_prefill_faster(u in 0u64..200_000, extra in 1u64..10_000, c in 0u64..200_000) {
            let model = LatencyModel::default();
            prop_assert!(model.prefill(u + extra, c) > model.prefill(u, c));
        }

        #[test]
        fn longer_cached_context_never_prefills_faster(u in 1u64..50_000, c in 0u64..200_000, extra in 1u64..10_000) {
            let model = LatencyModel::default();
            prop_assert!(model.prefill(u, c + extra) >= model.prefill(u, c));
        }

        #[test]
        fn a_cache_hit_is_never_slower_than_recompute(len in 1u64..100_000, hit in 0u64..100_000) {
            let model = LatencyModel::default();
            let hit = hit.min(len - 1);
            prop_assert!(model.prefill(len - hit, hit) <= model.prefill(len, 0));
        }
    }
}
