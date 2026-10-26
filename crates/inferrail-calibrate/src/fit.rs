//! Least squares for `ttft = a*u + b*u*(c + u/2) + d` with every
//! coefficient kept non negative.

/// One measurement: `u` tokens computed while `c` were cached, and the time
/// to first token it took.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub uncached: f64,
    pub cached: f64,
    pub ttft: f64,
}

impl Point {
    fn features(&self) -> [f64; 3] {
        let u = self.uncached;
        [u, u * (self.cached + u / 2.0), 1.0]
    }
}

/// The fitted coefficients and how well they explain the points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coefficients {
    pub token: f64,
    pub pair: f64,
    pub fixed: f64,
    pub r_squared: f64,
}

impl Coefficients {
    pub fn predict(&self, point: &Point) -> f64 {
        let [u, pairs, one] = point.features();
        self.token * u + self.pair * pairs + self.fixed * one
    }
}

/// Solves the normal equations for the features in `use_`, with columns
/// scaled to unit size so the attention term's large values do not swamp
/// the others.
fn least_squares(points: &[Point], use_: [bool; 3]) -> Option<[f64; 3]> {
    let columns: Vec<usize> = (0..3).filter(|&j| use_[j]).collect();
    let k = columns.len();
    let mut scale = [1.0f64; 3];
    for &j in &columns {
        let max = points
            .iter()
            .map(|p| p.features()[j].abs())
            .fold(0.0, f64::max);
        if max == 0.0 {
            return None;
        }
        scale[j] = max;
    }
    let mut a = vec![vec![0.0f64; k + 1]; k];
    for point in points {
        let x = point.features();
        for (r, &i) in columns.iter().enumerate() {
            for (c, &j) in columns.iter().enumerate() {
                a[r][c] += x[i] / scale[i] * x[j] / scale[j];
            }
            a[r][k] += x[i] / scale[i] * point.ttft;
        }
    }
    for col in 0..k {
        let pivot = (col..k).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        let pivot = a[col].clone();
        for (row, cells) in a.iter_mut().enumerate() {
            if row != col {
                let factor = cells[col] / pivot[col];
                for (cell, p) in cells.iter_mut().zip(&pivot).skip(col) {
                    *cell -= factor * p;
                }
            }
        }
    }
    let mut out = [0.0; 3];
    for (r, &j) in columns.iter().enumerate() {
        out[j] = a[r][k] / a[r][r] / scale[j];
    }
    Some(out)
}

fn sse(points: &[Point], coefficients: &[f64; 3]) -> f64 {
    points
        .iter()
        .map(|p| {
            let x = p.features();
            let predicted: f64 = x.iter().zip(coefficients).map(|(x, c)| x * c).sum();
            (predicted - p.ttft).powi(2)
        })
        .sum()
}

/// The best fit with no negative coefficient: every subset of the three
/// terms is solved and the feasible one with the least error wins.
pub fn fit(points: &[Point]) -> Option<Coefficients> {
    let mut best: Option<([f64; 3], f64)> = None;
    for mask in 1u8..8 {
        let use_ = [mask & 1 != 0, mask & 2 != 0, mask & 4 != 0];
        let Some(solution) = least_squares(points, use_) else {
            continue;
        };
        if solution.iter().any(|&c| c < 0.0 || !c.is_finite()) {
            continue;
        }
        let error = sse(points, &solution);
        if best.is_none_or(|(_, e)| error < e) {
            best = Some((solution, error));
        }
    }
    let (solution, error) = best?;
    let mean = points.iter().map(|p| p.ttft).sum::<f64>() / points.len() as f64;
    let total: f64 = points.iter().map(|p| (p.ttft - mean).powi(2)).sum();
    Some(Coefficients {
        token: solution[0],
        pair: solution[1],
        fixed: solution[2],
        r_squared: if total > 0.0 {
            1.0 - error / total
        } else {
            1.0
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn points(a: f64, b: f64, d: f64, noise: impl Fn(usize) -> f64) -> Vec<Point> {
        let mut out = Vec::new();
        for (i, (u, c)) in [0.0, 1024.0, 4096.0, 8192.0]
            .iter()
            .flat_map(|&c| [32.0, 256.0, 1024.0, 4096.0].map(|u| (u, c)))
            .enumerate()
        {
            let ttft = a * u + b * u * (c + u / 2.0) + d;
            out.push(Point {
                uncached: u,
                cached: c,
                ttft: ttft * (1.0 + noise(i)),
            });
        }
        out
    }

    fn within(got: f64, want: f64, tolerance: f64) -> bool {
        (got - want).abs() <= tolerance * want
    }

    #[test]
    fn exact_points_give_back_their_coefficients() {
        let fitted = fit(&points(100e-6, 3.3e-9, 5e-3, |_| 0.0)).unwrap();
        assert!(within(fitted.token, 100e-6, 1e-9), "{fitted:?}");
        assert!(within(fitted.pair, 3.3e-9, 1e-9), "{fitted:?}");
        assert!(within(fitted.fixed, 5e-3, 1e-9), "{fitted:?}");
        assert!(fitted.r_squared > 0.999_999);
    }

    #[test]
    fn negative_coefficients_are_never_returned() {
        let fitted = fit(&points(100e-6, 0.0, 0.0, |i| {
            if i % 2 == 0 { 0.02 } else { -0.02 }
        }))
        .unwrap();
        assert!(fitted.pair >= 0.0 && fitted.fixed >= 0.0, "{fitted:?}");
        assert!(within(fitted.token, 100e-6, 0.05), "{fitted:?}");
    }

    #[test]
    fn too_few_distinct_points_fit_nothing() {
        assert_eq!(fit(&[]), None);
        let flat = [Point {
            uncached: 0.0,
            cached: 0.0,
            ttft: 0.0,
        }];
        assert!(fit(&flat).is_none_or(|c| c.token == 0.0 && c.pair == 0.0));
    }

    proptest! {
        #[test]
        fn small_noise_moves_predictions_little(
            a in 10e-6f64..400e-6,
            b in 0.5e-9f64..20e-9,
            d in 1e-3f64..40e-3,
            seed: u64,
        ) {
            let noise = |i: usize| {
                let x = seed.wrapping_add(i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                ((x >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.002
            };
            let exact = points(a, b, d, |_| 0.0);
            let fitted = fit(&points(a, b, d, noise)).unwrap();
            let scale = exact.iter().map(|p| p.ttft).fold(0.0, f64::max);
            for point in &exact {
                let predicted = fitted.predict(point);
                prop_assert!((predicted - point.ttft).abs() <= 0.005 * scale, "{:?} at {:?}", fitted, point);
            }
            prop_assert!(within(fitted.token, a, 0.05), "{:?}", fitted);
        }
    }
}
