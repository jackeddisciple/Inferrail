//! Routing policy trait, baseline policies and the TTFT cost model.

pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name_matches_package() {
        assert_eq!(super::CRATE_NAME, "inferrail-policy");
    }
}
