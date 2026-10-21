//! Shared types, OpenAI wire schema, errors and configuration for Inferrail.

pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name_matches_package() {
        assert_eq!(super::CRATE_NAME, "inferrail-core");
    }
}
