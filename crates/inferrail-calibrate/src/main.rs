//! Profiles an engine and fits the TTFT cost model parameters.

fn banner() -> String {
    format!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
}

fn main() {
    println!("{}", banner());
}

#[cfg(test)]
mod tests {
    #[test]
    fn banner_names_the_binary() {
        assert!(super::banner().starts_with("inferrail-calibrate "));
    }
}
