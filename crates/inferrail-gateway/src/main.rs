//! OpenAI compatible gateway that routes requests across engine replicas.

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
        assert!(super::banner().starts_with("inferrail-gateway "));
    }
}
