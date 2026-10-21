//! Replica discovery, health, admission, rate limits and the admin API.

pub mod checker;
pub mod fleet;
pub mod health;
pub mod replica;

pub use checker::{CheckerConfig, Probe, run_checker};
pub use fleet::{Fleet, MembershipChange, Registry};
pub use health::{Clock, Ejection, Health, HealthPolicy, Nanos};
pub use replica::{Endpoint, Lease, Limits, Origin, Replica, ReplicaSpec};
