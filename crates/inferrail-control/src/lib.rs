//! Replica discovery, health, admission, rate limits and the admin API.

pub mod checker;
pub mod discovery;
pub mod fleet;
pub mod health;
pub mod replica;

pub use checker::{CheckerConfig, Probe, run_checker};
pub use discovery::{Resolution, Resolve, SystemResolver, run_dns};
pub use fleet::{Fleet, MembershipChange, Registry};
pub use health::{Clock, Ejection, Health, HealthPolicy, Nanos};
pub use replica::{Cards, Endpoint, Lease, Limits, Origin, Replica, ReplicaSpec};
