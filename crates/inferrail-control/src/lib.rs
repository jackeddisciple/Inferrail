//! Replica discovery, health, admission, rate limits and the admin API.

pub mod fleet;
pub mod replica;

pub use fleet::{Fleet, MembershipChange, Registry};
pub use replica::{Endpoint, Lease, Limits, Origin, Replica, ReplicaSpec};
