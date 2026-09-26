//! `PyO3` bindings for [`fhy_core::term`] (S10 of
//! `docs/design/python-switch.md`).
//!
//! - `renaming.rs`: `_rs.AlphaRenaming`, the Rust renaming with the Python
//!   identifier objects it was built from (D-S10-5).
//! - `adapter.rs`: the Python terms and binders as implementations of the
//!   core's traits, and the context they report errors to (D-S10-7).
//! - `binder.rs`: the functions `BinderMixin`'s derived methods call.
//! - `derived.rs`: the roles, the plans and the walks of
//!   `DerivedEquivalenceMixin` (D-S10-8, D-S10-9).
//! - `mapping.rs`: `is_identifier_mapping_alpha_equivalent_under`
//!   (D-S10-11).
//!
//! These call a node's own Python hooks, its children's methods, its
//! dataclass fields and user comparators per node, the term package's
//! exception to P3's granularity rule (N-S10-1), and compare the values
//! that are Rust-backed without calling Python.

mod adapter;
mod binder;
mod derived;
mod mapping;
mod renaming;

pub(crate) use binder::{
    binder_get_free_identifiers, binder_is_alpha_equivalent_under, binder_substitute,
};
pub(crate) use derived::{
    PyEquivalenceRole, derived_is_alpha_equivalent_under, derived_is_structurally_equivalent,
};
pub(crate) use mapping::is_identifier_mapping_alpha_equivalent_under;
pub(crate) use renaming::{PyAlphaRenaming, read_renaming};
