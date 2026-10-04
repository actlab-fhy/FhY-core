//! Terms: syntax trees whose leaves may reference identifiers and whose
//! interior nodes may bind them.
//!
//! - [`AlphaRenaming`] is the correspondence of identifiers between two
//!   terms under comparison: a stack of binder frames over a free renaming.
//! - [`AlphaEquivalence`] compares two terms up to renaming their bound
//!   identifiers, [`FreeIdentifiers`] reports a term's free identifiers, and
//!   [`Term`] joins both with substitution.
//! - [`Binder`] derives all three for a node that binds identifiers over
//!   its scoped children, with substitution that avoids capture.
//! - [`AlphaEquivalence`] is implemented for `Option`, slices, `Vec`, arrays,
//!   `Box`, `Rc`, `Arc` and tuples of terms, which compare their elements in
//!   order under one renaming.
//! - [`is_mapping_alpha_equivalent_under`] compares two maps keyed by
//!   identifiers.
//!
//! A comparison or a scope that runs code another implementation defines
//! can fail, with the trait's associated `Error`. An
//! [`Expression`](crate::expression::Expression) is a term that binds
//! nothing, and cannot fail: its errors are
//! [`Infallible`](std::convert::Infallible).

mod binder;
mod containers;
mod error;
mod mapping;
mod renaming;

pub use binder::{AlphaEquivalence, Binder, FreeIdentifiers, Term};
pub use error::{BinderPairingError, NonInjectiveRenamingError, RenamingPart};
pub use mapping::is_mapping_alpha_equivalent_under;
pub use renaming::{AlphaRenaming, RenamingMap};
