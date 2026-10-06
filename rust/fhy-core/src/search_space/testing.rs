//! Checks that an implementation of [`Variable`] or [`Alternative`] keeps
//! the [contract](super#implementing-variable-and-alternative), for the
//! implementing crate's tests.
//!
//! Test-only, not stable API: it is behind the `testing` feature, and may
//! change in any release.
//!
//! Each check takes sample values of one or more implementing types and a
//! resolver that reads their foreign parts back, and checks:
//!
//! - clause 1, [`ContractClause::StableGetters`]: every getter answers the
//!   same on a second call;
//! - clause 2, [`ContractClause::UniqueKind`]: the kind is not empty, is
//!   not a plain kind, is the same for every sample of one type and
//!   different for samples of different types;
//! - clause 3, [`ContractClause::DistinctBoundIdentifiers`]: an
//!   alternative's bound identifiers are distinct;
//! - clause 4, [`ContractClause::EquivalenceHooks`]: for every pair of
//!   samples of one kind, each hook answers `Ok`, answers `true` for a
//!   sample and itself, answers the same in both directions (the alpha
//!   hook under a renaming pairing the two samples' names, in each
//!   direction), and the alpha hook answers `true` under the empty
//!   renaming where the structural hook does;
//! - clause 6, [`ContractClause::WireForm`]: `to_foreign` gives a part
//!   whose type id is the kind, which the resolver builds back into a
//!   value of the kind structurally equivalent to the sample.
//!
//! Clause 5, that every identifier the data binds is declared, cannot be
//! checked from outside the implementation.

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use std::error::Error;
use std::fmt;

use crate::foreign::{BoxError, Part, Resolve};

use super::alternative::Alternative;
use super::variable::Variable;

/// A clause of the implementor contract, numbered as the
/// [module documentation](super#implementing-variable-and-alternative)
/// numbers them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ContractClause {
    /// Clause 1: every getter answers the same for the value's life.
    StableGetters,
    /// Clause 2: the kind is unique to the type and stable.
    UniqueKind,
    /// Clause 3: the bound identifiers are distinct and deterministic.
    DistinctBoundIdentifiers,
    /// Clause 4: the hooks are equivalences, the structural one implying
    /// the alpha one, and fail with `Err`.
    EquivalenceHooks,
    /// Clause 6: `to_foreign` round-trips through the resolver.
    WireForm,
}

/// A sample that breaks a clause of the implementor contract.
///
/// Displays as ``the implementation of kind `k` breaks clause N: ``
/// followed by what the check saw.
#[derive(Debug)]
pub struct ConformanceViolation {
    clause: ContractClause,
    kind: String,
    message: String,
    source: Option<BoxError>,
}

impl ConformanceViolation {
    /// Return the clause the sample breaks.
    #[must_use]
    pub fn clause(&self) -> ContractClause {
        todo!()
    }

    /// Return the kind of the sample that breaks it.
    #[must_use]
    pub fn kind(&self) -> &str {
        todo!()
    }
}

impl fmt::Display for ConformanceViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

impl Error for ConformanceViolation {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        todo!()
    }
}

/// Check that the variables `samples` keep the implementor contract, their
/// foreign parts read back by `resolver`.
///
/// Give at least two samples of each type, some equal and some different
/// in the implementation's own data, so that the pairwise checks have
/// something to compare.
///
/// # Errors
///
/// Returns the first violation found: the samples in order, each clause in
/// order, then each pair of samples in order.
pub fn check_variable_conformance<R: Resolve<Part<dyn Variable>> + ?Sized>(
    samples: &[Part<dyn Variable>],
    resolver: &R,
) -> Result<(), ConformanceViolation> {
    todo!()
}

/// Check that the alternatives `samples` keep the implementor contract,
/// their foreign parts read back by `resolver`.
///
/// Give at least two samples of each type, some equal and some different
/// in the implementation's own data, so that the pairwise checks have
/// something to compare.
///
/// # Errors
///
/// Returns the first violation found: the samples in order, each clause in
/// order, then each pair of samples in order.
pub fn check_alternative_conformance<R: Resolve<Part<dyn Alternative>> + ?Sized>(
    samples: &[Part<dyn Alternative>],
    resolver: &R,
) -> Result<(), ConformanceViolation> {
    todo!()
}
