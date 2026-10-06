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

use std::any::TypeId;
use std::error::Error;
use std::fmt;

use crate::foreign::{BoxError, Foreign, ForeignError, Part, Resolve};
use crate::identifier::Identifier;
use crate::term::AlphaRenaming;

use super::alternative::{Alternative, PlainAlternative};
use super::choice::first_repeat;
use super::equivalence::{alternative_labels, enter_frame};
use super::error::EquivalenceError;
use super::variable::{PlainVariable, Variable};

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
        self.clause
    }

    /// Return the kind of the sample that breaks it.
    #[must_use]
    pub fn kind(&self) -> &str {
        &self.kind
    }
}

impl fmt::Display for ConformanceViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let number = match self.clause {
            ContractClause::StableGetters => 1,
            ContractClause::UniqueKind => 2,
            ContractClause::DistinctBoundIdentifiers => 3,
            ContractClause::EquivalenceHooks => 4,
            ContractClause::WireForm => 6,
        };
        write!(
            f,
            "the implementation of kind `{}` breaks clause {number}: {}",
            self.kind, self.message
        )
    }
}

impl Error for ConformanceViolation {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn Error + 'static))
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
    check_samples(samples, |foreign| resolver.resolve(foreign))
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
    check_samples(samples, |foreign| resolver.resolve(foreign))
}

/// What the checks read of a sample, a variable or an alternative.
trait Sample: Sized {
    /// The kind of this module's own implementation.
    const PLAIN_KIND: &'static str;

    /// Return the sample's kind.
    fn kind(&self) -> String;

    /// Return the id of the sample's implementing type.
    fn type_id(&self) -> TypeId;

    /// Return the getter that answers differently on a second call, if
    /// one does.
    fn find_unstable_getter(&self) -> Option<&'static str>;

    /// Return the sample's bound identifiers.
    fn bound_identifiers(&self) -> Result<Vec<Identifier>, BoxError>;

    /// Return the names the sample binds compared on its own, in order.
    fn labels(&self) -> Result<Vec<Identifier>, BoxError>;

    /// Return the structural hook's answer for `other`.
    fn structural_hook(&self, other: &Self) -> Result<bool, BoxError>;

    /// Return the alpha hook's answer for `other` under `renaming`.
    fn alpha_hook(&self, other: &Self, renaming: &AlphaRenaming) -> Result<bool, BoxError>;

    /// Return the sample's foreign part.
    fn to_foreign(&self) -> Result<Foreign, ForeignError>;

    /// Return whether `other` is structurally equivalent, as the core
    /// compares them.
    fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError>;
}

impl Sample for Part<dyn Variable> {
    const PLAIN_KIND: &'static str = PlainVariable::KIND;

    fn kind(&self) -> String {
        self.get().kind().into_owned()
    }

    fn type_id(&self) -> TypeId {
        self.get().as_any().type_id()
    }

    fn find_unstable_getter(&self) -> Option<&'static str> {
        let part = self.get();
        if part.name() != part.name() {
            return Some("name");
        }
        if part.kind() != part.kind() {
            return Some("kind");
        }
        if part.param() != part.param() {
            return Some("param");
        }
        if part.notes() != part.notes() {
            return Some("notes");
        }
        None
    }

    fn bound_identifiers(&self) -> Result<Vec<Identifier>, BoxError> {
        Ok(Vec::new())
    }

    fn labels(&self) -> Result<Vec<Identifier>, BoxError> {
        Ok(vec![self.get().name().clone()])
    }

    fn structural_hook(&self, other: &Self) -> Result<bool, BoxError> {
        self.get().is_extension_structurally_equivalent(other.get())
    }

    fn alpha_hook(&self, other: &Self, renaming: &AlphaRenaming) -> Result<bool, BoxError> {
        self.get()
            .is_extension_alpha_equivalent_under(other.get(), renaming)
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        self.get().to_foreign()
    }

    fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError> {
        Self::is_structurally_equivalent(self, other)
    }
}

impl Sample for Part<dyn Alternative> {
    const PLAIN_KIND: &'static str = PlainAlternative::KIND;

    fn kind(&self) -> String {
        self.get().kind().into_owned()
    }

    fn type_id(&self) -> TypeId {
        self.get().as_any().type_id()
    }

    fn find_unstable_getter(&self) -> Option<&'static str> {
        let part = self.get();
        if part.name() != part.name() {
            return Some("name");
        }
        if part.kind() != part.kind() {
            return Some("kind");
        }
        if part.variables() != part.variables() {
            return Some("variables");
        }
        if part.choices() != part.choices() {
            return Some("choices");
        }
        if part.notes() != part.notes() {
            return Some("notes");
        }
        match (part.bound_identifiers(), part.bound_identifiers()) {
            (Ok(first), Ok(second)) if first != second => Some("bound_identifiers"),
            _ => None,
        }
    }

    fn bound_identifiers(&self) -> Result<Vec<Identifier>, BoxError> {
        self.get().bound_identifiers()
    }

    fn labels(&self) -> Result<Vec<Identifier>, BoxError> {
        alternative_labels(self.get())
    }

    fn structural_hook(&self, other: &Self) -> Result<bool, BoxError> {
        self.get().is_extension_structurally_equivalent(other.get())
    }

    fn alpha_hook(&self, other: &Self, renaming: &AlphaRenaming) -> Result<bool, BoxError> {
        self.get()
            .is_extension_alpha_equivalent_under(other.get(), renaming)
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        self.get().to_foreign()
    }

    fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError> {
        Self::is_structurally_equivalent(self, other)
    }
}

/// Return the violation of `clause` by the sample of `kind`.
fn violation(
    clause: ContractClause,
    kind: &str,
    message: impl Into<String>,
) -> ConformanceViolation {
    ConformanceViolation {
        clause,
        kind: kind.to_owned(),
        message: message.into(),
        source: None,
    }
}

/// Return the violation of `clause` by the sample of `kind` whose hook
/// failed with `source`.
fn failure(
    clause: ContractClause,
    kind: &str,
    message: impl Into<String>,
    source: impl Into<BoxError>,
) -> ConformanceViolation {
    ConformanceViolation {
        source: Some(source.into()),
        ..violation(clause, kind, message)
    }
}

/// Return the mapping of a hook's failure with `source` to a violation of
/// the hooks' clause by the sample of `kind`.
fn hook_failed<'a>(
    kind: &'a str,
    message: &'static str,
) -> impl FnOnce(BoxError) -> ConformanceViolation + 'a {
    move |source| failure(ContractClause::EquivalenceHooks, kind, message, source)
}

/// Return the mapping of a wire form's failure with `source` to a
/// violation of the wire-form clause by the sample of `kind`.
fn wire_failed<'a, E: Into<BoxError>>(
    kind: &'a str,
    message: &'static str,
) -> impl FnOnce(E) -> ConformanceViolation + 'a {
    move |source| failure(ContractClause::WireForm, kind, message, source)
}

/// Check `samples` against the contract, reading their foreign parts back
/// with `resolve`.
fn check_samples<T: Sample>(
    samples: &[T],
    resolve: impl Fn(&Foreign) -> Result<T, ForeignError>,
) -> Result<(), ConformanceViolation> {
    for (position, sample) in samples.iter().enumerate() {
        check_sample(sample, &samples[..position], &resolve)?;
    }
    for (position, left) in samples.iter().enumerate() {
        for right in &samples[position + 1..] {
            if left.kind() == right.kind() {
                check_pair(left, right)?;
            }
        }
    }
    Ok(())
}

/// Check the clauses that `sample` keeps on its own, and its kind against
/// the `earlier` samples'.
fn check_sample<T: Sample>(
    sample: &T,
    earlier: &[T],
    resolve: &impl Fn(&Foreign) -> Result<T, ForeignError>,
) -> Result<(), ConformanceViolation> {
    let kind = sample.kind();
    if let Some(getter) = sample.find_unstable_getter() {
        return Err(violation(
            ContractClause::StableGetters,
            &kind,
            format!("`{getter}` answers differently on a second call"),
        ));
    }
    check_kind(sample, &kind, earlier)?;
    let labels = check_bound_identifiers(sample, &kind)?;
    check_reflexive_hooks(sample, &kind, &labels)?;
    check_wire_form(sample, &kind, resolve)
}

/// Check that `kind`, `sample`'s, is not empty, not a plain kind, and the
/// kind of the `earlier` samples of its type only.
fn check_kind<T: Sample>(
    sample: &T,
    kind: &str,
    earlier: &[T],
) -> Result<(), ConformanceViolation> {
    if kind.is_empty() {
        return Err(violation(
            ContractClause::UniqueKind,
            kind,
            "the kind is empty",
        ));
    }
    if kind == T::PLAIN_KIND {
        return Err(violation(
            ContractClause::UniqueKind,
            kind,
            "the kind is this module's plain kind",
        ));
    }
    for other in earlier {
        let is_same_type = other.type_id() == sample.type_id();
        let is_same_kind = other.kind() == kind;
        if is_same_type && !is_same_kind {
            return Err(violation(
                ContractClause::UniqueKind,
                kind,
                format!("one type answers the kinds `{}` and `{kind}`", other.kind()),
            ));
        }
        if !is_same_type && is_same_kind {
            return Err(violation(
                ContractClause::UniqueKind,
                kind,
                "two types answer this kind",
            ));
        }
    }
    Ok(())
}

/// Check that `sample`'s bound identifiers are distinct, and return the
/// names it binds compared on its own.
fn check_bound_identifiers<T: Sample>(
    sample: &T,
    kind: &str,
) -> Result<Vec<Identifier>, ConformanceViolation> {
    let bound_failed = |source| {
        failure(
            ContractClause::DistinctBoundIdentifiers,
            kind,
            "the bound identifiers fail",
            source,
        )
    };
    let bound = sample.bound_identifiers().map_err(bound_failed)?;
    if let Some(repeated) = first_repeat(&bound) {
        return Err(violation(
            ContractClause::DistinctBoundIdentifiers,
            kind,
            format!("the bound identifiers repeat {repeated:?}"),
        ));
    }
    sample.labels().map_err(bound_failed)
}

/// Check that the hooks accept `sample` and itself, the alpha hook under
/// the frame pairing its `labels` with themselves.
fn check_reflexive_hooks<T: Sample>(
    sample: &T,
    kind: &str,
    labels: &[Identifier],
) -> Result<(), ConformanceViolation> {
    if !sample
        .structural_hook(sample)
        .map_err(hook_failed(kind, "the structural hook fails"))?
    {
        return Err(violation(
            ContractClause::EquivalenceHooks,
            kind,
            "the structural hook refuses a sample and itself",
        ));
    }
    let frame = enter_frame(&AlphaRenaming::default(), labels, labels).ok_or_else(|| {
        violation(
            ContractClause::DistinctBoundIdentifiers,
            kind,
            "the names the sample binds repeat",
        )
    })?;
    if !sample
        .alpha_hook(sample, &frame)
        .map_err(hook_failed(kind, "the alpha hook fails"))?
    {
        return Err(violation(
            ContractClause::EquivalenceHooks,
            kind,
            "the alpha hook refuses a sample and itself",
        ));
    }
    Ok(())
}

/// Check that `sample`'s foreign part carries its kind and reads back, by
/// `resolve`, into a structurally equivalent value.
fn check_wire_form<T: Sample>(
    sample: &T,
    kind: &str,
    resolve: &impl Fn(&Foreign) -> Result<T, ForeignError>,
) -> Result<(), ConformanceViolation> {
    let foreign = sample
        .to_foreign()
        .map_err(wire_failed(kind, "`to_foreign` fails"))?;
    if foreign.type_id() != kind {
        return Err(violation(
            ContractClause::WireForm,
            kind,
            format!("`to_foreign` writes the type id `{}`", foreign.type_id()),
        ));
    }
    let resolved = resolve(&foreign).map_err(wire_failed(kind, "the resolver refuses the part"))?;
    let is_equivalent = resolved.kind() == kind
        && sample
            .is_structurally_equivalent(&resolved)
            .map_err(wire_failed(kind, "comparing the part read back fails"))?;
    if !is_equivalent {
        return Err(violation(
            ContractClause::WireForm,
            kind,
            "the part read back is not structurally equivalent to the sample",
        ));
    }
    Ok(())
}

/// Check that the hooks answer `left` and `right`, two samples of one
/// kind, as equivalence relations do.
fn check_pair<T: Sample>(left: &T, right: &T) -> Result<(), ConformanceViolation> {
    let kind = left.kind();
    let forward = left
        .structural_hook(right)
        .map_err(hook_failed(&kind, "the structural hook fails"))?;
    let backward = right
        .structural_hook(left)
        .map_err(hook_failed(&kind, "the structural hook fails"))?;
    if forward != backward {
        return Err(violation(
            ContractClause::EquivalenceHooks,
            &kind,
            "the structural hook answers differently in the two directions",
        ));
    }
    if forward
        && !left
            .alpha_hook(right, &AlphaRenaming::default())
            .map_err(hook_failed(&kind, "the alpha hook fails"))?
    {
        return Err(violation(
            ContractClause::EquivalenceHooks,
            &kind,
            "the structural hook accepts what the alpha hook refuses under the empty renaming",
        ));
    }
    let left_labels = left
        .labels()
        .map_err(hook_failed(&kind, "the bound identifiers fail"))?;
    let right_labels = right
        .labels()
        .map_err(hook_failed(&kind, "the bound identifiers fail"))?;
    let renaming = AlphaRenaming::default();
    if let (Some(forward_frame), Some(backward_frame)) = (
        enter_frame(&renaming, &left_labels, &right_labels),
        enter_frame(&renaming, &right_labels, &left_labels),
    ) {
        let forward = left
            .alpha_hook(right, &forward_frame)
            .map_err(hook_failed(&kind, "the alpha hook fails"))?;
        let backward = right
            .alpha_hook(left, &backward_frame)
            .map_err(hook_failed(&kind, "the alpha hook fails"))?;
        if forward != backward {
            return Err(violation(
                ContractClause::EquivalenceHooks,
                &kind,
                "the alpha hook answers differently in the two directions",
            ));
        }
    }
    Ok(())
}
