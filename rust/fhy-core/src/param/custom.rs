//! [`CustomDomain`]: a domain of a kind this module does not define, such
//! as one a language binding defines.

use std::hash::Hasher;

use crate::constraint::{Constraint, Outcome, Value};
use crate::expression::SymbolType;
use crate::foreign::{BoxError, ForeignPart, impl_part, is_same_part};
use crate::identifier::Identifier;

use super::context::ParamContext;
use super::domain::{IntervalProfile, ParamDomain, Side};

/// A domain of a kind this module does not define.
///
/// It answers what the built-in kinds answer, each through its own hook,
/// and the procedures of this module call the hooks as they need them.
/// The procedures that decide or build pass their [`ParamContext`], so an
/// implementation can ask the solver, report events, or call this
/// module's own procedures. Every hook but [`eq_part`](Self::eq_part) and
/// [`hash_part`](Self::hash_part) is fallible, and a failure is the error
/// of the operation that asked, as [`ParamError::Custom`](super::ParamError::Custom).
pub trait CustomDomain: ForeignPart {
    /// Return the sort the solver reasons about the domain's values in, or
    /// `None` for a non-numeric domain.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn symbol_type(&self) -> Result<Option<SymbolType>, BoxError>;

    /// Return whether `value` lies in the domain's value set.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn is_value_admissible(&self, value: &Value) -> Result<bool, BoxError>;

    /// Refuse `constraint` on `variable` if the domain forbids it.
    ///
    /// # Errors
    ///
    /// Returns the implementation's refusal.
    fn validate_constraint(
        &self,
        constraint: &Constraint,
        variable: &Identifier,
    ) -> Result<(), BoxError>;

    /// Return the constraints the domain imposes on `variable`.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn implied_constraints(&self, variable: &Identifier) -> Result<Vec<Constraint>, BoxError>;

    /// Return what interval arithmetic reads from the domain, or `None`.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn interval_profile(&self) -> Result<Option<IntervalProfile>, BoxError>;

    /// Return whether the domain's value set is a subset of `other`'s.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn is_value_set_subset(
        &self,
        other: &ParamDomain,
        context: &ParamContext<'_>,
    ) -> Result<bool, BoxError>;

    /// Decide whether `own`'s constrained set is a subset of `other`'s.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn feasibility_subset(
        &self,
        own: Side<'_>,
        other_domain: &ParamDomain,
        other: Side<'_>,
        context: &ParamContext<'_>,
    ) -> Result<Outcome, BoxError>;

    /// Decide whether some admissible value satisfies `side`'s constraints.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn has_feasible_value(
        &self,
        side: Side<'_>,
        context: &ParamContext<'_>,
    ) -> Result<Outcome, BoxError>;

    /// Return the domain and constraints of the union of the two value
    /// sets, over `variable`, or `None` if the kind represents no union.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn union(
        &self,
        own: Side<'_>,
        other_domain: &ParamDomain,
        other: Side<'_>,
        variable: &Identifier,
        context: &ParamContext<'_>,
    ) -> Result<Option<(ParamDomain, Vec<Constraint>)>, BoxError>;

    /// Return the domain and constraints of the intersection of the two
    /// value sets, over `variable`.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn intersection(
        &self,
        own: Side<'_>,
        other_domain: &ParamDomain,
        other: Side<'_>,
        variable: &Identifier,
        context: &ParamContext<'_>,
    ) -> Result<(ParamDomain, Vec<Constraint>), BoxError>;

    /// Return whether `other` is a structurally identical domain, for `==`
    /// on a [`Part<dyn CustomDomain>`](crate::foreign::Part) and for
    /// [`ParamDomain::is_structurally_equivalent`].
    ///
    /// It must be an equivalence relation, symmetric included, and agree
    /// with [`hash_part`](Self::hash_part). The default is identity: the
    /// same domain.
    fn eq_part(&self, other: &dyn CustomDomain) -> bool {
        is_same_part(self, other)
    }

    /// Feed the domain's hash to `state`, consistently with
    /// [`eq_part`](Self::eq_part). The default feeds nothing.
    fn hash_part(&self, state: &mut dyn Hasher) {
        let _ = state;
    }
}

impl_part!(CustomDomain);
