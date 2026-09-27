//! [`CustomDomain`]: a domain of a kind this module does not define, such
//! as one a language binding defines.

use std::any::Any;
use std::fmt;

use crate::constraint::{Constraint, Outcome, Value};
use crate::expression::SymbolType;
use crate::foreign::BoxError;
use crate::foreign::{Foreign, ForeignError};
use crate::identifier::Identifier;

use super::domain::{IntervalProfile, ParamDomain, Side};

/// A domain of a kind this module does not define.
///
/// It answers what the built-in kinds answer, each through its own hook,
/// and the procedures of this module call the hooks as they need them.
pub trait CustomDomain: Send + Sync + fmt::Debug {
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
    fn is_value_set_subset(&self, other: &ParamDomain) -> Result<bool, BoxError>;

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
    ) -> Result<Outcome, BoxError>;

    /// Decide whether some admissible value satisfies `side`'s constraints.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn has_feasible_value(&self, side: Side<'_>) -> Result<Outcome, BoxError>;

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
    ) -> Result<(ParamDomain, Vec<Constraint>), BoxError>;

    /// Return whether `other` is a structurally identical domain.
    fn is_structurally_equivalent(&self, other: &ParamDomain) -> bool;

    /// Return the domain as a [`Foreign`] part, for serialization.
    ///
    /// # Errors
    ///
    /// The default returns [`ForeignError::NoWireForm`]: the domain
    /// cannot be serialized.
    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        Err(ForeignError::NoWireForm {
            type_name: "custom domain".to_owned(),
        })
    }

    /// Return the domain as [`Any`], so an implementation can recognize its
    /// own domains.
    fn as_any(&self) -> &dyn Any;
}
