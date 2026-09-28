//! [`ParamAssignment`]: a param bound to one value.

use std::fmt;
use std::hash::{Hash, Hasher};

use crate::constraint::{Binding, Bindings, Value};
use crate::term::{AlphaEquivalence, AlphaRenaming};

use super::context::ParamContext;
use super::decide::Evaluation;
use super::error::{AssignmentError, ParamError};
use super::parameter::{Param, ValueCheck};
use super::value::are_values_equal;

/// A param bound to one value it admits.
///
/// Cloning one shares its param.
#[derive(Debug, Clone)]
pub struct ParamAssignment {
    param: Param,
    value: Value,
}

impl ParamAssignment {
    /// Return the assignment of `value` to `param`, checked without
    /// bindings: the value must be admissible and satisfy every constraint
    /// provably.
    ///
    /// # Errors
    ///
    /// Returns [`AssignmentError::Inadmissible`],
    /// [`AssignmentError::ViolatedConstraint`] and
    /// [`AssignmentError::UnverifiedConstraint`], and what the evaluation returns.
    pub fn new(
        param: Param,
        value: Value,
        context: &ParamContext<'_>,
    ) -> Result<Self, AssignmentError> {
        let environment = param.environment(Binding::Value(value.clone()), &Bindings::new())?;
        match param.check_value(&environment, context)? {
            ValueCheck::Inadmissible => Err(AssignmentError::Inadmissible),
            ValueCheck::Violated { member } => Err(AssignmentError::ViolatedConstraint { member }),
            ValueCheck::Undecided { member } => {
                Err(AssignmentError::UnverifiedConstraint { member })
            }
            _ => Ok(Self { param, value }),
        }
    }

    /// Return the assignment of `value` to `param`, refusing only a value
    /// that is provably invalid: inadmissible, or violating a constraint.
    /// A constraint the value leaves undecided, such as a dependent one
    /// whose other bindings a payload does not carry, is accepted.
    ///
    /// # Errors
    ///
    /// Returns [`AssignmentError::Inadmissible`] and
    /// [`AssignmentError::ViolatedConstraint`], and what the evaluation returns.
    pub fn restore(
        param: Param,
        value: Value,
        context: &ParamContext<'_>,
    ) -> Result<Self, AssignmentError> {
        let binding = Binding::Value(value.clone());
        if !param.is_value_admissible(&binding)? {
            return Err(AssignmentError::Inadmissible);
        }
        let environment = param.environment(binding, &Bindings::new())?;
        let evaluation: Evaluation = param.evaluate_constraints(&environment, context)?;
        if evaluation.outcome() == crate::constraint::Outcome::Violated {
            if let Some(member) = evaluation.deciding_member() {
                return Err(AssignmentError::ViolatedConstraint { member });
            }
        }
        Ok(Self { param, value })
    }

    /// Return the assignment of `value` to `param` without a check, for a
    /// value the caller checked, as with bindings a payload does not carry.
    #[must_use]
    pub const fn new_unvalidated(param: Param, value: Value) -> Self {
        Self { param, value }
    }

    /// Return the param.
    #[must_use]
    pub const fn param(&self) -> &Param {
        &self.param
    }

    /// Return the value.
    #[must_use]
    pub const fn value(&self) -> &Value {
        &self.value
    }

    /// Return whether `other` assigns a structurally equivalent param a
    /// value equal type-strictly.
    #[must_use]
    pub fn is_structurally_equivalent(&self, other: &Self) -> bool {
        self.param.is_structurally_equivalent(&other.param)
            && are_values_equal(&self.value, &other.value)
    }
}

impl PartialEq for ParamAssignment {
    /// Compare the params structurally and the values as [`Value`]'s `==`
    /// does. That is [`is_structurally_equivalent`](Self::is_structurally_equivalent),
    /// except that a NaN value equals a NaN, so `==` is an equivalence.
    fn eq(&self, other: &Self) -> bool {
        self.param == other.param && self.value == other.value
    }
}

impl Eq for ParamAssignment {}

impl Hash for ParamAssignment {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.param.hash(state);
        self.value.hash(state);
    }
}

impl fmt::Display for ParamAssignment {
    /// Write the variable and the value: `x = 3`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.param.variable(), self.value)
    }
}

impl AlphaEquivalence for ParamAssignment {
    type Error = ParamError;

    /// Compare the params under `renaming`, and the values type-strictly.
    ///
    /// # Errors
    ///
    /// Returns the params' comparison error.
    fn is_alpha_equivalent_under(
        &self,
        other: &Self,
        renaming: &AlphaRenaming,
    ) -> Result<bool, ParamError> {
        Ok(self
            .param
            .is_alpha_equivalent_under(&other.param, renaming)?
            && are_values_equal(&self.value, &other.value))
    }
}
