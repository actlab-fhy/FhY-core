//! [`ParamAssignment`]: a param bound to one value.

use crate::constraint::{Binding, Bindings, Value};
use crate::term::{AlphaEquivalence, AlphaRenaming};

use super::context::ParamContext;
use super::decide::Evaluation;
use super::error::ParamError;
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
    /// Returns [`ParamError::Inadmissible`],
    /// [`ParamError::ViolatedConstraint`] and
    /// [`ParamError::UnverifiedConstraint`], and what the evaluation returns.
    pub fn new(param: Param, value: Value, context: &ParamContext<'_>) -> Result<Self, ParamError> {
        let environment = param.environment(Binding::Value(value.clone()), &Bindings::new())?;
        match param.check_value(&environment, context)? {
            ValueCheck::Inadmissible => Err(ParamError::Inadmissible),
            ValueCheck::Violated { member } => Err(ParamError::ViolatedConstraint { member }),
            ValueCheck::Undecided { member } => Err(ParamError::UnverifiedConstraint { member }),
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
    /// Returns [`ParamError::Inadmissible`] and
    /// [`ParamError::ViolatedConstraint`], and what the evaluation returns.
    pub fn restore(
        param: Param,
        value: Value,
        context: &ParamContext<'_>,
    ) -> Result<Self, ParamError> {
        let binding = Binding::Value(value.clone());
        if !param.is_value_admissible(&binding)? {
            return Err(ParamError::Inadmissible);
        }
        let environment = param.environment(binding, &Bindings::new())?;
        let evaluation: Evaluation = param.evaluate_constraints(&environment, context)?;
        if evaluation.outcome() == crate::constraint::Outcome::Violated {
            if let Some(member) = evaluation.deciding_member() {
                return Err(ParamError::ViolatedConstraint { member });
            }
        }
        Ok(Self { param, value })
    }

    /// Return the assignment of `value` to `param` without a check, for a
    /// value the caller checked, as with bindings a payload does not carry.
    #[must_use]
    pub fn new_unchecked(param: Param, value: Value) -> Self {
        Self { param, value }
    }

    /// Return the param.
    #[must_use]
    pub fn param(&self) -> &Param {
        &self.param
    }

    /// Return the value.
    #[must_use]
    pub fn value(&self) -> &Value {
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

impl AlphaEquivalence for ParamAssignment {
    /// Compare the params under `renaming`, and the values type-strictly.
    fn is_alpha_equivalent_under(&self, other: &Self, renaming: &AlphaRenaming) -> bool {
        self.param.is_alpha_equivalent_under(&other.param, renaming)
            && are_values_equal(&self.value, &other.value)
    }
}
