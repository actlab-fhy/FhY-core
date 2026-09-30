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
        Self::new_with_bindings(param, value, &Bindings::new(), context)
    }

    /// Return the assignment of `value` to `param`, checked under
    /// `bindings`, which supply the other variables its constraints depend
    /// on: the value must be admissible and satisfy every constraint
    /// provably.
    ///
    /// # Errors
    ///
    /// Returns [`AssignmentError::BindingsBindVariable`] if `bindings` binds
    /// the param's variable, [`AssignmentError::Inadmissible`],
    /// [`AssignmentError::ViolatedConstraint`] and
    /// [`AssignmentError::UnverifiedConstraint`], and what the evaluation returns.
    pub fn new_with_bindings(
        param: Param,
        value: Value,
        bindings: &Bindings,
        context: &ParamContext<'_>,
    ) -> Result<Self, AssignmentError> {
        let environment = param.environment(Binding::Value(value.clone()), bindings)?;
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
        Self::restore_with_bindings(param, value, &Bindings::new(), context)
    }

    /// Return the assignment of `value` to `param` as [`restore`](Self::restore)
    /// does, with the constraints evaluated under `bindings`.
    ///
    /// # Errors
    ///
    /// Returns [`AssignmentError::BindingsBindVariable`] if `bindings` binds
    /// the param's variable, [`AssignmentError::Inadmissible`] and
    /// [`AssignmentError::ViolatedConstraint`], and what the evaluation returns.
    pub fn restore_with_bindings(
        param: Param,
        value: Value,
        bindings: &Bindings,
        context: &ParamContext<'_>,
    ) -> Result<Self, AssignmentError> {
        let binding = Binding::Value(value.clone());
        if !param.is_value_admissible(&binding)? {
            return Err(AssignmentError::Inadmissible);
        }
        let environment = param.environment(binding, bindings)?;
        let evaluation: Evaluation = param.evaluate_constraints(&environment, context)?;
        if evaluation.outcome() == crate::constraint::Outcome::Violated {
            if let Some(member) = evaluation.deciding_member() {
                return Err(AssignmentError::ViolatedConstraint { member });
            }
        }
        Ok(Self { param, value })
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

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;

    use rstest::rstest;

    use crate::constraint::Value;
    use crate::expression::{BigInt, Decimal};
    use crate::identifier::Identifier;
    use crate::solver::Solver;

    use super::super::domain::{IntegerDomain, ParamDomain, RealDomain, Sign, ZeroInclusion};
    use super::*;

    fn build_param_of(domain: ParamDomain, variable: &Identifier) -> Param {
        let solver = Solver::new();
        Param::new(domain, variable.clone(), [], &ParamContext::new(&solver)).expect("a param")
    }

    fn build_param(domain: ParamDomain) -> Param {
        build_param_of(domain, &Identifier::new("p"))
    }

    /// Return the assignment of `value` to `param`, admissible or not.
    fn assign_without_check(param: &Param, value: Value) -> ParamAssignment {
        ParamAssignment {
            param: param.clone(),
            value,
        }
    }

    fn int(value: i64) -> Value {
        Value::Int(BigInt::from(value))
    }

    fn decimal(text: &str) -> Value {
        Value::Decimal(text.parse::<Decimal>().expect("a decimal"))
    }

    fn hash_of(assignment: &ParamAssignment) -> u64 {
        let mut hasher = DefaultHasher::new();
        assignment.hash(&mut hasher);
        hasher.finish()
    }

    /// Two assignments of one param are structurally equivalent exactly
    /// when their values are equal type-strictly: of one kind, at every
    /// depth, so `1` and `True` differ inside a tuple, the zeros are equal,
    /// and a NaN equals nothing; `==` differs only for the NaN.
    #[rstest]
    #[case::int_against_float(int(5), Value::Float(5.0), false)]
    #[case::zeros(Value::Float(-0.0), Value::Float(0.0), true)]
    #[case::nan(Value::Float(f64::NAN), Value::Float(f64::NAN), false)]
    #[case::decimal(decimal("0.1"), decimal("0.10"), true)]
    #[case::decimal_against_float(decimal("0.5"), Value::Float(0.5), false)]
    #[case::tuple_of_zeros(
        Value::Tuple(vec![Value::Float(-0.0)]),
        Value::Tuple(vec![Value::Float(0.0)]),
        true
    )]
    #[case::one_against_true_in_a_tuple(
        Value::Tuple(vec![int(1)]),
        Value::Tuple(vec![Value::Bool(true)]),
        false
    )]
    #[case::tuple_order(
        Value::Tuple(vec![int(1), int(2)]),
        Value::Tuple(vec![int(2), int(1)]),
        false
    )]
    #[case::frozenset_order(
        Value::FrozenSet(vec![int(1), int(2)]),
        Value::FrozenSet(vec![int(2), int(1), int(1)]),
        true
    )]
    #[case::one_against_true_in_a_frozenset(
        Value::FrozenSet(vec![int(1)]),
        Value::FrozenSet(vec![Value::Bool(true)]),
        false
    )]
    fn assignment_equivalence_compares_values_type_strictly(
        #[case] left: Value,
        #[case] right: Value,
        #[case] equivalent: bool,
    ) {
        let param = build_param(ParamDomain::from(IntegerDomain::new(
            Sign::Any,
            ZeroInclusion::Included,
        )));
        let is_nan = matches!(left, Value::Float(value) if value.is_nan());
        let left = assign_without_check(&param, left);
        let right = assign_without_check(&param, right);

        assert_eq!(left.is_structurally_equivalent(&right), equivalent);
        assert_eq!(right.is_structurally_equivalent(&left), equivalent);
        assert_eq!(left == right, equivalent || is_nan);
    }

    /// `==` is an equivalence: a NaN value equals itself and hashes like
    /// any other, which the type-strict value equality of
    /// `is_structurally_equivalent` denies.
    #[test]
    fn a_nan_assignment_equals_itself_and_hashes_by_its_param() {
        let variable = Identifier::new("x");
        let param = build_param_of(ParamDomain::from(RealDomain), &variable);
        let other = build_param_of(ParamDomain::from(RealDomain), &variable);
        let nan = assign_without_check(&param, Value::Float(f64::NAN));

        assert_eq!(nan, nan.clone());
        assert!(!nan.is_structurally_equivalent(&nan));
        assert_eq!(
            hash_of(&nan),
            hash_of(&assign_without_check(&other, Value::Float(f64::NAN)))
        );
    }
}
