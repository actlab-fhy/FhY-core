//! [`Param`]: a variable ranging over a value domain, narrowed by
//! constraints.

use std::sync::Arc;

use crate::constraint::{Binding, Bindings, Constraint, ConstraintSystem, Outcome};
use crate::expression::{LiteralValue, SymbolType};
use crate::identifier::Identifier;
use crate::term::{AlphaEquivalence, AlphaRenaming};

use super::context::ParamContext;
use super::decide::{Evaluation, evaluate_constraints};
use super::domain::{IntervalProfile, ParamDomain, Side};
use super::error::ParamError;
use super::interval::{
    BoundSide, Interval, Operand, bound_constraint, build_interval_param, check_natural_bound,
    coerce_to_interval, combine_ends, effective_interval, multiply_intervals, operand_profile,
};

/// The answer to whether a value is valid for a param.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ValueCheck {
    /// The value is admissible and provably satisfies every constraint.
    Valid,
    /// The value is not admissible in the domain.
    Inadmissible,
    /// The value provably violates the constraint at `member`, in canonical
    /// order.
    Violated {
        /// The constraint's position.
        member: usize,
    },
    /// The value could not be verified against the constraint at `member`.
    Undecided {
        /// The constraint's position.
        member: usize,
    },
}

/// A variable ranging over a [`ParamDomain`], narrowed by constraints.
///
/// Construction validates each constraint (its scope must hold the
/// variable, and the domain must allow it), drops those structurally
/// equivalent to an earlier one, appends the domain's implied constraints
/// unless an equivalent is present, and keeps the result as a
/// [`ConstraintSystem`], in canonical order. Cloning one shares it.
#[derive(Debug, Clone)]
pub struct Param(Arc<ParamInner>);

#[derive(Debug)]
struct ParamInner {
    domain: ParamDomain,
    variable: Identifier,
    system: ConstraintSystem,
}

/// Return whether some constraint of `constraints` is structurally
/// equivalent to `constraint`.
fn holds_equivalent(constraints: &[Constraint], constraint: &Constraint) -> bool {
    constraints
        .iter()
        .any(|existing| existing.is_structurally_equivalent(constraint))
}

impl Param {
    /// Return the param over `domain` whose variable is `variable`,
    /// narrowed by `constraints`.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::NativeConstantVariable`] for a variable that is
    /// a native constant's identifier, [`ParamError::OutOfScope`] for a
    /// constraint whose scope lacks the variable, the domain's refusal of a
    /// constraint, and a custom domain's or constraint's failure.
    pub fn new(
        domain: ParamDomain,
        variable: Identifier,
        constraints: impl IntoIterator<Item = Constraint>,
        context: &ParamContext<'_>,
    ) -> Result<Self, ParamError> {
        if context.is_native_constant(&variable) {
            return Err(ParamError::NativeConstantVariable(variable));
        }
        let mut accumulated: Vec<Constraint> = Vec::new();
        for constraint in constraints {
            validate_constraint(&domain, &variable, &constraint)?;
            if !holds_equivalent(&accumulated, &constraint) {
                accumulated.push(constraint);
            }
        }
        for implied in domain.implied_constraints(&variable)? {
            if !holds_equivalent(&accumulated, &implied) {
                accumulated.push(implied);
            }
        }
        Ok(Self(Arc::new(ParamInner {
            domain,
            variable,
            system: ConstraintSystem::new(accumulated),
        })))
    }

    /// Return whether `other` is this very param, or a clone of it.
    #[must_use]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        Arc::ptr_eq(&this.0, &other.0)
    }

    /// Return the domain.
    #[must_use]
    pub fn domain(&self) -> &ParamDomain {
        &self.0.domain
    }

    /// Return the variable.
    #[must_use]
    pub fn variable(&self) -> &Identifier {
        &self.0.variable
    }

    /// Return the constraints as their system.
    #[must_use]
    pub fn constraint_system(&self) -> &ConstraintSystem {
        &self.0.system
    }

    /// Return the constraints, in canonical order.
    #[must_use]
    pub fn constraints(&self) -> &[Constraint] {
        self.0.system.constraints()
    }

    /// Return the domain's numeric sort, or `None` for a non-numeric one.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::Custom`] for a custom domain that fails.
    pub fn symbol_type(&self) -> Result<Option<SymbolType>, ParamError> {
        self.0.domain.symbol_type()
    }

    /// Return the constraints and variable as a [`Side`] of a question.
    fn side(&self) -> Side<'_> {
        Side::new(self.constraints(), self.variable())
    }

    /// Refuse `constraint` unless its scope holds the variable and the
    /// domain allows it.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::OutOfScope`], and the domain's refusal.
    pub fn validate_constraint(&self, constraint: &Constraint) -> Result<(), ParamError> {
        validate_constraint(self.domain(), self.variable(), constraint)
    }

    /// Return the param with its constraints replaced by `constraints`,
    /// validated, deduplicated and canonicalized.
    ///
    /// # Errors
    ///
    /// Returns what [`Param::new`] returns.
    pub fn with_constraints(
        &self,
        constraints: impl IntoIterator<Item = Constraint>,
        context: &ParamContext<'_>,
    ) -> Result<Self, ParamError> {
        Self::new(
            self.domain().clone(),
            self.variable().clone(),
            constraints,
            context,
        )
    }

    /// Return the param with `constraint` added, or a clone of this one
    /// when an equivalent constraint is present.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::OutOfScope`], and the domain's refusal.
    pub fn with_constraint(
        &self,
        constraint: Constraint,
        context: &ParamContext<'_>,
    ) -> Result<Self, ParamError> {
        self.validate_constraint(&constraint)?;
        if holds_equivalent(self.constraints(), &constraint) {
            return Ok(self.clone());
        }
        let mut constraints = self.constraints().to_vec();
        constraints.push(constraint);
        self.with_constraints(constraints, context)
    }

    /// Return the param with the bound `variable <cmp> value` of `side`
    /// added, or a clone of this one when an equivalent is present.
    ///
    /// On a domain whose profile is non-negative, an integer bound must be
    /// a literal the natural numbers admit.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::NaturalBound`] for a bound the gate refuses,
    /// and what [`Param::with_constraint`] returns.
    pub fn with_bound(
        &self,
        value: &LiteralValue,
        side: BoundSide,
        is_inclusive: bool,
        context: &ParamContext<'_>,
    ) -> Result<Self, ParamError> {
        check_natural_bound(self.domain().interval_profile()?, value, side, is_inclusive)?;
        self.with_constraint(
            bound_constraint(self.variable(), value, side, is_inclusive),
            context,
        )
    }

    /// Return the environment of a value check: `value` bound to the
    /// variable, then `bindings`.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::BindingsBindVariable`] if `bindings` binds the
    /// variable.
    pub fn environment(&self, value: Binding, bindings: &Bindings) -> Result<Bindings, ParamError> {
        if bindings.get(self.variable()).is_some() {
            return Err(ParamError::BindingsBindVariable(self.variable().clone()));
        }
        let mut environment = Bindings::new();
        environment.insert(self.variable().clone(), value);
        for (identifier, binding) in bindings.iter() {
            environment.insert(identifier.clone(), binding.clone());
        }
        Ok(environment)
    }

    /// Return whether `value` lies in the domain's value set; an expression
    /// lies in none.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::Custom`] for a custom domain that fails.
    pub fn is_value_admissible(&self, value: &Binding) -> Result<bool, ParamError> {
        match value {
            Binding::Value(value) => self.domain().is_value_admissible(value),
            Binding::Expression(_) => Ok(false),
        }
    }

    /// Evaluate the constraints under `environment`, member by member, as
    /// [`evaluate_constraints`] does.
    ///
    /// # Errors
    ///
    /// Returns what [`evaluate_constraints`] returns.
    pub fn evaluate_constraints(
        &self,
        environment: &Bindings,
        context: &ParamContext<'_>,
    ) -> Result<Evaluation, ParamError> {
        evaluate_constraints(self.constraints(), environment, context)
    }

    /// Decide whether the value `environment` binds to the variable is
    /// valid: admissible, then satisfying every constraint provably.
    ///
    /// # Errors
    ///
    /// Returns what [`Param::evaluate_constraints`] returns.
    pub fn check_value(
        &self,
        environment: &Bindings,
        context: &ParamContext<'_>,
    ) -> Result<ValueCheck, ParamError> {
        let is_admissible = match environment.get(self.variable()) {
            Some(value) => self.is_value_admissible(value)?,
            None => false,
        };
        if !is_admissible {
            return Ok(ValueCheck::Inadmissible);
        }
        let evaluation = self.evaluate_constraints(environment, context)?;
        Ok(match (evaluation.outcome(), evaluation.deciding_member()) {
            (Outcome::Violated, Some(member)) => ValueCheck::Violated { member },
            (Outcome::Undecided, Some(member)) => ValueCheck::Undecided { member },
            _ => ValueCheck::Valid,
        })
    }

    /// Decide whether some value satisfies the domain and every constraint.
    ///
    /// # Errors
    ///
    /// Returns what [`ParamDomain::has_feasible_value`] returns.
    pub fn check_feasibility(&self, context: &ParamContext<'_>) -> Result<Outcome, ParamError> {
        self.domain().has_feasible_value(self.side(), context)
    }

    /// Decide whether this param's feasible set is a subset of `other`'s.
    ///
    /// # Errors
    ///
    /// Returns what [`ParamDomain::feasibility_subset`] returns.
    pub fn check_subset(
        &self,
        other: &Self,
        context: &ParamContext<'_>,
    ) -> Result<Outcome, ParamError> {
        self.domain()
            .feasibility_subset(self.side(), other.domain(), other.side(), context)
    }

    /// Return whether this param's value set is a subset of `other`'s.
    ///
    /// # Errors
    ///
    /// Returns what [`ParamDomain::is_value_set_subset`] returns.
    pub fn is_value_set_subset(&self, other: &Self) -> Result<bool, ParamError> {
        self.domain().is_value_set_subset(other.domain())
    }

    /// Return the param over `variable` admitting exactly the values valid
    /// for either param.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::UnsupportedUnion`] for a domain that
    /// represents no union, and what [`ParamDomain::union`] returns.
    pub fn union(
        &self,
        other: &Self,
        variable: Identifier,
        context: &ParamContext<'_>,
    ) -> Result<Self, ParamError> {
        let Some((domain, constraints)) = self.domain().union(
            self.side(),
            other.domain(),
            other.side(),
            &variable,
            context,
        )?
        else {
            return Err(ParamError::UnsupportedUnion(self.domain().kind()));
        };
        Self::new(domain, variable, constraints, context)
    }

    /// Return the param over `variable` admitting exactly the values valid
    /// for both params.
    ///
    /// An interval param and an integer param whose constraints are all
    /// bounds intersect as two interval params.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::EmptyParamIntersection`] for an intersection
    /// provably empty (its conjunction infeasible, or an operand proven
    /// infeasible), what [`ParamDomain::intersection`] returns, and the
    /// coercion's errors.
    pub fn intersection(
        &self,
        other: &Self,
        variable: Identifier,
        context: &ParamContext<'_>,
    ) -> Result<Self, ParamError> {
        let (left, right) = coerce_intersection_operands(self, other, context)?;
        let (domain, constraints) = left.domain().intersection(
            left.side(),
            right.domain(),
            right.side(),
            &variable,
            context,
        )?;
        let result = Self::new(domain, variable, constraints, context)?;
        let is_empty = match result.check_feasibility(context)? {
            Outcome::Violated => true,
            Outcome::Satisfied => false,
            Outcome::Undecided => {
                left.check_feasibility(context)? == Outcome::Violated
                    || right.check_feasibility(context)? == Outcome::Violated
            }
        };
        if is_empty {
            return Err(ParamError::EmptyParamIntersection);
        }
        Ok(result)
    }

    /// Return the interval operands of an operation of this param with
    /// `other`: this param (coerced to `other`'s profile when it is no
    /// operand as it stands), `other` coerced to its profile, and the
    /// profile; `None` when neither is an operand.
    fn resolve_operands(
        &self,
        other: &Operand,
        context: &ParamContext<'_>,
    ) -> Result<Option<(Self, Self, IntervalProfile)>, ParamError> {
        if let Some(profile) = operand_profile(self)? {
            let coerced = coerce_to_interval(profile, other, context)?;
            return Ok(Some((self.clone(), coerced, profile)));
        }
        let Operand::Param(other_param) = other else {
            return Ok(None);
        };
        let Some(other_profile) = operand_profile(other_param)? else {
            return Ok(None);
        };
        let coerced_self =
            coerce_to_interval(other_profile, &Operand::Param(self.clone()), context)?;
        coerced_self.resolve_operands(other, context)
    }

    /// Return the effective intervals of two operands.
    fn intervals(left: &Self, right: &Self) -> Result<(Interval, Interval), ParamError> {
        Ok((
            effective_interval(left.constraints(), left.variable())?,
            effective_interval(right.constraints(), right.variable())?,
        ))
    }

    /// Return the sum of this param and `other`, over a fresh variable.
    ///
    /// The result is natural when both operands' profiles are, admitting
    /// zero when the left one does.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::NotAnIntervalOperand`] when neither operand is
    /// an interval operand, and the coercion's and the bounds' errors.
    pub fn checked_add(
        &self,
        other: &Operand,
        context: &ParamContext<'_>,
    ) -> Result<Self, ParamError> {
        let (left, coerced, profile) = self
            .resolve_operands(other, context)?
            .ok_or(ParamError::NotAnIntervalOperand)?;
        let (own, others) = Self::intervals(&left, &coerced)?;
        let interval = Interval {
            min: combine_ends(own.min.as_ref(), others.min.as_ref(), |a, b| a + b),
            max: combine_ends(own.max.as_ref(), others.max.as_ref(), |a, b| a + b),
        };
        let coerced_profile = operand_profile(&coerced)?.ok_or(ParamError::NotAnIntervalOperand)?;
        let natural =
            (profile.non_negative && coerced_profile.non_negative).then_some(profile.zero_included);
        build_interval_param(&interval, profile, natural, context)
    }

    /// Return this param minus `other`, over a fresh variable. The result
    /// is never natural.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::NotAnIntervalOperand`] when neither operand is
    /// an interval operand, and the coercion's and the bounds' errors.
    pub fn checked_sub(
        &self,
        other: &Operand,
        context: &ParamContext<'_>,
    ) -> Result<Self, ParamError> {
        let (left, coerced, profile) = self
            .resolve_operands(other, context)?
            .ok_or(ParamError::NotAnIntervalOperand)?;
        let (own, others) = Self::intervals(&left, &coerced)?;
        let interval = Interval {
            min: combine_ends(own.min.as_ref(), others.max.as_ref(), |a, b| a - b),
            max: combine_ends(own.max.as_ref(), others.min.as_ref(), |a, b| a - b),
        };
        build_interval_param(&interval, profile, None, context)
    }

    /// Return `other` minus this param.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::NotAnIntervalOperand`] when this param is no
    /// interval operand, and the coercion's and the bounds' errors.
    pub fn checked_reverse_sub(
        &self,
        other: &Operand,
        context: &ParamContext<'_>,
    ) -> Result<Self, ParamError> {
        let profile = operand_profile(self)?.ok_or(ParamError::NotAnIntervalOperand)?;
        coerce_to_interval(profile, other, context)?
            .checked_sub(&Operand::Param(self.clone()), context)
    }

    /// Return the product of this param and `other`, over a fresh variable.
    ///
    /// The result is natural when both operands' profiles are, admitting
    /// zero when either does.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::NotAnIntervalOperand`] when neither operand is
    /// an interval operand, and the coercion's and the bounds' errors.
    pub fn checked_mul(
        &self,
        other: &Operand,
        context: &ParamContext<'_>,
    ) -> Result<Self, ParamError> {
        let (left, coerced, profile) = self
            .resolve_operands(other, context)?
            .ok_or(ParamError::NotAnIntervalOperand)?;
        let coerced_profile = operand_profile(&coerced)?.ok_or(ParamError::NotAnIntervalOperand)?;
        let (own, others) = Self::intervals(&left, &coerced)?;
        let interval = multiply_intervals(&own, &others);
        let natural = (profile.non_negative && coerced_profile.non_negative)
            .then_some(profile.zero_included || coerced_profile.zero_included);
        build_interval_param(&interval, profile, natural, context)
    }

    /// Return the negation of this interval param, over a fresh variable.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::NotAnIntervalOperand`] for a param that is no
    /// interval operand, and the bounds' errors.
    pub fn checked_neg(&self, context: &ParamContext<'_>) -> Result<Self, ParamError> {
        let profile = operand_profile(self)?.ok_or(ParamError::NotAnIntervalOperand)?;
        let own = effective_interval(self.constraints(), self.variable())?;
        let interval = Interval {
            min: own.max.map(|max| -max),
            max: own.min.map(|min| -min),
        };
        build_interval_param(&interval, profile, None, context)
    }

    /// Return whether `other` has an equivalent domain, the same variable,
    /// and structurally equivalent constraints.
    #[must_use]
    pub fn is_structurally_equivalent(&self, other: &Self) -> bool {
        self.domain().is_structurally_equivalent(other.domain())
            && self.variable() == other.variable()
            && self
                .constraint_system()
                .is_structurally_equivalent(other.constraint_system())
    }
}

impl AlphaEquivalence for Param {
    /// Compare the domains structurally, then the constraints under
    /// `renaming` extended by the two variables, which the param binds.
    fn is_alpha_equivalent_under(&self, other: &Self, renaming: &AlphaRenaming) -> bool {
        if !self.domain().is_structurally_equivalent(other.domain()) {
            return false;
        }
        let mut extended = renaming.clone();
        if extended
            .enter_binders(
                std::slice::from_ref(self.variable()),
                std::slice::from_ref(other.variable()),
            )
            .is_err()
        {
            return false;
        }
        self.constraint_system()
            .is_alpha_equivalent_under(other.constraint_system(), &extended)
    }
}

/// Refuse `constraint` for a param over `domain` of `variable` unless its
/// scope holds the variable and the domain allows it.
fn validate_constraint(
    domain: &ParamDomain,
    variable: &Identifier,
    constraint: &Constraint,
) -> Result<(), ParamError> {
    if !constraint.free_identifiers().contains(variable) {
        return Err(ParamError::OutOfScope {
            constraint: constraint.clone(),
            variable: variable.clone(),
        });
    }
    domain.validate_constraint(constraint, variable)
}

/// Return the operands of an intersection, a pair of one interval param
/// and one integer param recast as two interval params.
fn coerce_intersection_operands(
    left: &Param,
    right: &Param,
    context: &ParamContext<'_>,
) -> Result<(Param, Param), ParamError> {
    let (Some(left_profile), Some(right_profile)) = (
        left.domain().interval_profile()?,
        right.domain().interval_profile()?,
    ) else {
        return Ok((left.clone(), right.clone()));
    };
    if left_profile.admits_only_bounds && !right_profile.admits_only_bounds {
        return Ok((
            left.clone(),
            coerce_to_interval(left_profile, &Operand::Param(right.clone()), context)?,
        ));
    }
    if right_profile.admits_only_bounds && !left_profile.admits_only_bounds {
        return Ok((
            coerce_to_interval(right_profile, &Operand::Param(left.clone()), context)?,
            right.clone(),
        ));
    }
    Ok((left.clone(), right.clone()))
}
