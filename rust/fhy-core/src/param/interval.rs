//! Bounds and interval arithmetic: bound constraints and the natural-number
//! gates, the exact order of bound literals, the bounds an interval param
//! carries, and the arithmetic of interval operands.

use std::cmp::Ordering;

use num_bigint::BigInt;
use num_traits::{Signed, Zero};

use crate::constraint::{Constraint, EquationConstraint};
use crate::expression::{BinaryOperation, Expression, ExpressionKind, LiteralValue};
use crate::identifier::Identifier;

use super::context::ParamContext;
use super::domain::{
    IntervalIntegerDomain, IntervalProfile, ParamDomain, Sign, ZeroInclusion, is_bound_expression,
};
use super::error::{IntervalError, ParamBuildError};
use super::parameter::Param;

/// Whether a bound includes its value.
#[expect(
    clippy::exhaustive_enums,
    reason = "a bound is inclusive or exclusive, which callers match"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Inclusivity {
    /// The bound's value is admissible: `>=` or `<=`.
    Inclusive,
    /// The bound's value is not admissible: `>` or `<`.
    Exclusive,
}

impl Inclusivity {
    /// Return [`Inclusive`](Self::Inclusive) if `is_inclusive`, and
    /// [`Exclusive`](Self::Exclusive) otherwise.
    #[must_use]
    pub const fn inclusive_if(is_inclusive: bool) -> Self {
        if is_inclusive {
            Self::Inclusive
        } else {
            Self::Exclusive
        }
    }

    /// Return whether the bound is [`Inclusive`](Self::Inclusive).
    #[must_use]
    pub const fn is_inclusive(self) -> bool {
        matches!(self, Self::Inclusive)
    }
}

/// Which side of an interval a bound closes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BoundSide {
    /// A lower bound: `x >= k` or `x > k`.
    Lower,
    /// An upper bound: `x <= k` or `x < k`.
    Upper,
}

/// An operand of interval arithmetic: an integer, the exact interval it
/// denotes, or a param.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Operand {
    /// An integer.
    Integer(BigInt),
    /// A param.
    Param(Param),
}

/// Return the bound constraint `variable <cmp> value` of `side`, inclusive
/// or not.
pub(super) fn bound_constraint(
    variable: &Identifier,
    value: &LiteralValue,
    side: BoundSide,
    is_inclusive: bool,
) -> Constraint {
    let reference = Expression::from(variable);
    let literal = Expression::literal(value.clone());
    let expression = match (side, is_inclusive) {
        (BoundSide::Lower, true) => reference.greater_equal(literal),
        (BoundSide::Lower, false) => reference.greater(literal),
        (BoundSide::Upper, true) => reference.less_equal(literal),
        (BoundSide::Upper, false) => reference.less(literal),
    };
    Constraint::from(EquationConstraint::new(expression))
}

/// Return whether the integer `bound` is an admissible natural-domain
/// literal for `side`: the gate judges the literal, not the set it bounds.
fn is_valid_natural_bound(
    bound: &BigInt,
    side: BoundSide,
    zero_included: bool,
    is_inclusive: bool,
) -> bool {
    let least = match side {
        BoundSide::Lower => u8::from(zero_included != is_inclusive),
        BoundSide::Upper => u8::from(!zero_included) + u8::from(!is_inclusive),
    };
    *bound >= BigInt::from(least)
}

/// Refuse the bound `value` of `side` that the natural-number gate of a
/// non-negative `profile` refuses; only an integer bound is judged.
pub(super) fn check_natural_bound(
    profile: Option<IntervalProfile>,
    value: &LiteralValue,
    side: BoundSide,
    is_inclusive: bool,
) -> Result<(), ParamBuildError> {
    let Some(profile) = profile.filter(IntervalProfile::is_non_negative) else {
        return Ok(());
    };
    let LiteralValue::Int(bound) = value else {
        return Ok(());
    };
    if is_valid_natural_bound(bound, side, profile.is_zero_included(), is_inclusive) {
        return Ok(());
    }
    Err(ParamBuildError::NaturalBound {
        side,
        zero_included: profile.is_zero_included(),
        is_inclusive,
        is_negative: bound.is_negative(),
    })
}

/// Refuse bounds that enclose no value in any number system: a lower bound
/// above the upper, or equal to it with an exclusive side.
///
/// The bounds compare by the exact values they denote, never rounded: a
/// decimal `0.1` lies below the float `0.1`. A NaN bound compares with
/// nothing, so it is never refused.
///
/// # Errors
///
/// Returns [`ParamBuildError::UnorderedBounds`] for such bounds.
pub fn check_bounds_are_ordered(
    lower: &LiteralValue,
    upper: &LiteralValue,
    lower_inclusivity: Inclusivity,
    upper_inclusivity: Inclusivity,
) -> Result<(), ParamBuildError> {
    match lower.exact_cmp(upper) {
        Some(Ordering::Greater) => Err(ParamBuildError::UnorderedBounds),
        Some(Ordering::Equal)
            if !(lower_inclusivity.is_inclusive() && upper_inclusivity.is_inclusive()) =>
        {
            Err(ParamBuildError::UnorderedBounds)
        }
        _ => Ok(()),
    }
}

/// Return the operation `operation` read from the other side, as `k <cmp>
/// x` reads `x <inverse> k`.
fn invert_comparison(operation: BinaryOperation) -> BinaryOperation {
    match operation {
        BinaryOperation::Greater => BinaryOperation::Less,
        BinaryOperation::GreaterEqual => BinaryOperation::LessEqual,
        BinaryOperation::Less => BinaryOperation::Greater,
        BinaryOperation::LessEqual => BinaryOperation::GreaterEqual,
        other => other,
    }
}

/// A decoded bound: its side, its integer, and whether it is inclusive.
struct DecodedBound {
    side: BoundSide,
    value: BigInt,
    is_inclusive: bool,
}

/// Decode the bound `constraint` of `variable`.
///
/// # Errors
///
/// Returns [`IntervalError::MalformedBound`] for a constraint that is no
/// equation bound of `variable`.
fn decode_bound(
    constraint: &Constraint,
    variable: &Identifier,
) -> Result<DecodedBound, IntervalError> {
    let Constraint::Equation(equation) = constraint else {
        return Err(IntervalError::MalformedBound);
    };
    let expression = equation.expression();
    if !is_bound_expression(expression) {
        return Err(IntervalError::MalformedBound);
    }
    let ExpressionKind::Binary(binary) = expression.kind() else {
        return Err(IntervalError::MalformedBound);
    };
    let (operation, literal) = match (binary.left().kind(), binary.right().kind()) {
        (ExpressionKind::Identifier(identifier), ExpressionKind::Literal(literal))
            if identifier == variable =>
        {
            (binary.operation(), literal)
        }
        (ExpressionKind::Literal(literal), ExpressionKind::Identifier(identifier))
            if identifier == variable =>
        {
            (invert_comparison(binary.operation()), literal)
        }
        _ => return Err(IntervalError::MalformedBound),
    };
    let LiteralValue::Int(value) = literal else {
        return Err(IntervalError::MalformedBound);
    };
    Ok(DecodedBound {
        side: if matches!(
            operation,
            BinaryOperation::Greater | BinaryOperation::GreaterEqual
        ) {
            BoundSide::Lower
        } else {
            BoundSide::Upper
        },
        value: value.clone(),
        is_inclusive: matches!(
            operation,
            BinaryOperation::GreaterEqual | BinaryOperation::LessEqual
        ),
    })
}

/// The effective integer interval of a param's bounds: each end `None` when
/// unbounded.
pub(super) struct Interval {
    pub(super) min: Option<BigInt>,
    pub(super) max: Option<BigInt>,
}

/// Return the effective integer interval the bound `constraints` of
/// `variable` carry.
///
/// # Errors
///
/// Returns [`IntervalError::MalformedBound`] for a constraint that is no
/// bound, and [`IntervalError::EmptyInterval`] for bounds that enclose no
/// integer.
pub(super) fn effective_interval(
    constraints: &[Constraint],
    variable: &Identifier,
) -> Result<Interval, IntervalError> {
    let mut min: Option<BigInt> = None;
    let mut max: Option<BigInt> = None;
    for constraint in constraints {
        let bound = decode_bound(constraint, variable)?;
        match bound.side {
            BoundSide::Lower => {
                let effective = if bound.is_inclusive {
                    bound.value
                } else {
                    bound.value + 1
                };
                min = Some(match min {
                    Some(min) => min.max(effective),
                    None => effective,
                });
            }
            BoundSide::Upper => {
                let effective = if bound.is_inclusive {
                    bound.value
                } else {
                    bound.value - 1
                };
                max = Some(match max {
                    Some(max) => max.min(effective),
                    None => effective,
                });
            }
        }
    }
    if let (Some(min), Some(max)) = (&min, &max) {
        if min > max {
            return Err(IntervalError::Build(ParamBuildError::EmptyInterval(
                variable.clone(),
            )));
        }
    }
    Ok(Interval { min, max })
}

/// An end of an extended-integer interval.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum End {
    NegativeInfinity,
    Finite(BigInt),
    PositiveInfinity,
}

impl End {
    fn lower(value: Option<&BigInt>) -> Self {
        value.map_or(Self::NegativeInfinity, |value| Self::Finite(value.clone()))
    }

    fn upper(value: Option<&BigInt>) -> Self {
        value.map_or(Self::PositiveInfinity, |value| Self::Finite(value.clone()))
    }

    fn finite(self) -> Option<BigInt> {
        match self {
            Self::Finite(value) => Some(value),
            Self::NegativeInfinity | Self::PositiveInfinity => None,
        }
    }

    /// Return the sign of the end: -1, 0 or 1.
    fn sign(&self) -> i8 {
        match self {
            Self::Finite(value) if value.is_zero() => 0,
            Self::Finite(value) if value.is_negative() => -1,
            Self::NegativeInfinity => -1,
            Self::PositiveInfinity | Self::Finite(_) => 1,
        }
    }

    /// Return the product of two ends, per interval-product set semantics:
    /// a finite zero forces zero even against an infinite end.
    fn multiply(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Finite(left), Self::Finite(right)) => Self::Finite(left * right),
            _ if self.sign() == 0 || other.sign() == 0 => Self::Finite(BigInt::zero()),
            _ if self.sign() * other.sign() > 0 => Self::PositiveInfinity,
            _ => Self::NegativeInfinity,
        }
    }
}

/// Return the product interval of two intervals, spanning the least and the
/// greatest of the four pairwise products of their ends.
pub(super) fn multiply_intervals(left: &Interval, right: &Interval) -> Interval {
    let left_ends = [End::lower(left.min.as_ref()), End::upper(left.max.as_ref())];
    let right_ends = [
        End::lower(right.min.as_ref()),
        End::upper(right.max.as_ref()),
    ];
    let products: Vec<End> = left_ends
        .iter()
        .flat_map(|left| right_ends.iter().map(move |right| left.multiply(right)))
        .collect();
    let least = products
        .iter()
        .min()
        .cloned()
        .unwrap_or(End::NegativeInfinity);
    let greatest = products
        .iter()
        .max()
        .cloned()
        .unwrap_or(End::PositiveInfinity);
    Interval {
        min: least.finite(),
        max: greatest.finite(),
    }
}

/// Return `left` combined with `right` by `combine`, unbounded if either is.
pub(super) fn combine_ends(
    left: Option<&BigInt>,
    right: Option<&BigInt>,
    combine: impl FnOnce(&BigInt, &BigInt) -> BigInt,
) -> Option<BigInt> {
    Some(combine(left?, right?))
}

/// Return whether `min` may be rendered as the exclusive `> min - 1` for
/// `profile`: never when inclusive bounds are preferred, and on a
/// non-negative profile only when the gate admits the shifted literal.
fn is_exclusive_lower_rendering_valid(profile: IntervalProfile, min: &BigInt) -> bool {
    !profile.is_inclusive_preferred()
        && (!profile.is_non_negative()
            || is_valid_natural_bound(
                &(min - 1),
                BoundSide::Lower,
                profile.is_zero_included(),
                false,
            ))
}

/// Return whether `max` may be rendered as the exclusive `< max + 1`.
fn is_exclusive_upper_rendering_valid(profile: IntervalProfile, max: &BigInt) -> bool {
    !profile.is_inclusive_preferred()
        && (!profile.is_non_negative()
            || is_valid_natural_bound(
                &(max + 1),
                BoundSide::Upper,
                profile.is_zero_included(),
                false,
            ))
}

/// Return `param` bounded to `interval`, each bound rendered as its
/// domain's profile prefers.
fn apply_interval(
    param: Param,
    profile: IntervalProfile,
    interval: &Interval,
    context: &ParamContext<'_>,
) -> Result<Param, IntervalError> {
    let mut param = param;
    if let Some(min) = &interval.min {
        param = if is_exclusive_lower_rendering_valid(profile, min) {
            param.with_bound(
                &LiteralValue::Int(min - 1),
                BoundSide::Lower,
                false,
                context,
            )?
        } else {
            param.with_bound(
                &LiteralValue::Int(min.clone()),
                BoundSide::Lower,
                true,
                context,
            )?
        };
    }
    if let Some(max) = &interval.max {
        param = if is_exclusive_upper_rendering_valid(profile, max) {
            param.with_bound(
                &LiteralValue::Int(max + 1),
                BoundSide::Upper,
                false,
                context,
            )?
        } else {
            param.with_bound(
                &LiteralValue::Int(max.clone()),
                BoundSide::Upper,
                true,
                context,
            )?
        };
    }
    Ok(param)
}

/// Return a fresh interval param bounded to `interval`, over an interval
/// domain rendering as `template` prefers, restricted to the naturals as
/// `natural` says.
pub(super) fn build_interval_param(
    interval: &Interval,
    template: IntervalProfile,
    natural: Option<bool>,
    context: &ParamContext<'_>,
) -> Result<Param, IntervalError> {
    let domain = match natural {
        Some(zero_included) => IntervalIntegerDomain::new(
            Inclusivity::inclusive_if(template.is_inclusive_preferred()),
            Sign::NonNegative,
            ZeroInclusion::included_if(zero_included),
        ),
        None => IntervalIntegerDomain::new(
            Inclusivity::inclusive_if(template.is_inclusive_preferred()),
            Sign::Any,
            ZeroInclusion::Included,
        ),
    };
    let profile = ParamDomain::from(domain)
        .interval_profile()
        .map_err(profile_error)?
        .unwrap_or(template);
    let param = Param::new(
        ParamDomain::from(domain),
        Identifier::new("param"),
        Vec::new(),
        context,
    )?;
    apply_interval(param, profile, interval, context)
}

/// Return the interval error of a domain's failed profile, a custom
/// domain's error.
fn profile_error(error: super::error::ParamError) -> IntervalError {
    IntervalError::Custom(error.into_custom())
}

/// Return `param`'s profile if it is an interval operand as it stands: its
/// profile admits only bounds.
pub(super) fn operand_profile(param: &Param) -> Result<Option<IntervalProfile>, IntervalError> {
    Ok(param
        .domain()
        .interval_profile()
        .map_err(profile_error)?
        .filter(IntervalProfile::is_bounds_only))
}

/// Return `other` as an interval operand rendering as `template` prefers:
/// an integer as the exact interval it denotes; an operand param as it
/// stands; an integer param whose constraints are all bounds recast over an
/// interval domain.
///
/// # Errors
///
/// Returns [`IntervalError::NonBoundOperand`] for a param with a constraint
/// that is no bound, and [`IntervalError::UnsupportedOperand`] for a param
/// whose domain has no profile.
pub(super) fn coerce_to_interval(
    template: IntervalProfile,
    other: &Operand,
    context: &ParamContext<'_>,
) -> Result<Param, IntervalError> {
    match other {
        Operand::Integer(value) => {
            let domain = IntervalIntegerDomain::new(
                Inclusivity::inclusive_if(template.is_inclusive_preferred()),
                Sign::Any,
                ZeroInclusion::Included,
            );
            let literal = LiteralValue::Int(value.clone());
            Param::new(
                ParamDomain::from(domain),
                Identifier::new("param"),
                Vec::new(),
                context,
            )?
            .with_bound(&literal, BoundSide::Lower, true, context)?
            .with_bound(&literal, BoundSide::Upper, true, context)
            .map_err(IntervalError::from)
        }
        Operand::Param(param) => {
            let Some(profile) = param.domain().interval_profile().map_err(profile_error)? else {
                return Err(IntervalError::UnsupportedOperand);
            };
            if profile.is_bounds_only() {
                return Ok(param.clone());
            }
            for constraint in param.constraints() {
                let expression = constraint
                    .to_expression()
                    .map_err(|error| IntervalError::NonBoundOperand(Some(error)))?;
                if !is_bound_expression(&expression) {
                    return Err(IntervalError::NonBoundOperand(None));
                }
            }
            Param::new(
                ParamDomain::from(IntervalIntegerDomain::new(
                    Inclusivity::inclusive_if(template.is_inclusive_preferred()),
                    Sign::Any,
                    ZeroInclusion::Included,
                )),
                param.variable().clone(),
                param.constraints().to_vec(),
                context,
            )
            .map_err(IntervalError::from)
        }
    }
}
