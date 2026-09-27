//! The union and intersection of domains.

use crate::constraint::{Constraint, Member, MemberSet, Value};
use crate::identifier::Identifier;

use super::context::ParamContext;
use super::decide::is_value_valid_for;
use super::domain::{
    CategoricalDomain, DomainKind, IntegerDomain, IntervalIntegerDomain, OrdinalDomain,
    ParamDomain, RealDomain, Side, Sign, ZeroInclusion,
};
use super::error::{ParamError, SetOperation};
use super::interval::Inclusivity;
use super::screen::merge_intersection_constraints;
use super::value::member_value;

/// Return the members of `members` that are valid for `side` over `domain`,
/// in order.
fn effective_values(
    domain: &ParamDomain,
    side: Side<'_>,
    members: &[Member],
    context: &ParamContext<'_>,
) -> Result<Vec<Member>, ParamError> {
    let mut valid = Vec::with_capacity(members.len());
    for member in members {
        if is_value_valid_for(domain, side, &member_value(member), context)? {
            valid.push(member.clone());
        }
    }
    Ok(valid)
}

/// Return the values of a finite domain, in its order.
fn finite_members(domain: &ParamDomain) -> &[Member] {
    match domain {
        ParamDomain::Ordinal(domain) => domain.values(),
        ParamDomain::Categorical(domain) => domain.values(),
        ParamDomain::Permutation(domain) => domain.values(),
        _ => &[],
    }
}

/// Return the type-strict union of two member sequences: `own` in full,
/// then each member of `other` that no member collected equals.
fn merge_values(own: Vec<Member>, other: Vec<Member>) -> Vec<Member> {
    let mut merged = own;
    for member in other {
        if !MemberSet::new(merged.iter().cloned()).contains(&member) {
            merged.push(member);
        }
    }
    merged
}

/// Return the members of `own` that `other` holds, type-strictly.
fn intersect_values(own: Vec<Member>, other: &[Member]) -> Vec<Member> {
    let other = MemberSet::new(other.iter().cloned());
    own.into_iter()
        .filter(|member| other.contains(member))
        .collect()
}

/// Return the finite domain of `kind` holding `members`.
fn build_finite(kind: DomainKind, members: &[Member]) -> Result<ParamDomain, ParamError> {
    let values: Vec<Value> = members.iter().map(member_value).collect();
    Ok(match kind {
        DomainKind::Ordinal => ParamDomain::from(OrdinalDomain::new(values)?),
        _ => ParamDomain::from(CategoricalDomain::new(values)?),
    })
}

/// Refuse `other_domain` unless it is of `own_domain`'s kind.
fn require_same_kind(
    operation: SetOperation,
    own_domain: &ParamDomain,
    other_domain: &ParamDomain,
) -> Result<(), ParamError> {
    if own_domain.kind() == other_domain.kind() {
        Ok(())
    } else {
        Err(ParamError::KindMismatch {
            operation,
            own: own_domain.kind(),
            other: other_domain.kind(),
        })
    }
}

/// Return the union of two domains' value sets, as
/// [`ParamDomain::union`] describes.
pub(super) fn union(
    own_domain: &ParamDomain,
    own: Side<'_>,
    other_domain: &ParamDomain,
    other: Side<'_>,
    variable: &Identifier,
    context: &ParamContext<'_>,
) -> Result<Option<(ParamDomain, Vec<Constraint>)>, ParamError> {
    match own_domain {
        ParamDomain::Ordinal(_) | ParamDomain::Categorical(_) => {
            require_same_kind(SetOperation::Union, own_domain, other_domain)?;
            let own_values =
                effective_values(own_domain, own, finite_members(own_domain), context)?;
            let other_values =
                effective_values(other_domain, other, finite_members(other_domain), context)?;
            let merged = merge_values(own_values, other_values);
            if merged.is_empty() {
                return Err(ParamError::EmptyUnion(own_domain.kind()));
            }
            Ok(Some((
                build_finite(own_domain.kind(), &merged)?,
                Vec::new(),
            )))
        }
        ParamDomain::Custom(custom) => custom
            .get()
            .union(own, other_domain, other, variable, context)
            .map_err(ParamError::Custom),
        _ => Ok(None),
    }
}

/// Return the `(non_negative, zero_included)` restrictions an intersection
/// of two integer domains inherits: non-negative if either is, and without
/// zero if either non-negative side excludes it.
fn merge_restrictions(left: (bool, bool), right: (bool, bool)) -> (bool, bool) {
    let (left_non_negative, left_zero) = left;
    let (right_non_negative, right_zero) = right;
    (
        left_non_negative || right_non_negative,
        (!left_non_negative || left_zero) && (!right_non_negative || right_zero),
    )
}

/// Return the intersection of two domains' value sets, as
/// [`ParamDomain::intersection`] describes.
pub(super) fn intersection(
    own_domain: &ParamDomain,
    own: Side<'_>,
    other_domain: &ParamDomain,
    other: Side<'_>,
    variable: &Identifier,
    context: &ParamContext<'_>,
) -> Result<(ParamDomain, Vec<Constraint>), ParamError> {
    if let ParamDomain::Custom(custom) = own_domain {
        return custom
            .get()
            .intersection(own, other_domain, other, variable, context)
            .map_err(ParamError::Custom);
    }
    require_same_kind(SetOperation::Intersection, own_domain, other_domain)?;
    match (own_domain, other_domain) {
        (ParamDomain::Integer(left), ParamDomain::Integer(right)) => {
            let (non_negative, zero_included) = merge_restrictions(
                (left.is_non_negative(), left.is_zero_included()),
                (right.is_non_negative(), right.is_zero_included()),
            );
            Ok((
                ParamDomain::from(IntegerDomain::new(
                    Sign::non_negative_if(non_negative),
                    ZeroInclusion::included_if(zero_included),
                )),
                merge_intersection_constraints(own, other, variable)?,
            ))
        }
        (ParamDomain::IntervalInteger(left), ParamDomain::IntervalInteger(right)) => {
            let (non_negative, zero_included) = merge_restrictions(
                (left.is_non_negative(), left.is_zero_included()),
                (right.is_non_negative(), right.is_zero_included()),
            );
            Ok((
                ParamDomain::from(IntervalIntegerDomain::new(
                    Inclusivity::inclusive_if(left.is_inclusive_preferred()),
                    Sign::non_negative_if(non_negative),
                    ZeroInclusion::included_if(zero_included),
                )),
                merge_intersection_constraints(own, other, variable)?,
            ))
        }
        (ParamDomain::Real(_), ParamDomain::Real(_)) => Ok((
            ParamDomain::from(RealDomain),
            merge_intersection_constraints(own, other, variable)?,
        )),
        (ParamDomain::Permutation(_), ParamDomain::Permutation(_)) => {
            if !(own_domain.is_value_set_subset(other_domain, context)?
                && other_domain.is_value_set_subset(own_domain, context)?)
            {
                return Err(ParamError::DifferentPermutationMembers);
            }
            Ok((
                own_domain.clone(),
                merge_intersection_constraints(own, other, variable)?,
            ))
        }
        _ => {
            let own_values =
                effective_values(own_domain, own, finite_members(own_domain), context)?;
            let other_values =
                effective_values(other_domain, other, finite_members(other_domain), context)?;
            let common = intersect_values(own_values, &other_values);
            if common.is_empty() {
                return Err(ParamError::EmptyIntersection(own_domain.kind()));
            }
            Ok((build_finite(own_domain.kind(), &common)?, Vec::new()))
        }
    }
}
