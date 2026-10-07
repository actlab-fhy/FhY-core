//! What a static step over a decision of a space is: its kind and its
//! domain, derived from a choice's alternatives or a variable's param; how
//! a configuration grows by one value; and the walk over a domain's
//! coordinates that the oracles share.

use std::num::NonZeroU64;
use std::slice;

use num_bigint::{BigInt, BigUint};

use crate::constraint::{Member, MemberKind, Value};
use crate::identifier::Identifier;
use crate::param::{
    IntervalError, Param, ParamBuildError, ParamContext, ParamDomain, effective_interval,
};

use super::configuration::Configuration;
use super::domain::{
    ChoiceDomain, Coordinate, DecisionKind, OrderDomain, StepDomain, StridedDomain, StridedRun,
};
use super::error::{ConfigurationError, TraceError};
use super::rng::Rng;
use super::space::Decision;

/// The domain a param's values offer a step, if it has a finite one.
pub(super) enum ParamStepDomain {
    /// A finite domain.
    Finite(StepDomain),
    /// No value: an integer domain whose bounds enclose no integer. A step
    /// over it is a dead end, and it counts no configuration.
    Empty,
    /// An unbounded integer or a real domain.
    Unbounded,
    /// A custom domain.
    Unknown,
}

/// Return the kind of a static step over `decision`.
pub(super) fn decision_kind(decision: Decision<'_>) -> DecisionKind {
    match decision {
        Decision::Choice(_) => DecisionKind::choice(),
        Decision::Variable(variable) => DecisionKind::of_variable(&variable.get().kind()),
    }
}

/// Return the domain a static step over `decision` offers.
///
/// # Errors
///
/// Returns [`TraceError::Hook`] for a failing
/// [`search_domain`](super::Variable::search_domain),
/// [`TraceError::DeadEnd`] for a variable whose param admits no value, and
/// [`TraceError::NotEnumerable`] for a variable with no finite domain.
pub(super) fn decision_domain(decision: Decision<'_>) -> Result<StepDomain, TraceError> {
    match decision {
        Decision::Choice(choice) => {
            let names = choice
                .alternatives()
                .iter()
                .map(|alternative| Value::Identifier(alternative.get().name().clone()))
                .collect();
            ChoiceDomain::new(names)
                .map(StepDomain::from)
                .map_err(TraceError::Domain)
        }
        Decision::Variable(variable) => {
            let part = variable.get();
            let offered = part.search_domain().map_err(|source| TraceError::Hook {
                decision: part.name().clone(),
                source,
            })?;
            if let Some(domain) = offered {
                return Ok(domain);
            }
            match param_step_domain(part.param()) {
                ParamStepDomain::Finite(domain) => Ok(domain),
                ParamStepDomain::Empty => Err(TraceError::DeadEnd {
                    decision: part.name().clone(),
                }),
                ParamStepDomain::Unbounded | ParamStepDomain::Unknown => {
                    Err(TraceError::NotEnumerable {
                        decision: part.name().clone(),
                    })
                }
            }
        }
    }
}

/// Return the domain `param`'s values offer a step: its categories or
/// ordinal values, the orderings of its permutation members, or the
/// integers of its interval when both ends are bounded, none when they
/// enclose no integer.
pub(super) fn param_step_domain(param: &Param) -> ParamStepDomain {
    let values = |members: &[Member]| members.iter().map(member_value).collect::<Vec<_>>();
    let finite = |domain: Result<StepDomain, _>| {
        domain.map_or(ParamStepDomain::Unknown, ParamStepDomain::Finite)
    };
    match param.domain() {
        ParamDomain::Categorical(domain) => {
            finite(ChoiceDomain::new(values(domain.values())).map(StepDomain::from))
        }
        ParamDomain::Ordinal(domain) => {
            finite(ChoiceDomain::new(values(domain.values())).map(StepDomain::from))
        }
        ParamDomain::Permutation(domain) => {
            finite(OrderDomain::new(values(domain.values())).map(StepDomain::from))
        }
        ParamDomain::Integer(domain) => {
            interval_domain(param, domain.is_non_negative(), domain.is_zero_included())
        }
        ParamDomain::IntervalInteger(domain) => {
            interval_domain(param, domain.is_non_negative(), domain.is_zero_included())
        }
        ParamDomain::Real(_) => ParamStepDomain::Unbounded,
        ParamDomain::Custom(_) => ParamStepDomain::Unknown,
    }
}

/// Return whether every constraint of `param` is a bound of its variable.
pub(super) fn are_bounds(param: &Param) -> bool {
    param
        .constraints()
        .iter()
        .all(|constraint| effective_interval(slice::from_ref(constraint), param.variable()).is_ok())
}

/// Return the strided run over the integers `param`'s bound constraints
/// and its sign enclose, if both ends are bounded: empty when the bounds,
/// or a lower bound above an upper, enclose none.
fn interval_domain(
    param: &Param,
    is_non_negative: bool,
    is_zero_included: bool,
) -> ParamStepDomain {
    let variable = param.variable();
    let bounds: Vec<_> = param
        .constraints()
        .iter()
        .filter(|constraint| effective_interval(slice::from_ref(*constraint), variable).is_ok())
        .cloned()
        .collect();
    let interval = match effective_interval(&bounds, variable) {
        Ok(interval) => interval,
        Err(IntervalError::Build(ParamBuildError::EmptyInterval(_))) => {
            return ParamStepDomain::Empty;
        }
        Err(_) => return ParamStepDomain::Unbounded,
    };
    let implied = is_non_negative.then(|| BigInt::from(u8::from(!is_zero_included)));
    let lower = match (interval.min, implied) {
        (Some(min), Some(implied)) => Some(min.max(implied)),
        (min, implied) => min.or(implied),
    };
    let (Some(lower), Some(upper)) = (lower, interval.max) else {
        return ParamStepDomain::Unbounded;
    };
    if lower > upper {
        return ParamStepDomain::Empty;
    }
    StridedRun::new(lower, upper + 1, BigUint::from(1_u8))
        .and_then(|run| StridedDomain::new(vec![run]))
        .map_or(ParamStepDomain::Unbounded, |domain| {
            ParamStepDomain::Finite(StepDomain::from(domain))
        })
}

/// Return the value `member` holds.
pub(super) fn member_value(member: &Member) -> Value {
    match member.kind() {
        MemberKind::Bool(value) => Value::Bool(value),
        MemberKind::Int(value) => Value::Int(value.clone()),
        MemberKind::Float(value) => Value::Float(value),
        MemberKind::Str(value) => Value::Str(value.to_owned()),
        MemberKind::Identifier(value) => Value::Identifier(value.clone()),
        MemberKind::Tuple(members) => Value::Tuple(members.iter().map(member_value).collect()),
        MemberKind::FrozenSet(members) => {
            Value::FrozenSet(members.iter().map(member_value).collect())
        }
        MemberKind::Opaque(part) => Value::Opaque(part.clone()),
    }
}

/// Return `configuration` with the decision `name` given `value`, or `None`
/// when the value is not admissible: its param refuses it, or it completes
/// a forbidden clause that holds.
///
/// # Errors
///
/// Returns [`TraceError::Configuration`] for any other refusal.
pub(super) fn try_extend(
    configuration: &Configuration,
    name: &Identifier,
    value: Value,
    context: &ParamContext<'_>,
) -> Result<Option<Configuration>, TraceError> {
    match configuration.extended(name.clone(), value, context) {
        Ok(extended) => Ok(Some(extended)),
        Err(errors) => {
            let is_inadmissible = errors.errors().iter().all(|problem| {
                matches!(
                    problem,
                    ConfigurationError::Forbidden { .. } | ConfigurationError::Assignment { .. }
                )
            });
            if is_inadmissible {
                Ok(None)
            } else {
                Err(TraceError::Configuration(errors))
            }
        }
    }
}

/// Return the first coordinate of `domain`: index 0, or the identity
/// ordering.
pub(super) fn first_coordinate(domain: &StepDomain) -> Coordinate {
    match domain {
        StepDomain::Order(order) => Coordinate::Order(identity_positions(order.elements().len())),
        _ => Coordinate::Index(0),
    }
}

/// Return the coordinate after `coordinate`, in lexicographic order, of
/// the domain whose coordinates `contains` holds, or `None` after the last.
pub(super) fn next_coordinate(
    contains: impl Fn(&Coordinate) -> bool,
    coordinate: &Coordinate,
) -> Option<Coordinate> {
    match coordinate {
        Coordinate::Index(index) => {
            let next = Coordinate::Index(index.checked_add(1)?);
            contains(&next).then_some(next)
        }
        Coordinate::Order(positions) => {
            let mut next = positions.clone();
            next_permutation(&mut next).then_some(Coordinate::Order(next))
        }
    }
}

/// Return the positions `0..count` in order.
pub(super) fn identity_positions(count: usize) -> Box<[u32]> {
    (0..count)
        .map(|position| u32::try_from(position).unwrap_or(u32::MAX))
        .collect()
}

/// Move `positions` to the next permutation in lexicographic order; return
/// `false`, leaving it unchanged, after the last.
pub(super) fn next_permutation(positions: &mut [u32]) -> bool {
    let Some(pivot) = positions.windows(2).rposition(|pair| pair[0] < pair[1]) else {
        return false;
    };
    let Some(successor) = positions
        .iter()
        .rposition(|&position| position > positions[pivot])
    else {
        return false;
    };
    positions.swap(pivot, successor);
    positions[pivot + 1..].reverse();
    true
}

/// Return the permutation of `count` positions of lexicographic rank
/// `rank`, which is below `count!`.
pub(super) fn unrank_permutation(count: usize, rank: &BigUint) -> Box<[u32]> {
    let mut remaining: Vec<u32> = identity_positions(count).into_vec();
    let mut rank = rank.clone();
    let mut positions = Vec::with_capacity(count);
    for left in (0..count).rev() {
        let block: BigUint = (1..=left).map(BigUint::from).product();
        let digit = usize::try_from(&rank / &block).unwrap_or(0);
        rank %= &block;
        positions.push(remaining.remove(digit.min(remaining.len().saturating_sub(1))));
    }
    positions.into()
}

/// Return a coordinate of `domain` drawn uniformly with `rng`: an index by
/// `below`, an ordering by shuffling the positions.
pub(super) fn draw_coordinate(domain: &StepDomain, rng: &mut Rng) -> Coordinate {
    match domain {
        StepDomain::Order(order) => {
            let mut positions = identity_positions(order.elements().len());
            rng.shuffle(&mut positions);
            Coordinate::Order(positions)
        }
        StepDomain::Choice(choice) => Coordinate::Index(draw_index(choice.cardinality(), rng)),
        StepDomain::Strided(strided) => Coordinate::Index(draw_index(strided.cardinality(), rng)),
    }
}

/// Return an index below `count`, which is at least one, drawn with `rng`.
fn draw_index(count: u64, rng: &mut Rng) -> u64 {
    NonZeroU64::new(count).map_or(0, |bound| rng.below(bound))
}
