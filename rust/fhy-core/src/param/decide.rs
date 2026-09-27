//! The decision procedures of domains: evaluating constraints under
//! bindings, enumerating in-set candidates, and asking the solver about
//! screened systems.

use std::collections::HashMap;

use crate::constraint::{
    Binding, Bindings, Constraint, ConstraintError, ConstraintSystem, Member, MemberKind,
    MemberSet, Outcome, Polarity, SetConstraint, Value,
};
use crate::expression::SymbolType;
use crate::identifier::Identifier;
use crate::solver::CheckLimits;

use super::context::{MemberForwarder, ParamContext, ParamEvent, QuestionForwarder};
use super::domain::{ParamDomain, Side};
use super::error::{ParamBuildError, ParamError};
use super::screen::{Screened, rename_system, screen};
use super::value::member_value;

/// The outcome of a conjunction of constraints under bindings, and the
/// member it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Evaluation {
    outcome: Outcome,
    deciding_member: Option<usize>,
}

impl Evaluation {
    /// Return the conjunction's outcome: [`Outcome::Violated`] at the first
    /// violated member, [`Outcome::Satisfied`] if every member is, and
    /// [`Outcome::Undecided`] otherwise.
    #[must_use]
    pub fn outcome(&self) -> Outcome {
        self.outcome
    }

    /// Return the position of the member the outcome came from: the first
    /// violated member, else the first undecided one; `None` for a
    /// satisfied conjunction.
    #[must_use]
    pub fn deciding_member(&self) -> Option<usize> {
        self.deciding_member
    }
}

/// Evaluate `constraint` alone under `bindings`, reporting its events.
fn evaluate_member(
    constraint: &Constraint,
    bindings: &Bindings,
    context: &ParamContext<'_>,
) -> Result<Outcome, crate::constraint::ConstraintError> {
    let forwarder = MemberForwarder {
        context,
        constraint,
        bindings,
    };
    constraint.evaluate(bindings, &context.constraint_context_with(&forwarder))
}

/// Evaluate the conjunction of `constraints` under `bindings`, member by
/// member in order.
///
/// A violated member answers at once. An undecided one is reported as
/// [`ParamEvent::UndecidedMember`], and so is one whose evaluation fails
/// with an error the context's observer judges undecidable, after
/// [`ParamEvent::BridgeFailed`]. Each member's own events are reported as
/// [`ParamEvent::Member`].
///
/// # Errors
///
/// Returns the first member's error the observer does not judge
/// undecidable.
pub fn evaluate_constraints(
    constraints: &[Constraint],
    bindings: &Bindings,
    context: &ParamContext<'_>,
) -> Result<Evaluation, ConstraintError> {
    let mut first_undecided = None;
    for (index, constraint) in constraints.iter().enumerate() {
        let outcome = match evaluate_member(constraint, bindings, context) {
            Ok(outcome) => outcome,
            Err(error) if context.is_undecidable(&error) => {
                context.notify(&ParamEvent::BridgeFailed {
                    constraint,
                    bindings,
                    error: &error,
                });
                Outcome::Undecided
            }
            Err(error) => return Err(error),
        };
        match outcome {
            Outcome::Violated => {
                return Ok(Evaluation {
                    outcome: Outcome::Violated,
                    deciding_member: Some(index),
                });
            }
            Outcome::Undecided => {
                context.notify(&ParamEvent::UndecidedMember { constraint });
                first_undecided.get_or_insert(index);
            }
            Outcome::Satisfied => {}
        }
    }
    Ok(Evaluation {
        outcome: if first_undecided.is_some() {
            Outcome::Undecided
        } else {
            Outcome::Satisfied
        },
        deciding_member: first_undecided,
    })
}

/// Return whether every one of `constraints`, each evaluated alone under
/// `bindings`, is satisfied, stopping at the first that is not.
///
/// # Errors
///
/// Returns the first member's error.
pub fn are_all_constraints_satisfied(
    constraints: &[Constraint],
    bindings: &Bindings,
    context: &ParamContext<'_>,
) -> Result<bool, ConstraintError> {
    for constraint in constraints {
        if evaluate_member(constraint, bindings, context)? != Outcome::Satisfied {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Return the question-level error of a domain's failure to give its
/// implied constraints: a custom domain's own error, as its other hooks'.
pub(super) fn restriction_error(error: ParamBuildError) -> ParamError {
    match error {
        ParamBuildError::Custom(error) => ParamError::Custom(error),
        other => ParamError::Build(other),
    }
}

/// Return `side`'s constraints with each of `domain`'s implied constraints
/// on its variable the side does not already hold, so a domain-level
/// procedure respects the domain's restriction as a param's side, which
/// holds them, does.
pub(super) fn restricted(
    domain: &ParamDomain,
    side: Side<'_>,
) -> Result<Vec<Constraint>, ParamError> {
    let mut constraints = side.constraints().to_vec();
    for implied in domain
        .implied_constraints(side.variable())
        .map_err(restriction_error)?
    {
        if !constraints
            .iter()
            .any(|constraint| constraint.is_structurally_equivalent(&implied))
        {
            constraints.push(implied);
        }
    }
    Ok(constraints)
}

/// Return the bindings of `value` to `variable`.
fn bind(variable: &Identifier, value: Value) -> Bindings {
    Bindings::from_iter([(variable.clone(), Binding::Value(value))])
}

/// Return whether `value` is admissible in `domain` and, bound to
/// `side`'s variable, satisfies each of `side`'s constraints alone.
pub(super) fn is_value_valid_for(
    domain: &ParamDomain,
    side: Side<'_>,
    value: &Value,
    context: &ParamContext<'_>,
) -> Result<bool, ParamError> {
    if !domain.is_value_admissible(value)? {
        return Ok(false);
    }
    Ok(are_all_constraints_satisfied(
        side.constraints(),
        &bind(side.variable(), value.clone()),
        context,
    )?)
}

/// Return the set constraints of `constraints` of `polarity`.
fn set_constraints(
    constraints: &[Constraint],
    polarity: Polarity,
) -> impl Iterator<Item = &SetConstraint> {
    constraints
        .iter()
        .filter_map(move |constraint| match constraint {
            Constraint::Set(set) if set.polarity() == polarity => Some(set),
            _ => None,
        })
}

/// Return whether `constraints` holds an in-set constraint.
fn has_in_set(constraints: &[Constraint]) -> bool {
    set_constraints(constraints, Polarity::In).next().is_some()
}

/// Return the in-set candidates of `constraints`: the members of the
/// first in-set constraint, in canonical order, held by every other
/// in-set constraint and by no not-in-set constraint.
fn in_set_candidates(constraints: &[Constraint]) -> Vec<Member> {
    let mut in_sets = set_constraints(constraints, Polarity::In);
    let Some(first) = in_sets.next() else {
        return Vec::new();
    };
    let others: Vec<&MemberSet> = in_sets.map(SetConstraint::members).collect();
    let excluded: Vec<&MemberSet> = set_constraints(constraints, Polarity::NotIn)
        .map(SetConstraint::members)
        .collect();
    first
        .members()
        .iter()
        .filter(|candidate| {
            others.iter().all(|members| members.contains(candidate))
                && !excluded.iter().any(|members| members.contains(candidate))
        })
        .cloned()
        .collect()
}

/// Return the equations of `constraints`.
fn equations(constraints: &[Constraint]) -> Vec<Constraint> {
    constraints
        .iter()
        .filter(|constraint| matches!(constraint, Constraint::Equation(_)))
        .cloned()
        .collect()
}

/// Return the outcome of `candidate` for `side` over `domain`:
/// [`Outcome::Violated`] if the domain does not admit it, and otherwise the
/// outcome of `side`'s equations, in canonical order, with it bound.
fn evaluate_candidate(
    domain: &ParamDomain,
    variable: &Identifier,
    equations: &[Constraint],
    candidate: &Member,
    context: &ParamContext<'_>,
) -> Result<Outcome, ParamError> {
    let value = member_value(candidate);
    if !domain.is_value_admissible(&value)? {
        return Ok(Outcome::Violated);
    }
    Ok(evaluate_constraints(equations, &bind(variable, value), context)?.outcome())
}

/// Return the equations of `side` in the canonical order of their system.
fn canonical_equations(side: Side<'_>) -> Result<Vec<Constraint>, ParamError> {
    Ok(ConstraintSystem::new(equations(side.constraints()))
        .map_err(ParamError::Constraint)?
        .constraints()
        .to_vec())
}

/// Decide feasibility from each in-set candidate's outcome: satisfied at
/// the first candidate decided so, violated when every candidate is decided
/// violated, and otherwise undecided, reported as
/// [`ParamEvent::EnumerationUndecided`].
fn decide_feasibility_by_enumeration(
    domain: &ParamDomain,
    side: Side<'_>,
    context: &ParamContext<'_>,
) -> Result<Outcome, ParamError> {
    let equations = canonical_equations(side)?;
    let mut undecided = Vec::new();
    for candidate in in_set_candidates(side.constraints()) {
        match evaluate_candidate(domain, side.variable(), &equations, &candidate, context)? {
            Outcome::Satisfied => return Ok(Outcome::Satisfied),
            Outcome::Undecided => undecided.push(candidate),
            Outcome::Violated => {}
        }
    }
    if undecided.is_empty() {
        return Ok(Outcome::Violated);
    }
    context.notify(&ParamEvent::EnumerationUndecided {
        variable: side.variable(),
        candidates: &undecided,
    });
    Ok(Outcome::Undecided)
}

/// Return `other`'s outcome for `candidate`: violated if `other_domain`
/// does not admit it or a set constraint refuses it, and otherwise the
/// outcome of `other`'s equations with it bound.
fn evaluate_candidate_against_other(
    other_domain: &ParamDomain,
    other: Side<'_>,
    candidate: &Member,
    context: &ParamContext<'_>,
) -> Result<Outcome, ParamError> {
    let value = member_value(candidate);
    if !other_domain.is_value_admissible(&value)? {
        return Ok(Outcome::Violated);
    }
    for constraint in other.constraints() {
        if let Constraint::Set(set) = constraint {
            let is_member = set.members().contains(candidate);
            let is_refused = match set.polarity() {
                Polarity::NotIn => is_member,
                _ => !is_member,
            };
            if is_refused {
                return Ok(Outcome::Violated);
            }
        }
    }
    Ok(evaluate_constraints(
        &canonical_equations(other)?,
        &bind(other.variable(), value),
        context,
    )?
    .outcome())
}

/// Decide the subset relation from `own`'s in-set candidates, as
/// [`compute_constraint_implication_subset`] describes.
fn decide_subset_by_enumerating_own(
    own_domain: &ParamDomain,
    own: Side<'_>,
    other_domain: &ParamDomain,
    other: Side<'_>,
    context: &ParamContext<'_>,
) -> Result<Outcome, ParamError> {
    let equations = canonical_equations(own)?;
    let mut undecided = Vec::new();
    for candidate in in_set_candidates(own.constraints()) {
        let own_outcome =
            evaluate_candidate(own_domain, own.variable(), &equations, &candidate, context)?;
        if own_outcome == Outcome::Violated {
            continue;
        }
        let other_outcome =
            evaluate_candidate_against_other(other_domain, other, &candidate, context)?;
        if other_outcome == Outcome::Satisfied {
            continue;
        }
        if own_outcome == Outcome::Satisfied && other_outcome == Outcome::Violated {
            return Ok(Outcome::Violated);
        }
        undecided.push(candidate);
    }
    if undecided.is_empty() {
        return Ok(Outcome::Satisfied);
    }
    context.notify(&ParamEvent::SubsetEnumerationUndecided {
        own: own.variable(),
        other: other.variable(),
        candidates: &undecided,
    });
    Ok(Outcome::Undecided)
}

/// Ask whether `system` is satisfiable with `variable` of `symbol_type`.
fn ask_satisfiability(
    system: &ConstraintSystem,
    variable: &Identifier,
    symbol_type: SymbolType,
    context: &ParamContext<'_>,
) -> Result<Outcome, ParamError> {
    let symbol_types = HashMap::from([(variable.clone(), symbol_type)]);
    let forwarder = QuestionForwarder {
        context,
        system,
        symbol_types: &symbol_types,
    };
    system
        .check_satisfiability(
            &symbol_types,
            CheckLimits::new(),
            &context.constraint_context_with(&forwarder),
        )
        .map_err(ParamError::Constraint)
}

/// Return whether `own` provably admits a value outside `permitted`: only
/// on an exact screened system, and only when the solver decides the
/// system with the exclusion satisfiable.
fn does_own_admit_a_value_outside(
    own: Side<'_>,
    permitted: Vec<Member>,
    symbol_type: SymbolType,
    context: &ParamContext<'_>,
) -> Result<bool, ParamError> {
    let Screened { system, is_exact } = screen(own.constraints(), own.variable(), context)?;
    if !is_exact {
        return Ok(false);
    }
    let common = Identifier::new("var");
    let renamed = rename_system(&system, own.variable(), &common)?;
    let permitted_count = permitted.len();
    let exclusion = SetConstraint::new(common.clone(), MemberSet::new(permitted), Polarity::NotIn);
    if exclusion.to_expression().is_err() {
        return Ok(false);
    }
    let mut members = renamed.constraints().to_vec();
    members.push(Constraint::from(exclusion));
    let witness = ConstraintSystem::new(members).map_err(ParamError::Constraint)?;
    if ask_satisfiability(&witness, &common, symbol_type, context)? != Outcome::Satisfied {
        return Ok(false);
    }
    context.notify(&ParamEvent::WitnessOutside {
        variable: own.variable(),
        permitted: permitted_count,
    });
    Ok(true)
}

/// Return whether one of `constraints` of `polarity` on `variable` holds a
/// float member.
fn has_float_member(constraints: &[Constraint], variable: &Identifier, polarity: Polarity) -> bool {
    set_constraints(constraints, polarity).any(|set| {
        set.variable() == variable
            && set
                .members()
                .iter()
                .any(|member| matches!(member.kind(), MemberKind::Float(_)))
    })
}

/// Decide whether `own`'s admissible set over `own_domain` is a subset of
/// `other`'s over `other_domain`, reasoning about their values in
/// `symbol_type`.
///
/// Each side holds its domain's implied constraints as well as its own, as
/// a param's side does, so the domains' restrictions count.
///
/// 1. When `own` holds an in-set constraint, its candidates are
///    enumerated: a candidate decided in `own` and decided out of `other`
///    is a counterexample ([`Outcome::Violated`]); `other` deciding every
///    candidate not decided out of `own` proves the relation
///    ([`Outcome::Satisfied`]); anything else is undecided, reported as
///    [`ParamEvent::SubsetEnumerationUndecided`].
/// 2. When only `other` holds one, its candidates not decided out are the
///    permitted values, and a value `own` provably admits outside them,
///    on an exact screened system, is a counterexample, reported as
///    [`ParamEvent::WitnessOutside`].
/// 3. Otherwise, or when no witness is found, both screened systems are
///    moved onto one fresh variable and the solver is asked whether `own`
///    implies `other`. An answer that rests on a weakened side is
///    undecided, reported as [`ParamEvent::ImplicationDowngraded`]: a
///    violation with an inexact antecedent, a satisfaction with an inexact
///    consequent, and over the REAL sort a satisfaction with a float member
///    in `own`'s not-in-set or `other`'s in-set constraints, which the sort
///    conflates with other kinds. A solver that gives up is reported as
///    [`ParamEvent::ImplicationUndecided`].
///
/// # Errors
///
/// Returns a constraint's error, and [`ParamError::Custom`] for a custom
/// domain that fails.
pub fn compute_constraint_implication_subset(
    own_domain: &ParamDomain,
    own: Side<'_>,
    other_domain: &ParamDomain,
    other: Side<'_>,
    symbol_type: SymbolType,
    context: &ParamContext<'_>,
) -> Result<Outcome, ParamError> {
    let own_constraints = restricted(own_domain, own)?;
    let other_constraints = restricted(other_domain, other)?;
    implication_subset(
        own_domain,
        Side::new(&own_constraints, own.variable()),
        other_domain,
        Side::new(&other_constraints, other.variable()),
        symbol_type,
        context,
    )
}

/// Decide the subset relation of two sides that hold their domains'
/// implied constraints, as [`compute_constraint_implication_subset`]
/// describes.
fn implication_subset(
    own_domain: &ParamDomain,
    own: Side<'_>,
    other_domain: &ParamDomain,
    other: Side<'_>,
    symbol_type: SymbolType,
    context: &ParamContext<'_>,
) -> Result<Outcome, ParamError> {
    if has_in_set(own.constraints()) {
        return decide_subset_by_enumerating_own(own_domain, own, other_domain, other, context);
    }
    if has_in_set(other.constraints()) {
        let equations = canonical_equations(other)?;
        let mut permitted = Vec::new();
        for candidate in in_set_candidates(other.constraints()) {
            let outcome = evaluate_candidate(
                other_domain,
                other.variable(),
                &equations,
                &candidate,
                context,
            )?;
            if outcome != Outcome::Violated {
                permitted.push(candidate);
            }
        }
        if does_own_admit_a_value_outside(own, permitted, symbol_type, context)? {
            return Ok(Outcome::Violated);
        }
    }
    let common = Identifier::new("var");
    let own_screened = screen(own.constraints(), own.variable(), context)?;
    let other_screened = screen(other.constraints(), other.variable(), context)?;
    let antecedent = rename_system(&own_screened.system, own.variable(), &common)?;
    let consequent = rename_system(&other_screened.system, other.variable(), &common)?;
    let symbol_types = HashMap::from([(common.clone(), symbol_type)]);
    let forwarder = QuestionForwarder {
        context,
        system: &antecedent,
        symbol_types: &symbol_types,
    };
    let outcome = antecedent
        .check_implication(
            &consequent,
            &symbol_types,
            CheckLimits::new(),
            &context.constraint_context_with(&forwarder),
        )
        .map_err(ParamError::Constraint)?;
    if outcome == Outcome::Undecided {
        context.notify(&ParamEvent::ImplicationUndecided {
            own: own.variable(),
            other: other.variable(),
        });
    }
    let is_real = symbol_type == SymbolType::Real;
    let is_own_narrowed =
        is_real && has_float_member(own.constraints(), own.variable(), Polarity::NotIn);
    let is_other_widened =
        is_real && has_float_member(other.constraints(), other.variable(), Polarity::In);
    let is_unproven = (outcome == Outcome::Violated && !own_screened.is_exact)
        || (outcome == Outcome::Satisfied
            && (!other_screened.is_exact || is_own_narrowed || is_other_widened));
    if !is_unproven {
        return Ok(outcome);
    }
    context.notify(&ParamEvent::ImplicationDowngraded {
        outcome,
        own: own.variable(),
        other: other.variable(),
    });
    Ok(Outcome::Undecided)
}

/// Decide whether some value of a numeric `domain` of `symbol_type`
/// satisfies `side`: by enumeration with an in-set constraint, and
/// otherwise by asking the solver about the screened system, with the
/// downgrades of an inexact system and of the REAL sort.
fn numeric_has_feasible_value(
    domain: &ParamDomain,
    symbol_type: SymbolType,
    side: Side<'_>,
    context: &ParamContext<'_>,
) -> Result<Outcome, ParamError> {
    if has_in_set(side.constraints()) {
        return decide_feasibility_by_enumeration(domain, side, context);
    }
    let variable = side.variable();
    let Screened { system, is_exact } = screen(side.constraints(), variable, context)?;
    let outcome = ask_satisfiability(&system, variable, symbol_type, context)?;
    if outcome == Outcome::Satisfied && !is_exact {
        context.notify(&ParamEvent::SatisfiedOnInexactSystem { variable });
        return Ok(Outcome::Undecided);
    }
    if outcome == Outcome::Violated
        && symbol_type == SymbolType::Real
        && has_float_member(side.constraints(), variable, Polarity::NotIn)
    {
        context.notify(&ParamEvent::ViolatedUnderKindConflation { variable });
        return Ok(Outcome::Undecided);
    }
    if outcome == Outcome::Undecided {
        context.notify(&ParamEvent::SatisfiabilityUndecided { variable });
    }
    Ok(outcome)
}

/// Return the permutations of `members`, in lexicographic order of their
/// positions, as tuples.
struct Permutations<'a> {
    members: &'a [Member],
    positions: Option<Vec<usize>>,
}

impl<'a> Permutations<'a> {
    fn new(members: &'a [Member]) -> Self {
        Self {
            members,
            positions: Some((0..members.len()).collect()),
        }
    }
}

impl Iterator for Permutations<'_> {
    type Item = Value;

    fn next(&mut self) -> Option<Value> {
        let positions = self.positions.as_mut()?;
        let value = Value::Tuple(
            positions
                .iter()
                .map(|&position| member_value(&self.members[position]))
                .collect(),
        );
        // Advance to the next permutation in lexicographic order.
        let length = positions.len();
        match (1..length)
            .rev()
            .find(|&index| positions[index - 1] < positions[index])
        {
            None => self.positions = None,
            Some(pivot) => {
                let successor = (pivot..length)
                    .rev()
                    .find(|&index| positions[index] > positions[pivot - 1])
                    .unwrap_or(pivot);
                positions.swap(pivot - 1, successor);
                positions[pivot..].reverse();
            }
        }
        Some(value)
    }
}

/// Return the values a finite `domain` enumerates: its members, or its
/// permutations.
fn finite_values(domain: &ParamDomain) -> Option<Box<dyn Iterator<Item = Value> + '_>> {
    match domain {
        ParamDomain::Ordinal(domain) => Some(Box::new(domain.values().iter().map(member_value))),
        ParamDomain::Categorical(domain) => {
            Some(Box::new(domain.values().iter().map(member_value)))
        }
        ParamDomain::Permutation(domain) => Some(Box::new(Permutations::new(domain.values()))),
        _ => None,
    }
}

/// Decide whether some value of `domain` satisfies `side`, as
/// [`ParamDomain::has_feasible_value`] describes.
pub(super) fn has_feasible_value(
    domain: &ParamDomain,
    side: Side<'_>,
    context: &ParamContext<'_>,
) -> Result<Outcome, ParamError> {
    let constraints = match domain {
        ParamDomain::Custom(_) => side.constraints().to_vec(),
        _ => restricted(domain, side)?,
    };
    let side = Side::new(&constraints, side.variable());
    match domain {
        ParamDomain::Integer(_) | ParamDomain::IntervalInteger(_) => {
            numeric_has_feasible_value(domain, SymbolType::Int, side, context)
        }
        ParamDomain::Real(_) => numeric_has_feasible_value(domain, SymbolType::Real, side, context),
        ParamDomain::Custom(custom) => custom
            .get()
            .has_feasible_value(side, context)
            .map_err(ParamError::Custom),
        _ => {
            let Some(values) = finite_values(domain) else {
                return Ok(Outcome::Undecided);
            };
            for value in values {
                if is_value_valid_for(domain, side, &value, context)? {
                    return Ok(Outcome::Satisfied);
                }
            }
            Ok(Outcome::Violated)
        }
    }
}

/// Decide whether `own`'s set over `own_domain` is a subset of `other`'s
/// over `other_domain`, as [`ParamDomain::feasibility_subset`] describes.
pub(super) fn feasibility_subset(
    own_domain: &ParamDomain,
    own: Side<'_>,
    other_domain: &ParamDomain,
    other: Side<'_>,
    context: &ParamContext<'_>,
) -> Result<Outcome, ParamError> {
    match own_domain {
        ParamDomain::Integer(_) | ParamDomain::IntervalInteger(_) | ParamDomain::Real(_) => {
            let Some(symbol_type) = own_domain.symbol_type()? else {
                return Ok(Outcome::Violated);
            };
            if other_domain.symbol_type()? != Some(symbol_type) {
                return Ok(Outcome::Violated);
            }
            compute_constraint_implication_subset(
                own_domain,
                own,
                other_domain,
                other,
                symbol_type,
                context,
            )
        }
        ParamDomain::Custom(custom) => custom
            .get()
            .feasibility_subset(own, other_domain, other, context)
            .map_err(ParamError::Custom),
        _ => {
            let is_same_kind = own_domain.kind() == other_domain.kind();
            let is_comparable = match (own_domain, other_domain) {
                (ParamDomain::Permutation(left), ParamDomain::Permutation(right)) => {
                    left.values().len() == right.values().len()
                }
                _ => is_same_kind,
            };
            if !is_comparable {
                return Ok(Outcome::Violated);
            }
            let own_constraints = restricted(own_domain, own)?;
            let other_constraints = restricted(other_domain, other)?;
            let own = Side::new(&own_constraints, own.variable());
            let other = Side::new(&other_constraints, other.variable());
            let Some(values) = finite_values(own_domain) else {
                return Ok(Outcome::Violated);
            };
            for value in values {
                if !is_value_valid_for(own_domain, own, &value, context)? {
                    continue;
                }
                if !is_value_valid_for(other_domain, other, &value, context)? {
                    return Ok(Outcome::Violated);
                }
            }
            Ok(Outcome::Satisfied)
        }
    }
}
