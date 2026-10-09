//! Screening a param's constraints into a system the solver can be asked
//! about, and moving constraints from one variable to another.

use std::collections::HashMap;

use crate::constraint::{
    Constraint, ConstraintError, ConstraintSystem, EquationConstraint, Member, MemberSet, Polarity,
    SetConstraint,
};
use crate::expression::Expression;
use crate::identifier::Identifier;

use super::context::{ParamContext, ParamEvent, ScreenReason};
use super::domain::Side;
use super::error::ParamError;

/// A screened system, and whether it denotes exactly the value set of the
/// constraints it was screened from.
#[derive(Debug)]
pub(super) struct Screened {
    pub(super) system: ConstraintSystem,
    pub(super) is_exact: bool,
}

/// Return the system of `constraints` the solver can be asked about for
/// `variable`, and whether it lost nothing.
///
/// It keeps an equation scoped exactly to `variable`; a set constraint on
/// `variable`, an in-set one only if every member lifts to an expression,
/// a not-in-set one narrowed to the members that lift and dropped if none
/// does. Each drop or narrowing is reported as [`ParamEvent::Screened`];
/// a constraint of another kind is dropped without an event. Any drop or
/// narrowing makes the system inexact.
///
/// # Errors
///
/// Returns [`ParamError::Constraint`] if building the system fails, which a
/// system of built-in constraints never does.
pub(super) fn screen(
    constraints: &[Constraint],
    variable: &Identifier,
    context: &ParamContext<'_>,
) -> Result<Screened, ParamError> {
    let mut members = Vec::with_capacity(constraints.len());
    let mut is_exact = true;
    let report = |constraint: &Constraint, reason: ScreenReason<'_>| {
        context.notify(&ParamEvent::Screened {
            constraint,
            variable,
            reason,
        });
    };
    for constraint in constraints {
        match constraint {
            Constraint::Equation(equation) => {
                let scope = equation.free_identifiers();
                if scope.len() == 1 && scope.contains(variable) {
                    members.push(constraint.clone());
                } else {
                    report(constraint, ScreenReason::DependentScope);
                    is_exact = false;
                }
            }
            Constraint::Set(set) if set.variable() != variable => {
                report(constraint, ScreenReason::ForeignVariable);
                is_exact = false;
            }
            Constraint::Set(set) => match set.polarity() {
                Polarity::NotIn => {
                    let (liftable, excluded): (Vec<Member>, Vec<Member>) = set
                        .members()
                        .iter()
                        .cloned()
                        .partition(Member::lifts_to_expression);
                    if excluded.is_empty() {
                        members.push(constraint.clone());
                    } else if liftable.is_empty() {
                        report(constraint, ScreenReason::NoLiftableMember);
                        is_exact = false;
                    } else {
                        report(
                            constraint,
                            ScreenReason::Narrowed {
                                liftable: &liftable,
                                excluded: &excluded,
                            },
                        );
                        members.push(Constraint::from(SetConstraint::new(
                            variable.clone(),
                            MemberSet::new(liftable),
                            Polarity::NotIn,
                        )));
                        is_exact = false;
                    }
                }
                _ => match set.to_expression() {
                    Ok(_) => members.push(constraint.clone()),
                    Err(error) => {
                        report(constraint, ScreenReason::UnliftableMember(&error));
                        is_exact = false;
                    }
                },
            },
            _ => is_exact = false,
        }
    }
    Ok(Screened {
        system: ConstraintSystem::new(members).map_err(ParamError::Constraint)?,
        is_exact,
    })
}

/// Return `expression` with `old` replaced by a reference to `new`.
fn rename_in_expression(
    expression: &Expression,
    old: &Identifier,
    new: &Identifier,
) -> Result<Expression, ParamError> {
    expression
        .substitute(&HashMap::from([(old.clone(), Expression::from(new))]))
        .map_err(|error| ParamError::Constraint(ConstraintError::Substitution(error)))
}

/// Return `constraint` rescoped from `old` to `new`: an equation with `old`
/// substituted, which changes nothing where `old` is not free; a set
/// constraint on `new`, which must constrain `old`.
///
/// # Errors
///
/// Returns [`ParamError::Rescope`] for a set constraint on another variable,
/// and [`ParamError::UnexpectedConstraintKind`] for a custom constraint.
pub(super) fn rename_constraint(
    constraint: &Constraint,
    old: &Identifier,
    new: &Identifier,
) -> Result<Constraint, ParamError> {
    match constraint {
        Constraint::Equation(equation) => Ok(Constraint::from(EquationConstraint::new(
            rename_in_expression(equation.expression(), old, new)?,
        ))),
        Constraint::Set(set) => {
            if set.variable() != old {
                return Err(ParamError::Rescope {
                    from: old.clone(),
                    to: new.clone(),
                    variable: set.variable().clone(),
                });
            }
            Ok(Constraint::from(SetConstraint::new(
                new.clone(),
                set.members().clone(),
                set.polarity(),
            )))
        }
        _ => Err(ParamError::UnexpectedConstraintKind),
    }
}

/// Return `system` with each member rescoped from `old` to `new`.
pub(super) fn rename_system(
    system: &ConstraintSystem,
    old: &Identifier,
    new: &Identifier,
) -> Result<ConstraintSystem, ParamError> {
    let renamed = system
        .constraints()
        .iter()
        .map(|constraint| rename_constraint(constraint, old, new))
        .collect::<Result<Vec<_>, _>>()?;
    ConstraintSystem::new(renamed).map_err(ParamError::Constraint)
}

/// Return `constraints` with each equation's `operand` replaced by
/// `variable`; set and custom constraints are kept as they are.
fn substitute_operand(
    constraints: Vec<Constraint>,
    operand: &Identifier,
    variable: &Identifier,
) -> Result<Vec<Constraint>, ParamError> {
    constraints
        .into_iter()
        .map(|constraint| match &constraint {
            Constraint::Equation(equation) => Ok(Constraint::from(EquationConstraint::new(
                rename_in_expression(equation.expression(), operand, variable)?,
            ))),
            _ => Ok(constraint),
        })
        .collect()
}

/// Return both sides' constraints rescoped to `variable`, `own`'s first,
/// each side's references to the other side's variable substituted by
/// `variable` too.
pub(super) fn merge_intersection_constraints(
    own: Side<'_>,
    other: Side<'_>,
    variable: &Identifier,
) -> Result<Vec<Constraint>, ParamError> {
    let rescope = |side: Side<'_>| {
        side.constraints()
            .iter()
            .map(|constraint| rename_constraint(constraint, side.variable(), variable))
            .collect::<Result<Vec<_>, _>>()
    };
    let mut merged = substitute_operand(rescope(own)?, other.variable(), variable)?;
    merged.extend(substitute_operand(
        rescope(other)?,
        own.variable(),
        variable,
    )?);
    Ok(merged)
}
