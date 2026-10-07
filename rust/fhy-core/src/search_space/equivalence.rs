//! The equivalence walk over spaces and their parts: the frame pairing two
//! spaces' names, the correspondence of params, domains and constraint
//! systems under it, and the structural comparison with names compared by
//! `==`.
//!
//! An identifier member of a domain or of a set constraint is a reference:
//! it corresponds to the other side's member only as
//! [`AlphaRenaming::is_corresponding`] says, a categorical domain's and a
//! set constraint's members as a bijection. A constraint system's members
//! pair up in any order, since its canonical order follows identifiers'
//! ids, which a renaming changes.

use crate::constraint::{Constraint, ConstraintError, ConstraintSystem, Member, MemberKind, Value};
use crate::foreign::{BoxError, Part};
use crate::identifier::Identifier;
use crate::param::{Param, ParamDomain};
use crate::term::{AlphaEquivalence, AlphaRenaming};

use super::alternative::Alternative;
use super::choice::Choice;
use super::error::EquivalenceError;
use super::space::Space;
use super::variable::Variable;

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

/// Return `renaming` with one more frame pairing `left` with `right`, or
/// `None` when they cannot pair: lists of different lengths, or a list
/// repeating an identifier.
pub(super) fn enter_frame(
    renaming: &AlphaRenaming,
    left: &[Identifier],
    right: &[Identifier],
) -> Option<AlphaRenaming> {
    let mut extended = renaming.clone();
    extended.enter_binders(left, right).ok()?;
    Some(extended)
}

/// Return the names of `alternative` in canonical order: its name, its
/// bound identifiers, its variables' names, then its sub-choices' names.
pub(super) fn alternative_labels(
    alternative: &dyn Alternative,
) -> Result<Vec<Identifier>, BoxError> {
    let mut labels = vec![alternative.name().clone()];
    labels.extend(alternative.bound_identifiers()?);
    labels.extend(
        alternative
            .variables()
            .iter()
            .map(|variable| variable.get().name().clone()),
    );
    for choice in alternative.choices() {
        labels.extend(choice.labels().iter().cloned());
    }
    Ok(labels)
}

// ---------------------------------------------------------------------------
// Members, values, domains, constraints and params
// ---------------------------------------------------------------------------

/// Return whether `left` corresponds to `right` under `renaming`.
fn do_members_correspond(left: &Member, right: &Member, renaming: &AlphaRenaming) -> bool {
    match (left.kind(), right.kind()) {
        (MemberKind::Identifier(left), MemberKind::Identifier(right)) => {
            renaming.is_corresponding(left, right)
        }
        (MemberKind::Tuple(left), MemberKind::Tuple(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| do_members_correspond(left, right, renaming))
        }
        (MemberKind::FrozenSet(left), MemberKind::FrozenSet(right)) => {
            let (left, right): (Vec<&Member>, Vec<&Member>) =
                (left.iter().collect(), right.iter().collect());
            are_member_sets_corresponding(&left, &right, renaming)
        }
        _ => left == right,
    }
}

/// Return whether the distinct members `left` and `right` correspond as a
/// bijection: as many, each left member corresponding to a right one.
/// Correspondence under one renaming is injective, so this is a bijection.
fn are_member_sets_corresponding(
    left: &[&Member],
    right: &[&Member],
    renaming: &AlphaRenaming,
) -> bool {
    left.len() == right.len()
        && left.iter().all(|member| {
            right
                .iter()
                .any(|other| do_members_correspond(member, other, renaming))
        })
}

/// Return whether the value `left` corresponds to `right` under
/// `renaming`: identifiers through it, tuples in order, frozen sets in any
/// order, and anything else by `==`.
pub(super) fn do_values_correspond(left: &Value, right: &Value, renaming: &AlphaRenaming) -> bool {
    match (left, right) {
        (Value::Identifier(left), Value::Identifier(right)) => {
            renaming.is_corresponding(left, right)
        }
        (Value::Tuple(left), Value::Tuple(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| do_values_correspond(left, right, renaming))
        }
        (Value::FrozenSet(left), Value::FrozenSet(right)) => {
            left.iter().all(|value| {
                right
                    .iter()
                    .any(|other| do_values_correspond(value, other, renaming))
            }) && right.iter().all(|value| {
                left.iter()
                    .any(|other| do_values_correspond(other, value, renaming))
            })
        }
        _ => left == right,
    }
}

/// Return whether the domains correspond under `renaming`: a categorical
/// domain's members as a bijection, an ordinal or permutation domain's in
/// order, and any other domain structurally.
fn do_domains_correspond(
    left: &ParamDomain,
    right: &ParamDomain,
    renaming: &AlphaRenaming,
) -> bool {
    let in_order = |left: &[Member], right: &[Member]| {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(left, right)| do_members_correspond(left, right, renaming))
    };
    match (left, right) {
        (ParamDomain::Categorical(left), ParamDomain::Categorical(right)) => {
            let (left, right): (Vec<&Member>, Vec<&Member>) = (
                left.values().iter().collect(),
                right.values().iter().collect(),
            );
            are_member_sets_corresponding(&left, &right, renaming)
        }
        (ParamDomain::Ordinal(left), ParamDomain::Ordinal(right)) => {
            in_order(left.values(), right.values())
        }
        (ParamDomain::Permutation(left), ParamDomain::Permutation(right)) => {
            in_order(left.values(), right.values())
        }
        _ => left.is_structurally_equivalent(right),
    }
}

/// Return whether the constraints correspond under `renaming`: set
/// constraints by polarity, variable and members, and the others through
/// their alpha equivalence.
fn do_constraints_correspond(
    left: &Constraint,
    right: &Constraint,
    renaming: &AlphaRenaming,
) -> Result<bool, ConstraintError> {
    match (left, right) {
        (Constraint::Set(left), Constraint::Set(right)) => {
            let (left_members, right_members): (Vec<&Member>, Vec<&Member>) = (
                left.members().iter().collect(),
                right.members().iter().collect(),
            );
            Ok(left.polarity() == right.polarity()
                && renaming.is_corresponding(left.variable(), right.variable())
                && are_member_sets_corresponding(&left_members, &right_members, renaming))
        }
        _ => left.is_alpha_equivalent_under(right, renaming),
    }
}

/// Return whether the systems' members pair up under `renaming`, in any
/// order.
///
/// Correspondence under one renaming relates each member to the members
/// equal to it up to that renaming, so matching greedily finds a pairing
/// whenever one exists.
pub(super) fn do_systems_correspond(
    left: &ConstraintSystem,
    right: &ConstraintSystem,
    renaming: &AlphaRenaming,
) -> Result<bool, ConstraintError> {
    let left = left.constraints();
    let mut unmatched: Vec<&Constraint> = right.constraints().iter().collect();
    if left.len() != unmatched.len() {
        return Ok(false);
    }
    for member in left {
        let mut found = None;
        for (position, candidate) in unmatched.iter().enumerate() {
            if do_constraints_correspond(member, candidate, renaming)? {
                found = Some(position);
                break;
            }
        }
        let Some(position) = found else {
            return Ok(false);
        };
        unmatched.swap_remove(position);
    }
    Ok(true)
}

/// Return whether the params correspond under `renaming`: their domains,
/// then their constraints under one more frame pairing their variables.
fn do_params_correspond(
    left: &Param,
    right: &Param,
    renaming: &AlphaRenaming,
) -> Result<bool, EquivalenceError> {
    if !do_domains_correspond(left.domain(), right.domain(), renaming) {
        return Ok(false);
    }
    let Some(extended) = enter_frame(
        renaming,
        std::slice::from_ref(left.variable()),
        std::slice::from_ref(right.variable()),
    ) else {
        return Ok(false);
    };
    do_systems_correspond(
        left.constraint_system(),
        right.constraint_system(),
        &extended,
    )
    .map_err(EquivalenceError::Constraint)
}

// ---------------------------------------------------------------------------
// The alpha walk, under a frame that already pairs the names
// ---------------------------------------------------------------------------

/// Return whether the variables correspond under `renaming`, which pairs
/// their names.
pub(super) fn do_variables_correspond(
    left: &dyn Variable,
    right: &dyn Variable,
    renaming: &AlphaRenaming,
) -> Result<bool, EquivalenceError> {
    if left.kind() != right.kind()
        || left.notes() != right.notes()
        || !do_params_correspond(left.param(), right.param(), renaming)?
    {
        return Ok(false);
    }
    left.is_extension_alpha_equivalent_under(right, renaming)
        .map_err(EquivalenceError::Extension)
}

/// Return whether the alternatives correspond under `renaming`, which
/// pairs every name they hold.
///
/// They must bind as many identifiers: the frame pairs the names in
/// canonical order, so only then does it pair each one's bound
/// identifiers with the other's, whatever the implementation's hook
/// answers.
pub(super) fn do_alternatives_correspond(
    left: &dyn Alternative,
    right: &dyn Alternative,
    renaming: &AlphaRenaming,
) -> Result<bool, EquivalenceError> {
    if left.kind() != right.kind()
        || left.notes() != right.notes()
        || left.variables().len() != right.variables().len()
        || left.choices().len() != right.choices().len()
    {
        return Ok(false);
    }
    let bound = |alternative: &dyn Alternative| {
        alternative
            .bound_identifiers()
            .map(|bound| bound.len())
            .map_err(EquivalenceError::Extension)
    };
    if bound(left)? != bound(right)? {
        return Ok(false);
    }
    for (left, right) in left.variables().iter().zip(right.variables()) {
        if !do_variables_correspond(left.get(), right.get(), renaming)? {
            return Ok(false);
        }
    }
    for (left, right) in left.choices().iter().zip(right.choices()) {
        if !do_choices_correspond(left, right, renaming)? {
            return Ok(false);
        }
    }
    left.is_extension_alpha_equivalent_under(right, renaming)
        .map_err(EquivalenceError::Extension)
}

/// Return whether the choices correspond under `renaming`, which pairs
/// every name they hold.
pub(super) fn do_choices_correspond(
    left: &Choice,
    right: &Choice,
    renaming: &AlphaRenaming,
) -> Result<bool, EquivalenceError> {
    if left.notes() != right.notes()
        || left.alternatives().len() != right.alternatives().len()
        || left.labels().len() != right.labels().len()
    {
        return Ok(false);
    }
    for (left, right) in left.alternatives().iter().zip(right.alternatives()) {
        if !do_alternatives_correspond(left.get(), right.get(), renaming)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Return the frame pairing the two spaces' names on top of `renaming`, or
/// `None` when the spaces hold different numbers of names.
pub(super) fn space_frame(
    left: &Space,
    right: &Space,
    renaming: &AlphaRenaming,
) -> Option<AlphaRenaming> {
    enter_frame(renaming, left.labels(), right.labels())
}

/// Return whether the spaces correspond under `frame`, which pairs their
/// names.
pub(super) fn do_spaces_correspond(
    left: &Space,
    right: &Space,
    frame: &AlphaRenaming,
) -> Result<bool, EquivalenceError> {
    if left.notes() != right.notes()
        || left.variables().len() != right.variables().len()
        || left.choices().len() != right.choices().len()
        || left.decision_count() != right.decision_count()
        || left.forbidden().len() != right.forbidden().len()
    {
        return Ok(false);
    }
    for (left, right) in left.variables().iter().zip(right.variables()) {
        if !do_variables_correspond(left.get(), right.get(), frame)? {
            return Ok(false);
        }
    }
    for (left, right) in left.choices().iter().zip(right.choices()) {
        if !do_choices_correspond(left, right, frame)? {
            return Ok(false);
        }
    }
    for position in 0..left.decision_count() {
        let is_corresponding = match (left.condition_at(position), right.condition_at(position)) {
            (None, None) => true,
            (Some(left), Some(right)) => {
                do_systems_correspond(left, right, frame).map_err(EquivalenceError::Constraint)?
            }
            _ => false,
        };
        if !is_corresponding {
            return Ok(false);
        }
    }
    for (left, right) in left.forbidden().iter().zip(right.forbidden()) {
        if !do_systems_correspond(left.when(), right.when(), frame)
            .map_err(EquivalenceError::Constraint)?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// The standalone alpha relations
// ---------------------------------------------------------------------------

/// Return whether the variables are alpha-equivalent on their own.
pub(super) fn is_variable_alpha_equivalent(
    left: &Part<dyn Variable>,
    right: &Part<dyn Variable>,
    renaming: &AlphaRenaming,
) -> Result<bool, EquivalenceError> {
    let Some(frame) = enter_frame(
        renaming,
        std::slice::from_ref(left.get().name()),
        std::slice::from_ref(right.get().name()),
    ) else {
        return Ok(false);
    };
    do_variables_correspond(left.get(), right.get(), &frame)
}

/// Return whether the alternatives are alpha-equivalent on their own.
pub(super) fn is_alternative_alpha_equivalent(
    left: &Part<dyn Alternative>,
    right: &Part<dyn Alternative>,
    renaming: &AlphaRenaming,
) -> Result<bool, EquivalenceError> {
    let left_labels = alternative_labels(left.get()).map_err(EquivalenceError::Extension)?;
    let right_labels = alternative_labels(right.get()).map_err(EquivalenceError::Extension)?;
    let Some(frame) = enter_frame(renaming, &left_labels, &right_labels) else {
        return Ok(false);
    };
    do_alternatives_correspond(left.get(), right.get(), &frame)
}

/// Return whether the choices are alpha-equivalent on their own.
pub(super) fn is_choice_alpha_equivalent(
    left: &Choice,
    right: &Choice,
    renaming: &AlphaRenaming,
) -> Result<bool, EquivalenceError> {
    let Some(frame) = enter_frame(renaming, left.labels(), right.labels()) else {
        return Ok(false);
    };
    do_choices_correspond(left, right, &frame)
}

/// Return whether the spaces are alpha-equivalent.
pub(super) fn is_space_alpha_equivalent(
    left: &Space,
    right: &Space,
    renaming: &AlphaRenaming,
) -> Result<bool, EquivalenceError> {
    let Some(frame) = space_frame(left, right, renaming) else {
        return Ok(false);
    };
    do_spaces_correspond(left, right, &frame)
}

// ---------------------------------------------------------------------------
// The structural relations
// ---------------------------------------------------------------------------

/// Return whether the variables are structurally equivalent.
pub(super) fn is_variable_structurally_equivalent(
    left: &dyn Variable,
    right: &dyn Variable,
) -> Result<bool, EquivalenceError> {
    if left.kind() != right.kind()
        || left.name() != right.name()
        || left.notes() != right.notes()
        || !left.param().is_structurally_equivalent(right.param())
    {
        return Ok(false);
    }
    left.is_extension_structurally_equivalent(right)
        .map_err(EquivalenceError::Extension)
}

/// Return whether the alternatives are structurally equivalent.
pub(super) fn is_alternative_structurally_equivalent(
    left: &dyn Alternative,
    right: &dyn Alternative,
) -> Result<bool, EquivalenceError> {
    if left.kind() != right.kind()
        || left.name() != right.name()
        || left.notes() != right.notes()
        || left.variables().len() != right.variables().len()
        || left.choices().len() != right.choices().len()
    {
        return Ok(false);
    }
    let left_bound = left
        .bound_identifiers()
        .map_err(EquivalenceError::Extension)?;
    let right_bound = right
        .bound_identifiers()
        .map_err(EquivalenceError::Extension)?;
    if left_bound != right_bound {
        return Ok(false);
    }
    for (left, right) in left.variables().iter().zip(right.variables()) {
        if !is_variable_structurally_equivalent(left.get(), right.get())? {
            return Ok(false);
        }
    }
    for (left, right) in left.choices().iter().zip(right.choices()) {
        if !is_choice_structurally_equivalent(left, right)? {
            return Ok(false);
        }
    }
    left.is_extension_structurally_equivalent(right)
        .map_err(EquivalenceError::Extension)
}

/// Return whether the choices are structurally equivalent.
pub(super) fn is_choice_structurally_equivalent(
    left: &Choice,
    right: &Choice,
) -> Result<bool, EquivalenceError> {
    if left.name() != right.name()
        || left.notes() != right.notes()
        || left.alternatives().len() != right.alternatives().len()
    {
        return Ok(false);
    }
    for (left, right) in left.alternatives().iter().zip(right.alternatives()) {
        if !is_alternative_structurally_equivalent(left.get(), right.get())? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Return whether the spaces are structurally equivalent.
pub(super) fn is_space_structurally_equivalent(
    left: &Space,
    right: &Space,
) -> Result<bool, EquivalenceError> {
    if left.name() != right.name()
        || left.notes() != right.notes()
        || left.variables().len() != right.variables().len()
        || left.choices().len() != right.choices().len()
        || left.conditions().len() != right.conditions().len()
        || left.forbidden().len() != right.forbidden().len()
    {
        return Ok(false);
    }
    for (left, right) in left.variables().iter().zip(right.variables()) {
        if !is_variable_structurally_equivalent(left.get(), right.get())? {
            return Ok(false);
        }
    }
    for (left, right) in left.choices().iter().zip(right.choices()) {
        if !is_choice_structurally_equivalent(left, right)? {
            return Ok(false);
        }
    }
    let conditions_match = left
        .conditions()
        .iter()
        .zip(right.conditions())
        .all(|(left, right)| {
            left.target() == right.target() && left.when().is_structurally_equivalent(right.when())
        });
    let forbidden_match = left
        .forbidden()
        .iter()
        .zip(right.forbidden())
        .all(|(left, right)| left.when().is_structurally_equivalent(right.when()));
    Ok(conditions_match && forbidden_match)
}
