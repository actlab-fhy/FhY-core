//! The ordinal order of values, a sort that tolerates comparisons that are
//! no order, and the conversions between members and values.

use std::cmp::Ordering;

use num_bigint::BigInt;

use crate::constraint::{Member, MemberKind, Value};
use crate::expression::ExactNumber;
use crate::foreign::BoxError;

/// Return the value `member` holds.
pub(crate) fn member_value(member: &Member) -> Value {
    match member.kind() {
        MemberKind::Bool(value) => Value::Bool(value),
        MemberKind::Int(value) => Value::Int(value.clone()),
        MemberKind::Float(value) => Value::Float(value),
        MemberKind::Str(value) => Value::Str(value.to_owned()),
        MemberKind::Tuple(members) => Value::Tuple(members.iter().map(member_value).collect()),
        MemberKind::FrozenSet(members) => {
            Value::FrozenSet(members.iter().map(member_value).collect())
        }
        MemberKind::Opaque(value) => Value::Opaque(value.clone()),
    }
}

/// Return the rank of a leaf member's kind among values that tie in the
/// ordinal order: `bool`, `float`, `int`, then the others.
fn tie_rank(member: &Member) -> u8 {
    match member.kind() {
        MemberKind::Bool(_) => 0,
        MemberKind::Float(_) => 1,
        MemberKind::Int(_) => 2,
        MemberKind::Str(_) => 3,
        MemberKind::Tuple(_) | MemberKind::FrozenSet(_) | MemberKind::Opaque(_) => 4,
    }
}

/// Return the numeric order of two numbers, exactly.
fn compare_numbers(left: &Member, right: &Member) -> Option<Ordering> {
    let as_int = |member: &Member| match member.kind() {
        MemberKind::Bool(value) => Some(BigInt::from(u8::from(value))),
        MemberKind::Int(value) => Some(value.clone()),
        _ => None,
    };
    let as_float = |member: &Member| match member.kind() {
        MemberKind::Float(value) => Some(value),
        _ => None,
    };
    match (as_int(left), as_int(right), as_float(left), as_float(right)) {
        (Some(left), Some(right), _, _) => Some(left.cmp(&right)),
        (None, None, Some(left), Some(right)) => left.partial_cmp(&right),
        (Some(left), None, _, Some(right)) => {
            Some(ExactNumber::integer(left).cmp(&ExactNumber::of_f64(right)?))
        }
        (None, Some(right), Some(left), _) => {
            Some(ExactNumber::of_f64(left)?.cmp(&ExactNumber::integer(right)))
        }
        _ => None,
    }
}

/// Return whether `member` is a number of the ordinal order.
fn is_number(member: &Member) -> bool {
    matches!(
        member.kind(),
        MemberKind::Bool(_) | MemberKind::Int(_) | MemberKind::Float(_)
    )
}

/// Return how two leaf members order in an ordinal domain, before ties are
/// broken: numbers numerically, strings by code point, opaque values by
/// their producer's order, and `None` for a pair that does not order.
fn compare_ordinal_values(left: &Member, right: &Member) -> Result<Option<Ordering>, BoxError> {
    if is_number(left) && is_number(right) {
        return Ok(compare_numbers(left, right));
    }
    match (left.kind(), right.kind()) {
        (MemberKind::Str(left), MemberKind::Str(right)) => Ok(Some(left.cmp(right))),
        (MemberKind::Opaque(left), MemberKind::Opaque(right)) => {
            left.get().order_against(right.get())
        }
        _ => Ok(None),
    }
}

/// Return how two leaf members order in an ordinal domain: by value, then
/// equal values by kind (`bool`, `float`, `int`); `Equal` for values the
/// order does not tell apart, and `None` for values that do not order.
///
/// # Errors
///
/// Returns an opaque value's error from ordering it.
pub(crate) fn compare_ordinal(left: &Member, right: &Member) -> Result<Option<Ordering>, BoxError> {
    Ok(compare_ordinal_values(left, right)?
        .map(|ordering| ordering.then_with(|| tie_rank(left).cmp(&tie_rank(right)))))
}

/// Sort `items` stably by `compare`, merging runs bottom-up.
///
/// A comparison that is not a total order gives some order, never a
/// panic, which [`slice::sort_by`] does not promise. The first error the
/// comparison answers stops the sort and is returned. The runs merged are
/// of indices, so no item is cloned; the items move once, into the order
/// found.
pub(crate) fn sort_tolerantly<T, E>(
    items: Vec<T>,
    mut compare: impl FnMut(&T, &T) -> Result<Ordering, E>,
) -> Result<Vec<T>, E> {
    let length = items.len();
    let mut current: Vec<usize> = (0..length).collect();
    let mut merged = Vec::with_capacity(length);
    let mut width = 1;
    while width < length {
        merged.clear();
        let mut start = 0;
        while start < length {
            let middle = (start + width).min(length);
            let end = (start + 2 * width).min(length);
            let (mut left, mut right) = (start, middle);
            while left < middle && right < end {
                if compare(&items[current[right]], &items[current[left]])? == Ordering::Less {
                    merged.push(current[right]);
                    right += 1;
                } else {
                    merged.push(current[left]);
                    left += 1;
                }
            }
            merged.extend_from_slice(&current[left..middle]);
            merged.extend_from_slice(&current[right..end]);
            start = end;
        }
        std::mem::swap(&mut current, &mut merged);
        width *= 2;
    }
    let mut slots: Vec<Option<T>> = items.into_iter().map(Some).collect();
    Ok(current
        .into_iter()
        .filter_map(|index| slots[index].take())
        .collect())
}

/// Return whether two values are equal type-strictly: of one kind and
/// equal, at every depth; opaque values by their producer, and a NaN equal
/// to nothing.
#[expect(
    clippy::float_cmp,
    reason = "type-strict equality of floats is IEEE equality, as Python's `==` is"
)]
pub(crate) fn are_values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Bool(left), Value::Bool(right)) => left == right,
        (Value::Int(left), Value::Int(right)) => left == right,
        (Value::Float(left), Value::Float(right)) => left == right,
        (Value::Decimal(left), Value::Decimal(right)) => left == right,
        (Value::Str(left), Value::Str(right)) => left == right,
        (Value::Tuple(left), Value::Tuple(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| are_values_equal(left, right))
        }
        (Value::FrozenSet(left), Value::FrozenSet(right)) => {
            left.iter()
                .all(|value| right.iter().any(|other| are_values_equal(value, other)))
                && right
                    .iter()
                    .all(|value| left.iter().any(|other| are_values_equal(value, other)))
        }
        (Value::Opaque(left), Value::Opaque(right)) => left == right,
        _ => false,
    }
}
