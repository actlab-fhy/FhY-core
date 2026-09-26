//! The ordinal order of values, a sort that tolerates comparisons that are
//! no order, and the conversions between members and values.

use std::cmp::Ordering;

use num_bigint::BigInt;
use num_traits::FromPrimitive;

use crate::constraint::{Member, MemberKind, Value};

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

/// Return the exact order of the integer `left` and the float `right`,
/// which is not a NaN.
fn compare_int_with_float(left: &BigInt, right: f64) -> Ordering {
    if right.is_infinite() {
        return if right > 0.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let floor = right.floor();
    let Some(floor_int) = BigInt::from_f64(floor) else {
        // A finite float always has an integral floor.
        return Ordering::Equal;
    };
    match left.cmp(&floor_int) {
        Ordering::Equal if right > floor => Ordering::Less,
        ordering => ordering,
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
        (Some(left), None, _, Some(right)) => Some(compare_int_with_float(&left, right)),
        (None, Some(right), Some(left), _) => Some(compare_int_with_float(&right, left).reverse()),
        (None, None, Some(left), Some(right)) => left.partial_cmp(&right),
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
fn compare_ordinal_values(left: &Member, right: &Member) -> Option<Ordering> {
    if is_number(left) && is_number(right) {
        return compare_numbers(left, right);
    }
    match (left.kind(), right.kind()) {
        (MemberKind::Str(left), MemberKind::Str(right)) => Some(left.cmp(right)),
        (MemberKind::Opaque(left), MemberKind::Opaque(right)) => {
            left.get().order_against(right.get())
        }
        _ => None,
    }
}

/// Return how two leaf members order in an ordinal domain: by value, then
/// equal values by kind (`bool`, `float`, `int`); `Equal` for values the
/// order does not tell apart, and `None` for values that do not order.
pub(crate) fn compare_ordinal(left: &Member, right: &Member) -> Option<Ordering> {
    let ordering = compare_ordinal_values(left, right)?;
    Some(ordering.then_with(|| tie_rank(left).cmp(&tie_rank(right))))
}

/// Sort `items` stably by `compare`, merging runs bottom-up.
///
/// A comparison that is not a total order gives some order, never a
/// panic, which [`slice::sort_by`] does not promise. The first `None` the
/// comparison answers stops the sort and is returned as `Err`.
pub(crate) fn sort_tolerantly<T: Clone>(
    items: Vec<T>,
    mut compare: impl FnMut(&T, &T) -> Option<Ordering>,
) -> Result<Vec<T>, ()> {
    let mut current = items;
    let length = current.len();
    let mut width = 1;
    while width < length {
        let mut merged = Vec::with_capacity(length);
        let mut start = 0;
        while start < length {
            let middle = (start + width).min(length);
            let end = (start + 2 * width).min(length);
            let (mut left, mut right) = (start, middle);
            while left < middle && right < end {
                match compare(&current[right], &current[left]) {
                    None => return Err(()),
                    Some(Ordering::Less) => {
                        merged.push(current[right].clone());
                        right += 1;
                    }
                    Some(_) => {
                        merged.push(current[left].clone());
                        left += 1;
                    }
                }
            }
            merged.extend_from_slice(&current[left..middle]);
            merged.extend_from_slice(&current[right..end]);
            start = end;
        }
        current = merged;
        width *= 2;
    }
    Ok(current)
}
