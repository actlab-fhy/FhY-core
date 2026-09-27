//! Values a constraint is evaluated against, and the members a set
//! constraint holds.
//!
//! A [`Value`] is any value a caller can bind to an identifier besides an
//! expression. A [`Member`] is a value a set constraint can hold. It has no
//! decimal and no NaN at any depth, and its zeros are positive. A
//! [`MemberSet`] holds distinct members in one canonical order.
//!
//! Equality is type-strict: a Boolean, an integer and a float never compare
//! equal, whatever they hold. A value only its producer can compare, such as
//! a user object of the Python binding, is an opaque value: a
//! [`Part<dyn OpaqueValue>`](Part).

use std::borrow::Cow;
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::mem;

use crate::expression::{BigInt, Decimal, LiteralValue, write_float};
use crate::foreign::{BoxError, ForeignPart, Part, impl_part, is_same_part};

/// A value only its producer can compare.
///
/// Two opaque values are equal when [`eq_part`](Self::eq_part) says so;
/// the implementation decides whether values of different types can be. A
/// member-shaped opaque value can be a [`Member`], ordered after every
/// other kind by its [`ordering_key`](Self::ordering_key), which the member
/// reads once, when it is built.
pub trait OpaqueValue: ForeignPart {
    /// Return whether the value can be a member of a set constraint: a fact
    /// fixed when the value is built, which calls no code of its producer.
    fn is_member_shaped(&self) -> bool;

    /// Return whether the value equals `other`, for `==` on a
    /// [`Part<dyn OpaqueValue>`](Part).
    ///
    /// It must be an equivalence relation, symmetric included, and agree
    /// with [`ordering_key`](Self::ordering_key) and
    /// [`hash_part`](Self::hash_part): equal values have equal keys and
    /// hashes. The default is identity: the same value.
    fn eq_part(&self, other: &dyn OpaqueValue) -> bool {
        is_same_part(self, other)
    }

    /// Feed the value's hash to `state`, consistently with
    /// [`eq_part`](Self::eq_part). The default feeds nothing.
    fn hash_part(&self, state: &mut dyn Hasher) {
        let _ = state;
    }

    /// Check that the value can be looked up in a set, as a bound value
    /// is before its membership is decided.
    ///
    /// # Errors
    ///
    /// Returns the producer's error when it cannot be.
    fn check_hashable(&self) -> Result<(), BoxError>;

    /// Return a text equal for equal values, which orders opaque members.
    ///
    /// A member reads it once, when it is built.
    ///
    /// # Errors
    ///
    /// Returns the producer's error, which fails building the member.
    fn ordering_key(&self) -> Result<Cow<'_, str>, BoxError>;

    /// Return how the value orders against `other` by its producer's own
    /// order, as an ordinal param orders its values, or `None` when the two
    /// do not order.
    ///
    /// # Errors
    ///
    /// Returns the producer's error. The default orders nothing.
    fn order_against(&self, other: &dyn OpaqueValue) -> Result<Option<Ordering>, BoxError> {
        let _ = other;
        Ok(None)
    }
}

impl_part!(OpaqueValue);

/// A value bound to an identifier, besides an expression.
///
/// More kinds may be added, so a `match` on a value outside this crate
/// needs a wildcard arm.
///
/// Tuples and sets nest, and dropping, comparing and building a member of a
/// value recurse once per level, as [`Provenance`](crate::provenance::Provenance)
/// does; a value built through the API is as deep as its caller makes it.
/// Decoding refuses a value nested deeper than
/// [`MAX_VALUE_DEPTH`](super::wire::MAX_VALUE_DEPTH), so no payload
/// builds a deeper one.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Value {
    /// A Boolean.
    Bool(bool),
    /// An integer of any size.
    Int(BigInt),
    /// A binary float: any `f64`, NaN and the infinities included.
    Float(f64),
    /// An exact, normalized decimal.
    Decimal(Decimal),
    /// A string.
    Str(String),
    /// A sequence of values.
    Tuple(Vec<Value>),
    /// An unordered collection of values.
    FrozenSet(Vec<Value>),
    /// A value only its producer can compare.
    Opaque(Part<dyn OpaqueValue>),
}

impl From<LiteralValue> for Value {
    /// Return the value of the literal `value`, of the same kind.
    fn from(value: LiteralValue) -> Self {
        match value {
            LiteralValue::Bool(value) => Self::Bool(value),
            LiteralValue::Int(value) => Self::Int(value),
            LiteralValue::Float(value) => Self::Float(value),
            LiteralValue::Decimal(value) => Self::Decimal(value),
        }
    }
}

impl PartialEq for Value {
    /// Compare structurally and type-strictly: of one kind and equal, at
    /// every depth. A float equals a float of the same number, `-0.0` and
    /// `0.0` included, and a NaN equals a NaN, so `==` is an equivalence; a
    /// frozen set equals one holding equal values, in any order and with
    /// any repeats; an opaque value compares through its
    /// [`eq_part`](OpaqueValue::eq_part).
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bool(left), Self::Bool(right)) => left == right,
            (Self::Int(left), Self::Int(right)) => left == right,
            (Self::Float(left), Self::Float(right)) => are_floats_equivalent(*left, *right),
            (Self::Decimal(left), Self::Decimal(right)) => left == right,
            (Self::Str(left), Self::Str(right)) => left == right,
            (Self::Tuple(left), Self::Tuple(right)) => left == right,
            (Self::FrozenSet(left), Self::FrozenSet(right)) => {
                left.iter().all(|value| right.contains(value))
                    && right.iter().all(|value| left.contains(value))
            }
            (Self::Opaque(left), Self::Opaque(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl Hash for Value {
    /// Feed the kind, then the contents, consistently with `==`: a float's
    /// bits with `-0.0` folded into `0.0`, nothing more for a NaN, a frozen
    /// set's distinct element hashes in sorted order, and an opaque value's
    /// [`hash_part`](OpaqueValue::hash_part).
    fn hash<H: Hasher>(&self, state: &mut H) {
        mem::discriminant(self).hash(state);
        match self {
            Self::Bool(value) => value.hash(state),
            Self::Int(value) => value.hash(state),
            Self::Float(value) => {
                if !value.is_nan() {
                    (value + 0.0).to_bits().hash(state);
                }
            }
            Self::Decimal(value) => value.hash(state),
            Self::Str(value) => value.hash(state),
            Self::Tuple(values) => values.hash(state),
            Self::FrozenSet(values) => {
                let mut hashes: Vec<u64> = values
                    .iter()
                    .map(|value| {
                        let mut hasher = DefaultHasher::new();
                        value.hash(&mut hasher);
                        hasher.finish()
                    })
                    .collect();
                hashes.sort_unstable();
                hashes.dedup();
                hashes.hash(state);
            }
            Self::Opaque(value) => value.hash(state),
        }
    }
}

impl fmt::Display for Value {
    /// Write the value for people, as a literal expression writes the
    /// kinds it shares (`true`, `3`, `0.5`): a string quoted and escaped,
    /// a tuple as `(1, 2)`, `(1,)` or `()`, a frozen set as `{1, 2}` or
    /// `{}`, and an opaque value as its type name in angle brackets. The
    /// text is not parsed back, and distinct values may write alike (the
    /// integer `1` and the float `1.0`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(value) => write!(f, "{value}"),
            Self::Int(value) => write!(f, "{value}"),
            Self::Float(value) => write_float(*value, f),
            Self::Decimal(value) => write!(f, "{value}"),
            Self::Str(value) => write!(f, "{value:?}"),
            Self::Tuple(values) => write_tuple(f, values),
            Self::FrozenSet(values) => write_braced(f, values),
            Self::Opaque(value) => write!(f, "<{}>", value.get().type_name()),
        }
    }
}

/// Return whether two floats are the same number, or both NaN.
fn are_floats_equivalent(left: f64, right: f64) -> bool {
    left.partial_cmp(&right) == Some(Ordering::Equal) || (left.is_nan() && right.is_nan())
}

/// Write `items` as a tuple: `(a, b)`, `(a,)` or `()`.
fn write_tuple<T: fmt::Display>(f: &mut fmt::Formatter<'_>, items: &[T]) -> fmt::Result {
    f.write_str("(")?;
    write_separated(f, items)?;
    if items.len() == 1 {
        f.write_str(",")?;
    }
    f.write_str(")")
}

/// Write `items` in braces, separated by `, `: `{a, b}`, or `{}`.
fn write_braced<T: fmt::Display>(f: &mut fmt::Formatter<'_>, items: &[T]) -> fmt::Result {
    f.write_str("{")?;
    write_separated(f, items)?;
    f.write_str("}")
}

/// Write `items` separated by `, `.
fn write_separated<T: fmt::Display>(f: &mut fmt::Formatter<'_>, items: &[T]) -> fmt::Result {
    for (position, item) in items.iter().enumerate() {
        if position > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{item}")?;
    }
    Ok(())
}

impl Value {
    /// Return whether the value could be a member: whether it and every
    /// value it holds is a Boolean, an integer, a float, a string, a tuple, a
    /// frozen set, or a member-shaped opaque value. A NaN float is
    /// member-shaped, although no member is one.
    #[must_use]
    pub fn is_member_shaped(&self) -> bool {
        let mut pending = vec![self];
        while let Some(value) = pending.pop() {
            match value {
                Self::Bool(_) | Self::Int(_) | Self::Float(_) | Self::Str(_) => {}
                Self::Decimal(_) => return false,
                Self::Opaque(value) => {
                    if !value.get().is_member_shaped() {
                        return false;
                    }
                }
                Self::Tuple(values) | Self::FrozenSet(values) => pending.extend(values),
            }
        }
        true
    }

    /// Check that every opaque value the value holds can be looked up in a
    /// set.
    ///
    /// # Errors
    ///
    /// Returns the first opaque value's error, in pre-order.
    pub fn check_hashable(&self) -> Result<(), BoxError> {
        let mut pending = vec![self];
        while let Some(value) = pending.pop() {
            match value {
                Self::Opaque(value) => value.get().check_hashable()?,
                Self::Tuple(values) | Self::FrozenSet(values) => {
                    pending.extend(values.iter().rev());
                }
                Self::Bool(_) | Self::Int(_) | Self::Float(_) | Self::Decimal(_) | Self::Str(_) => {
                }
            }
        }
        Ok(())
    }
}

/// Why a [`Value`] cannot be a [`Member`].
#[derive(Debug)]
#[non_exhaustive]
pub enum MemberError {
    /// The value is, or holds, a NaN, which is unequal to itself.
    Nan,
    /// The value is, or holds, a decimal, which is no member kind.
    Decimal,
    /// The value is, or holds, an opaque value that is not member-shaped.
    NotMemberShaped {
        /// The name of the opaque value's type.
        type_name: String,
    },
    /// The value is, or holds, an opaque value whose
    /// [`ordering_key`](OpaqueValue::ordering_key) failed.
    OrderingKey {
        /// The name of the opaque value's type.
        type_name: String,
        /// The producer's error.
        source: BoxError,
    },
}

impl fmt::Display for MemberError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nan => f.write_str(
                "a member is, or holds, a NaN, which is unequal to itself and could never \
                 match a bound value",
            ),
            Self::Decimal => {
                f.write_str("a member is, or holds, a decimal, which is no member kind")
            }
            Self::NotMemberShaped { type_name } => write!(
                f,
                "a member is, or holds, a value of type {type_name}, which is no member kind"
            ),
            Self::OrderingKey { type_name, .. } => {
                write!(f, "the ordering key of a member of type {type_name} failed")
            }
        }
    }
}

impl Error for MemberError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OrderingKey { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

/// A value a set constraint can hold: a Boolean, an integer, a float, a
/// string, a tuple of members, a set of members, or a member-shaped opaque
/// value.
///
/// A member holds no NaN, and its float zeros are positive, so `-0.0` and
/// `0.0` make one member. Members compare type-strictly, and a
/// [`MemberSet`] orders them canonically.
#[derive(Debug, Clone)]
pub struct Member(MemberValue);

#[derive(Debug, Clone)]
enum MemberValue {
    Bool(bool),
    Int(BigInt),
    Float(f64),
    Str(String),
    Tuple(Vec<Member>),
    FrozenSet(MemberSet),
    Opaque(Part<dyn OpaqueValue>, String),
}

/// A view of a [`Member`]'s kind and contents.
#[expect(
    clippy::exhaustive_enums,
    reason = "the member kinds, which the binding converts one by one"
)]
#[derive(Debug, Clone, Copy)]
pub enum MemberKind<'a> {
    /// A Boolean.
    Bool(bool),
    /// An integer.
    Int(&'a BigInt),
    /// A float, never a NaN nor `-0.0`.
    Float(f64),
    /// A string.
    Str(&'a str),
    /// A tuple of members.
    Tuple(&'a [Member]),
    /// A set of members.
    FrozenSet(&'a MemberSet),
    /// A member-shaped opaque value.
    Opaque(&'a Part<dyn OpaqueValue>),
}

impl TryFrom<Value> for Member {
    type Error = MemberError;

    /// Return the member of `value`.
    ///
    /// # Errors
    ///
    /// Returns [`MemberError`] if `value` is, or holds, a NaN, a decimal or
    /// an opaque value that is not member-shaped, whichever comes first in
    /// pre-order, or an opaque value whose ordering key fails.
    fn try_from(value: Value) -> Result<Self, MemberError> {
        check_member(&value)?;
        build_member(value)
    }
}

impl Member {
    /// Return the member's kind and contents.
    #[must_use]
    pub fn kind(&self) -> MemberKind<'_> {
        match &self.0 {
            MemberValue::Bool(value) => MemberKind::Bool(*value),
            MemberValue::Int(value) => MemberKind::Int(value),
            MemberValue::Float(value) => MemberKind::Float(*value),
            MemberValue::Str(value) => MemberKind::Str(value),
            MemberValue::Tuple(members) => MemberKind::Tuple(members),
            MemberValue::FrozenSet(members) => MemberKind::FrozenSet(members),
            MemberValue::Opaque(value, _) => MemberKind::Opaque(value),
        }
    }

    /// Return whether the member lifts to a literal expression: a Boolean,
    /// an integer or a float. A string does not, since literal equality
    /// would canonicalize it against numeric members, and neither does a
    /// container or an opaque value.
    #[must_use]
    pub fn lifts_to_expression(&self) -> bool {
        matches!(
            self.0,
            MemberValue::Bool(_) | MemberValue::Int(_) | MemberValue::Float(_)
        )
    }

    /// Return the literal the member lifts to, or `None` if it does not
    /// lift.
    #[must_use]
    pub fn to_literal(&self) -> Option<LiteralValue> {
        match &self.0 {
            MemberValue::Bool(value) => Some(LiteralValue::Bool(*value)),
            MemberValue::Int(value) => Some(LiteralValue::Int(value.clone())),
            MemberValue::Float(value) => Some(LiteralValue::Float(*value)),
            MemberValue::Str(_)
            | MemberValue::Tuple(_)
            | MemberValue::FrozenSet(_)
            | MemberValue::Opaque(..) => None,
        }
    }

    /// Return the name of the member's kind: `bool`, `int`, `float`, `str`,
    /// `tuple`, `frozenset`, or an opaque value's type name.
    #[must_use]
    pub fn kind_name(&self) -> Cow<'_, str> {
        match &self.0 {
            MemberValue::Bool(_) => Cow::Borrowed("bool"),
            MemberValue::Int(_) => Cow::Borrowed("int"),
            MemberValue::Float(_) => Cow::Borrowed("float"),
            MemberValue::Str(_) => Cow::Borrowed("str"),
            MemberValue::Tuple(_) => Cow::Borrowed("tuple"),
            MemberValue::FrozenSet(_) => Cow::Borrowed("frozenset"),
            MemberValue::Opaque(value, _) => value.get().type_name(),
        }
    }

    /// Return the opaque member's ordering key, which is its value's key.
    pub(super) fn opaque_key(&self) -> Option<&str> {
        match &self.0 {
            MemberValue::Opaque(_, key) => Some(key),
            _ => None,
        }
    }

    /// Return the rank of the member's kind in the canonical order.
    fn rank(&self) -> u8 {
        match &self.0 {
            MemberValue::Bool(_) => 0,
            MemberValue::Float(_) => 1,
            MemberValue::FrozenSet(_) => 2,
            MemberValue::Int(_) => 3,
            MemberValue::Str(_) => 4,
            MemberValue::Tuple(_) => 5,
            MemberValue::Opaque(..) => 6,
        }
    }
}

impl PartialEq for Member {
    /// Compare type-strictly.
    fn eq(&self, other: &Self) -> bool {
        compare_canonically(self, other) == Ordering::Equal && are_opaque_parts_equal(self, other)
    }
}

impl PartialOrd for Member {
    /// Order canonically, as [`MemberSet`] does; opaque members with equal
    /// ordering keys that are not equal have no order.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match compare_canonically(self, other) {
            Ordering::Equal if !are_opaque_parts_equal(self, other) => None,
            ordering => Some(ordering),
        }
    }
}

impl Eq for Member {}

impl Hash for Member {
    /// Feed the kind, then the contents, consistently with `==`; an opaque
    /// member feeds its ordering key, which equal members share.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.rank().hash(state);
        match &self.0 {
            MemberValue::Bool(value) => value.hash(state),
            MemberValue::Int(value) => value.hash(state),
            MemberValue::Float(value) => value.to_bits().hash(state),
            MemberValue::Str(value) | MemberValue::Opaque(_, value) => value.hash(state),
            MemberValue::Tuple(members) => members.hash(state),
            MemberValue::FrozenSet(members) => members.hash(state),
        }
    }
}

impl fmt::Display for Member {
    /// Write the member as its [`Value`] writes, a frozen set's members in
    /// canonical order.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            MemberValue::Bool(value) => write!(f, "{value}"),
            MemberValue::Int(value) => write!(f, "{value}"),
            MemberValue::Float(value) => write_float(*value, f),
            MemberValue::Str(value) => write!(f, "{value:?}"),
            MemberValue::Tuple(members) => write_tuple(f, members),
            MemberValue::FrozenSet(members) => write!(f, "{members}"),
            MemberValue::Opaque(value, _) => write!(f, "<{}>", value.get().type_name()),
        }
    }
}

/// Return the canonical order of `left` and `right`, in which opaque values
/// compare by their keys alone.
fn compare_canonically(left: &Member, right: &Member) -> Ordering {
    match (&left.0, &right.0) {
        (MemberValue::Bool(left), MemberValue::Bool(right)) => left.cmp(right),
        (MemberValue::Int(left), MemberValue::Int(right)) => left.cmp(right),
        (MemberValue::Float(left), MemberValue::Float(right)) => left.total_cmp(right),
        (MemberValue::Str(left), MemberValue::Str(right))
        | (MemberValue::Opaque(_, left), MemberValue::Opaque(_, right)) => left.cmp(right),
        (MemberValue::Tuple(left), MemberValue::Tuple(right)) => compare_sequences(left, right),
        (MemberValue::FrozenSet(left), MemberValue::FrozenSet(right)) => {
            compare_sequences(&left.members, &right.members)
        }
        _ => left.rank().cmp(&right.rank()),
    }
}

/// Return the lexicographic canonical order of two member sequences.
fn compare_sequences(left: &[Member], right: &[Member]) -> Ordering {
    for (left, right) in left.iter().zip(right) {
        let ordering = compare_canonically(left, right);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.len().cmp(&right.len())
}

/// Return whether the opaque values of two members in the same canonical
/// place are equal, pairwise; members without opaque values are.
fn are_opaque_parts_equal(left: &Member, right: &Member) -> bool {
    match (&left.0, &right.0) {
        (MemberValue::Opaque(left, _), MemberValue::Opaque(right, _)) => left == right,
        (MemberValue::Tuple(left), MemberValue::Tuple(right)) => left
            .iter()
            .zip(right)
            .all(|(left, right)| are_opaque_parts_equal(left, right)),
        (MemberValue::FrozenSet(left), MemberValue::FrozenSet(right)) => left == right,
        _ => true,
    }
}

/// Check that `value` can be a member, in pre-order.
fn check_member(value: &Value) -> Result<(), MemberError> {
    let mut pending = vec![value];
    while let Some(value) = pending.pop() {
        match value {
            Value::Float(number) if number.is_nan() => return Err(MemberError::Nan),
            Value::Decimal(_) => return Err(MemberError::Decimal),
            Value::Opaque(opaque) if !opaque.get().is_member_shaped() => {
                return Err(MemberError::NotMemberShaped {
                    type_name: opaque.get().type_name().into_owned(),
                });
            }
            Value::Tuple(values) | Value::FrozenSet(values) => pending.extend(values.iter().rev()),
            Value::Bool(_) | Value::Int(_) | Value::Float(_) | Value::Str(_) | Value::Opaque(_) => {
            }
        }
    }
    Ok(())
}

/// Return the member of the checked `value`.
///
/// The recursion follows the nesting of containers, which a caller builds
/// by hand and is shallow.
fn build_member(value: Value) -> Result<Member, MemberError> {
    Ok(Member(match value {
        Value::Bool(value) => MemberValue::Bool(value),
        Value::Int(value) => MemberValue::Int(value),
        Value::Float(value) => MemberValue::Float(value + 0.0),
        Value::Str(value) => MemberValue::Str(value),
        Value::Tuple(values) => MemberValue::Tuple(
            values
                .into_iter()
                .map(build_member)
                .collect::<Result<_, _>>()?,
        ),
        Value::FrozenSet(values) => MemberValue::FrozenSet(
            values
                .into_iter()
                .map(build_member)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .collect(),
        ),
        Value::Opaque(value) => {
            let key = match value.get().ordering_key() {
                Ok(key) => key.into_owned(),
                Err(source) => {
                    return Err(MemberError::OrderingKey {
                        type_name: value.get().type_name().into_owned(),
                        source,
                    });
                }
            };
            MemberValue::Opaque(value, key)
        }
        Value::Decimal(_) => unreachable!("a checked value holds no decimal"),
    }))
}

/// Return whether `value` is, or holds, an opaque value.
fn holds_opaque(value: &Value) -> bool {
    let mut pending = vec![value];
    while let Some(value) = pending.pop() {
        match value {
            Value::Opaque(_) => return true,
            Value::Tuple(values) | Value::FrozenSet(values) => pending.extend(values),
            Value::Bool(_)
            | Value::Int(_)
            | Value::Float(_)
            | Value::Decimal(_)
            | Value::Str(_) => {}
        }
    }
    false
}

/// Return whether the member-shaped `value` equals `member`
/// type-strictly. A NaN equals nothing.
fn is_value_equal_to_member(value: &Value, member: &Member) -> bool {
    match (value, &member.0) {
        (Value::Bool(value), MemberValue::Bool(member)) => value == member,
        (Value::Int(value), MemberValue::Int(member)) => value == member,
        (Value::Float(value), MemberValue::Float(member)) => {
            // A member's zero is positive, and it is never a NaN.
            (value + 0.0).to_bits() == member.to_bits()
        }
        (Value::Str(value), MemberValue::Str(member)) => value == member,
        (Value::Tuple(values), MemberValue::Tuple(members)) => {
            values.len() == members.len()
                && values
                    .iter()
                    .zip(members)
                    .all(|(value, member)| is_value_equal_to_member(value, member))
        }
        (Value::FrozenSet(values), MemberValue::FrozenSet(members)) => {
            values.iter().all(|value| members.contains_value(value))
                && members.iter().all(|member| {
                    values
                        .iter()
                        .any(|value| is_value_equal_to_member(value, member))
                })
        }
        (Value::Opaque(value), MemberValue::Opaque(member, _)) => {
            value.get().is_member_shaped() && member == value
        }
        _ => false,
    }
}

/// Distinct members in canonical order.
///
/// The order is by kind, `bool`, `float`, `frozenset`, `int`, `str`,
/// `tuple`, then opaque values, and within a kind by value: `false` before
/// `true`, numbers numerically, strings by code point, tuples element by
/// element, sets by their members in canonical order, and opaque values by
/// their ordering keys, keeping the order they were given in among equal
/// keys. Two sets are equal when they hold equal members.
#[derive(Debug, Clone, Default)]
pub struct MemberSet {
    members: Vec<Member>,
}

impl MemberSet {
    /// Return the set of `members`, keeping the first of equal members.
    pub fn new(members: impl IntoIterator<Item = Member>) -> Self {
        let mut sorted: Vec<Member> = members.into_iter().collect();
        sorted.sort_by(compare_canonically);
        let mut distinct: Vec<Member> = Vec::with_capacity(sorted.len());
        let mut run_start = 0;
        for member in sorted {
            if distinct
                .last()
                .is_none_or(|last| compare_canonically(last, &member) != Ordering::Equal)
            {
                run_start = distinct.len();
            } else if distinct[run_start..]
                .iter()
                .any(|kept| are_opaque_parts_equal(kept, &member))
            {
                continue;
            }
            distinct.push(member);
        }
        Self { members: distinct }
    }

    /// Return the members in canonical order.
    pub fn iter(&self) -> std::slice::Iter<'_, Member> {
        self.members.iter()
    }

    /// Return the number of members.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Return whether the set holds no member.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Return whether the set holds `member`.
    #[must_use]
    pub fn contains(&self, member: &Member) -> bool {
        let start = self
            .members
            .partition_point(|held| compare_canonically(held, member) == Ordering::Less);
        self.members[start..]
            .iter()
            .take_while(|held| compare_canonically(held, member) == Ordering::Equal)
            .any(|held| are_opaque_parts_equal(held, member))
    }

    /// Return whether the set holds a member equal to `value`. A NaN, a
    /// decimal, and a value that is not member-shaped, at any depth, equal
    /// no member.
    #[must_use]
    pub fn contains_value(&self, value: &Value) -> bool {
        if !value.is_member_shaped() {
            return false;
        }
        if holds_opaque(value) {
            // An opaque value's ordering key may cost its producer more than
            // the comparisons, so it is compared with each member instead.
            return self
                .members
                .iter()
                .any(|member| is_value_equal_to_member(value, member));
        }
        match Member::try_from(value.clone()) {
            Ok(member) => self.contains(&member),
            Err(_nan) => false,
        }
    }
}

impl PartialEq for MemberSet {
    /// Compare as sets: the same number of members, each held by the other.
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|member| other.contains(member))
    }
}

impl Eq for MemberSet {}

impl Hash for MemberSet {
    /// Feed the members in canonical order: equal sets hold members of equal
    /// hashes in the same order, since members that tie in it share their
    /// hash.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.members.hash(state);
    }
}

impl fmt::Display for MemberSet {
    /// Write the members in canonical order, in braces: `{1, 2}`, or `{}`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_braced(f, &self.members)
    }
}

impl<'a> IntoIterator for &'a MemberSet {
    type Item = &'a Member;
    type IntoIter = std::slice::Iter<'a, Member>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl FromIterator<Member> for MemberSet {
    fn from_iter<I: IntoIterator<Item = Member>>(members: I) -> Self {
        Self::new(members)
    }
}
