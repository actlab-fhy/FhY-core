//! The domains of a search's steps: the sets a step may be answered from,
//! each numbering its values with [`Coordinate`]s, the [`DomainSignature`]
//! a replay compares, and the open [`DecisionKind`] of a step.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use num_bigint::{BigInt, BigUint};
use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::constraint::Value;
use crate::expression::{integer_text, serialize_display_text};

use super::error::{EmptyKind, StepDomainError};
use super::space::Space;

/// What a step is about: a namespaced name, the same in every process.
///
/// A static step's kind is [`DecisionKind::CHOICE`] for a choice and the
/// variable's own [`kind`](super::Variable::kind) for a variable. A dynamic
/// step's kind is its caller's, such as `moga.cir.address`. Two kinds are
/// equal when their names are.
///
/// `serde` writes the name as a string, and reading refuses an empty one.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct DecisionKind(Arc<str>);

impl DecisionKind {
    /// The kind of a static step over a choice.
    pub const CHOICE: &'static str = "search_space.choice";

    /// Return the kind named `kind`.
    ///
    /// # Errors
    ///
    /// Returns [`EmptyKind`] for an empty name.
    pub fn new(kind: &str) -> Result<Self, EmptyKind> {
        if kind.is_empty() {
            return Err(EmptyKind);
        }
        Ok(Self(Arc::from(kind)))
    }

    /// Return the kind of a static step over a choice,
    /// [`CHOICE`](Self::CHOICE).
    #[must_use]
    pub fn choice() -> Self {
        Self(Arc::from(Self::CHOICE))
    }

    /// Return the kind of a static step over a variable of kind `kind`;
    /// an empty kind, which breaks the implementor contract, is read as
    /// the plain variable's.
    pub(super) fn of_variable(kind: &str) -> Self {
        let kind = if kind.is_empty() {
            super::PlainVariable::KIND
        } else {
            kind
        };
        Self(Arc::from(kind))
    }

    /// Return the kind's name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for DecisionKind {
    type Error = EmptyKind;

    fn try_from(kind: String) -> Result<Self, Self::Error> {
        Self::new(&kind)
    }
}

impl From<DecisionKind> for String {
    fn from(kind: DecisionKind) -> Self {
        kind.0.to_string()
    }
}

/// Displays the kind's name.
impl fmt::Display for DecisionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A step's answer: a position in its domain.
///
/// `serde` writes `{"index": <integer>}` or `{"order": [<integer>, ..]}`,
/// externally tagged, so non-self-describing formats read it too.
#[expect(
    clippy::exhaustive_enums,
    reason = "an answer is an index or a permutation of positions"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Coordinate {
    /// The position of a value in a choice domain, or of an integer among a
    /// strided domain's runs flattened in order.
    Index(u64),
    /// A permutation of an order domain's element positions: the `k`-th
    /// entry is the position, among the elements, of the one placed
    /// `k`-th.
    Order(Box<[u32]>),
}

/// One or more distinct values; a step over it answers one of them.
///
/// Values are distinct by [`Value`]'s type-strict `==`, so `true` and the
/// integer `1` are two values. Cloning shares the values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChoiceDomain(Arc<[Value]>);

impl ChoiceDomain {
    /// Return the domain of `values`, in the order given.
    ///
    /// # Errors
    ///
    /// In order: [`StepDomainError::EmptyChoice`] for no value,
    /// [`StepDomainError::NanValue`] for a NaN float, at any depth, and
    /// [`StepDomainError::RepeatedValue`] for two equal values.
    pub fn new(values: Vec<Value>) -> Result<Self, StepDomainError> {
        if values.is_empty() {
            return Err(StepDomainError::EmptyChoice);
        }
        check_distinct_values(&values)?;
        Ok(Self(values.into()))
    }

    /// Return the values, in the order given.
    #[must_use]
    pub fn values(&self) -> &[Value] {
        &self.0
    }

    /// Return the number of values.
    #[must_use]
    pub fn cardinality(&self) -> u64 {
        count_of(self.0.len())
    }

    /// Return the value at `index`, or `None` past the last.
    #[must_use]
    pub fn value_at(&self, index: u64) -> Option<&Value> {
        usize::try_from(index)
            .ok()
            .and_then(|index| self.0.get(index))
    }

    /// Return the position of the value equal to `value`, or `None`.
    #[must_use]
    pub fn coordinate_of(&self, value: &Value) -> Option<u64> {
        self.0
            .iter()
            .position(|candidate| candidate == value)
            .map(count_of)
    }

    /// Return whether a value of the domain equals `value`.
    #[must_use]
    pub fn admits(&self, value: &Value) -> bool {
        self.0.contains(value)
    }
}

/// The permutations of one or more distinct elements; a step over it
/// answers one ordering, a [`Value::Tuple`] holding each element once.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OrderDomain(Arc<[Value]>);

impl OrderDomain {
    /// Return the domain of the orderings of `elements`, whose canonical
    /// order is the order given.
    ///
    /// # Errors
    ///
    /// In order: [`StepDomainError::EmptyOrder`] for no element,
    /// [`StepDomainError::NanValue`] for a NaN float,
    /// [`StepDomainError::RepeatedValue`] for two equal elements, and
    /// [`StepDomainError::TooLarge`] for more than `u32::MAX` elements.
    pub fn new(elements: Vec<Value>) -> Result<Self, StepDomainError> {
        if elements.is_empty() {
            return Err(StepDomainError::EmptyOrder);
        }
        check_distinct_values(&elements)?;
        if u32::try_from(elements.len()).is_err() {
            return Err(StepDomainError::TooLarge);
        }
        Ok(Self(elements.into()))
    }

    /// Return the elements, in their canonical order.
    #[must_use]
    pub fn elements(&self) -> &[Value] {
        &self.0
    }

    /// Return the number of orderings, `n!` for `n` elements.
    #[must_use]
    pub fn cardinality(&self) -> BigUint {
        factorial(self.0.len())
    }

    /// Return the ordering `positions` names, outermost first, or `None`
    /// when `positions` is not a permutation of the element positions.
    #[must_use]
    pub fn value_at(&self, positions: &[u32]) -> Option<Value> {
        if !is_permutation(positions, self.0.len()) {
            return None;
        }
        positions
            .iter()
            .map(|&position| {
                usize::try_from(position)
                    .ok()
                    .and_then(|position| self.0.get(position))
                    .cloned()
            })
            .collect::<Option<Vec<Value>>>()
            .map(Value::Tuple)
    }

    /// Return the positions `value` orders the elements in, or `None` when
    /// `value` is not a tuple holding each element once.
    #[must_use]
    pub fn coordinate_of(&self, value: &Value) -> Option<Box<[u32]>> {
        let Value::Tuple(ordering) = value else {
            return None;
        };
        if ordering.len() != self.0.len() {
            return None;
        }
        let positions = ordering
            .iter()
            .map(|element| {
                self.0
                    .iter()
                    .position(|candidate| candidate == element)
                    .and_then(|position| u32::try_from(position).ok())
            })
            .collect::<Option<Box<[u32]>>>()?;
        is_permutation(&positions, self.0.len()).then_some(positions)
    }

    /// Return whether `value` is an ordering of the elements.
    #[must_use]
    pub fn admits(&self, value: &Value) -> bool {
        self.coordinate_of(value).is_some()
    }
}

/// A run of integers: `start`, `start + stride`, `start + 2 * stride`, ...,
/// each below `stop`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StridedRun {
    start: BigInt,
    stop: BigInt,
    stride: BigUint,
}

impl StridedRun {
    /// Return the run from `start` below `stop` by `stride`.
    ///
    /// # Errors
    ///
    /// In order: [`StepDomainError::EmptyRun`] when `stop` is not above
    /// `start`, and [`StepDomainError::ZeroStride`] for a zero stride.
    pub fn new(start: BigInt, stop: BigInt, stride: BigUint) -> Result<Self, StepDomainError> {
        if stop <= start {
            return Err(StepDomainError::EmptyRun);
        }
        if stride == BigUint::ZERO {
            return Err(StepDomainError::ZeroStride);
        }
        Ok(Self {
            start,
            stop,
            stride,
        })
    }

    /// Return the run's first integer.
    #[must_use]
    pub fn start(&self) -> &BigInt {
        &self.start
    }

    /// Return the bound every integer of the run is below.
    #[must_use]
    pub fn stop(&self) -> &BigInt {
        &self.stop
    }

    /// Return the distance between consecutive integers of the run.
    #[must_use]
    pub fn stride(&self) -> &BigUint {
        &self.stride
    }

    /// Return the number of integers the run holds.
    #[must_use]
    pub fn width(&self) -> BigUint {
        let span = (&self.stop - &self.start).magnitude().clone();
        (span + &self.stride - BigUint::from(1_u8)) / &self.stride
    }

    /// Return whether `value` is an integer of the run.
    #[must_use]
    pub fn admits(&self, value: &BigInt) -> bool {
        self.offset_of(value).is_some()
    }

    /// Return the position of `value` among the run's integers, if it is
    /// one.
    fn offset_of(&self, value: &BigInt) -> Option<BigUint> {
        if *value < self.start || *value >= self.stop {
            return None;
        }
        let distance = (value - &self.start).magnitude().clone();
        (&distance % &self.stride == BigUint::ZERO).then(|| distance / &self.stride)
    }

    /// Return the integer at `offset` in the run.
    fn value_at(&self, offset: u64) -> BigInt {
        &self.start + BigInt::from(&self.stride * BigUint::from(offset))
    }
}

/// A union of disjoint, ascending [`StridedRun`]s; a step over it answers
/// one of their integers.
///
/// Its coordinates number the runs' integers in order, the first run's
/// first, so a uniform index is uniform over the union.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StridedDomain(Arc<StridedRuns>);

/// The runs of a [`StridedDomain`] and their widths.
#[derive(Debug, PartialEq, Eq, Hash)]
struct StridedRuns {
    runs: Box<[StridedRun]>,
    /// Each run's number of integers, which fits a `u64`.
    widths: Box<[u64]>,
}

impl StridedDomain {
    /// Return the domain of `runs`.
    ///
    /// # Errors
    ///
    /// In order: [`StepDomainError::EmptyRuns`] for no run,
    /// [`StepDomainError::UnorderedRuns`] for a run starting below the
    /// previous run's stop, and [`StepDomainError::TooLarge`] for more than
    /// `u64::MAX` integers in all.
    pub fn new(runs: Vec<StridedRun>) -> Result<Self, StepDomainError> {
        if runs.is_empty() {
            return Err(StepDomainError::EmptyRuns);
        }
        if let Some(index) = (1..runs.len()).find(|&index| runs[index].start < runs[index - 1].stop)
        {
            return Err(StepDomainError::UnorderedRuns { index });
        }
        let mut total: u64 = 0;
        let mut widths = Vec::with_capacity(runs.len());
        for run in &runs {
            let width =
                u64::try_from(run.width()).map_err(|_too_wide| StepDomainError::TooLarge)?;
            total = total.checked_add(width).ok_or(StepDomainError::TooLarge)?;
            widths.push(width);
        }
        Ok(Self(Arc::new(StridedRuns {
            runs: runs.into(),
            widths: widths.into(),
        })))
    }

    /// Return the runs, ascending.
    #[must_use]
    pub fn runs(&self) -> &[StridedRun] {
        &self.0.runs
    }

    /// Return the runs and their widths.
    fn measured(&self) -> impl Iterator<Item = (&StridedRun, u64)> {
        self.0.runs.iter().zip(self.0.widths.iter().copied())
    }

    /// Return the number of integers the runs hold.
    #[must_use]
    pub fn cardinality(&self) -> u64 {
        self.0.widths.iter().sum()
    }

    /// Return the integer at `index`, or `None` past the last.
    #[must_use]
    pub fn value_at(&self, index: u64) -> Option<BigInt> {
        let mut remaining = index;
        for (run, width) in self.measured() {
            if remaining < width {
                return Some(run.value_at(remaining));
            }
            remaining -= width;
        }
        None
    }

    /// Return the position of `value` among the runs' integers, or `None`.
    #[must_use]
    pub fn coordinate_of(&self, value: &BigInt) -> Option<u64> {
        let mut before: u64 = 0;
        for (run, width) in self.measured() {
            if let Some(offset) = run.offset_of(value) {
                return u64::try_from(offset).ok().map(|offset| before + offset);
            }
            before += width;
        }
        None
    }

    /// Return whether `value` is an integer value of the runs. Only a
    /// [`Value::Int`] can be.
    #[must_use]
    pub fn admits(&self, value: &Value) -> bool {
        matches!(value, Value::Int(integer) if self.coordinate_of(integer).is_some())
    }
}

/// The set a step may be answered from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StepDomain {
    /// One of a list of values.
    Choice(ChoiceDomain),
    /// An ordering of a list of elements.
    Order(OrderDomain),
    /// An integer of a union of strided runs.
    Strided(StridedDomain),
}

impl StepDomain {
    /// Return the number of values.
    #[must_use]
    pub fn cardinality(&self) -> BigUint {
        match self {
            Self::Choice(domain) => BigUint::from(domain.cardinality()),
            Self::Order(domain) => domain.cardinality(),
            Self::Strided(domain) => BigUint::from(domain.cardinality()),
        }
    }

    /// Return whether `coordinate` names a value of the domain: an index
    /// below the cardinality of a choice or strided domain, a permutation
    /// of an order domain's positions.
    #[must_use]
    pub fn contains(&self, coordinate: &Coordinate) -> bool {
        match (self, coordinate) {
            (Self::Choice(domain), Coordinate::Index(index)) => *index < domain.cardinality(),
            (Self::Strided(domain), Coordinate::Index(index)) => *index < domain.cardinality(),
            (Self::Order(domain), Coordinate::Order(positions)) => {
                is_permutation(positions, domain.elements().len())
            }
            _ => false,
        }
    }

    /// Return the value `coordinate` names, or `None` when it names none.
    #[must_use]
    pub fn value_at(&self, coordinate: &Coordinate) -> Option<Value> {
        match (self, coordinate) {
            (Self::Choice(domain), Coordinate::Index(index)) => domain.value_at(*index).cloned(),
            (Self::Strided(domain), Coordinate::Index(index)) => {
                domain.value_at(*index).map(Value::Int)
            }
            (Self::Order(domain), Coordinate::Order(positions)) => domain.value_at(positions),
            _ => None,
        }
    }

    /// Return the coordinate of `value`, or `None` when the domain does not
    /// admit it.
    #[must_use]
    pub fn coordinate_of(&self, value: &Value) -> Option<Coordinate> {
        match self {
            Self::Choice(domain) => domain.coordinate_of(value).map(Coordinate::Index),
            Self::Strided(domain) => match value {
                Value::Int(integer) => domain.coordinate_of(integer).map(Coordinate::Index),
                _ => None,
            },
            Self::Order(domain) => domain.coordinate_of(value).map(Coordinate::Order),
        }
    }

    /// Return whether the domain admits `value`.
    #[must_use]
    pub fn admits(&self, value: &Value) -> bool {
        self.coordinate_of(value).is_some()
    }

    /// Return the domain's signature, every identifier written as
    /// `identifier`, as a dynamic step's is.
    #[must_use]
    pub fn signature(&self) -> DomainSignature {
        self.signature_in(None)
    }

    /// Return the domain's signature, every identifier `space` binds written
    /// as its position among the space's names.
    pub(super) fn signature_in(&self, space: Option<&Space>) -> DomainSignature {
        let members = |values: &[Value]| -> Arc<[MemberSignature]> {
            values
                .iter()
                .map(|value| MemberSignature::of(value, space))
                .collect()
        };
        DomainSignature(match self {
            Self::Choice(domain) => SignatureRepr::Choice(members(domain.values())),
            Self::Order(domain) => SignatureRepr::Order(members(domain.elements())),
            Self::Strided(domain) => SignatureRepr::Strided(domain.clone()),
        })
    }
}

impl From<ChoiceDomain> for StepDomain {
    fn from(domain: ChoiceDomain) -> Self {
        Self::Choice(domain)
    }
}

impl From<OrderDomain> for StepDomain {
    fn from(domain: OrderDomain) -> Self {
        Self::Order(domain)
    }
}

impl From<StridedDomain> for StepDomain {
    fn from(domain: StridedDomain) -> Self {
        Self::Strided(domain)
    }
}

/// What a replay compares of a step's domain: the parts that mean the same
/// thing in every module and every process.
///
/// - The shape and the cardinality.
/// - A choice's values and an order's elements, each as itself if it is
///   plain (a Boolean, integer, float, decimal or string, or a tuple or
///   frozen set of plain values), as its position among the space's names
///   if it is an identifier the step's space binds, as `identifier` for
///   any other identifier, and as `opaque` for any other value: an opaque
///   value, or a tuple or frozen set holding an identifier or an opaque
///   value.
/// - A strided domain's runs.
///
/// `serde` writes `{"choice": [..]}`, `{"order": [..]}` or `{"strided":
/// [{"start", "stop", "stride"}, ..]}`, each integer of a run as its
/// decimal digits in a string, as a value's integer is written, and a
/// member as `{"value": <value>}`, `{"bound": <position>}`, `"identifier"`
/// or `"opaque"`; reading refuses a run that [`StridedRun::new`] or
/// [`StridedDomain::new`] refuses.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DomainSignature(SignatureRepr);

/// The parts of a [`DomainSignature`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum SignatureRepr {
    Choice(Arc<[MemberSignature]>),
    Order(Arc<[MemberSignature]>),
    Strided(StridedDomain),
}

/// One value of a choice or element of an order, in a signature.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum MemberSignature {
    Value(Value),
    Bound(u64),
    Identifier,
    Opaque,
}

impl DomainSignature {
    /// Return the number of values of the domain it was taken from.
    #[must_use]
    pub fn cardinality(&self) -> BigUint {
        match &self.0 {
            SignatureRepr::Choice(members) => BigUint::from(members.len()),
            SignatureRepr::Order(members) => factorial(members.len()),
            SignatureRepr::Strided(domain) => BigUint::from(domain.cardinality()),
        }
    }

    /// Return the shape: `choice`, `order` or `strided`.
    #[must_use]
    pub fn shape(&self) -> &'static str {
        match &self.0 {
            SignatureRepr::Choice(_) => "choice",
            SignatureRepr::Order(_) => "order",
            SignatureRepr::Strided(_) => "strided",
        }
    }

    /// Return whether `coordinate` names a value of the domain it was taken
    /// from, as [`StepDomain::contains`] says.
    #[must_use]
    pub fn contains(&self, coordinate: &Coordinate) -> bool {
        match (&self.0, coordinate) {
            (SignatureRepr::Choice(members), Coordinate::Index(index)) => {
                usize::try_from(*index).is_ok_and(|index| index < members.len())
            }
            (SignatureRepr::Strided(domain), Coordinate::Index(index)) => {
                *index < domain.cardinality()
            }
            (SignatureRepr::Order(members), Coordinate::Order(positions)) => {
                is_permutation(positions, members.len())
            }
            _ => false,
        }
    }
}

impl MemberSignature {
    /// Return the signature of `value`, an identifier `space` binds written
    /// as its position among the space's names.
    fn of(value: &Value, space: Option<&Space>) -> Self {
        if is_plain(value) {
            return Self::Value(value.clone());
        }
        match value {
            Value::Identifier(identifier) => space
                .and_then(|space| space.label_position(identifier))
                .map_or(Self::Identifier, |position| Self::Bound(count_of(position))),
            _ => Self::Opaque,
        }
    }
}

/// The wire form of a [`DomainSignature`].
#[derive(Serialize, Deserialize)]
#[serde(rename = "DomainSignature", rename_all = "snake_case")]
enum SignatureWire {
    Choice(Vec<MemberWire>),
    Order(Vec<MemberWire>),
    Strided(Vec<RunWire>),
}

/// The wire form of a [`MemberSignature`].
#[derive(Serialize, Deserialize)]
#[serde(rename = "MemberSignature", rename_all = "snake_case")]
enum MemberWire {
    Value(Value),
    Bound(u64),
    Identifier,
    Opaque,
}

/// The wire form of a [`StridedRun`]: its integers' decimal digits.
#[derive(Serialize, Deserialize)]
#[serde(rename = "StridedRun", deny_unknown_fields)]
struct RunWire {
    #[serde(
        serialize_with = "serialize_display_text",
        deserialize_with = "integer_text::deserialize"
    )]
    start: BigInt,
    #[serde(
        serialize_with = "serialize_display_text",
        deserialize_with = "integer_text::deserialize"
    )]
    stop: BigInt,
    #[serde(
        serialize_with = "serialize_display_text",
        deserialize_with = "integer_text::deserialize"
    )]
    stride: BigInt,
}

impl From<&MemberSignature> for MemberWire {
    fn from(member: &MemberSignature) -> Self {
        match member {
            MemberSignature::Value(value) => Self::Value(value.clone()),
            MemberSignature::Bound(position) => Self::Bound(*position),
            MemberSignature::Identifier => Self::Identifier,
            MemberSignature::Opaque => Self::Opaque,
        }
    }
}

impl From<MemberWire> for MemberSignature {
    fn from(member: MemberWire) -> Self {
        match member {
            MemberWire::Value(value) => Self::Value(value),
            MemberWire::Bound(position) => Self::Bound(position),
            MemberWire::Identifier => Self::Identifier,
            MemberWire::Opaque => Self::Opaque,
        }
    }
}

/// Serializes the shape of the type's documentation.
impl Serialize for DomainSignature {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let members = |members: &[MemberSignature]| members.iter().map(MemberWire::from).collect();
        let wire = match &self.0 {
            SignatureRepr::Choice(choices) => SignatureWire::Choice(members(choices)),
            SignatureRepr::Order(elements) => SignatureWire::Order(members(elements)),
            SignatureRepr::Strided(domain) => SignatureWire::Strided(
                domain
                    .runs()
                    .iter()
                    .map(|run| RunWire {
                        start: run.start.clone(),
                        stop: run.stop.clone(),
                        stride: BigInt::from(run.stride.clone()),
                    })
                    .collect(),
            ),
        };
        wire.serialize(serializer)
    }
}

/// Deserializes the shape of the type's documentation.
impl<'de> Deserialize<'de> for DomainSignature {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let members = |members: Vec<MemberWire>| -> Arc<[MemberSignature]> {
            members.into_iter().map(MemberSignature::from).collect()
        };
        Ok(Self(match SignatureWire::deserialize(deserializer)? {
            SignatureWire::Choice(choices) => SignatureRepr::Choice(members(choices)),
            SignatureWire::Order(elements) => SignatureRepr::Order(members(elements)),
            SignatureWire::Strided(runs) => {
                let runs = runs
                    .into_iter()
                    .map(|run| {
                        let stride = run.stride.to_biguint().ok_or_else(|| {
                            de::Error::custom("a run's stride must not be negative")
                        })?;
                        StridedRun::new(run.start, run.stop, stride).map_err(de::Error::custom)
                    })
                    .collect::<Result<Vec<_>, D::Error>>()?;
                SignatureRepr::Strided(StridedDomain::new(runs).map_err(de::Error::custom)?)
            }
        }))
    }
}

/// Return `count` as a `u64`, which every count of values in memory fits.
fn count_of(count: usize) -> u64 {
    u64::try_from(count).unwrap_or(u64::MAX)
}

/// Return `n!`.
fn factorial(n: usize) -> BigUint {
    (1..=n).map(BigUint::from).product()
}

/// Return whether `positions` holds each of `0..count` once.
pub(super) fn is_permutation(positions: &[u32], count: usize) -> bool {
    if positions.len() != count {
        return false;
    }
    let mut seen = vec![false; count];
    positions.iter().all(|&position| {
        usize::try_from(position)
            .ok()
            .and_then(|position| seen.get_mut(position))
            .is_some_and(|slot| !std::mem::replace(slot, true))
    })
}

/// Return whether `value` is, or holds, a NaN float.
fn holds_nan(value: &Value) -> bool {
    match value {
        Value::Float(float) => float.is_nan(),
        Value::Tuple(values) | Value::FrozenSet(values) => values.iter().any(holds_nan),
        _ => false,
    }
}

/// Return whether `value` is plain: a Boolean, integer, float, decimal or
/// string, or a tuple or frozen set of plain values.
fn is_plain(value: &Value) -> bool {
    match value {
        Value::Bool(_) | Value::Int(_) | Value::Float(_) | Value::Decimal(_) | Value::Str(_) => {
            true
        }
        Value::Tuple(values) | Value::FrozenSet(values) => values.iter().all(is_plain),
        _ => false,
    }
}

/// Refuse a NaN in `values`, then two equal values.
fn check_distinct_values(values: &[Value]) -> Result<(), StepDomainError> {
    if let Some(index) = values.iter().position(holds_nan) {
        return Err(StepDomainError::NanValue { index });
    }
    let mut seen: HashMap<&Value, usize> = HashMap::with_capacity(values.len());
    for (second, value) in values.iter().enumerate() {
        if let Some(&first) = seen.get(value) {
            return Err(StepDomainError::RepeatedValue { first, second });
        }
        seen.insert(value, second);
    }
    Ok(())
}
