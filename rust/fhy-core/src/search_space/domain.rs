//! The domains of a search's steps: the sets a step may be answered from,
//! each numbering its values with [`Coordinate`]s, the [`DomainSignature`]
//! a replay compares, and the open [`DecisionKind`] of a step.

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use std::fmt;
use std::sync::Arc;

use num_bigint::{BigInt, BigUint};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::constraint::Value;

use super::error::{EmptyKind, StepDomainError};

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
        todo!()
    }

    /// Return the kind of a static step over a choice,
    /// [`CHOICE`](Self::CHOICE).
    #[must_use]
    pub fn choice() -> Self {
        todo!()
    }

    /// Return the kind's name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        todo!()
    }
}

impl TryFrom<String> for DecisionKind {
    type Error = EmptyKind;

    fn try_from(kind: String) -> Result<Self, Self::Error> {
        todo!()
    }
}

impl From<DecisionKind> for String {
    fn from(kind: DecisionKind) -> Self {
        todo!()
    }
}

/// Displays the kind's name.
impl fmt::Display for DecisionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

/// A step's answer: a position in its domain.
///
/// `serde` writes an index as an integer and an order as an array of
/// integers.
#[expect(
    clippy::exhaustive_enums,
    reason = "an answer is an index or a permutation of positions"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(untagged)]
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
        todo!()
    }

    /// Return the values, in the order given.
    #[must_use]
    pub fn values(&self) -> &[Value] {
        todo!()
    }

    /// Return the number of values.
    #[must_use]
    pub fn cardinality(&self) -> u64 {
        todo!()
    }

    /// Return the value at `index`, or `None` past the last.
    #[must_use]
    pub fn value_at(&self, index: u64) -> Option<&Value> {
        todo!()
    }

    /// Return the position of the value equal to `value`, or `None`.
    #[must_use]
    pub fn coordinate_of(&self, value: &Value) -> Option<u64> {
        todo!()
    }

    /// Return whether a value of the domain equals `value`.
    #[must_use]
    pub fn admits(&self, value: &Value) -> bool {
        todo!()
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
        todo!()
    }

    /// Return the elements, in their canonical order.
    #[must_use]
    pub fn elements(&self) -> &[Value] {
        todo!()
    }

    /// Return the number of orderings, `n!` for `n` elements.
    #[must_use]
    pub fn cardinality(&self) -> BigUint {
        todo!()
    }

    /// Return the ordering `positions` names, outermost first, or `None`
    /// when `positions` is not a permutation of the element positions.
    #[must_use]
    pub fn value_at(&self, positions: &[u32]) -> Option<Value> {
        todo!()
    }

    /// Return the positions `value` orders the elements in, or `None` when
    /// `value` is not a tuple holding each element once.
    #[must_use]
    pub fn coordinate_of(&self, value: &Value) -> Option<Box<[u32]>> {
        todo!()
    }

    /// Return whether `value` is an ordering of the elements.
    #[must_use]
    pub fn admits(&self, value: &Value) -> bool {
        todo!()
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
        todo!()
    }

    /// Return the run's first integer.
    #[must_use]
    pub fn start(&self) -> &BigInt {
        todo!()
    }

    /// Return the bound every integer of the run is below.
    #[must_use]
    pub fn stop(&self) -> &BigInt {
        todo!()
    }

    /// Return the distance between consecutive integers of the run.
    #[must_use]
    pub fn stride(&self) -> &BigUint {
        todo!()
    }

    /// Return the number of integers the run holds.
    #[must_use]
    pub fn width(&self) -> BigUint {
        todo!()
    }

    /// Return whether `value` is an integer of the run.
    #[must_use]
    pub fn admits(&self, value: &BigInt) -> bool {
        todo!()
    }
}

/// A union of disjoint, ascending [`StridedRun`]s; a step over it answers
/// one of their integers.
///
/// Its coordinates number the runs' integers in order, the first run's
/// first, so a uniform index is uniform over the union.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StridedDomain(Arc<[StridedRun]>);

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
        todo!()
    }

    /// Return the runs, ascending.
    #[must_use]
    pub fn runs(&self) -> &[StridedRun] {
        todo!()
    }

    /// Return the number of integers the runs hold.
    #[must_use]
    pub fn cardinality(&self) -> u64 {
        todo!()
    }

    /// Return the integer at `index`, or `None` past the last.
    #[must_use]
    pub fn value_at(&self, index: u64) -> Option<BigInt> {
        todo!()
    }

    /// Return the position of `value` among the runs' integers, or `None`.
    #[must_use]
    pub fn coordinate_of(&self, value: &BigInt) -> Option<u64> {
        todo!()
    }

    /// Return whether `value` is an integer value of the runs. Only a
    /// [`Value::Int`] can be.
    #[must_use]
    pub fn admits(&self, value: &Value) -> bool {
        todo!()
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
        todo!()
    }

    /// Return whether `coordinate` names a value of the domain: an index
    /// below the cardinality of a choice or strided domain, a permutation
    /// of an order domain's positions.
    #[must_use]
    pub fn contains(&self, coordinate: &Coordinate) -> bool {
        todo!()
    }

    /// Return the value `coordinate` names, or `None` when it names none.
    #[must_use]
    pub fn value_at(&self, coordinate: &Coordinate) -> Option<Value> {
        todo!()
    }

    /// Return the coordinate of `value`, or `None` when the domain does not
    /// admit it.
    #[must_use]
    pub fn coordinate_of(&self, value: &Value) -> Option<Coordinate> {
        todo!()
    }

    /// Return whether the domain admits `value`.
    #[must_use]
    pub fn admits(&self, value: &Value) -> bool {
        todo!()
    }

    /// Return the domain's signature, every identifier written as
    /// `identifier`, as a dynamic step's is.
    #[must_use]
    pub fn signature(&self) -> DomainSignature {
        todo!()
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
///   any other identifier, and as `opaque` for an opaque value.
/// - A strided domain's runs.
///
/// `serde` writes `{"choice": [..]}`, `{"order": [..]}` or `{"strided":
/// [{"start", "stop", "stride"}, ..]}`, a member as `{"value": <value>}`,
/// `{"bound": <position>}`, `"identifier"` or `"opaque"`; reading refuses
/// a run that [`StridedRun::new`] or [`StridedDomain::new`] refuses.
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
        todo!()
    }

    /// Return the shape: `choice`, `order` or `strided`.
    #[must_use]
    pub fn shape(&self) -> &'static str {
        todo!()
    }

    /// Return whether `coordinate` names a value of the domain it was taken
    /// from, as [`StepDomain::contains`] says.
    #[must_use]
    pub fn contains(&self, coordinate: &Coordinate) -> bool {
        todo!()
    }
}

/// Serializes the shape of the type's documentation.
impl Serialize for DomainSignature {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        todo!()
    }
}

/// Deserializes the shape of the type's documentation.
impl<'de> Deserialize<'de> for DomainSignature {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        todo!()
    }
}
