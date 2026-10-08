//! [`Trace`]: the steps of one run of a search, recorded in the order they
//! were asked, and [`TraceStep`], one of them.

use std::fmt;
use std::sync::Arc;

use num_bigint::BigUint;
use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::constraint::Value;
use crate::identifier::Identifier;

use super::domain::{Coordinate, DecisionKind, DomainSignature, StepDomain};
use super::error::TraceError;

/// One step of a run: what was asked, the signature of the domain it was
/// asked over, and the answer.
///
/// A **static** step asks a decision of a [`Space`](super::Space): it
/// records the decision's canonical position in its space and its name as
/// the subject. A **dynamic** step asks about a subject the space does not
/// declare, over a domain its caller built.
///
/// `==` and `Hash` compare every field.
///
/// `serde` writes `{"kind", "subject", "decision", "domain", "coordinate",
/// "value"}`, `decision` and `value` `null` when absent, a value that is
/// or holds an opaque value `null`; reading refuses a coordinate its
/// signature does not contain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TraceStep {
    kind: DecisionKind,
    subject: Identifier,
    decision: Option<u32>,
    signature: DomainSignature,
    coordinate: Coordinate,
    value: Option<Value>,
}

impl TraceStep {
    /// Return the dynamic step of `kind` about `subject` over `domain`,
    /// answered with `coordinate`; its value is the one `coordinate` names.
    ///
    /// # Errors
    ///
    /// Returns [`TraceError::CoordinateOutOfDomain`], with position 0, when
    /// `coordinate` names no value of `domain`.
    pub fn dynamic(
        kind: DecisionKind,
        subject: Identifier,
        domain: &StepDomain,
        coordinate: Coordinate,
    ) -> Result<Self, TraceError> {
        let value = domain
            .value_at(&coordinate)
            .ok_or(TraceError::CoordinateOutOfDomain { position: 0 })?;
        Ok(Self {
            kind,
            subject,
            decision: None,
            signature: domain.signature(),
            coordinate,
            value: Some(value),
        })
    }

    /// Return the static step of `kind` asking the decision `subject` at
    /// canonical `decision` over a domain of signature `signature`,
    /// answered with `coordinate`, which names `value`.
    pub(super) fn of_decision(
        kind: DecisionKind,
        subject: Identifier,
        decision: usize,
        signature: DomainSignature,
        coordinate: Coordinate,
        value: Value,
    ) -> Self {
        Self {
            kind,
            subject,
            decision: Some(u32::try_from(decision).unwrap_or(u32::MAX)),
            signature,
            coordinate,
            value: Some(value),
        }
    }

    /// Return what the step is about.
    #[must_use]
    pub fn kind(&self) -> &DecisionKind {
        &self.kind
    }

    /// Return the step's subject: a static step's decision name.
    #[must_use]
    pub fn subject(&self) -> &Identifier {
        &self.subject
    }

    /// Return a static step's decision, by its canonical position in its
    /// space, or `None` for a dynamic step.
    #[must_use]
    pub fn decision(&self) -> Option<usize> {
        self.decision
            .and_then(|decision| usize::try_from(decision).ok())
    }

    /// Return the signature of the domain the step was asked over.
    #[must_use]
    pub fn signature(&self) -> &DomainSignature {
        &self.signature
    }

    /// Return the number of values the step's domain held.
    #[must_use]
    pub fn cardinality(&self) -> BigUint {
        self.signature.cardinality()
    }

    /// Return the answer.
    #[must_use]
    pub fn coordinate(&self) -> &Coordinate {
        &self.coordinate
    }

    /// Return the value answered, or `None` for one a trace read back does
    /// not hold (an opaque value is not written).
    #[must_use]
    pub fn value(&self) -> Option<&Value> {
        self.value.as_ref()
    }
}

/// The steps of one run, in the order they were asked: a point of a search
/// space and the path that reached it.
///
/// Cloning shares the steps. `==` and `Hash` compare the steps; two traces
/// recorded over different modules differ in their subjects, so compare
/// their [`coordinates`](Self::coordinates) to ask whether they took the
/// same answers.
///
/// Displays as `0 steps`, or as the count and the count of each kind, in
/// the order the kinds first occur: `5 steps (2 search_space.choice, 3
/// moga.cir.address)`.
///
/// `serde` writes `{"steps": [..]}`, each step as `{"kind", "subject",
/// "decision", "domain", "coordinate", "value"}`, `decision` and `value`
/// `null` when absent, and a value that is or holds an opaque value as
/// `null`. Reading refuses a coordinate its signature does not contain.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Trace(Arc<[TraceStep]>);

impl Trace {
    /// Return the trace of `steps`, in the order given.
    #[must_use]
    pub fn new(steps: Vec<TraceStep>) -> Self {
        Self(steps.into())
    }

    /// Return the steps, in the order asked.
    #[must_use]
    pub fn steps(&self) -> &[TraceStep] {
        &self.0
    }

    /// Return the number of steps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return whether the trace has no step.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Return the steps' coordinates, in the order asked: the point as a
    /// bare vector.
    pub fn coordinates(&self) -> impl ExactSizeIterator<Item = &Coordinate> + '_ {
        self.steps().iter().map(TraceStep::coordinate)
    }

    /// Return the steps of `kind`, in the order asked.
    pub fn of_kind(&self, kind: &DecisionKind) -> impl Iterator<Item = &TraceStep> + use<'_> {
        let kind = kind.clone();
        self.steps().iter().filter(move |step| *step.kind() == kind)
    }

    /// Return the product of the steps' cardinalities: the number of points
    /// sharing this path's shape. One for an empty trace.
    #[must_use]
    pub fn traversed_cardinality(&self) -> BigUint {
        self.0.iter().map(TraceStep::cardinality).product()
    }
}

impl fmt::Display for Trace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut counts: Vec<(&DecisionKind, usize)> = Vec::new();
        for step in self.0.iter() {
            match counts.iter_mut().find(|(kind, _)| *kind == step.kind()) {
                Some((_, count)) => *count += 1,
                None => counts.push((step.kind(), 1)),
            }
        }
        write!(f, "{} steps", self.0.len())?;
        if counts.is_empty() {
            return Ok(());
        }
        f.write_str(" (")?;
        for (position, (kind, count)) in counts.iter().enumerate() {
            if position > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{count} {kind}")?;
        }
        f.write_str(")")
    }
}

/// The wire form of a [`TraceStep`], written.
#[derive(Serialize)]
#[serde(rename = "TraceStep")]
struct StepRef<'a> {
    kind: &'a DecisionKind,
    subject: &'a Identifier,
    decision: Option<u32>,
    domain: &'a DomainSignature,
    coordinate: &'a Coordinate,
    value: Option<&'a Value>,
}

/// The wire form of a [`TraceStep`], read.
#[derive(Deserialize)]
#[serde(rename = "TraceStep", deny_unknown_fields)]
struct StepWire {
    kind: DecisionKind,
    subject: Identifier,
    decision: Option<u32>,
    domain: DomainSignature,
    coordinate: Coordinate,
    value: Option<Value>,
}

/// Return whether `value` is, or holds, an opaque value.
fn holds_opaque(value: &Value) -> bool {
    match value {
        Value::Opaque(_) => true,
        Value::Tuple(values) | Value::FrozenSet(values) => values.iter().any(holds_opaque),
        _ => false,
    }
}

/// Serializes the shape of the type's documentation.
impl Serialize for TraceStep {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        StepRef {
            kind: &self.kind,
            subject: &self.subject,
            decision: self.decision,
            domain: &self.signature,
            coordinate: &self.coordinate,
            value: self.value.as_ref().filter(|value| !holds_opaque(value)),
        }
        .serialize(serializer)
    }
}

/// Deserializes the shape of the type's documentation.
impl<'de> Deserialize<'de> for TraceStep {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = StepWire::deserialize(deserializer)?;
        if !wire.domain.contains(&wire.coordinate) {
            return Err(de::Error::custom(
                "a step's coordinate names no value of its domain",
            ));
        }
        Ok(Self {
            kind: wire.kind,
            subject: wire.subject,
            decision: wire.decision,
            signature: wire.domain,
            coordinate: wire.coordinate,
            value: wire.value,
        })
    }
}

/// The wire form of a [`Trace`], written.
#[derive(Serialize)]
#[serde(rename = "Trace")]
struct TraceRef<'a> {
    steps: &'a [TraceStep],
}

/// The wire form of a [`Trace`], read.
#[derive(Deserialize)]
#[serde(rename = "Trace", deny_unknown_fields)]
struct TraceWire {
    steps: Vec<TraceStep>,
}

/// Serializes the shape of the type's documentation.
impl Serialize for Trace {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        TraceRef { steps: &self.0 }.serialize(serializer)
    }
}

/// Deserializes the shape of the type's documentation.
impl<'de> Deserialize<'de> for Trace {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::new(TraceWire::deserialize(deserializer)?.steps))
    }
}
