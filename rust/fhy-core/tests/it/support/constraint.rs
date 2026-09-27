//! Helpers of the constraint tests: members and values, a test-local
//! opaque value, and an observer that records its events.

use std::any::Any;
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use fhy_core::constraint::{
    Bindings, Constraint, CustomConstraint, Event, Member, MemberSet, Observer, Opaque,
    OpaqueValue, Outcome, Value,
};
use fhy_core::expression::{BigInt, Expression};
use fhy_core::foreign::BoxError;
use fhy_core::identifier::Identifier;
use fhy_core::solver::QueryKind;
use fhy_core::term::AlphaRenaming;

/// Return the integer value `value`.
pub(crate) fn int(value: i64) -> Value {
    Value::Int(BigInt::from(value))
}

/// Return the string value `text`.
pub(crate) fn text(text: &str) -> Value {
    Value::Str(text.to_owned())
}

/// Return the member of `value`.
///
/// # Panics
///
/// Panics if `value` cannot be a member.
pub(crate) fn member(value: Value) -> Member {
    Member::try_from_value(value).expect("the value is a member")
}

/// Return the set of the members of `values`.
///
/// # Panics
///
/// Panics if a value cannot be a member.
pub(crate) fn member_set(values: impl IntoIterator<Item = Value>) -> MemberSet {
    values.into_iter().map(member).collect()
}

/// Return the set of the integers `values`.
pub(crate) fn int_set(values: impl IntoIterator<Item = i64>) -> MemberSet {
    member_set(values.into_iter().map(int))
}

/// Return the bindings of `entries`.
pub(crate) fn bind<B: Into<fhy_core::constraint::Binding>>(
    entries: impl IntoIterator<Item = (Identifier, B)>,
) -> Bindings {
    entries.into_iter().collect()
}

/// An error a test value reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TestValueError(pub(crate) String);

impl fmt::Display for TestValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TestValueError {}

/// An opaque value of a test type: equal to another of the same type
/// name and payload, keyed by its `key`, which need not tell unequal
/// values apart.
#[derive(Debug, Clone)]
pub(crate) struct TestOpaque {
    pub(crate) type_name: &'static str,
    pub(crate) payload: i64,
    pub(crate) key: String,
    pub(crate) is_member_shaped: bool,
    pub(crate) is_hashable: bool,
}

impl TestOpaque {
    /// Return a member-shaped, hashable value of type `Token`, keyed by its
    /// payload.
    pub(crate) fn token(payload: i64) -> Self {
        Self {
            type_name: "Token",
            payload,
            key: format!("Token:{payload}"),
            is_member_shaped: true,
            is_hashable: true,
        }
    }

    /// Return a member-shaped, hashable value whose key is the same for
    /// every payload.
    pub(crate) fn colliding(payload: i64) -> Self {
        Self {
            key: "Colliding".to_owned(),
            type_name: "Colliding",
            ..Self::token(payload)
        }
    }

    /// Return this value as an [`Opaque`] value.
    pub(crate) fn into_value(self) -> Value {
        Value::Opaque(Opaque::new(self))
    }
}

impl OpaqueValue for TestOpaque {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.type_name)
    }

    fn is_member_shaped(&self) -> bool {
        self.is_member_shaped
    }

    fn is_equal(&self, other: &dyn OpaqueValue) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.type_name == self.type_name && other.payload == self.payload)
    }

    fn check_hashable(&self) -> Result<(), BoxError> {
        if self.is_hashable {
            Ok(())
        } else {
            Err(Box::new(TestValueError(format!(
                "{} is unhashable",
                self.type_name
            ))))
        }
    }

    fn ordering_key(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.key)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// An owned copy of an [`Event`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RecordedEvent {
    Unbound(Identifier),
    SymbolicBinding(Identifier, Expression),
    BoundNativeConstants(Vec<Identifier>),
    Residual(Expression, bool),
    InMember(usize, Box<RecordedEvent>),
    UndecidedMember(usize),
    Refused(QueryKind),
    GaveUp(QueryKind, String),
}

impl RecordedEvent {
    /// Return the owned copy of `event`.
    fn of(event: &Event<'_>) -> Self {
        match *event {
            Event::Unbound { variable } => Self::Unbound(variable.clone()),
            Event::SymbolicBinding { variable, binding } => {
                Self::SymbolicBinding(variable.clone(), binding.clone())
            }
            Event::BoundNativeConstants { identifiers } => {
                Self::BoundNativeConstants(identifiers.to_vec())
            }
            Event::Residual {
                residual,
                has_free_identifiers,
            } => Self::Residual(residual.clone(), has_free_identifiers),
            Event::InMember { index, event } => Self::InMember(index, Box::new(Self::of(event))),
            Event::UndecidedMember { index } => Self::UndecidedMember(index),
            Event::Refused { kind, .. } => Self::Refused(kind),
            Event::GaveUp { kind, reason } => Self::GaveUp(kind, reason.to_owned()),
            _ => unreachable!("the constraint tests know every event"),
        }
    }
}

/// An [`Observer`] that records every event.
#[derive(Debug, Default)]
pub(crate) struct RecordingObserver {
    events: Mutex<Vec<RecordedEvent>>,
}

impl RecordingObserver {
    /// Return the events so far, in order.
    pub(crate) fn events(&self) -> Vec<RecordedEvent> {
        self.events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl Observer for RecordingObserver {
    fn notify(&self, event: &Event<'_>) {
        self.events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(RecordedEvent::of(event));
    }
}

/// A custom constraint of a test kind: keyed by its label, converting to
/// its expression, and answering its outcome while recording its label in
/// a shared log.
#[derive(Debug)]
pub(crate) struct TestCustom {
    pub(crate) label: String,
    pub(crate) expression: Expression,
    pub(crate) outcome: Outcome,
    pub(crate) log: Arc<Mutex<Vec<String>>>,
}

impl TestCustom {
    /// Return the custom constraint labelled `label`, answering `outcome`
    /// and converting to `expression`, logging to `log`.
    pub(crate) fn build(
        label: &str,
        expression: Expression,
        outcome: Outcome,
        log: &Arc<Mutex<Vec<String>>>,
    ) -> Constraint {
        Constraint::Custom(Arc::new(Self {
            label: label.to_owned(),
            expression,
            outcome,
            log: Arc::clone(log),
        }))
    }
}

impl CustomConstraint for TestCustom {
    fn free_identifiers(&self) -> HashSet<Identifier> {
        self.expression.free_identifiers()
    }

    fn evaluate(&self, bindings: &Bindings) -> Result<Outcome, BoxError> {
        let source = bindings
            .source()
            .and_then(|source| source.downcast_ref::<String>())
            .map_or("none", String::as_str);
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(format!("{}:{source}", self.label));
        Ok(self.outcome)
    }

    fn to_expression(&self) -> Result<Expression, BoxError> {
        Ok(self.expression.clone())
    }

    fn ordering_key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("custom|{}", self.label))
    }

    fn is_structurally_equivalent(&self, other: &dyn CustomConstraint) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.label == self.label)
    }

    fn is_alpha_equivalent_under(
        &self,
        other: &dyn CustomConstraint,
        _renaming: &AlphaRenaming,
    ) -> bool {
        self.is_structurally_equivalent(other)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
