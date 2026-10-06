//! What a search measures and which way is better: [`Direction`] and
//! [`Objective`]; the record of one measured configuration,
//! [`Measurement`]; and what measures, [`Measurer`].

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::diagnostic::Note;
use crate::foreign::BoxError;

use super::configuration::ConfigurationKey;
use super::error::MeasurementError;

/// Which way an objective's values are better.
///
/// `serde` writes `"minimize"`, `"maximize"` or `"report"`, as `Display`
/// does.
#[expect(
    clippy::exhaustive_enums,
    reason = "a value is better lower, better higher, or never compared"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    /// Lower is better: a cost.
    Minimize,
    /// Higher is better: a benefit.
    Maximize,
    /// Recorded, never compared: a diagnostic.
    Report,
}

impl Direction {
    /// Return the direction's name: `minimize`, `maximize` or `report`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Minimize => "minimize",
            Self::Maximize => "maximize",
            Self::Report => "report",
        }
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A named quantity a search measures, and which way it is better.
///
/// Its name is a string, stable across processes, so measurements made in
/// different processes name the same objectives alike. Two objectives are
/// equal when their names and directions are. Displays as
/// `<name> (<direction>)`; `serde` writes `{"name", "direction"}` and
/// refuses an empty name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Objective {
    name: Arc<str>,
    direction: Direction,
}

impl Objective {
    /// Return the objective `name`, better in `direction`.
    ///
    /// # Errors
    ///
    /// Returns [`MeasurementError::EmptyName`] for an empty name.
    pub fn new(name: &str, direction: Direction) -> Result<Self, MeasurementError> {
        todo!()
    }

    /// Return the objective's name.
    #[must_use]
    pub fn name(&self) -> &str {
        todo!()
    }

    /// Return which way the objective's values are better.
    #[must_use]
    pub fn direction(&self) -> Direction {
        todo!()
    }

    /// Compare the values `left` and `right` of the objective:
    /// [`Greater`](Ordering::Greater) when `left` is better,
    /// [`Less`](Ordering::Less) when it is worse, [`Equal`](Ordering::Equal)
    /// when neither is, and `None` for a [`Report`](Direction::Report)
    /// objective, which is never compared.
    ///
    /// A NaN is worse than every number, in either direction, and equal to
    /// another NaN, so it never wins a comparison and any number beats it.
    #[must_use]
    pub fn compare(&self, left: f64, right: f64) -> Option<Ordering> {
        todo!()
    }
}

impl fmt::Display for Objective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

/// Serializes `{"name", "direction"}`.
impl Serialize for Objective {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        todo!()
    }
}

/// Deserializes `{"name", "direction"}`, refusing an empty name.
impl<'de> Deserialize<'de> for Objective {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        todo!()
    }
}

/// How a measurement of a configuration went.
///
/// `serde` writes `"ok"`, `{"infeasible": {"reason"}}`, `{"failed":
/// {"reason"}}` or `"timeout"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MeasurementStatus {
    /// The configuration was measured: the measurement holds a value per
    /// objective.
    Ok,
    /// The configuration cannot be realized: data about the space, such as
    /// a lowering pipeline's rejection, with the stage in the reason.
    Infeasible {
        /// Why.
        reason: String,
    },
    /// The measurement was attempted and broke, such as a crashed
    /// simulator: a fault of the measuring, not of the configuration.
    Failed {
        /// Why.
        reason: String,
    },
    /// The measurement ran out of time.
    Timeout,
}

/// The record of one measured configuration: its key, its status, a
/// finite value per objective when the status is
/// [`Ok`](MeasurementStatus::Ok) and none otherwise, and notes.
///
/// A value of `-0.0` is kept as `0.0`. `==` and `Hash` compare the key,
/// the status, the values in order and the notes. `serde` writes `{"key",
/// "status", "values": [{"objective", "value"}, ..], "notes"}`, the key in
/// [`ConfigurationKeyData`](super::wire::ConfigurationKeyData)'s shape,
/// and reads it through the constructors' checks;
/// [`MeasurementData`](super::wire::MeasurementData) reads a key holding
/// another crate's opaque values.
#[derive(Debug, Clone)]
pub struct Measurement(Arc<MeasurementInner>);

/// The fields of a [`Measurement`].
#[derive(Debug)]
struct MeasurementInner {
    key: ConfigurationKey,
    status: MeasurementStatus,
    values: Vec<(Objective, f64)>,
    notes: Vec<Note>,
}

impl Measurement {
    /// Return the successful measurement of the configuration `key`, a value
    /// per objective in the order given.
    ///
    /// # Errors
    ///
    /// In order: [`MeasurementError::NoValues`] for no value;
    /// [`MeasurementError::RepeatedObjective`] for two values whose
    /// objectives share a name; [`MeasurementError::NonFiniteValue`] for a
    /// NaN or an infinity.
    pub fn ok(
        key: ConfigurationKey,
        values: Vec<(Objective, f64)>,
    ) -> Result<Self, MeasurementError> {
        todo!()
    }

    /// Return the measurement of a configuration `key` that cannot be
    /// realized, for `reason`.
    #[must_use]
    pub fn infeasible(key: ConfigurationKey, reason: impl Into<String>) -> Self {
        todo!()
    }

    /// Return the measurement of the configuration `key` that broke, for
    /// `reason`.
    #[must_use]
    pub fn failed(key: ConfigurationKey, reason: impl Into<String>) -> Self {
        todo!()
    }

    /// Return the measurement of the configuration `key` that ran out of
    /// time.
    #[must_use]
    pub fn timeout(key: ConfigurationKey) -> Self {
        todo!()
    }

    /// Return the measurement with `notes` in place of its own.
    #[must_use]
    pub fn with_notes(self, notes: Vec<Note>) -> Self {
        todo!()
    }

    /// Return the key of the configuration measured.
    #[must_use]
    pub fn key(&self) -> &ConfigurationKey {
        todo!()
    }

    /// Return how the measurement went.
    #[must_use]
    pub fn status(&self) -> &MeasurementStatus {
        todo!()
    }

    /// Return whether the measurement succeeded.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        todo!()
    }

    /// Return the values, one per objective, in the order given; none for
    /// a measurement that did not succeed.
    #[must_use]
    pub fn values(&self) -> &[(Objective, f64)] {
        todo!()
    }

    /// Return the value of the objective named `objective`, if the
    /// measurement holds one.
    #[must_use]
    pub fn value(&self, objective: &str) -> Option<f64> {
        todo!()
    }

    /// Return the notes.
    #[must_use]
    pub fn notes(&self) -> &[Note] {
        todo!()
    }

    /// Return whether this measurement dominates `other`: at least as good
    /// on every compared objective and better on one, the
    /// [`Report`](Direction::Report) objectives not compared.
    ///
    /// # Errors
    ///
    /// Returns [`MeasurementError::NotOk`] when either measurement did not
    /// succeed, and [`MeasurementError::DifferentObjectives`] when their
    /// objectives differ, by name or direction, in any order.
    pub fn dominates(&self, other: &Self) -> Result<bool, MeasurementError> {
        todo!()
    }
}

impl PartialEq for Measurement {
    fn eq(&self, other: &Self) -> bool {
        todo!()
    }
}

impl Eq for Measurement {}

impl Hash for Measurement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        todo!()
    }
}

/// Measures subjects of type `S`, such as a lowered program, as one
/// configuration of a space.
///
/// A subject that cannot be measured is an `Ok` measurement with a failing
/// status ([`Infeasible`](MeasurementStatus::Infeasible),
/// [`Failed`](MeasurementStatus::Failed),
/// [`Timeout`](MeasurementStatus::Timeout)); an `Err` is a fault of the
/// measurer itself, which stops the search. A successful measurement holds
/// a value for each of [`objectives`](Self::objectives), and only those.
pub trait Measurer<S: ?Sized> {
    /// Return the objectives every successful measurement holds a value
    /// for.
    fn objectives(&self) -> &[Objective];

    /// Measure `subject`, a realization of the configuration `key`.
    ///
    /// # Errors
    ///
    /// Returns the measurer's own fault.
    fn measure(&mut self, key: &ConfigurationKey, subject: &S) -> Result<Measurement, BoxError>;
}
