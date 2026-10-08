//! What a search measures and which way is better: [`Direction`] and
//! [`Objective`]; what a measurement is of, [`MeasurementKey`]; the record
//! of one measured point, [`Measurement`]; what measures, [`Measurer`]; and
//! the measurements no other one dominates, [`non_dominated`].

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::diagnostic::Note;
use crate::foreign::BoxError;

use super::configuration::ConfigurationKey;
use super::error::MeasurementError;
use super::trace::TraceKey;

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
        if name.is_empty() {
            return Err(MeasurementError::EmptyName);
        }
        Ok(Self {
            name: Arc::from(name),
            direction,
        })
    }

    /// Return the objective's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return which way the objective's values are better.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.direction
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
        if self.direction == Direction::Report {
            return None;
        }
        let ordering = match (left.is_nan(), right.is_nan()) {
            (true, true) => return Some(Ordering::Equal),
            (true, false) => return Some(Ordering::Less),
            (false, true) => return Some(Ordering::Greater),
            (false, false) => left.partial_cmp(&right)?,
        };
        Some(match self.direction {
            Direction::Minimize => ordering.reverse(),
            Direction::Maximize | Direction::Report => ordering,
        })
    }
}

impl fmt::Display for Objective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name, self.direction)
    }
}

/// The wire form of an [`Objective`].
#[derive(Serialize, Deserialize)]
#[serde(rename = "Objective", deny_unknown_fields)]
struct ObjectiveWire {
    name: String,
    direction: Direction,
}

/// Serializes `{"name", "direction"}`.
impl Serialize for Objective {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ObjectiveWire {
            name: self.name.to_string(),
            direction: self.direction,
        }
        .serialize(serializer)
    }
}

/// Deserializes `{"name", "direction"}`, refusing an empty name.
impl<'de> Deserialize<'de> for Objective {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = ObjectiveWire::deserialize(deserializer)?;
        Self::new(&wire.name, wire.direction).map_err(de::Error::custom)
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

/// What a [`Measurement`] is of: a configuration of a space, by its
/// [`ConfigurationKey`], or one run of a stream, static and dynamic steps
/// alike, by its [`TraceKey`].
///
/// Key a measurement by its configuration when the realization depends on
/// the configuration alone, and by its trace when it also depends on a
/// run's dynamic steps, such as where an allocator placed a buffer.
///
/// `==` and `Hash` compare the variant and its key, and a key compares
/// equal to the [`ConfigurationKey`] or [`TraceKey`] it holds. `serde`
/// writes `{"configuration": <key>}` or `{"trace": <key>}`, a
/// configuration key refusing an opaque value as
/// [`ConfigurationKey`]'s own decoding does.
#[expect(
    clippy::exhaustive_enums,
    reason = "a measurement is of a configuration or of a run, and callers match both"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementKey {
    /// A configuration of a space.
    Configuration(ConfigurationKey),
    /// One run of a stream.
    Trace(TraceKey),
}

impl From<ConfigurationKey> for MeasurementKey {
    #[expect(
        unused_variables,
        clippy::todo,
        reason = "interface stub; bodies are todo!() until implementation"
    )]
    fn from(key: ConfigurationKey) -> Self {
        todo!()
    }
}

impl From<TraceKey> for MeasurementKey {
    #[expect(
        unused_variables,
        clippy::todo,
        reason = "interface stub; bodies are todo!() until implementation"
    )]
    fn from(key: TraceKey) -> Self {
        todo!()
    }
}

impl PartialEq<ConfigurationKey> for MeasurementKey {
    /// Return whether this is a configuration's key equal to `other`.
    #[expect(
        unused_variables,
        clippy::todo,
        reason = "interface stub; bodies are todo!() until implementation"
    )]
    fn eq(&self, other: &ConfigurationKey) -> bool {
        todo!()
    }
}

impl PartialEq<TraceKey> for MeasurementKey {
    /// Return whether this is a trace's key equal to `other`.
    #[expect(
        unused_variables,
        clippy::todo,
        reason = "interface stub; bodies are todo!() until implementation"
    )]
    fn eq(&self, other: &TraceKey) -> bool {
        todo!()
    }
}

/// The record of one measured point: its [`MeasurementKey`], its status, a
/// finite value per objective when the status is
/// [`Ok`](MeasurementStatus::Ok) and none otherwise, and notes.
///
/// A value of `-0.0` is kept as `0.0`. `==` and `Hash` compare the key,
/// the status, the values in order and the notes. `serde` writes `{"key",
/// "status", "values": [{"objective", "value"}, ..], "notes"}`, the key as
/// `{"configuration": ..}`, in
/// [`ConfigurationKeyData`](super::wire::ConfigurationKeyData)'s shape, or
/// as `{"trace": ..}`, in [`TraceKey`]'s, and reads it through the
/// constructors' checks; [`MeasurementData`](super::wire::MeasurementData)
/// reads a configuration key holding another crate's opaque values.
#[derive(Debug, Clone)]
pub struct Measurement(Arc<MeasurementInner>);

/// The fields of a [`Measurement`].
#[derive(Debug, Clone)]
struct MeasurementInner {
    key: MeasurementKey,
    status: MeasurementStatus,
    values: Vec<(Objective, f64)>,
    notes: Vec<Note>,
}

impl Measurement {
    /// Return the successful measurement of `key`, a configuration's or a
    /// trace's, a value per objective in the order given.
    ///
    /// # Errors
    ///
    /// In order: [`MeasurementError::NoValues`] for no value;
    /// [`MeasurementError::RepeatedObjective`] for two values whose
    /// objectives share a name; [`MeasurementError::NonFiniteValue`] for a
    /// NaN or an infinity.
    pub fn ok(
        key: impl Into<MeasurementKey>,
        values: Vec<(Objective, f64)>,
    ) -> Result<Self, MeasurementError> {
        if values.is_empty() {
            return Err(MeasurementError::NoValues);
        }
        let mut names = HashSet::with_capacity(values.len());
        if let Some((repeated, _)) = values
            .iter()
            .find(|(objective, _)| !names.insert(objective.name()))
        {
            return Err(MeasurementError::RepeatedObjective {
                name: repeated.name().to_owned(),
            });
        }
        if let Some((objective, _)) = values.iter().find(|(_, value)| !value.is_finite()) {
            return Err(MeasurementError::NonFiniteValue {
                objective: objective.name().to_owned(),
            });
        }
        let values = values
            .into_iter()
            .map(|(objective, value)| (objective, without_negative_zero(value)))
            .collect();
        Ok(Self::of(key.into(), MeasurementStatus::Ok, values))
    }

    /// Return the measurement of `key` with `status` and `values`, and no
    /// notes.
    fn of(key: MeasurementKey, status: MeasurementStatus, values: Vec<(Objective, f64)>) -> Self {
        Self(Arc::new(MeasurementInner {
            key,
            status,
            values,
            notes: Vec::new(),
        }))
    }

    /// Return the measurement of `key` that cannot be realized, for
    /// `reason`.
    #[must_use]
    pub fn infeasible(key: impl Into<MeasurementKey>, reason: impl Into<String>) -> Self {
        Self::of(
            key.into(),
            MeasurementStatus::Infeasible {
                reason: reason.into(),
            },
            Vec::new(),
        )
    }

    /// Return the measurement of `key` that broke, for `reason`.
    #[must_use]
    pub fn failed(key: impl Into<MeasurementKey>, reason: impl Into<String>) -> Self {
        Self::of(
            key.into(),
            MeasurementStatus::Failed {
                reason: reason.into(),
            },
            Vec::new(),
        )
    }

    /// Return the measurement of `key` that ran out of time.
    #[must_use]
    pub fn timeout(key: impl Into<MeasurementKey>) -> Self {
        Self::of(key.into(), MeasurementStatus::Timeout, Vec::new())
    }

    /// Return the measurement with `notes` in place of its own.
    #[must_use]
    pub fn with_notes(self, notes: Vec<Note>) -> Self {
        let mut inner = Arc::unwrap_or_clone(self.0);
        inner.notes = notes;
        Self(Arc::new(inner))
    }

    /// Return the key of what was measured.
    #[must_use]
    pub fn key(&self) -> &MeasurementKey {
        &self.0.key
    }

    /// Return how the measurement went.
    #[must_use]
    pub fn status(&self) -> &MeasurementStatus {
        &self.0.status
    }

    /// Return whether the measurement succeeded.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.0.status == MeasurementStatus::Ok
    }

    /// Return the values, one per objective, in the order given; none for
    /// a measurement that did not succeed.
    #[must_use]
    pub fn values(&self) -> &[(Objective, f64)] {
        &self.0.values
    }

    /// Return the value of the objective named `objective`, if the
    /// measurement holds one.
    #[must_use]
    pub fn value(&self, objective: &str) -> Option<f64> {
        self.0
            .values
            .iter()
            .find(|(held, _)| held.name() == objective)
            .map(|(_, value)| *value)
    }

    /// Return the notes.
    #[must_use]
    pub fn notes(&self) -> &[Note] {
        &self.0.notes
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
        if !self.is_ok() || !other.is_ok() {
            return Err(MeasurementError::NotOk);
        }
        if self.0.values.len() != other.0.values.len() {
            return Err(MeasurementError::DifferentObjectives);
        }
        let mut is_better_somewhere = false;
        for (objective, value) in &self.0.values {
            let theirs = other
                .0
                .values
                .iter()
                .find(|(held, _)| held == objective)
                .map(|(_, theirs)| *theirs)
                .ok_or(MeasurementError::DifferentObjectives)?;
            match objective.compare(*value, theirs) {
                Some(Ordering::Less) => return Ok(false),
                Some(Ordering::Greater) => is_better_somewhere = true,
                Some(Ordering::Equal) | None => {}
            }
        }
        Ok(is_better_somewhere)
    }
}

/// Return `value`, `-0.0` made `0.0`.
fn without_negative_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

impl PartialEq for Measurement {
    fn eq(&self, other: &Self) -> bool {
        let (left, right) = (&self.0, &other.0);
        left.key == right.key
            && left.status == right.status
            && left.notes == right.notes
            && left.values.len() == right.values.len()
            && left.values.iter().zip(&right.values).all(
                |((left_objective, left_value), (right_objective, right_value))| {
                    left_objective == right_objective
                        && left_value.to_bits() == right_value.to_bits()
                },
            )
    }
}

/// A measurement's values are finite and never `-0.0`, so comparing their
/// bits is `==` on them.
impl Eq for Measurement {}

impl Hash for Measurement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let inner = &self.0;
        inner.key.hash(state);
        inner.status.hash(state);
        inner.values.len().hash(state);
        for (objective, value) in &inner.values {
            objective.hash(state);
            value.to_bits().hash(state);
        }
        inner.notes.hash(state);
    }
}

/// Measures subjects of type `S`, such as a lowered program, as one
/// configuration of a space or one run of a stream.
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

    /// Measure `subject`, a realization of the configuration or the run
    /// `key` names.
    ///
    /// # Errors
    ///
    /// Returns the measurer's own fault.
    fn measure(&mut self, key: &MeasurementKey, subject: &S) -> Result<Measurement, BoxError>;
}

/// Return the successful measurements of `measurements` that no other
/// successful one [dominates](Measurement::dominates), in the order given:
/// the Pareto front.
///
/// A measurement that did not succeed is left out. Neither of two
/// measurements with equal values dominates the other, so both are kept
/// unless a third dominates them.
///
/// # Errors
///
/// Returns [`MeasurementError::DifferentObjectives`] when two successful
/// measurements are over different objectives.
#[expect(
    unused_variables,
    clippy::todo,
    reason = "interface stub; bodies are todo!() until implementation"
)]
pub fn non_dominated(measurements: &[Measurement]) -> Result<Vec<&Measurement>, MeasurementError> {
    todo!()
}
