//! [`Recorder`]: drives one run of a search, asking its oracle each step,
//! checking and recording the answers, and, over a space, growing the
//! run's configuration; and [`Recorded`], what a finished run gives.

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use crate::constraint::Value;
use crate::identifier::Identifier;
use crate::param::ParamContext;

use super::configuration::Configuration;
use super::domain::{Coordinate, DecisionKind, StepDomain};
use super::error::TraceError;
use super::oracle::SearchOracle;
use super::space::Space;
use super::trace::{Trace, TraceStep};

/// One run of a search: every step asked through it is put to the oracle
/// given with it, checked and recorded.
///
/// - [`new`](Self::new) records dynamic steps only.
/// - [`over`](Self::over) also asks the decisions of a space, starting
///   from its empty configuration, and grows the configuration with each
///   answer.
/// - [`realizing`](Self::realizing) asks the decisions of a
///   configuration's space, answering those the configuration assigns from
///   it, without asking the oracle, and every other step from the oracle.
///
/// The recorder holds only the run's state: each step takes the oracle
/// that answers it and the context its values are checked under, so a run
/// may span calls that each borrow them anew. A step whose answer is
/// refused stops the run and is not recorded; [`trace`](Self::trace) still
/// gives the steps recorded before it. Cloning copies the run's state.
#[derive(Debug, Clone, Default)]
pub struct Recorder {
    space: Option<Space>,
    configuration: Option<Configuration>,
    preset: Option<Configuration>,
    steps: Vec<TraceStep>,
}

impl Recorder {
    /// Return the recorder of a run of dynamic steps.
    #[must_use]
    pub fn new() -> Self {
        todo!()
    }

    /// Return the recorder of a run over `space`, from its empty
    /// configuration.
    #[must_use]
    pub fn over(space: &Space) -> Self {
        todo!()
    }

    /// Return the recorder of a run realizing `configuration`: a decision
    /// it assigns is answered with its value, everything else by the
    /// oracle.
    #[must_use]
    pub fn realizing(configuration: &Configuration) -> Self {
        todo!()
    }

    /// Ask the decision `decision` of the space of `oracle`, checked under
    /// `context`, and return its value: a variable's value, or the
    /// identifier value of the chosen alternative's name.
    ///
    /// # Errors
    ///
    /// In order: [`TraceError::NoSpace`] for a recorder over no space;
    /// [`TraceError::UnknownDecision`]; [`TraceError::AlreadyDecided`];
    /// [`TraceError::NotActive`] for a decision that is inactive or
    /// pending; [`TraceError::NotEnumerable`] or [`TraceError::Hook`] when
    /// its domain cannot be built; [`TraceError::Oracle`];
    /// [`TraceError::CoordinateOutOfDomain`]; [`TraceError::Inadmissible`];
    /// and [`TraceError::Configuration`].
    pub fn decide(
        &mut self,
        decision: &Identifier,
        oracle: &mut dyn SearchOracle,
        context: &ParamContext<'_>,
    ) -> Result<Value, TraceError> {
        todo!()
    }

    /// Ask `oracle` the dynamic step of `kind` about `subject` over
    /// `domain`, and return the answer, which the caller maps to its own
    /// object.
    ///
    /// # Errors
    ///
    /// [`TraceError::Oracle`] and [`TraceError::CoordinateOutOfDomain`].
    pub fn decide_dynamic(
        &mut self,
        kind: &DecisionKind,
        subject: &Identifier,
        domain: &StepDomain,
        oracle: &mut dyn SearchOracle,
        context: &ParamContext<'_>,
    ) -> Result<Coordinate, TraceError> {
        todo!()
    }

    /// Return the steps recorded so far, also after a refused step.
    #[must_use]
    pub fn trace(&self) -> Trace {
        todo!()
    }

    /// Return the run's configuration so far, or `None` for a recorder over
    /// no space.
    #[must_use]
    pub fn configuration(&self) -> Option<&Configuration> {
        todo!()
    }

    /// Finish the run.
    ///
    /// # Errors
    ///
    /// Returns [`TraceError::Unasked`] for a realizing recorder that never
    /// asked a decision its configuration assigns.
    pub fn finish(self) -> Result<Recorded, TraceError> {
        todo!()
    }
}

/// A finished run: its trace and, over a space, its configuration.
#[derive(Debug, Clone)]
pub struct Recorded {
    trace: Trace,
    configuration: Option<Configuration>,
}

impl Recorded {
    /// Return the run's trace.
    #[must_use]
    pub fn trace(&self) -> &Trace {
        todo!()
    }

    /// Return the run's configuration, or `None` for a run over no space.
    #[must_use]
    pub fn configuration(&self) -> Option<&Configuration> {
        todo!()
    }

    /// Return the trace and the configuration.
    #[must_use]
    pub fn into_parts(self) -> (Trace, Option<Configuration>) {
        todo!()
    }
}
