//! [`Recorder`]: drives one run of a search, asking its oracle each step,
//! checking and recording the answers, and, over a space, growing the
//! run's configuration; and [`Recorded`], what a finished run gives.

use crate::constraint::Value;
use crate::identifier::Identifier;
use crate::param::ParamContext;

use super::configuration::{Activity, Configuration};
use super::domain::{Coordinate, DecisionKind, StepDomain};
use super::error::TraceError;
use super::oracle::{PendingStep, SearchOracle};
use super::space::Space;
use super::step::{decision_domain, decision_kind, try_extend};
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
///
/// An oracle's error stops the run as [`TraceError::Oracle`], except one
/// that is itself a [`TraceError`], such as the
/// [`DeadEnd`](TraceError::DeadEnd) of
/// [`PendingStep::draw_uniform`](super::PendingStep::draw_uniform), which
/// stops it as that error.
#[derive(Debug, Clone, Default)]
pub struct Recorder {
    /// The run's configuration so far, for a run over a space.
    configuration: Option<Configuration>,
    /// The configuration a realizing run answers from.
    preset: Option<Configuration>,
    steps: Vec<TraceStep>,
}

impl Recorder {
    /// Return the recorder of a run of dynamic steps.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the recorder of a run over `space`, from its empty
    /// configuration.
    #[must_use]
    pub fn over(space: &Space) -> Self {
        Self {
            configuration: Some(Configuration::empty(space)),
            preset: None,
            steps: Vec::new(),
        }
    }

    /// Return the recorder of a run realizing `configuration`: a decision
    /// it assigns is answered with its value, everything else by the
    /// oracle.
    #[must_use]
    pub fn realizing(configuration: &Configuration) -> Self {
        Self {
            configuration: Some(Configuration::empty(configuration.space())),
            preset: Some(configuration.clone()),
            steps: Vec::new(),
        }
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
        let configuration = self.configuration.as_ref().ok_or(TraceError::NoSpace)?;
        let space = configuration.space();
        let unknown = || TraceError::UnknownDecision {
            name: decision.clone(),
        };
        let canonical = space.position(decision).ok_or_else(unknown)?;
        if configuration.value(decision).is_some() {
            return Err(TraceError::AlreadyDecided {
                name: decision.clone(),
            });
        }
        match configuration.activity(decision) {
            Some(Activity::Active) => {}
            Some(activity) => {
                return Err(TraceError::NotActive {
                    name: decision.clone(),
                    activity,
                });
            }
            None => return Err(unknown()),
        }
        let node = space.decision_at(canonical);
        let kind = decision_kind(node);
        let domain = decision_domain(node)?;
        let position = self.steps.len();
        let preset = self
            .preset
            .as_ref()
            .and_then(|preset| preset.value(decision));
        let (coordinate, admitted) = if let Some(value) = preset {
            let coordinate = domain
                .coordinate_of(value)
                .ok_or(TraceError::CoordinateOutOfDomain { position })?;
            (coordinate, None)
        } else {
            let step = PendingStep::of_decision(
                &kind,
                configuration,
                decision,
                &domain,
                position,
                context,
            )
            .ok_or_else(unknown)?;
            let coordinate = ask(oracle, &step)?;
            // An oracle that checked its answer's admissibility, as the
            // shipped ones do, left the grown configuration on the step.
            let admitted = step.take_admitted(&coordinate);
            (coordinate, admitted)
        };
        let value = domain
            .value_at(&coordinate)
            .ok_or(TraceError::CoordinateOutOfDomain { position })?;
        let extended =
            match admitted {
                Some(extended) => extended,
                None => try_extend(configuration, decision, value.clone(), context)?.ok_or_else(
                    || TraceError::Inadmissible {
                        position,
                        coordinate: coordinate.clone(),
                    },
                )?,
            };
        let signature = domain.signature_in(Some(space));
        self.steps.push(TraceStep::of_decision(
            kind,
            decision.clone(),
            canonical,
            signature,
            coordinate,
            value.clone(),
        ));
        self.configuration = Some(extended);
        Ok(value)
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
        let position = self.steps.len();
        let step = PendingStep::dynamic(kind, subject, domain, position, context);
        let coordinate = ask(oracle, &step)?;
        let recorded =
            TraceStep::dynamic(kind.clone(), subject.clone(), domain, coordinate.clone())
                .map_err(|_outside| TraceError::CoordinateOutOfDomain { position })?;
        self.steps.push(recorded);
        Ok(coordinate)
    }

    /// Return the steps recorded so far, also after a refused step.
    #[must_use]
    pub fn trace(&self) -> Trace {
        Trace::new(self.steps.clone())
    }

    /// Return the run's configuration so far, or `None` for a recorder over
    /// no space.
    #[must_use]
    pub fn configuration(&self) -> Option<&Configuration> {
        self.configuration.as_ref()
    }

    /// Finish the run.
    ///
    /// # Errors
    ///
    /// Returns [`TraceError::Unasked`] for a realizing recorder that never
    /// asked a decision its configuration assigns.
    pub fn finish(self) -> Result<Recorded, TraceError> {
        if let (Some(preset), Some(configuration)) = (&self.preset, &self.configuration) {
            let unasked: Vec<Identifier> = preset
                .entries()
                .map(|(name, _)| name)
                .filter(|name| configuration.value(name).is_none())
                .cloned()
                .collect();
            if !unasked.is_empty() {
                return Err(TraceError::Unasked { decisions: unasked });
            }
        }
        Ok(Recorded {
            trace: Trace::new(self.steps),
            configuration: self.configuration,
        })
    }
}

/// Ask `oracle` `step`, its error stopping the run as the type documents.
fn ask(oracle: &mut dyn SearchOracle, step: &PendingStep<'_>) -> Result<Coordinate, TraceError> {
    oracle
        .decide(step)
        .map_err(|source| match source.downcast::<TraceError>() {
            Ok(error) => *error,
            Err(source) => TraceError::Oracle {
                position: step.position(),
                source,
            },
        })
}

/// A finished run: its trace and, over a space, its configuration.
#[derive(Debug, Clone)]
pub struct Recorded {
    trace: Trace,
    configuration: Option<Configuration>,
}

impl Recorded {
    /// Return the run of `trace` and `configuration`.
    pub(super) fn new(trace: Trace, configuration: Option<Configuration>) -> Self {
        Self {
            trace,
            configuration,
        }
    }

    /// Return the run's trace.
    #[must_use]
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    /// Return the run's configuration, or `None` for a run over no space.
    #[must_use]
    pub fn configuration(&self) -> Option<&Configuration> {
        self.configuration.as_ref()
    }

    /// Return the trace and the configuration.
    #[must_use]
    pub fn into_parts(self) -> (Trace, Option<Configuration>) {
        (self.trace, self.configuration)
    }
}
