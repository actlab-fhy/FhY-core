//! [`SearchOracle`]: what answers the steps of a run, the
//! [`PendingStep`] it is asked, and the oracles this module ships:
//! [`RandomOracle`], [`ReplayOracle`] and [`ExhaustiveOracle`].

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use std::fmt;

use crate::foreign::BoxError;
use crate::identifier::Identifier;
use crate::param::ParamContext;

use super::configuration::Configuration;
use super::domain::{Coordinate, DecisionKind, DomainSignature, StepDomain};
use super::error::{ReplayError, TraceError};
use super::rng::Rng;
use super::space::Decision;
use super::trace::Trace;

/// Answers the steps of a run, one at a time.
///
/// An oracle answers a [`Coordinate`] of the step's domain. A
/// [`Recorder`](super::Recorder) checks every answer: one outside the
/// domain or not [admissible](PendingStep::admits) stops the run, and is
/// not recorded. An oracle that fails returns its own error, which stops
/// the run as [`TraceError::Oracle`].
///
/// An oracle answers one run on one thread; it may keep state across the
/// steps of a run and across runs.
pub trait SearchOracle {
    /// Return the answer to `step`.
    ///
    /// # Errors
    ///
    /// Returns the oracle's error, which stops the run.
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError>;
}

impl<O: SearchOracle + ?Sized> SearchOracle for &mut O {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        (**self).decide(step)
    }
}

impl<O: SearchOracle + ?Sized> SearchOracle for Box<O> {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        (**self).decide(step)
    }
}

/// A step as an oracle is asked it: what it is about, its domain, and,
/// for a static step, the decision and the run's configuration so far.
///
/// Displays as `<kind> step for <subject> over <n> value(s)`.
pub struct PendingStep<'a> {
    kind: &'a DecisionKind,
    subject: &'a Identifier,
    domain: &'a StepDomain,
    position: usize,
    decision: Option<Decision<'a>>,
    configuration: Option<&'a Configuration>,
    context: &'a ParamContext<'a>,
}

impl<'a> PendingStep<'a> {
    /// Return the dynamic step of `kind` about `subject` over `domain`, at
    /// `position` in its run: every value of `domain` is admissible.
    ///
    /// A [`Recorder`](super::Recorder) builds the steps of a run; this is
    /// for asking an oracle outside one, as a test of an oracle does.
    #[must_use]
    pub fn dynamic(
        kind: &'a DecisionKind,
        subject: &'a Identifier,
        domain: &'a StepDomain,
        position: usize,
        context: &'a ParamContext<'a>,
    ) -> Self {
        todo!()
    }

    /// Return the static step of `kind` over `domain` asking the decision
    /// `decision` of `configuration`'s space, at `position` in its run:
    /// a coordinate is admissible when `configuration` with its value is
    /// accepted under `context`. `None` when the space has no such
    /// decision.
    ///
    /// A [`Recorder`](super::Recorder) builds the steps of a run, `kind`
    /// and `domain` the decision's own; this is for asking an oracle
    /// outside one.
    #[must_use]
    pub fn of_decision(
        kind: &'a DecisionKind,
        configuration: &'a Configuration,
        decision: &Identifier,
        domain: &'a StepDomain,
        position: usize,
        context: &'a ParamContext<'a>,
    ) -> Option<Self> {
        todo!()
    }

    /// Return what the step is about.
    #[must_use]
    pub fn kind(&self) -> &'a DecisionKind {
        todo!()
    }

    /// Return the step's subject: a static step's decision name.
    #[must_use]
    pub fn subject(&self) -> &'a Identifier {
        todo!()
    }

    /// Return the domain the step may be answered from.
    #[must_use]
    pub fn domain(&self) -> &'a StepDomain {
        todo!()
    }

    /// Return the step's position in its run's trace.
    #[must_use]
    pub fn position(&self) -> usize {
        todo!()
    }

    /// Return a static step's decision, or `None` for a dynamic step.
    #[must_use]
    pub fn decision(&self) -> Option<Decision<'a>> {
        todo!()
    }

    /// Return the run's configuration so far, for a static step, or
    /// `None`.
    #[must_use]
    pub fn configuration(&self) -> Option<&'a Configuration> {
        todo!()
    }

    /// Return whether `coordinate` is an admissible answer: it names a
    /// value of the domain and, for a static step, the run's configuration
    /// with that value is accepted. Every value of a dynamic step's domain
    /// is admissible.
    ///
    /// # Errors
    ///
    /// Returns [`TraceError::Configuration`] when the configuration with
    /// that value is refused for a reason other than its admissibility.
    pub fn admits(&self, coordinate: &Coordinate) -> Result<bool, TraceError> {
        todo!()
    }

    /// Return an admissible coordinate drawn uniformly from the domain's
    /// admissible ones.
    ///
    /// Draws a coordinate (`rng.below(n)` for an index; for an order, the
    /// positions shuffled with `rng`) up to 64 times, returning the first
    /// admissible one; then, for a domain of at most `2^16` values, lists
    /// the admissible coordinates in order and draws one with
    /// `rng.below`.
    ///
    /// # Errors
    ///
    /// Returns [`TraceError::DeadEnd`] when no coordinate is found
    /// admissible, and what [`admits`](Self::admits) returns.
    pub fn draw_uniform(&self, rng: &mut Rng) -> Result<Coordinate, TraceError> {
        todo!()
    }
}

impl fmt::Debug for PendingStep<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

impl fmt::Display for PendingStep<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

/// Answers every step with a coordinate drawn uniformly from its
/// admissible ones, [`PendingStep::draw_uniform`].
///
/// Uniform per step, not per configuration: over a conditional space, a
/// branch with fewer configurations below it is drawn as often as one with
/// more. [`Space::sample_uniform`](super::Space::sample_uniform) draws
/// uniformly per configuration.
#[derive(Debug, Clone)]
pub struct RandomOracle {
    rng: Rng,
}

impl RandomOracle {
    /// Return the oracle drawing from a generator seeded with `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        todo!()
    }

    /// Return the oracle drawing from `rng`.
    #[must_use]
    pub fn from_rng(rng: Rng) -> Self {
        todo!()
    }

    /// Return the generator, in its current state.
    #[must_use]
    pub fn rng(&self) -> &Rng {
        todo!()
    }
}

impl SearchOracle for RandomOracle {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        todo!()
    }
}

/// Answers the `n`-th step asked with the `n`-th step of a recorded
/// [`Trace`], refusing a run that leaves the trace's path.
///
/// Before answering it checks, in order, that a recorded step exists
/// ([`ReplayError::Exhausted`]), the kinds are equal
/// ([`ReplayError::KindMismatch`]), both steps are dynamic or both are
/// static over the same canonical position ([`ReplayError::DecisionMismatch`];
/// a dynamic step's subject is not compared), the signatures are equal
/// ([`ReplayError::DomainMismatch`]), and the recorded coordinate names a
/// value of the domain offered ([`ReplayError::CoordinateOutOfDomain`]) that
/// is admissible ([`ReplayError::Inadmissible`]). Its error reaches the
/// recorder as [`TraceError::Oracle`], the [`ReplayError`] as the source.
///
/// [`finish`](Self::finish) refuses a run that ended before asking every
/// recorded step.
#[derive(Debug, Clone)]
pub struct ReplayOracle {
    trace: Trace,
    position: usize,
}

impl ReplayOracle {
    /// Return the oracle replaying `trace` from its first step.
    #[must_use]
    pub fn new(trace: Trace) -> Self {
        todo!()
    }

    /// Return the trace replayed.
    #[must_use]
    pub fn trace(&self) -> &Trace {
        todo!()
    }

    /// Return whether every recorded step was answered.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        todo!()
    }

    /// Finish the replay.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayError::Unconsumed`] naming the first recorded step
    /// no run asked.
    pub fn finish(self) -> Result<(), ReplayError> {
        todo!()
    }
}

impl SearchOracle for ReplayOracle {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        todo!()
    }
}

/// Answers the runs of a deterministic stream so that, run after run, they
/// take every path once, in lexicographic order of their coordinates.
///
/// It keeps the path of the current run: per position, the coordinate
/// answered and the domain's cardinality. A run answers the path's
/// coordinates in order, then the first admissible coordinate of each new
/// step (the lowest index; an order's lexicographically first admissible
/// permutation). At a step with no admissible coordinate left it fails
/// with a backtrack error, which abandons the run
/// ([`is_backtrack`](Self::is_backtrack)). [`advance`](Self::advance)
/// moves to the next path.
///
/// A step of a replayed position whose domain signature differs from the
/// one the path recorded fails with [`ReplayError::DomainMismatch`]: the
/// stream is not deterministic.
#[derive(Debug, Clone, Default)]
pub struct ExhaustiveOracle {
    path: Vec<(Coordinate, DomainSignature)>,
    position: usize,
}

impl ExhaustiveOracle {
    /// Return the oracle at its first path.
    #[must_use]
    pub fn new() -> Self {
        todo!()
    }

    /// Move to the next path: the next coordinate of the deepest position
    /// that has one, every later position dropped. Return `false` when
    /// every path was taken.
    pub fn advance(&mut self) -> bool {
        todo!()
    }

    /// Return whether `error` stopped a run because this oracle abandoned
    /// it at an exhausted branch.
    #[must_use]
    pub fn is_backtrack(error: &TraceError) -> bool {
        todo!()
    }
}

impl SearchOracle for ExhaustiveOracle {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        todo!()
    }
}
