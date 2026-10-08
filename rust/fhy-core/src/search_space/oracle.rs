//! [`SearchOracle`]: what answers the steps of a run, the
//! [`PendingStep`] it is asked, and the oracles this module ships:
//! [`RandomOracle`], [`ReplayOracle`] and [`ExhaustiveOracle`].

use std::cell::RefCell;
use std::error::Error;
use std::fmt;
use std::num::NonZeroU64;

use num_bigint::BigUint;

use crate::foreign::BoxError;
use crate::identifier::Identifier;
use crate::param::ParamContext;

use super::configuration::Configuration;
use super::domain::{Coordinate, DecisionKind, DomainSignature, StepDomain};
use super::error::{ReplayError, TraceError};
use super::rng::Rng;
use super::space::Decision;
use super::step::{draw_coordinate, first_coordinate, next_coordinate, try_extend};
use super::trace::Trace;

/// The most coordinates [`PendingStep::draw_uniform`] draws before it lists
/// the admissible ones.
const DRAWS_BEFORE_LISTING: usize = 64;

/// The largest domain [`PendingStep::draw_uniform`] lists the admissible
/// coordinates of.
const LISTED_DOMAIN_SIZE: u32 = 1 << 16;

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
    /// A static step's decision and its canonical position.
    decision: Option<(Decision<'a>, usize)>,
    configuration: Option<&'a Configuration>,
    context: &'a ParamContext<'a>,
    /// The last coordinate [`admits`](Self::admits) accepted, with the
    /// configuration it extended the run's to, so the recorder that asked
    /// the step need not check that configuration again.
    admitted: RefCell<Option<(Coordinate, Configuration)>>,
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
        Self {
            kind,
            subject,
            domain,
            position,
            decision: None,
            configuration: None,
            context,
            admitted: RefCell::new(None),
        }
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
        let space = configuration.space();
        let canonical = space.position(decision)?;
        let decision = space.decision_at(canonical);
        Some(Self {
            kind,
            subject: decision.name(),
            domain,
            position,
            decision: Some((decision, canonical)),
            configuration: Some(configuration),
            context,
            admitted: RefCell::new(None),
        })
    }

    /// Return what the step is about.
    #[must_use]
    pub fn kind(&self) -> &'a DecisionKind {
        self.kind
    }

    /// Return the step's subject: a static step's decision name.
    #[must_use]
    pub fn subject(&self) -> &'a Identifier {
        self.subject
    }

    /// Return the domain the step may be answered from.
    #[must_use]
    pub fn domain(&self) -> &'a StepDomain {
        self.domain
    }

    /// Return the step's position in its run's trace.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Return a static step's decision, or `None` for a dynamic step.
    #[must_use]
    pub fn decision(&self) -> Option<Decision<'a>> {
        self.decision.map(|(decision, _)| decision)
    }

    /// Return a static step's decision's canonical position in its space.
    pub(super) fn decision_position(&self) -> Option<usize> {
        self.decision.map(|(_, position)| position)
    }

    /// Return the signature of the step's domain, the names its space binds
    /// written by position for a static step.
    pub(super) fn signature(&self) -> DomainSignature {
        self.domain
            .signature_in(self.configuration.map(Configuration::space))
    }

    /// Return the run's configuration so far, for a static step, or
    /// `None`.
    #[must_use]
    pub fn configuration(&self) -> Option<&'a Configuration> {
        self.configuration
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
        let Some(configuration) = self.configuration else {
            return Ok(self.domain.contains(coordinate));
        };
        let Some(value) = self.domain.value_at(coordinate) else {
            return Ok(false);
        };
        let extended = try_extend(configuration, self.subject, value, self.context)?;
        let is_admissible = extended.is_some();
        if let Some(extended) = extended {
            *self.admitted.borrow_mut() = Some((coordinate.clone(), extended));
        }
        Ok(is_admissible)
    }

    /// Return the configuration the run's grows to with `coordinate`'s value,
    /// if [`admits`](Self::admits) accepted that coordinate last.
    pub(super) fn take_admitted(&self, coordinate: &Coordinate) -> Option<Configuration> {
        let mut admitted = self.admitted.borrow_mut();
        match admitted.take() {
            Some((accepted, extended)) if accepted == *coordinate => Some(extended),
            other => {
                *admitted = other;
                None
            }
        }
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
        for _ in 0..DRAWS_BEFORE_LISTING {
            let coordinate = draw_coordinate(self.domain, rng);
            if self.admits(&coordinate)? {
                return Ok(coordinate);
            }
        }
        let dead_end = || TraceError::DeadEnd {
            decision: self.subject.clone(),
        };
        if self.domain.cardinality() > BigUint::from(LISTED_DOMAIN_SIZE) {
            return Err(dead_end());
        }
        let admissible = self.list_admissible()?;
        let count = u64::try_from(admissible.len()).unwrap_or(u64::MAX);
        let bound = NonZeroU64::new(count).ok_or_else(dead_end)?;
        let chosen = usize::try_from(rng.below(bound)).unwrap_or(0);
        admissible.into_iter().nth(chosen).ok_or_else(dead_end)
    }

    /// Return the admissible coordinates of the step's domain, in
    /// lexicographic order.
    fn list_admissible(&self) -> Result<Vec<Coordinate>, TraceError> {
        let mut admissible = Vec::new();
        let mut coordinate = Some(first_coordinate(self.domain));
        while let Some(current) = coordinate {
            if self.admits(&current)? {
                admissible.push(current.clone());
            }
            coordinate = next_coordinate(|next| self.domain.contains(next), &current);
        }
        Ok(admissible)
    }

    /// Return the first admissible coordinate at or after `start`, in
    /// lexicographic order.
    pub(super) fn first_admissible_from(
        &self,
        start: Coordinate,
    ) -> Result<Option<Coordinate>, TraceError> {
        let mut coordinate = Some(start);
        while let Some(current) = coordinate {
            if self.admits(&current)? {
                return Ok(Some(current));
            }
            coordinate = next_coordinate(|next| self.domain.contains(next), &current);
        }
        Ok(None)
    }
}

impl fmt::Debug for PendingStep<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingStep")
            .field("kind", self.kind)
            .field("subject", self.subject)
            .field("domain", self.domain)
            .field("position", &self.position)
            .field("decision", &self.decision_position())
            .finish_non_exhaustive()
    }
}

impl fmt::Display for PendingStep<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} step for {} over {} value(s)",
            self.kind,
            self.subject,
            self.domain.cardinality()
        )
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
        Self::from_rng(Rng::new(seed))
    }

    /// Return the oracle drawing from `rng`.
    #[must_use]
    pub fn from_rng(rng: Rng) -> Self {
        Self { rng }
    }

    /// Return the generator, in its current state.
    #[must_use]
    pub fn rng(&self) -> &Rng {
        &self.rng
    }
}

impl SearchOracle for RandomOracle {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        Ok(step.draw_uniform(&mut self.rng)?)
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
        Self { trace, position: 0 }
    }

    /// Return the trace replayed.
    #[must_use]
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    /// Return whether every recorded step was answered.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.position >= self.trace.len()
    }

    /// Finish the replay.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayError::Unconsumed`] naming the first recorded step
    /// no run asked.
    pub fn finish(self) -> Result<(), ReplayError> {
        if self.is_exhausted() {
            Ok(())
        } else {
            Err(ReplayError::Unconsumed {
                position: self.position,
            })
        }
    }

    /// Return the answer to `step`, the recorded step at this position,
    /// checked as the type documents.
    fn answer(&self, step: &PendingStep<'_>) -> Result<Coordinate, ReplayError> {
        let position = self.position;
        let recorded = self
            .trace
            .steps()
            .get(position)
            .ok_or(ReplayError::Exhausted { position })?;
        if recorded.kind() != step.kind() {
            return Err(ReplayError::KindMismatch { position });
        }
        if recorded.decision() != step.decision_position() {
            return Err(ReplayError::DecisionMismatch { position });
        }
        if *recorded.signature() != step.signature() {
            return Err(ReplayError::DomainMismatch { position });
        }
        let coordinate = recorded.coordinate();
        if !step.domain().contains(coordinate) {
            return Err(ReplayError::CoordinateOutOfDomain { position });
        }
        let is_admissible = step
            .admits(coordinate)
            .map_err(|error| ReplayError::Trace(Box::new(error)))?;
        if !is_admissible {
            return Err(ReplayError::Inadmissible { position });
        }
        Ok(coordinate.clone())
    }
}

impl SearchOracle for ReplayOracle {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        let coordinate = self.answer(step)?;
        self.position += 1;
        Ok(coordinate)
    }
}

/// Answers the runs of a deterministic stream so that, run after run, they
/// take every path once, in lexicographic order of their coordinates.
///
/// It keeps the path of the current run: per position, the coordinate
/// answered and the domain's signature, and no value of the domain. A run answers the path's
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
    /// Per position of the current path: the coordinate answered and the
    /// signature of the domain it was answered over, which also says which
    /// coordinate comes next. No domain is kept, so the oracle holds none
    /// of a domain's values.
    path: Vec<(Coordinate, DomainSignature)>,
    position: usize,
}

/// The error an [`ExhaustiveOracle`] abandons a run with at a step that has
/// no admissible coordinate left.
#[derive(Debug)]
struct Backtrack;

impl fmt::Display for Backtrack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the branch has no admissible coordinate left")
    }
}

impl Error for Backtrack {}

impl ExhaustiveOracle {
    /// Return the oracle at its first path.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Move to the next path: the next coordinate of the deepest position
    /// that has one, every later position dropped. Return `false` when
    /// every path was taken.
    pub fn advance(&mut self) -> bool {
        self.position = 0;
        while let Some((coordinate, signature)) = self.path.pop() {
            if let Some(next) = next_coordinate(|next| signature.contains(next), &coordinate) {
                self.path.push((next, signature));
                return true;
            }
        }
        false
    }

    /// Return whether `error` stopped a run because this oracle abandoned
    /// it at an exhausted branch.
    #[must_use]
    pub fn is_backtrack(error: &TraceError) -> bool {
        matches!(error, TraceError::Oracle { source, .. } if source.is::<Backtrack>())
    }

    /// Return whether `error` stopped a run this oracle answers at a branch
    /// with no configuration, so that the run moves on to the next path:
    /// this oracle's backtrack, or the [`DeadEnd`](TraceError::DeadEnd) of
    /// a variable whose param admits no value, which is found before the
    /// oracle is asked. The oracle itself never answers with a dead end.
    #[must_use]
    pub fn is_dead_branch(error: &TraceError) -> bool {
        Self::is_backtrack(error) || matches!(error, TraceError::DeadEnd { .. })
    }

    /// Return the answer to `step`: the path's coordinate at this position
    /// or after it, or the first admissible coordinate of a new position.
    fn answer(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        let position = self.position;
        let signature = step.signature();
        let start = match self.path.get(position) {
            Some((coordinate, recorded)) => {
                if *recorded != signature {
                    return Err(Box::new(ReplayError::DomainMismatch { position }));
                }
                coordinate.clone()
            }
            None => first_coordinate(step.domain()),
        };
        let Some(coordinate) = step.first_admissible_from(start)? else {
            self.path.truncate(position);
            return Err(Box::new(Backtrack));
        };
        let entry = (coordinate.clone(), signature);
        match self.path.get_mut(position) {
            Some(slot) => *slot = entry,
            None => self.path.push(entry),
        }
        self.position += 1;
        Ok(coordinate)
    }
}

impl SearchOracle for ExhaustiveOracle {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        self.answer(step)
    }
}
