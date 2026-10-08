//! What a search does over a [`Space`] alone: sampling, per step and
//! uniformly over the complete configurations; completing a configuration;
//! replaying a trace into a configuration; enumerating and counting the
//! complete configurations; and mutating one or crossing two.

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64};

use num_bigint::BigUint;

use crate::foreign::BoxError;
use crate::identifier::Identifier;
use crate::param::ParamContext;

use super::configuration::{Activity, Configuration};
use super::counting::{count_space, sample_uniformly};
use super::domain::{Coordinate, StepDomain};
use super::error::{ReplayError, TraceError};
use super::oracle::{ExhaustiveOracle, PendingStep, SearchOracle};
use super::recorder::{Recorded, Recorder};
use super::rng::Rng;
use super::space::Space;
use super::step::{decision_domain, decision_kind, try_extend};
use super::trace::{Trace, TraceStep};

impl Space {
    /// Ask every active decision of the space of `oracle`, in
    /// [decision order](Self::decision_order), and return the complete
    /// configuration and its trace.
    ///
    /// Decision order puts each decision after those it depends on, so each
    /// decision is active or inactive when it is reached.
    ///
    /// # Errors
    ///
    /// Returns what [`Recorder::decide`](super::Recorder::decide) returns.
    pub fn sample(
        &self,
        oracle: &mut dyn SearchOracle,
        context: &ParamContext<'_>,
    ) -> Result<Recorded, TraceError> {
        walk(self, |_| true, oracle, context)
    }

    /// Return `configuration` completed: every decision it assigns keeps
    /// its value, and every other decision active when reached is asked of
    /// `oracle`, in [decision order](Self::decision_order), and its trace.
    ///
    /// The trace holds a step for every assigned decision, the assigned
    /// ones answered from `configuration` without asking `oracle`, so
    /// [`replay`](Self::replay) turns it into the completed configuration.
    /// A complete configuration is returned as it is, with its trace.
    ///
    /// # Errors
    ///
    /// Returns [`TraceError::OtherSpace`] for a configuration of another
    /// space, and what [`Recorder::decide`](super::Recorder::decide)
    /// returns.
    #[expect(
        unused_variables,
        clippy::todo,
        reason = "interface stub; bodies are todo!() until implementation"
    )]
    pub fn complete(
        &self,
        configuration: &Configuration,
        oracle: &mut dyn SearchOracle,
        context: &ParamContext<'_>,
    ) -> Result<Recorded, TraceError> {
        todo!()
    }

    /// Draw a complete configuration uniformly from all of them, and return
    /// it and its trace.
    ///
    /// The **relaxation** of the space drops its conditions, forbidden
    /// clauses and param constraints. It draws an index below the
    /// relaxation's count with `rng`, decodes it into a relaxed point,
    /// keeps the values of the decisions active under the space, and
    /// refuses the point if a value or a forbidden clause refuses it. A
    /// configuration `c` is the image of `m(c)` relaxed points, the product
    /// of the relaxed counts of the decisions a condition makes inactive
    /// while their parent is active, so it accepts with probability
    /// `1/m(c)`. Each complete configuration is then drawn with the same
    /// probability.
    ///
    /// # Errors
    ///
    /// Returns [`TraceError::NotEnumerable`] for a variable with no finite
    /// domain, [`TraceError::AttemptsExhausted`] after `attempts` refused
    /// points, and [`TraceError::Configuration`].
    pub fn sample_uniform(
        &self,
        rng: &mut Rng,
        context: &ParamContext<'_>,
        attempts: NonZeroU32,
    ) -> Result<Recorded, TraceError> {
        sample_uniformly(self, rng, context, attempts)
    }

    /// Return the configuration the static steps of `trace` describe,
    /// whatever order they were asked in; dynamic steps are skipped.
    ///
    /// It walks the space in decision order: each active decision takes the
    /// step at its canonical position, checked as
    /// [`ReplayOracle`](super::ReplayOracle) checks a step, and an inactive
    /// decision must have none.
    ///
    /// # Errors
    ///
    /// [`ReplayError::RepeatedStep`]; [`ReplayError::MissingStep`];
    /// [`ReplayError::DecisionMismatch`] for a step at a position the
    /// space does not have or for an inactive decision;
    /// [`ReplayError::KindMismatch`]; [`ReplayError::DomainMismatch`];
    /// [`ReplayError::CoordinateOutOfDomain`]; [`ReplayError::Inadmissible`];
    /// [`ReplayError::Configuration`]; and [`ReplayError::Trace`] when a
    /// step's domain cannot be built.
    pub fn replay(
        &self,
        trace: &Trace,
        context: &ParamContext<'_>,
    ) -> Result<Configuration, ReplayError> {
        replay_trace(self, trace, context)
    }

    /// Return the complete configurations of the space, each once, in
    /// lexicographic order of their coordinates in decision order.
    ///
    /// A variable whose param admits no value ends its branch with no
    /// configuration. A decision with no finite domain yields one
    /// [`TraceError::NotEnumerable`] and ends the enumeration.
    #[must_use]
    pub fn enumerate<'s, 'c>(&'s self, context: &'s ParamContext<'c>) -> Enumeration<'s, 'c> {
        Enumeration {
            space: self,
            context,
            oracle: ExhaustiveOracle::new(),
            is_done: false,
        }
    }

    /// Count the complete configurations of the space, checking at most
    /// `budget` configurations in each component that has no closed form.
    ///
    /// Top-level decisions are in one component when a condition or a
    /// forbidden clause names decisions of both; counts of components
    /// multiply. A component with no condition, no forbidden clause and no
    /// param constraint beyond the bounds of an integer variable is counted
    /// in closed form: a variable's values, a choice's sum over its
    /// alternatives of the product of their decisions'. Any other component
    /// is counted by enumerating it.
    ///
    /// The answers combine: an `Exact(0)` component makes the count
    /// `Exact(0)`; else an `Unknown` one makes it `Unknown`; else an
    /// `Unbounded` one makes it `Unbounded` when every other component has
    /// a configuration and `Unknown` otherwise; else an `AtLeast` one makes
    /// it `AtLeast` the product; else it is `Exact` the product.
    ///
    /// # Errors
    ///
    /// Returns [`TraceError::Configuration`] and [`TraceError::Hook`].
    pub fn cardinality(
        &self,
        context: &ParamContext<'_>,
        budget: u64,
    ) -> Result<Cardinality, TraceError> {
        count_space(self, context, budget)
    }

    /// Return `configuration` with one decision changed, the rest repaired,
    /// and its trace.
    ///
    /// A decision may change to another value (an ordering: two of its
    /// positions swapped) when some complete configuration keeping the
    /// values of the decisions before it in decision order takes that
    /// value; domains of more than `2^16` values are not changed. It picks
    /// uniformly a decision that may change, then uniformly one of its
    /// other values, and re-walks decision order keeping every other
    /// decision's value while it stays admissible and drawing every other
    /// step uniformly. A repair that reaches a dead end starts again from
    /// the pick.
    ///
    /// Which decisions may change is found per component, as
    /// [`cardinality`](Self::cardinality) splits the space. In a component
    /// counted in closed form every other value of a variable may be taken,
    /// and every other alternative of a choice whose decisions all admit a
    /// value. In any other component it searches the component's
    /// completions, at most 1024 runs per value, and takes a value whose
    /// search is unfinished as one that may be taken.
    ///
    /// # Errors
    ///
    /// In order: [`TraceError::OtherSpace`], [`TraceError::Incomplete`],
    /// [`TraceError::NothingToMutate`], [`TraceError::AttemptsExhausted`]
    /// after `attempts` dead ends, and what a step returns.
    pub fn mutate(
        &self,
        configuration: &Configuration,
        rng: &mut Rng,
        context: &ParamContext<'_>,
        attempts: NonZeroU32,
    ) -> Result<Recorded, TraceError> {
        mutate_configuration(self, configuration, rng, context, attempts)
    }
}

impl Space {
    /// Return a configuration crossing `first` and `second`, repaired to a
    /// complete one, and its trace.
    ///
    /// For each decision, in canonical order, a draw `rng.below(2)` picks
    /// the parent it inherits from: the first on 0, the second on 1. The
    /// run walks decision order as a [`GuidedOracle`](super::GuidedOracle)
    /// guided by the inherited coordinates walks it: a decision takes its
    /// picked parent's value when that parent assigns it and the value is
    /// admissible, else the other parent's under the same condition, and
    /// otherwise a value drawn uniformly from its admissible ones with
    /// `rng`. A repair that reaches a dead end starts again with new picks.
    /// The parents need not be complete.
    ///
    /// # Errors
    ///
    /// In order: [`TraceError::OtherSpace`] when either parent is of
    /// another space, [`TraceError::AttemptsExhausted`] after `attempts`
    /// dead ends, and what a step returns.
    #[expect(
        unused_variables,
        clippy::todo,
        reason = "interface stub; bodies are todo!() until implementation"
    )]
    pub fn crossover(
        &self,
        first: &Configuration,
        second: &Configuration,
        rng: &mut Rng,
        context: &ParamContext<'_>,
        attempts: NonZeroU32,
    ) -> Result<Recorded, TraceError> {
        todo!()
    }
}

impl Configuration {
    /// Return the trace of the configuration's assigned decisions, in
    /// decision order: the static steps
    /// [`Space::replay`] turns back into this configuration.
    ///
    /// # Errors
    ///
    /// Returns [`TraceError::NotEnumerable`] for an assigned variable with
    /// no finite domain, and [`TraceError::Hook`].
    pub fn trace(&self, context: &ParamContext<'_>) -> Result<Trace, TraceError> {
        let _ = context;
        let space = self.space();
        let mut steps = Vec::new();
        for &position in space.order_positions() {
            let node = space.decision_at(position);
            let Some(value) = self.value(node.name()) else {
                continue;
            };
            let domain = decision_domain(node)?;
            let coordinate =
                domain
                    .coordinate_of(value)
                    .ok_or(TraceError::CoordinateOutOfDomain {
                        position: steps.len(),
                    })?;
            steps.push(TraceStep::of_decision(
                decision_kind(node),
                node.name().clone(),
                position,
                domain.signature_in(Some(space)),
                coordinate,
                value.clone(),
            ));
        }
        Ok(Trace::new(steps))
    }
}

/// How many complete configurations a space has.
#[expect(
    clippy::exhaustive_enums,
    reason = "a count is exact, a lower bound, infinite or unknown"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cardinality {
    /// Exactly this many.
    Exact(BigUint),
    /// At least this many: the budget ran out.
    AtLeast(BigUint),
    /// Infinitely many, unless a constraint the count does not solve
    /// excludes every value: a configuration activates a variable whose
    /// domain is an unbounded integer or a real.
    Unbounded {
        /// The variable.
        decision: Identifier,
    },
    /// Not known: a configuration activates a variable with a custom
    /// domain and no [`search_domain`](super::Variable::search_domain).
    Unknown {
        /// The variable.
        decision: Identifier,
    },
}

/// The complete configurations of a space, from [`Space::enumerate`].
#[derive(Debug)]
pub struct Enumeration<'s, 'c> {
    space: &'s Space,
    context: &'s ParamContext<'c>,
    oracle: ExhaustiveOracle,
    is_done: bool,
}

impl Iterator for Enumeration<'_, '_> {
    type Item = Result<Configuration, TraceError>;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.is_done {
            let result = self.space.sample(&mut self.oracle, self.context);
            self.is_done = !self.oracle.advance();
            match result {
                Ok(recorded) => return recorded.into_parts().1.map(Ok),
                Err(error) if ExhaustiveOracle::is_dead_branch(&error) => {}
                Err(error) => {
                    self.is_done = true;
                    return Some(Err(error));
                }
            }
        }
        None
    }
}

/// Ask `oracle` every active decision of `space` whose canonical position
/// `includes` holds, in decision order.
///
/// # Errors
///
/// Returns what [`Recorder::decide`] returns, and
/// [`TraceError::NotActive`] for a decision still pending when reached.
pub(super) fn walk(
    space: &Space,
    includes: impl Fn(usize) -> bool,
    oracle: &mut dyn SearchOracle,
    context: &ParamContext<'_>,
) -> Result<Recorded, TraceError> {
    let mut recorder = Recorder::over(space);
    for &position in space.order_positions() {
        if !includes(position) {
            continue;
        }
        let name = space.decision_at(position).name();
        let activity = recorder
            .configuration()
            .and_then(|configuration| configuration.activity(name));
        match activity {
            Some(Activity::Active) => {
                recorder.decide(name, oracle, context)?;
            }
            Some(Activity::Inactive) => {}
            Some(activity) => {
                return Err(TraceError::NotActive {
                    name: name.clone(),
                    activity,
                });
            }
            None => return Err(TraceError::UnknownDecision { name: name.clone() }),
        }
    }
    recorder.finish()
}

/// Return the configuration of `space` the static steps of `trace`
/// describe, as [`Space::replay`] documents.
fn replay_trace(
    space: &Space,
    trace: &Trace,
    context: &ParamContext<'_>,
) -> Result<Configuration, ReplayError> {
    let mut steps: HashMap<usize, (usize, &TraceStep)> = HashMap::new();
    for (index, step) in trace.steps().iter().enumerate() {
        let Some(decision) = step.decision() else {
            continue;
        };
        if decision >= space.decision_count() {
            return Err(ReplayError::DecisionMismatch { position: index });
        }
        if steps.insert(decision, (index, step)).is_some() {
            return Err(ReplayError::RepeatedStep {
                decision: space.decision_at(decision).name().clone(),
            });
        }
    }
    let mut configuration = Configuration::empty(space);
    for &position in space.order_positions() {
        let node = space.decision_at(position);
        let name = node.name();
        let step = steps.remove(&position);
        if configuration.activity(name) != Some(Activity::Active) {
            if let Some((index, _)) = step {
                return Err(ReplayError::DecisionMismatch { position: index });
            }
            continue;
        }
        let (index, step) = step.ok_or_else(|| ReplayError::MissingStep {
            decision: name.clone(),
        })?;
        if *step.kind() != decision_kind(node) {
            return Err(ReplayError::KindMismatch { position: index });
        }
        let domain = decision_domain(node).map_err(|error| ReplayError::Trace(Box::new(error)))?;
        if *step.signature() != domain.signature_in(Some(space)) {
            return Err(ReplayError::DomainMismatch { position: index });
        }
        let value = domain
            .value_at(step.coordinate())
            .ok_or(ReplayError::CoordinateOutOfDomain { position: index })?;
        configuration = match try_extend(&configuration, name, value, context) {
            Ok(Some(extended)) => extended,
            Ok(None) => return Err(ReplayError::Inadmissible { position: index }),
            Err(TraceError::Configuration(errors)) => {
                return Err(ReplayError::Configuration(errors));
            }
            Err(error) => return Err(ReplayError::Trace(Box::new(error))),
        };
    }
    Ok(configuration)
}

/// The largest domain a mutation lists the other values of.
const LISTED_MUTATION_DOMAIN: u32 = 1 << 16;

/// The most runs a mutation searches for a completion of one changed
/// value before it takes that value as completable.
const COMPLETION_BUDGET: u64 = 1_024;

/// A decision a mutation may change: its canonical position and the other
/// coordinates it may take.
struct Candidate {
    position: usize,
    options: Vec<Coordinate>,
}

/// Return `configuration` mutated, as [`Space::mutate`] documents.
fn mutate_configuration(
    space: &Space,
    configuration: &Configuration,
    rng: &mut Rng,
    context: &ParamContext<'_>,
    attempts: NonZeroU32,
) -> Result<Recorded, TraceError> {
    if configuration.space() != space {
        return Err(TraceError::OtherSpace);
    }
    if !configuration.is_complete() {
        return Err(TraceError::Incomplete);
    }
    let (previous, candidates) = find_candidates(space, configuration, context)?;
    let Some(count) = NonZeroU64::new(u64::try_from(candidates.len()).unwrap_or(u64::MAX)) else {
        return Err(TraceError::NothingToMutate);
    };
    for _ in 0..attempts.get() {
        let candidate = &candidates[usize::try_from(rng.below(count)).unwrap_or(0)];
        let Some(options) = NonZeroU64::new(u64::try_from(candidate.options.len()).unwrap_or(0))
        else {
            continue;
        };
        let target = candidate.options[usize::try_from(rng.below(options)).unwrap_or(0)].clone();
        let mut oracle = GuidedOracle {
            target: (candidate.position, target),
            previous: &previous,
            rng,
        };
        match walk(space, |_| true, &mut oracle, context) {
            Ok(recorded) => return Ok(recorded),
            Err(TraceError::DeadEnd { .. } | TraceError::Inadmissible { .. }) => {}
            Err(error) => return Err(error),
        }
    }
    Err(TraceError::AttemptsExhausted {
        attempts: attempts.get(),
    })
}

/// Return each assigned decision's coordinate in `configuration`, by
/// canonical position, and the decisions with another coordinate that a
/// complete configuration keeping the decisions before them takes.
fn find_candidates(
    space: &Space,
    configuration: &Configuration,
    context: &ParamContext<'_>,
) -> Result<(HashMap<usize, Coordinate>, Vec<Candidate>), TraceError> {
    let mut previous = HashMap::new();
    let mut candidates = Vec::new();
    for &position in space.order_positions() {
        let node = space.decision_at(position);
        let Some(value) = configuration.value(node.name()) else {
            continue;
        };
        let domain = decision_domain(node)?;
        let coordinate = domain
            .coordinate_of(value)
            .ok_or(TraceError::CoordinateOutOfDomain { position: 0 })?;
        let mut options = Vec::new();
        for option in list_neighbours(&domain, &coordinate) {
            let mut pinned = previous.clone();
            pinned.insert(position, option.clone());
            if completes(space, &pinned, context)? {
                options.push(option);
            }
        }
        if !options.is_empty() {
            candidates.push(Candidate { position, options });
        }
        previous.insert(position, coordinate);
    }
    Ok((previous, candidates))
}

/// Return the coordinates a mutation may move `current` to: an ordering
/// with two positions swapped, or another index; none for a domain of
/// more than [`LISTED_MUTATION_DOMAIN`] values.
fn list_neighbours(domain: &StepDomain, current: &Coordinate) -> Vec<Coordinate> {
    if domain.cardinality() > BigUint::from(LISTED_MUTATION_DOMAIN) {
        return Vec::new();
    }
    match current {
        Coordinate::Order(positions) => {
            let mut swapped = Vec::new();
            for first in 0..positions.len() {
                for second in first + 1..positions.len() {
                    let mut neighbour = positions.clone();
                    neighbour.swap(first, second);
                    swapped.push(Coordinate::Order(neighbour));
                }
            }
            swapped
        }
        Coordinate::Index(index) => {
            let count = u64::try_from(domain.cardinality()).unwrap_or(0);
            (0..count)
                .filter(|other| other != index)
                .map(Coordinate::Index)
                .collect()
        }
    }
}

/// Return whether a complete configuration of `space` takes the `pinned`
/// coordinates, searching at most [`COMPLETION_BUDGET`] runs and taking an
/// unfinished search as a yes.
fn completes(
    space: &Space,
    pinned: &HashMap<usize, Coordinate>,
    context: &ParamContext<'_>,
) -> Result<bool, TraceError> {
    let mut oracle = PinnedOracle {
        pinned,
        rest: ExhaustiveOracle::new(),
    };
    for _ in 0..COMPLETION_BUDGET {
        match walk(space, |_| true, &mut oracle, context) {
            Ok(_) => return Ok(true),
            Err(TraceError::Inadmissible { .. }) => return Ok(false),
            Err(error) if ExhaustiveOracle::is_dead_branch(&error) => {}
            Err(error) => return Err(error),
        }
        if !oracle.rest.advance() {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Answers the pinned decisions with their coordinates and every other
/// step exhaustively.
struct PinnedOracle<'a> {
    pinned: &'a HashMap<usize, Coordinate>,
    rest: ExhaustiveOracle,
}

impl SearchOracle for PinnedOracle<'_> {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        match step
            .decision_position()
            .and_then(|position| self.pinned.get(&position))
        {
            Some(coordinate) => Ok(coordinate.clone()),
            None => self.rest.decide(step),
        }
    }
}

/// Answers a mutation's re-walk: the mutated decision with its new
/// coordinate, every other decision with its old one while admissible,
/// and the rest uniformly.
struct GuidedOracle<'a> {
    target: (usize, Coordinate),
    previous: &'a HashMap<usize, Coordinate>,
    rng: &'a mut Rng,
}

impl SearchOracle for GuidedOracle<'_> {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        let position = step.decision_position();
        if position == Some(self.target.0) {
            return Ok(self.target.1.clone());
        }
        if let Some(previous) = position.and_then(|position| self.previous.get(&position)) {
            if step.admits(previous)? {
                return Ok(previous.clone());
            }
        }
        Ok(step.draw_uniform(self.rng)?)
    }
}
