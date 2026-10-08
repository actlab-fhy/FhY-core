//! What a search does over a [`Space`] alone: sampling, per step and
//! uniformly over the complete configurations; completing a configuration;
//! replaying a trace into a configuration; enumerating and counting the
//! complete configurations; and mutating one or crossing two.

use std::borrow::Cow;
use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64};

use num_bigint::BigUint;

use crate::foreign::BoxError;
use crate::identifier::Identifier;
use crate::param::ParamContext;

use super::configuration::{Activity, Configuration};
use super::counting::{Components, Tree, admits_completion, count_space, sample_uniformly};
use super::domain::{Coordinate, StepDomain};
use super::error::{ReplayError, TraceError};
use super::oracle::{ExhaustiveOracle, PendingStep, SearchOracle};
use super::recorder::{Recorded, Recorder};
use super::rng::Rng;
use super::space::{Decision, Space};
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
    pub fn complete(
        &self,
        configuration: &Configuration,
        oracle: &mut dyn SearchOracle,
        context: &ParamContext<'_>,
    ) -> Result<Recorded, TraceError> {
        if configuration.space() != self {
            return Err(TraceError::OtherSpace);
        }
        let recorder = Recorder::realizing(configuration);
        walk_from(recorder, self, |_| true, oracle, context)
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

    /// Return a configuration crossing `first` and `second`, repaired to a
    /// complete one, and its trace.
    ///
    /// For each decision, in canonical order, a draw `rng.below(2)` picks
    /// the parent it inherits from: the first on 0, the second on 1. The
    /// run then walks decision order: a decision takes its picked parent's
    /// value when that parent assigns it and the value is admissible, else
    /// the other parent's under the same condition, and otherwise a value
    /// drawn uniformly from its admissible ones with `rng`. A repair that
    /// reaches a dead end starts again with new picks. The parents need not
    /// be complete.
    ///
    /// # Errors
    ///
    /// In order: [`TraceError::OtherSpace`] when either parent is of
    /// another space, [`TraceError::AttemptsExhausted`] after `attempts`
    /// dead ends, and what a step returns.
    pub fn crossover(
        &self,
        first: &Configuration,
        second: &Configuration,
        rng: &mut Rng,
        context: &ParamContext<'_>,
        attempts: NonZeroU32,
    ) -> Result<Recorded, TraceError> {
        cross_configurations(self, [first, second], rng, context, attempts)
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
    walk_from(Recorder::over(space), space, includes, oracle, context)
}

/// Ask `oracle`, through `recorder`, every active decision of `space`
/// whose canonical position `includes` holds, in decision order.
///
/// # Errors
///
/// As [`walk`].
fn walk_from(
    mut recorder: Recorder,
    space: &Space,
    includes: impl Fn(usize) -> bool,
    oracle: &mut dyn SearchOracle,
    context: &ParamContext<'_>,
) -> Result<Recorded, TraceError> {
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

/// Ask `oracle` every active decision of the space of `empty`, its empty
/// configuration, whose canonical position `includes` holds, in decision
/// order, starting from `empty`: for the runs of one search, which share
/// the empty configuration instead of each building it.
///
/// # Errors
///
/// As [`walk`].
fn walk_from_empty(
    empty: &Configuration,
    includes: impl Fn(usize) -> bool,
    oracle: &mut dyn SearchOracle,
    context: &ParamContext<'_>,
) -> Result<Recorded, TraceError> {
    let recorder = Recorder::over_empty(empty.clone());
    walk_from(recorder, empty.space(), includes, oracle, context)
}

/// Run `attempt` until it returns a configuration, at most `attempts`
/// times: a run that reaches a dead end or an inadmissible answer is run
/// again, and any other error stops.
///
/// # Errors
///
/// Returns [`TraceError::AttemptsExhausted`] when every attempt reached a
/// dead end, and the first other error a run returns.
fn retry_after_dead_ends(
    attempts: NonZeroU32,
    mut attempt: impl FnMut() -> Result<Recorded, TraceError>,
) -> Result<Recorded, TraceError> {
    for _ in 0..attempts.get() {
        match attempt() {
            Ok(recorded) => return Ok(recorded),
            Err(TraceError::DeadEnd { .. } | TraceError::Inadmissible { .. }) => {}
            Err(error) => return Err(error),
        }
    }
    Err(TraceError::AttemptsExhausted {
        attempts: attempts.get(),
    })
}

/// Return the first of `preferred` that `step` admits, or else a
/// coordinate drawn uniformly from its admissible ones with `rng`.
///
/// # Errors
///
/// Returns what [`PendingStep::admits`] and
/// [`PendingStep::draw_uniform`] return.
fn answer_first_admissible<'c>(
    step: &PendingStep<'_>,
    preferred: impl IntoIterator<Item = Cow<'c, Coordinate>>,
    rng: &mut Rng,
) -> Result<Coordinate, TraceError> {
    for coordinate in preferred {
        if step.admits(&coordinate)? {
            return Ok(coordinate.into_owned());
        }
    }
    step.draw_uniform(rng)
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

/// The largest domain a mutation changes a value of.
const LISTED_MUTATION_DOMAIN: u32 = 1 << 16;

/// The most runs a mutation searches for a completion of one changed
/// value before it takes that value as completable.
const COMPLETION_BUDGET: u64 = 1_024;

/// The other coordinates a mutation may move a decision to, in the order a
/// mutation draws them by index.
enum Options {
    /// Every index below `count` except `current`, in order.
    OtherIndices { count: u64, current: u64 },
    /// Every ordering `current` gives with two of its positions swapped:
    /// the first position, then the second, ascending.
    Swaps { current: Box<[u32]> },
    /// The coordinates listed.
    Listed(Vec<Coordinate>),
}

impl Options {
    /// Return every other coordinate of `domain` than `current`, or none
    /// for a domain of more than [`LISTED_MUTATION_DOMAIN`] values.
    fn all_others(domain: &StepDomain, current: &Coordinate) -> Self {
        if domain.cardinality() > BigUint::from(LISTED_MUTATION_DOMAIN) {
            return Self::Listed(Vec::new());
        }
        match current {
            Coordinate::Order(positions) => Self::Swaps {
                current: positions.clone(),
            },
            Coordinate::Index(index) => Self::OtherIndices {
                count: u64::try_from(domain.cardinality())
                    .expect("a domain of at most 2^16 values counts in a u64"),
                current: *index,
            },
        }
    }

    /// Return the number of coordinates.
    fn len(&self) -> u64 {
        match self {
            Self::OtherIndices { count, current } => {
                if current < count {
                    count - 1
                } else {
                    *count
                }
            }
            Self::Swaps { current } => {
                let length = u64::try_from(current.len()).unwrap_or(u64::MAX);
                length * length.saturating_sub(1) / 2
            }
            Self::Listed(listed) => u64::try_from(listed.len()).unwrap_or(u64::MAX),
        }
    }

    /// Return whether there is no coordinate.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the coordinate at `index`, below [`len`](Self::len).
    fn nth(&self, index: u64) -> Option<Coordinate> {
        match self {
            Self::OtherIndices { current, .. } => Some(Coordinate::Index(if index < *current {
                index
            } else {
                index + 1
            })),
            Self::Swaps { current } => {
                let mut remaining = usize::try_from(index).ok()?;
                for first in 0..current.len() {
                    let pairs = current.len() - first - 1;
                    if remaining < pairs {
                        let mut swapped = current.clone();
                        swapped.swap(first, first + 1 + remaining);
                        return Some(Coordinate::Order(swapped));
                    }
                    remaining -= pairs;
                }
                None
            }
            Self::Listed(listed) => listed.get(usize::try_from(index).ok()?).cloned(),
        }
    }

    /// Return the coordinates, in order.
    fn iter(&self) -> impl Iterator<Item = Coordinate> + '_ {
        (0..self.len()).map_while(|index| self.nth(index))
    }
}

/// A decision a mutation may change: its canonical position and the other
/// coordinates it may take, at least one.
struct Candidate {
    position: usize,
    options: Options,
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
    let empty = Configuration::empty(space);
    let (previous, candidates) = find_candidates(space, configuration, &empty, context)?;
    let Some(count) = NonZeroU64::new(u64::try_from(candidates.len()).unwrap_or(u64::MAX)) else {
        return Err(TraceError::NothingToMutate);
    };
    retry_after_dead_ends(attempts, || {
        let candidate = &candidates[usize::try_from(rng.below(count)).unwrap_or(0)];
        let options = NonZeroU64::new(candidate.options.len())
            .expect("a candidate is kept only with an option");
        let target = candidate
            .options
            .nth(rng.below(options))
            .expect("every index below the options' length names one");
        let mut oracle = RepairOracle {
            target: (candidate.position, target),
            previous: &previous,
            rng: &mut *rng,
        };
        walk_from_empty(&empty, |_| true, &mut oracle, context)
    })
}

/// Return each assigned decision's coordinate in `configuration`, by
/// canonical position, and the decisions with another coordinate that a
/// complete configuration keeping the decisions before them takes, in
/// decision order.
///
/// A decision of a component counted in closed form takes every other
/// coordinate of a variable, and every other alternative of a choice whose
/// relaxed count is not zero. A decision of any other component is
/// searched within its component, from `empty`, the space's empty
/// configuration, with the component's earlier decisions pinned.
fn find_candidates(
    space: &Space,
    configuration: &Configuration,
    empty: &Configuration,
    context: &ParamContext<'_>,
) -> Result<(HashMap<usize, Coordinate>, Vec<Candidate>), TraceError> {
    let tree = Tree::of(space);
    let components = Components::of(space, &tree);
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
        let all_others = Options::all_others(&domain, &coordinate);
        let component = components.component_of(position);
        let options = if components.is_closed_form(component) {
            match node {
                Decision::Choice(_) => {
                    find_completable_alternatives(space, &tree, position, &all_others)?
                }
                Decision::Variable(_) => all_others,
            }
        } else {
            let search = ComponentSearch {
                empty,
                includes: |member: usize| components.component_of(member) == component,
                previous: &previous,
                context,
            };
            search.find_options(position, &all_others)?
        };
        if !options.is_empty() {
            candidates.push(Candidate { position, options });
        }
        previous.insert(position, coordinate);
    }
    Ok((previous, candidates))
}

/// Return the alternatives of `others` the choice at `position` of a
/// component counted in closed form may move to: those whose relaxed count
/// is not zero.
///
/// # Errors
///
/// Returns what [`admits_completion`] returns.
fn find_completable_alternatives(
    space: &Space,
    tree: &Tree,
    position: usize,
    others: &Options,
) -> Result<Options, TraceError> {
    let mut listed = Vec::new();
    for option in others.iter() {
        let Coordinate::Index(index) = option else {
            continue;
        };
        let Ok(alternative) = usize::try_from(index) else {
            continue;
        };
        if admits_completion(space, tree, position, alternative)? {
            listed.push(option);
        }
    }
    Ok(Options::Listed(listed))
}

/// The completion search of one component no closed form counts: runs from
/// `empty`, the space's empty configuration, over the decisions `includes`
/// holds, each decision before the changed one answered with its
/// coordinate in `previous`.
struct ComponentSearch<'a, 'c, F> {
    empty: &'a Configuration,
    includes: F,
    previous: &'a HashMap<usize, Coordinate>,
    context: &'a ParamContext<'c>,
}

impl<F: Fn(usize) -> bool> ComponentSearch<'_, '_, F> {
    /// Return the coordinates of `others` the decision at `target` may move
    /// to: those a complete configuration of the component takes.
    ///
    /// # Errors
    ///
    /// Returns what a run returns other than a dead end.
    fn find_options(&self, target: usize, others: &Options) -> Result<Options, TraceError> {
        let mut listed = Vec::new();
        for option in others.iter() {
            if self.completes(target, &option)? {
                listed.push(option);
            }
        }
        Ok(Options::Listed(listed))
    }

    /// Return whether a complete configuration of the component takes
    /// `option` at `target`, searching at most [`COMPLETION_BUDGET`] runs
    /// and taking an unfinished search as a yes.
    ///
    /// # Errors
    ///
    /// Returns what a run returns other than a dead end.
    fn completes(&self, target: usize, option: &Coordinate) -> Result<bool, TraceError> {
        let mut oracle = PinnedOracle {
            previous: self.previous,
            target: (target, option),
            rest: ExhaustiveOracle::new(),
        };
        for _ in 0..COMPLETION_BUDGET {
            match walk_from_empty(self.empty, &self.includes, &mut oracle, self.context) {
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
}

/// Answers the changed decision with its new coordinate, each decision
/// before it with its old one, and every other step exhaustively.
struct PinnedOracle<'a> {
    previous: &'a HashMap<usize, Coordinate>,
    target: (usize, &'a Coordinate),
    rest: ExhaustiveOracle,
}

impl SearchOracle for PinnedOracle<'_> {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        let position = step.decision_position();
        if position == Some(self.target.0) {
            return Ok(self.target.1.clone());
        }
        match position.and_then(|position| self.previous.get(&position)) {
            Some(coordinate) => Ok(coordinate.clone()),
            None => self.rest.decide(step),
        }
    }
}

/// Answers a mutation's re-walk: the mutated decision with its new
/// coordinate, every other decision with its old one while admissible,
/// and the rest uniformly.
struct RepairOracle<'a> {
    target: (usize, Coordinate),
    previous: &'a HashMap<usize, Coordinate>,
    rng: &'a mut Rng,
}

impl SearchOracle for RepairOracle<'_> {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        let position = step.decision_position();
        if position == Some(self.target.0) {
            return Ok(self.target.1.clone());
        }
        let previous = position.and_then(|position| self.previous.get(&position));
        Ok(answer_first_admissible(
            step,
            previous.map(Cow::Borrowed),
            self.rng,
        )?)
    }
}

/// The number of parents a crossover picks between.
const PARENTS: NonZeroU64 = NonZeroU64::new(2).expect("two is positive");

/// Return the crossover of `parents`, as [`Space::crossover`] documents.
fn cross_configurations(
    space: &Space,
    parents: [&Configuration; 2],
    rng: &mut Rng,
    context: &ParamContext<'_>,
    attempts: NonZeroU32,
) -> Result<Recorded, TraceError> {
    if parents.iter().any(|parent| parent.space() != space) {
        return Err(TraceError::OtherSpace);
    }
    let empty = Configuration::empty(space);
    retry_after_dead_ends(attempts, || {
        let picks: Vec<bool> = (0..space.decision_count())
            .map(|_| rng.below(PARENTS) == 1)
            .collect();
        let mut oracle = CrossoverOracle {
            parents,
            picks: &picks,
            rng: &mut *rng,
        };
        walk_from_empty(&empty, |_| true, &mut oracle, context)
    })
}

/// Answers a crossover's walk: each decision with the value of the parent
/// its pick names (`true` for the second), else the other parent's, when
/// that parent assigns it and it is admissible, and otherwise uniformly.
struct CrossoverOracle<'a> {
    parents: [&'a Configuration; 2],
    /// Per decision, by canonical position, whether it inherits from the
    /// second parent.
    picks: &'a [bool],
    rng: &'a mut Rng,
}

impl SearchOracle for CrossoverOracle<'_> {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        let [first, second] = self.parents;
        let parents = match step.decision_position() {
            Some(position) if self.picks[position] => [Some(second), Some(first)],
            Some(_) => [Some(first), Some(second)],
            None => [None, None],
        };
        let inherited = parents.into_iter().flatten().filter_map(|parent| {
            let value = parent.value(step.subject())?;
            step.domain().coordinate_of(value).map(Cow::Owned)
        });
        Ok(answer_first_admissible(step, inherited, self.rng)?)
    }
}
