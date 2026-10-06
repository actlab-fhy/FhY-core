//! What a search does over a [`Space`] alone: sampling, per step and
//! uniformly over the complete configurations; replaying a trace into a
//! configuration; enumerating and counting the complete configurations;
//! and mutating one.

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use std::num::NonZeroU32;

use num_bigint::BigUint;

use crate::identifier::Identifier;
use crate::param::ParamContext;

use super::configuration::Configuration;
use super::error::{ReplayError, TraceError};
use super::oracle::{ExhaustiveOracle, SearchOracle};
use super::recorder::Recorded;
use super::rng::Rng;
use super::space::Space;
use super::trace::Trace;

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
        todo!()
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
        todo!()
    }

    /// Return the complete configurations of the space, each once, in
    /// lexicographic order of their coordinates in decision order.
    ///
    /// A decision with no finite domain yields one
    /// [`TraceError::NotEnumerable`] and ends the enumeration.
    #[must_use]
    pub fn enumerate<'s, 'c>(&'s self, context: &'s ParamContext<'c>) -> Enumeration<'s, 'c> {
        todo!()
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
        todo!()
    }

    /// Return `configuration` with one decision changed, the rest repaired,
    /// and its trace.
    ///
    /// It picks uniformly an active decision with at least two admissible
    /// values; gives it a new value (an order's two positions, drawn
    /// uniformly, swapped; another shape's other admissible coordinates
    /// drawn among uniformly); and re-walks decision order keeping every
    /// other decision's value while it stays admissible and drawing every
    /// other step uniformly. A dead end starts again from the pick.
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
        todo!()
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
        todo!()
    }
}
