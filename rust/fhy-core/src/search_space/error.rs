//! Errors of building spaces and configurations and of comparing them.

use std::error::Error;
use std::fmt;

use crate::constraint::{ConstraintError, Value};
use crate::foreign::BoxError;
use crate::identifier::Identifier;
use crate::param::AssignmentError;

use super::choice::MAX_CHOICE_DEPTH;
use super::configuration::Activity;
use super::domain::Coordinate;

/// Why a [`Space`](super::Space), a [`Choice`](super::Choice) or a
/// [`PlainAlternative`](super::PlainAlternative) cannot be built.
#[derive(Debug)]
#[non_exhaustive]
pub enum SpaceError {
    /// A name occurs twice among the names a space or one of its parts
    /// holds.
    DuplicateName {
        /// The name.
        name: Identifier,
    },
    /// A choice has no alternative.
    EmptyChoice {
        /// The choice's name.
        choice: Identifier,
    },
    /// A choice's sub-choices make it nest more than
    /// [`MAX_CHOICE_DEPTH`](super::MAX_CHOICE_DEPTH) levels of choices.
    ChoiceTooDeep {
        /// The choice's name.
        choice: Identifier,
    },
    /// A condition's target is not a decision of the space.
    UnknownConditionTarget {
        /// The target.
        target: Identifier,
    },
    /// A condition or a forbidden clause names an identifier that is not a
    /// decision of the space.
    UnknownReference {
        /// The identifier.
        name: Identifier,
    },
    /// An equation of a condition or a forbidden clause names a choice,
    /// which they name only in set constraints.
    EquationOverChoice {
        /// The choice's name.
        choice: Identifier,
    },
    /// A condition names no decision: its constraints name no identifier,
    /// so nothing decides when its target is active.
    EmptyCondition {
        /// The condition's target.
        target: Identifier,
    },
    /// A condition names its target or a decision under the target's
    /// alternatives.
    ConditionReferencesSubtree {
        /// The condition's target.
        target: Identifier,
        /// The decision it names.
        name: Identifier,
    },
    /// A forbidden clause names no decision.
    EmptyForbidden {
        /// The clause's position, in the order given.
        index: usize,
    },
    /// Decisions depend on each other in a cycle, each through its choice
    /// or its condition.
    CyclicDependency {
        /// The decisions on the cycle, from the one first in canonical
        /// order, each depending on the one before it, and the first on the
        /// last.
        cycle: Vec<Identifier>,
    },
    /// An alternative's [`bound_identifiers`](super::Alternative::bound_identifiers)
    /// failed.
    Hook {
        /// The alternative's name.
        alternative: Identifier,
        /// The implementation's error.
        source: BoxError,
    },
    /// A custom constraint's scope or key failed.
    Constraint(ConstraintError),
}

impl fmt::Display for SpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateName { name } => write!(f, "the name {name:?} is used more than once"),
            Self::EmptyChoice { choice } => write!(f, "the choice {choice:?} has no alternative"),
            Self::ChoiceTooDeep { choice } => write!(
                f,
                "the choice {choice:?} nests choices more than {MAX_CHOICE_DEPTH} levels deep"
            ),
            Self::UnknownConditionTarget { target } => {
                write!(
                    f,
                    "the condition's target {target:?} is not a decision of the space"
                )
            }
            Self::UnknownReference { name } => {
                write!(f, "{name:?} is not a decision of the space")
            }
            Self::EquationOverChoice { choice } => write!(
                f,
                "an equation names the choice {choice:?}, which conditions and forbidden \
                 clauses name only in set constraints"
            ),
            Self::EmptyCondition { target } => {
                write!(f, "the condition on {target:?} names no decision")
            }
            Self::ConditionReferencesSubtree { target, name } => write!(
                f,
                "the condition on {target:?} names {name:?}, which is the target or under it"
            ),
            Self::EmptyForbidden { index } => {
                write!(f, "the forbidden clause {index} names no decision")
            }
            Self::CyclicDependency { cycle } => {
                f.write_str("the decisions ")?;
                for (position, name) in cycle.iter().enumerate() {
                    if position > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{name:?}")?;
                }
                f.write_str(" depend on each other in a cycle")
            }
            Self::Hook { alternative, .. } => write!(
                f,
                "the bound identifiers of the alternative {alternative:?} failed"
            ),
            Self::Constraint(_) => f.write_str("a custom constraint failed"),
        }
    }
}

impl Error for SpaceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Hook { source, .. } => Some(&**source),
            Self::Constraint(error) => Some(error),
            _ => None,
        }
    }
}

/// One problem of a configuration's entries, found while building it.
#[derive(Debug)]
#[non_exhaustive]
pub enum ConfigurationError {
    /// An entry names no decision of the space.
    UnknownDecision {
        /// The name.
        name: Identifier,
    },
    /// An entry names a decision an earlier entry named.
    DuplicateEntry {
        /// The decision's name.
        name: Identifier,
    },
    /// An entry gives a value to a decision that is inactive or pending.
    InactiveDecision {
        /// The decision's name.
        name: Identifier,
    },
    /// A choice's value is not the identifier value of one of its
    /// alternatives' names.
    UnknownAlternative {
        /// The choice's name.
        choice: Identifier,
        /// The value.
        value: Value,
    },
    /// A variable's value cannot be assigned to its param.
    Assignment {
        /// The variable's name.
        variable: Identifier,
        /// Why, as [`ParamAssignment::new`](crate::param::ParamAssignment::new)
        /// refused it.
        error: AssignmentError,
    },
    /// A forbidden clause applies and holds.
    Forbidden {
        /// The clause's position in [`Space::forbidden`](super::Space::forbidden).
        index: usize,
    },
    /// A condition evaluated to undecided.
    UndecidedCondition {
        /// The condition's target.
        target: Identifier,
    },
    /// A forbidden clause that applies evaluated to undecided.
    UndecidedForbidden {
        /// The clause's position in [`Space::forbidden`](super::Space::forbidden).
        index: usize,
    },
    /// A condition failed to evaluate.
    FailedCondition {
        /// The condition's target.
        target: Identifier,
        /// The constraint's error.
        error: ConstraintError,
    },
    /// A forbidden clause that applies failed to evaluate.
    FailedForbidden {
        /// The clause's position in [`Space::forbidden`](super::Space::forbidden).
        index: usize,
        /// The constraint's error.
        error: ConstraintError,
    },
}

impl fmt::Display for ConfigurationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownDecision { name } => {
                write!(f, "{name:?} is not a decision of the space")
            }
            Self::DuplicateEntry { name } => {
                write!(f, "the decision {name:?} is given more than one value")
            }
            Self::InactiveDecision { name } => {
                write!(
                    f,
                    "the decision {name:?} is given a value but is not active"
                )
            }
            Self::UnknownAlternative { choice, value } => {
                write!(f, "the choice {choice:?} has no alternative {value}")
            }
            Self::Assignment { variable, .. } => write!(
                f,
                "the value of the variable {variable:?} cannot be assigned to its param"
            ),
            Self::Forbidden { index } => {
                write!(
                    f,
                    "the configuration takes the forbidden combination {index}"
                )
            }
            Self::UndecidedCondition { target } => {
                write!(f, "the condition on {target:?} could not be decided")
            }
            Self::UndecidedForbidden { index } => {
                write!(f, "the forbidden clause {index} could not be decided")
            }
            Self::FailedCondition { target, .. } => {
                write!(f, "the condition on {target:?} failed to evaluate")
            }
            Self::FailedForbidden { index, .. } => {
                write!(f, "the forbidden clause {index} failed to evaluate")
            }
        }
    }
}

impl Error for ConfigurationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Assignment { error, .. } => Some(error),
            Self::FailedCondition { error, .. } | Self::FailedForbidden { error, .. } => {
                Some(error)
            }
            _ => None,
        }
    }
}

/// Every problem found building a [`Configuration`](super::Configuration),
/// at least one, in the order the type documents.
///
/// Displays as `the configuration is invalid: ` and each problem, joined by
/// `; `.
#[derive(Debug)]
pub struct ConfigurationErrors(Vec<ConfigurationError>);

impl ConfigurationErrors {
    /// Return the problems, which `problems` must not be empty of.
    pub(super) fn new(problems: Vec<ConfigurationError>) -> Self {
        Self(problems)
    }

    /// Return the problems, in the order found.
    #[must_use]
    pub fn errors(&self) -> &[ConfigurationError] {
        &self.0
    }
}

impl fmt::Display for ConfigurationErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the configuration is invalid: ")?;
        for (position, problem) in self.0.iter().enumerate() {
            if position > 0 {
                f.write_str("; ")?;
            }
            write!(f, "{problem}")?;
        }
        Ok(())
    }
}

impl Error for ConfigurationErrors {}

/// Why comparing spaces, choices, alternatives, variables or
/// configurations failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum EquivalenceError {
    /// A custom constraint failed.
    Constraint(ConstraintError),
    /// An implementation's hook failed, a comparison hook or
    /// [`bound_identifiers`](super::Alternative::bound_identifiers).
    Extension(BoxError),
}

impl fmt::Display for EquivalenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constraint(_) => f.write_str("a custom constraint failed during the comparison"),
            Self::Extension(_) => {
                f.write_str("an implementation's hook failed during the comparison")
            }
        }
    }
}

impl Error for EquivalenceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Constraint(error) => Some(error),
            Self::Extension(error) => Some(&**error),
        }
    }
}

/// The name of a [`DecisionKind`](super::DecisionKind) is empty.
///
/// Displays as `a decision kind needs a name`.
#[expect(
    clippy::exhaustive_structs,
    reason = "a unit error with nothing to add"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyKind;

impl fmt::Display for EmptyKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a decision kind needs a name")
    }
}

impl Error for EmptyKind {}

/// Why a step's domain cannot be built.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum StepDomainError {
    /// A choice domain has no value.
    EmptyChoice,
    /// An order domain has no element.
    EmptyOrder,
    /// A strided domain has no run.
    EmptyRuns,
    /// A value or element is, or holds, a NaN float.
    NanValue {
        /// Its position, in the order given.
        index: usize,
    },
    /// Two values or elements are equal.
    RepeatedValue {
        /// The first one's position.
        first: usize,
        /// The second one's position.
        second: usize,
    },
    /// A strided run's stop is not above its start.
    EmptyRun,
    /// A strided run's stride is zero.
    ZeroStride,
    /// A strided run starts below the previous run's stop.
    UnorderedRuns {
        /// The run's position, in the order given.
        index: usize,
    },
    /// The domain holds more values than its coordinates can number.
    TooLarge,
}

impl fmt::Display for StepDomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyChoice => f.write_str("a choice domain needs at least one value"),
            Self::EmptyOrder => f.write_str("an order domain needs at least one element"),
            Self::EmptyRuns => f.write_str("a strided domain needs at least one run"),
            Self::NanValue { index } => write!(f, "the value at {index} is or holds a NaN"),
            Self::RepeatedValue { first, second } => {
                write!(f, "the values at {first} and {second} are equal")
            }
            Self::EmptyRun => f.write_str("a strided run's stop must be above its start"),
            Self::ZeroStride => f.write_str("a strided run's stride must be at least 1"),
            Self::UnorderedRuns { index } => {
                write!(f, "the run at {index} starts below the previous run's stop")
            }
            Self::TooLarge => f.write_str("the domain holds more values than coordinates number"),
        }
    }
}

impl Error for StepDomainError {}

/// Why a run of a search stopped: building a step, asking it, checking the
/// answer, or growing the run's configuration.
#[derive(Debug)]
#[non_exhaustive]
pub enum TraceError {
    /// A step's domain cannot be built.
    Domain(StepDomainError),
    /// A static step names no decision of the space.
    UnknownDecision {
        /// The name.
        name: Identifier,
    },
    /// A static step asks a decision the run decided already.
    AlreadyDecided {
        /// The decision's name.
        name: Identifier,
    },
    /// A static step asks a decision that is not active in the run's
    /// configuration so far.
    NotActive {
        /// The decision's name.
        name: Identifier,
        /// Its activity.
        activity: Activity,
    },
    /// A static step asks a variable whose domain is not finite: an
    /// unbounded integer, a real, or a custom domain its variable offers no
    /// [`search_domain`](super::Variable::search_domain) for.
    NotEnumerable {
        /// The variable's name.
        decision: Identifier,
    },
    /// A static step was asked of a recorder over no space.
    NoSpace,
    /// The oracle's answer names no value of the step's domain.
    CoordinateOutOfDomain {
        /// The step's position in the run.
        position: usize,
    },
    /// The oracle's answer names a value the run's configuration refuses.
    Inadmissible {
        /// The step's position in the run.
        position: usize,
        /// The answer.
        coordinate: Coordinate,
    },
    /// A step has no admissible value, or its variable's param admits
    /// none: an integer domain whose bounds enclose no integer.
    DeadEnd {
        /// The decision's name, or the dynamic step's subject.
        decision: Identifier,
    },
    /// The oracle failed.
    Oracle {
        /// The step's position in the run.
        position: usize,
        /// The oracle's error.
        source: BoxError,
    },
    /// A variable's [`search_domain`](super::Variable::search_domain)
    /// failed.
    Hook {
        /// The variable's name.
        decision: Identifier,
        /// The implementation's error.
        source: BoxError,
    },
    /// The run's configuration refused a value for a reason other than its
    /// admissibility: an undecided or failing condition or clause.
    ///
    /// Transparent: it writes the errors' text and forwards their source.
    Configuration(ConfigurationErrors),
    /// A realizing recorder finished before every decision its
    /// configuration assigns was asked.
    Unasked {
        /// Those decisions, in canonical order.
        decisions: Vec<Identifier>,
    },
    /// The configuration to mutate is not complete.
    Incomplete,
    /// The configuration to mutate is of another space.
    OtherSpace,
    /// The configuration to mutate has no decision with a second admissible
    /// value.
    NothingToMutate,
    /// A sampler or a mutation was refused as many times as it may try.
    AttemptsExhausted {
        /// The attempts made.
        attempts: u32,
    },
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Domain(_) => f.write_str("the step's domain is invalid"),
            Self::UnknownDecision { name } => write!(f, "{name:?} is not a decision of the space"),
            Self::AlreadyDecided { name } => {
                write!(f, "the decision {name:?} was decided already in this run")
            }
            Self::NotActive { name, activity } => {
                write!(f, "the decision {name:?} is {activity:?}, not active")
            }
            Self::NotEnumerable { decision } => {
                write!(
                    f,
                    "the variable {decision:?} has no finite domain to search"
                )
            }
            Self::NoSpace => f.write_str("a static step needs a recorder over a space"),
            Self::CoordinateOutOfDomain { position } => {
                write!(
                    f,
                    "the answer to step {position} names no value of its domain"
                )
            }
            Self::Inadmissible { position, .. } => {
                write!(f, "the answer to step {position} is not admissible")
            }
            Self::DeadEnd { decision } => {
                write!(f, "the step for {decision:?} has no admissible value")
            }
            Self::Oracle { position, .. } => write!(f, "the oracle failed at step {position}"),
            Self::Hook { decision, .. } => {
                write!(f, "the search domain of the variable {decision:?} failed")
            }
            Self::Configuration(errors) => write!(f, "{errors}"),
            Self::Unasked { decisions } => {
                f.write_str("the run never asked the assigned decisions ")?;
                for (position, name) in decisions.iter().enumerate() {
                    if position > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{name:?}")?;
                }
                Ok(())
            }
            Self::Incomplete => f.write_str("the configuration to mutate is not complete"),
            Self::OtherSpace => f.write_str("the configuration is of another space"),
            Self::NothingToMutate => {
                f.write_str("no decision of the configuration has another admissible value")
            }
            Self::AttemptsExhausted { attempts } => {
                write!(f, "every one of the {attempts} attempts was refused")
            }
        }
    }
}

impl Error for TraceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Domain(error) => Some(error),
            Self::Oracle { source, .. } | Self::Hook { source, .. } => Some(&**source),
            Self::Configuration(errors) => errors.source(),
            _ => None,
        }
    }
}

/// Why a replay does not describe the run replaying it.
#[derive(Debug)]
#[non_exhaustive]
pub enum ReplayError {
    /// The run asked a step past the trace's last.
    Exhausted {
        /// The step's position in the run.
        position: usize,
    },
    /// The step asked and the step recorded differ in kind.
    KindMismatch {
        /// The step's position.
        position: usize,
    },
    /// The step asked and the step recorded are not both dynamic, or are
    /// static over different decisions; or a trace holds a step for a
    /// decision that is inactive in the configuration it describes.
    DecisionMismatch {
        /// The step's position.
        position: usize,
    },
    /// The domain offered and the domain recorded have different
    /// signatures.
    DomainMismatch {
        /// The step's position.
        position: usize,
    },
    /// The recorded coordinate names no value of the domain offered.
    CoordinateOutOfDomain {
        /// The step's position.
        position: usize,
    },
    /// The recorded coordinate names a value the run's configuration
    /// refuses.
    Inadmissible {
        /// The step's position.
        position: usize,
    },
    /// The run ended before the step at `position` was asked.
    Unconsumed {
        /// The first unasked step's position.
        position: usize,
    },
    /// The trace holds no step for an active decision.
    MissingStep {
        /// The decision's name.
        decision: Identifier,
    },
    /// The trace holds two steps for one decision.
    RepeatedStep {
        /// The decision's name.
        decision: Identifier,
    },
    /// The configuration the steps describe is refused.
    ///
    /// Transparent: it writes the errors' text and forwards their source.
    Configuration(ConfigurationErrors),
    /// Building a step failed.
    ///
    /// Transparent: it writes the error's text and forwards its source.
    Trace(Box<TraceError>),
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhausted { position } => {
                write!(f, "the trace has no step {position} for the run to replay")
            }
            Self::KindMismatch { position } => {
                write!(
                    f,
                    "step {position} is of another kind than the one recorded"
                )
            }
            Self::DecisionMismatch { position } => {
                write!(
                    f,
                    "step {position} asks another decision than the one recorded"
                )
            }
            Self::DomainMismatch { position } => write!(
                f,
                "step {position} is offered another domain than the one recorded"
            ),
            Self::CoordinateOutOfDomain { position } => write!(
                f,
                "the recorded answer to step {position} names no value of the domain offered"
            ),
            Self::Inadmissible { position } => write!(
                f,
                "the recorded answer to step {position} is not admissible in this run"
            ),
            Self::Unconsumed { position } => {
                write!(
                    f,
                    "the run ended before asking the recorded step {position}"
                )
            }
            Self::MissingStep { decision } => {
                write!(f, "the trace holds no step for the decision {decision:?}")
            }
            Self::RepeatedStep { decision } => {
                write!(f, "the trace holds two steps for the decision {decision:?}")
            }
            Self::Configuration(errors) => write!(f, "{errors}"),
            Self::Trace(error) => write!(f, "{error}"),
        }
    }
}

impl Error for ReplayError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Configuration(errors) => errors.source(),
            Self::Trace(error) => error.source(),
            _ => None,
        }
    }
}

/// Why an objective or a measurement cannot be built, or two measurements
/// cannot be compared.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MeasurementError {
    /// An objective's name is empty.
    EmptyName,
    /// A successful measurement holds no value.
    NoValues,
    /// A measurement holds two values for one objective name.
    RepeatedObjective {
        /// The name.
        name: String,
    },
    /// A value is a NaN or infinite.
    NonFiniteValue {
        /// The objective's name.
        objective: String,
    },
    /// A measurement that did not succeed holds values: read from a
    /// payload, since no constructor builds one.
    UnexpectedValues,
    /// `dominates` was asked of a measurement that is not successful.
    NotOk,
    /// `dominates` was asked of measurements over different objectives.
    DifferentObjectives,
}

impl fmt::Display for MeasurementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => f.write_str("an objective needs a name"),
            Self::NoValues => f.write_str("a successful measurement needs at least one value"),
            Self::RepeatedObjective { name } => {
                write!(f, "the objective {name:?} has two values")
            }
            Self::NonFiniteValue { objective } => {
                write!(f, "the value of the objective {objective:?} is not finite")
            }
            Self::UnexpectedValues => {
                f.write_str("a measurement that did not succeed holds no values")
            }
            Self::NotOk => f.write_str("only successful measurements are compared"),
            Self::DifferentObjectives => {
                f.write_str("the measurements are over different objectives")
            }
        }
    }
}

impl Error for MeasurementError {}
