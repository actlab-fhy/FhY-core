//! Errors of building spaces and configurations and of comparing them.

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use std::error::Error;
use std::fmt;

use crate::constraint::{ConstraintError, Value};
use crate::foreign::BoxError;
use crate::identifier::Identifier;
use crate::param::AssignmentError;

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
        todo!()
    }

    /// Return the problems, in the order found.
    #[must_use]
    pub fn errors(&self) -> &[ConfigurationError] {
        todo!()
    }
}

impl fmt::Display for ConfigurationErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
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
