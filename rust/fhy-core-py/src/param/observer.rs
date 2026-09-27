//! The log records of the param procedures (D-S16-7): the core reports each
//! [`ParamEvent`], and the observer here logs it with the logger, level,
//! text and Python `repr`s the Python implementation logged; and it judges
//! which evaluation failures are undecided (D-S16-8).

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyFrozenSet, PyTuple};

use fhy_core::constraint::{
    Binding, Bindings, Constraint, ConstraintError, ConstraintEvent, Member,
};
use fhy_core::param::{ParamEvent, ParamObserver, ScreenReason};
use fhy_core::solver::QueryKind;

use crate::constraint::{
    DEBUG, LoggingObserver, WARNING, core_logger, join_items, log, member_to_python,
    record_pending_error, repr_text, system_logger,
};
use crate::solver::{is_pass_execution_failure, warn_hazard, warn_unknown};

use super::objects::{
    bindings_to_python, bound_identifiers_text, constraint_class_name, constraint_repr,
    identifier_object,
};

/// Return `fhy_core.symbolic.param.domains`'s logger.
fn domains_logger(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    static LOGGER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    LOGGER
        .get_or_try_init(py, || {
            py.import(intern!(py, "fhy_core.logger"))?
                .call_method1(
                    intern!(py, "get_logger"),
                    ("fhy_core.symbolic.param.domains",),
                )
                .map(Bound::unbind)
        })
        .map(|logger| logger.bind(py).clone())
}

/// Return whether `logger` handles `level`.
fn is_enabled(logger: &Bound<'_, PyAny>, level: u8) -> PyResult<bool> {
    logger
        .call_method1(intern!(logger.py(), "isEnabledFor"), (level,))?
        .is_truthy()
}

/// Return the `repr` of the Python object of `identifier`.
fn identifier_repr(identifier: &fhy_core::identifier::Identifier) -> String {
    format!("{identifier:?}")
}

/// Return the members `members` as `format_comma_separated_list` joins
/// their Python values.
fn join_members(py: Python<'_>, members: &[Member]) -> PyResult<String> {
    let values = members
        .iter()
        .map(|member| member_to_python(py, member))
        .collect::<PyResult<Vec<_>>>()?;
    join_items(PyTuple::new(py, values)?.as_any())
}

/// Return the `repr` of the tuple of `members`' Python values.
fn members_tuple_repr(py: Python<'_>, members: &[Member]) -> PyResult<String> {
    let values = members
        .iter()
        .map(|member| member_to_python(py, member))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(repr_text(PyTuple::new(py, values)?.as_any()))
}

/// Return the name of the solver function a question of `kind` stands
/// for, which its warnings name.
fn question_name(kind: QueryKind) -> &'static str {
    match kind {
        QueryKind::Implication => "does_expression_imply",
        _ => "check_expression_satisfiability",
    }
}

/// Return the level a constraint's `event` is logged at, if any.
fn member_event_level(event: &ConstraintEvent<'_>) -> Option<u8> {
    match event {
        ConstraintEvent::Unbound { .. } | ConstraintEvent::SymbolicBinding { .. } => Some(DEBUG),
        ConstraintEvent::BoundNativeConstants { .. } => Some(WARNING),
        ConstraintEvent::Residual {
            has_free_identifiers,
            ..
        } => Some(if *has_free_identifiers {
            DEBUG
        } else {
            WARNING
        }),
        _ => None,
    }
}

/// Log the event `event` of `constraint` evaluated under `bindings`, as the
/// constraint's own observer logs it.
fn log_member_event(
    py: Python<'_>,
    constraint: &Constraint,
    bindings: &Bindings,
    event: &ConstraintEvent<'_>,
) -> PyResult<()> {
    let Some(level) = member_event_level(event) else {
        return Ok(());
    };
    if !is_enabled(&core_logger(py)?, level)? {
        return Ok(());
    }
    let mapping = bindings_to_python(py, bindings)?;
    let (variable, bound) = match constraint {
        Constraint::Set(set) => {
            let variable = identifier_object(py, set.variable())?;
            let bound = match bindings.get(set.variable()) {
                Some(Binding::Value(_) | Binding::Expression(_)) => {
                    mapping.get_item(&variable)?.map(Bound::unbind)
                }
                None => None,
            };
            (Some(variable.unbind()), bound)
        }
        _ => (None, None),
    };
    LoggingObserver::new(
        constraint_class_name(py, constraint),
        variable,
        mapping.into_any().unbind(),
        bound,
    )
    .log_event(py, event)
}

/// Return the text of the WARNING of a constraint screening dropped or
/// narrowed.
fn screened_message(
    py: Python<'_>,
    constraint: &Constraint,
    variable: &str,
    reason: &ScreenReason<'_>,
) -> PyResult<String> {
    let rendered = constraint_repr(py, constraint);
    Ok(match reason {
        ScreenReason::DependentScope => {
            // A screened constraint is an equation, whose scope cannot fail.
            let mut scope: Vec<_> = constraint
                .free_identifiers()
                .unwrap_or_default()
                .into_iter()
                .collect();
            scope.sort_by_key(fhy_core::identifier::Identifier::id);
            let objects = scope
                .iter()
                .map(|identifier| identifier_object(py, identifier))
                .collect::<PyResult<Vec<_>>>()?;
            let scope = repr_text(PyFrozenSet::new(py, &objects)?.as_any());
            format!(
                "_build_screened_constraint_system: excluding dependent constraint {rendered} \
                 from the screened system for variable {variable}; its scope {scope} reaches \
                 beyond {variable}."
            )
        }
        ScreenReason::ForeignVariable => {
            let scoped = match constraint {
                Constraint::Set(set) => identifier_repr(set.variable()),
                _ => String::new(),
            };
            format!(
                "_build_screened_constraint_system: excluding {rendered} from the screened \
                 system for variable {variable}; it is scoped to {scoped} instead."
            )
        }
        ScreenReason::UnliftableMember(error) => format!(
            "_build_screened_constraint_system: excluding {rendered} for variable {variable}; \
             it does not lift to an expression ({error})."
        ),
        ScreenReason::NoLiftableMember => format!(
            "_build_screened_constraint_system: excluding {rendered} for variable {variable}; \
             none of its members lift to an expression."
        ),
        ScreenReason::Narrowed { liftable, excluded } => format!(
            "_build_screened_constraint_system: narrowing {rendered} for variable {variable} to \
             its liftable member(s) {}; excluded non-liftable member(s) {}.",
            members_tuple_repr(py, liftable)?,
            members_tuple_repr(py, excluded)?
        ),
        _ => format!("_build_screened_constraint_system: excluding {rendered}."),
    })
}

/// Log the event `event` of a procedure on the domains' logger.
fn log_domain_event(py: Python<'_>, event: &ParamEvent<'_>) -> PyResult<()> {
    let warning =
        |message: String| -> PyResult<()> { log(&domains_logger(py)?, WARNING, || message) };
    match *event {
        ParamEvent::EnumerationUndecided {
            variable,
            candidates,
        } => warning(format!(
            "_decide_feasibility_by_enumeration: equation constraints could not decide \
                 candidate(s) {} for variable {} and decided none feasible; reporting UNDECIDED.",
            join_members(py, candidates)?,
            identifier_repr(variable)
        )),
        ParamEvent::SubsetEnumerationUndecided {
            own,
            other,
            candidates,
        } => warning(format!(
            "_decide_subset_by_enumerating_own: candidate(s) {} of variable {} could not be \
                 decided against variable {} on both sides; reporting UNDECIDED.",
            join_members(py, candidates)?,
            identifier_repr(own),
            identifier_repr(other)
        )),
        ParamEvent::SatisfiedOnInexactSystem { variable } => warning(format!(
            "_numeric_has_feasible_value: the solver's SATISFIED answer for variable {} \
                 rests on constraints screening dropped or narrowed; reporting UNDECIDED.",
            identifier_repr(variable)
        )),
        ParamEvent::ViolatedUnderKindConflation { variable } => warning(format!(
            "_numeric_has_feasible_value: the solver's VIOLATED answer for variable {} rests \
                 on a not-in-set constraint whose float member the REAL sort conflates with \
                 another kind denoting the same number; reporting UNDECIDED.",
            identifier_repr(variable)
        )),
        ParamEvent::SatisfiabilityUndecided { variable } => warning(format!(
            "_numeric_has_feasible_value: the solver could not decide satisfiability for \
                 variable {}; reporting UNDECIDED.",
            identifier_repr(variable)
        )),
        ParamEvent::ImplicationUndecided { own, other } => warning(format!(
            "compute_constraint_implication_subset: the solver could not decide whether {} \
                 implies {}; reporting UNDECIDED.",
            identifier_repr(own),
            identifier_repr(other)
        )),
        ParamEvent::ImplicationDowngraded {
            outcome,
            own,
            other,
        } => warning(format!(
            "compute_constraint_implication_subset: the solver's {} answer to whether {} \
                 implies {} rests on constraints screening dropped or narrowed, or on a \
                 REAL-sort member Z3 conflates with another kind; reporting UNDECIDED.",
            match outcome {
                fhy_core::constraint::Outcome::Satisfied => "SATISFIED",
                fhy_core::constraint::Outcome::Violated => "VIOLATED",
                fhy_core::constraint::Outcome::Undecided => "UNDECIDED",
            },
            identifier_repr(own),
            identifier_repr(other)
        )),
        ParamEvent::WitnessOutside {
            variable,
            permitted,
        } => log(&domains_logger(py)?, DEBUG, || {
            format!(
                "_does_own_admit_a_value_outside: {} provably admits a value outside the \
                     {permitted} value(s) the other side permits.",
                identifier_repr(variable)
            )
        }),
        _ => Ok(()),
    }
}

/// The observer of one param question.
pub(super) struct PyParamObserver {
    /// The name of the solver's SMT backend, for the `unknown` warning.
    backend: String,
}

impl PyParamObserver {
    /// Return the observer of a question of a solver whose SMT backend is
    /// named `backend`.
    pub(super) fn new(backend: String) -> Self {
        Self { backend }
    }

    /// Log `event`.
    fn log_event(&self, py: Python<'_>, event: &ParamEvent<'_>) -> PyResult<()> {
        match *event {
            ParamEvent::Member {
                constraint,
                bindings,
                event,
            } => log_member_event(py, constraint, bindings, event),
            ParamEvent::UndecidedMember { constraint } => log(&system_logger(py)?, DEBUG, || {
                format!(
                    "ConstraintSystem.evaluate_with_bindings: member {} is undecided under \
                         the given bindings; the conjunction reports UNDECIDED unless a later \
                         member is violated",
                    constraint_repr(py, constraint)
                )
            }),
            ParamEvent::BridgeFailed {
                constraint,
                bindings,
                ..
            } => log(&domains_logger(py)?, WARNING, || {
                format!(
                    "evaluate_system_outcome: the expression bridge could not evaluate {} under \
                     bindings for {}; reporting UNDECIDED.",
                    constraint_repr(py, constraint),
                    bound_identifiers_text(bindings)
                )
            }),
            ParamEvent::Question {
                symbol_types,
                event,
                ..
            } => match *event {
                ConstraintEvent::Refused { kind, hazard } => {
                    warn_hazard(py, question_name(kind), hazard, symbol_types)
                }
                ConstraintEvent::GaveUp { kind, reason } => {
                    warn_unknown(py, question_name(kind), &self.backend, reason)
                }
                _ => Ok(()),
            },
            ParamEvent::Screened {
                constraint,
                variable,
                reason,
            } => {
                let logger = domains_logger(py)?;
                if !is_enabled(&logger, WARNING)? {
                    return Ok(());
                }
                let message =
                    screened_message(py, constraint, &identifier_repr(variable), &reason)?;
                log(&logger, WARNING, || message)
            }
            _ => log_domain_event(py, event),
        }
    }
}

impl ParamObserver for PyParamObserver {
    fn notify(&self, event: &ParamEvent<'_>) {
        Python::attach(|py| {
            if let Err(error) = self.log_event(py, event) {
                record_pending_error(error);
            }
        });
    }

    /// Judge undecided only a failure Python raises as
    /// `PassExecutionError`, as `evaluate_system_outcome` caught it.
    fn is_undecidable(&self, error: &ConstraintError) -> bool {
        match error {
            ConstraintError::Solve(error) => {
                Python::attach(|py| is_pass_execution_failure(py, error))
            }
            _ => false,
        }
    }
}
