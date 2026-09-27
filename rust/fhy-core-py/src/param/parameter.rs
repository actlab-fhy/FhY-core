//! `fhy_core._rs.Param` and `ParamAssignment`: the bases of the public
//! classes of `fhy_core.symbolic.param.core` (P2; D-S16-9), over the core's
//! [`Param`] and [`ParamAssignment`].
//!
//! A param keeps the Python objects of its domain, its variable and its
//! `ConstraintSystem`, whose members are the constraint objects it was
//! given, and new public-class objects of the constraints the core made.
//! Objects the binding builds from core values are built through a seed
//! handed to the public class's `__new__`, so they are not validated again.

use std::sync::{Arc, Mutex, PoisonError};

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyString, PyTuple, PyType};

use fhy_core::constraint::{Binding, Constraint, ConstraintError, Outcome, Value};
use fhy_core::expression::{ExpressionKind, LiteralValue};
use fhy_core::param::{
    BoundSide, Operand, Param, ParamAssignment, ParamContext, ParamDomain, ParamError, ValueCheck,
};
use fhy_core::term::AlphaEquivalence;

use crate::constraint::{
    PythonBindings, ReadBindings, constraint_error_to_py, outcome_to_python, read_binding,
    read_constraint, read_scoped_bindings, repr_text, type_name, with_pending_errors,
};
use crate::dataclass::OptionalArgument;
use crate::expression::{
    PyExpression, coerce_to_expression, read_big_int, try_get_native_constant_for_identifier,
};
use crate::frozen::build_frozen_mutation_error;
use crate::identifier::{
    deserialize_identifier, identifier_to_python, new_python_identifier, restore_identifier,
    serialize_identifier,
};
use crate::serialization::{
    construct_from_decoded_fields, deserialization_value_error_class, is_serialized_dict,
};
use crate::term::read_renaming;

use super::domains::run_with_context;
use super::error::{param_error, param_error_to_py};
use super::objects::{constraint_to_python, read_domain};
use super::value::{read_candidate, to_tuple};

/// The module of the public param classes.
const CORE: &str = "fhy_core.symbolic.param.core";

/// Return the public class `name` of `module`, imported once into `cell`.
fn import_class<'py>(
    py: Python<'py>,
    cell: &'static PyOnceLock<Py<PyType>>,
    module: &str,
    name: &str,
) -> PyResult<&'py Bound<'py, PyType>> {
    cell.import(py, module, name)
}

/// Return the public `Param` class.
fn param_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    import_class(py, &CLASS, CORE, "Param")
}

/// Return the public `ParamAssignment` class.
fn assignment_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    import_class(py, &CLASS, CORE, "ParamAssignment")
}

/// Return the public `ConstraintSystem` class.
fn system_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    import_class(
        py,
        &CLASS,
        "fhy_core.symbolic.constraint.system",
        "ConstraintSystem",
    )
}

/// Return `fhy_core.symbolic.param.domains.ParamDomain`.
fn domain_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    import_class(py, &CLASS, "fhy_core.symbolic.param.domains", "ParamDomain")
}

/// The Python objects a param holds beside its core.
struct ParamObjects {
    domain: Py<PyAny>,
    variable: Py<PyAny>,
    system: Py<PyAny>,
    constraints: Py<PyTuple>,
}

/// The state a param object is built from, handed to `__new__` as the
/// private keyword `_seed`.
#[pyclass(frozen, module = "fhy_core._rs", name = "_ParamSeed")]
pub(crate) struct PyParamSeed {
    state: Mutex<Option<(Param, ParamObjects)>>,
}

/// Return the Python object of each of `constraints`: the object of a
/// constraint `known` holds, by identity, and a new object otherwise.
fn constraint_objects<'py>(
    py: Python<'py>,
    constraints: &[Constraint],
    known: &[(Constraint, Bound<'py, PyAny>)],
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    constraints
        .iter()
        .map(|constraint| {
            known
                .iter()
                .find(|(core, _)| Constraint::ptr_eq(core, constraint))
                .map_or_else(
                    || constraint_to_python(py, constraint),
                    |(_, object)| Ok(object.clone()),
                )
        })
        .collect()
}

/// Return the Python `ConstraintSystem` of `param`'s constraints, whose
/// members are the objects `known` holds where it holds them.
fn system_object<'py>(
    py: Python<'py>,
    param: &Param,
    known: &[(Constraint, Bound<'py, PyAny>)],
) -> PyResult<Bound<'py, PyAny>> {
    let objects = constraint_objects(py, param.constraints(), known)?;
    system_class(py)?.call1((PyTuple::new(py, objects)?,))
}

/// Return a new instance of the public `Param` class holding `param`, with
/// the Python objects `domain` and `variable`, and a system whose members
/// are the objects `known` holds where it holds them.
fn build_param_object<'py>(
    py: Python<'py>,
    param: Param,
    domain: &Bound<'py, PyAny>,
    variable: &Bound<'py, PyAny>,
    known: &[(Constraint, Bound<'py, PyAny>)],
) -> PyResult<Bound<'py, PyAny>> {
    let system = system_object(py, &param, known)?;
    instantiate_param(py, param, domain, variable, &system)
}

/// Return a new instance of the public `Param` class holding `param` and
/// the Python objects given.
fn instantiate_param<'py>(
    py: Python<'py>,
    param: Param,
    domain: &Bound<'py, PyAny>,
    variable: &Bound<'py, PyAny>,
    system: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let constraints = system
        .getattr(intern!(py, "constraints"))?
        .cast_into::<PyTuple>()?;
    let objects = ParamObjects {
        domain: domain.clone().unbind(),
        variable: variable.clone().unbind(),
        system: system.clone().unbind(),
        constraints: constraints.unbind(),
    };
    let seed = Py::new(
        py,
        PyParamSeed {
            state: Mutex::new(Some((param, objects))),
        },
    )?;
    let class = param_class(py)?;
    let keywords = PyDict::new(py);
    keywords.set_item(intern!(py, "_seed"), seed)?;
    let instance =
        class.call_method(intern!(py, "__new__"), (class, py.None()), Some(&keywords))?;
    instance.call_method1(intern!(py, "__init__"), (domain, variable, system))?;
    Ok(instance)
}

/// Return the bound literal of the Python bound `value`, lifted as an
/// expression operand is.
fn read_bound_literal(value: &Bound<'_, PyAny>) -> PyResult<LiteralValue> {
    let expression = coerce_to_expression(value)?;
    let node = expression.cast::<PyExpression>()?;
    match node.get().expression().kind() {
        ExpressionKind::Literal(literal) => Ok(literal.clone()),
        _ => Err(PyTypeError::new_err(format!(
            "a bound must be a number, got {}.",
            type_name(value)
        ))),
    }
}

/// Check that the bounds `lower` and `upper` enclose some value in some
/// number system, comparing them exactly.
///
/// Raises `ParamError` for bounds that are reversed or equal with an
/// exclusive side, and `ValueError` for a `str` outside the literal grammar.
#[pyfunction]
pub(crate) fn check_param_bounds_are_ordered(
    lower: &Bound<'_, PyAny>,
    upper: &Bound<'_, PyAny>,
    is_lower_inclusive: bool,
    is_upper_inclusive: bool,
) -> PyResult<()> {
    let py = lower.py();
    let lower = read_bound_literal(lower)?;
    let upper = read_bound_literal(upper)?;
    fhy_core::param::check_bounds_are_ordered(
        &lower,
        &upper,
        is_lower_inclusive,
        is_upper_inclusive,
    )
    .map_err(|error| param_error_to_py(py, error, None))
}

/// A variable ranging over a domain, narrowed by constraints, backed by the
/// core's [`Param`].
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Param")]
pub(crate) struct PyParam {
    core: Param,
    objects: ParamObjects,
}

/// What a param error's message names, besides the param.
struct Site<'a, 'py> {
    /// The constraints' Python objects the operation was given.
    known: &'a [(Constraint, Bound<'py, PyAny>)],
    /// The other operand, if any.
    other: Option<&'a Bound<'py, PyAny>>,
}

impl PyParam {
    /// Return the exception of `error` raised by an operation of `this`.
    fn error_to_py(this: &Bound<'_, Self>, error: ParamError, site: &Site<'_, '_>) -> PyErr {
        let py = this.py();
        let objects = &this.get().objects;
        let variable = objects.variable.bind(py);
        match error {
            ParamError::NativeConstantVariable(_) => native_constant_error(variable),
            ParamError::OutOfScope { constraint, .. } => {
                out_of_scope_error(py, variable, &constraint, site.known)
            }
            ParamError::BindingsBindVariable(_) => param_error(
                py,
                format!(
                    "bindings must not include this parameter's own variable {}; its value is \
                     already supplied as `value`.",
                    repr_text(variable)
                ),
            ),
            ParamError::UnsupportedUnion(_) => PyTypeError::new_err(format!(
                "Union is not supported for domain kind {}.",
                type_name(objects.domain.bind(py))
            )),
            ParamError::UnsupportedOperand => PyTypeError::new_err(format!(
                "Unsupported operand type: {}",
                site.other.map_or_else(
                    || "?".to_owned(),
                    |other| repr_text(other.get_type().as_any())
                )
            )),
            ParamError::NonBoundOperand(cause) => {
                let refused = PyTypeError::new_err(ParamError::NonBoundOperand(None).to_string());
                if let Some(cause) = cause {
                    refused.set_cause(py, Some(constraint_error_to_py(py, cause, None)));
                }
                refused
            }
            other => param_error_to_py(py, other, site.other),
        }
    }

    /// Return the `ParamError` of a value check that failed.
    fn value_error(this: &Bound<'_, Self>, value: &Bound<'_, PyAny>, check: ValueCheck) -> PyErr {
        let py = this.py();
        let value = repr_text(value);
        let param = repr_text(this.as_any());
        let member = |member: usize| {
            this.get()
                .objects
                .constraints
                .bind(py)
                .get_item(member)
                .map_or_else(|_| "?".to_owned(), |constraint| repr_text(&constraint))
        };
        param_error(
            py,
            match check {
                ValueCheck::Violated { member: index } => format!(
                    "Value {value} violates constraint {} for parameter {param}.",
                    member(index)
                ),
                ValueCheck::Undecided { member: index } => format!(
                    "Value {value} could not be verified against constraint {} for parameter \
                     {param}.",
                    member(index)
                ),
                _ => format!("Value {value} is not admissible for parameter {param}."),
            },
        )
    }

    /// Return a param over this one's domain and variable, the core
    /// `param`, with a system whose members are the objects of `known`.
    fn rebuilt<'py>(
        this: &Bound<'py, Self>,
        param: Param,
        known: &[(Constraint, Bound<'py, PyAny>)],
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = this.py();
        if Param::ptr_eq(&param, &this.get().core) {
            return Ok(this.clone().into_any());
        }
        let objects = &this.get().objects;
        let mut known = known.to_vec();
        known.extend(this_known(this));
        build_param_object(
            py,
            param,
            objects.domain.bind(py),
            objects.variable.bind(py),
            &known,
        )
    }

    /// Return a fresh param holding the core `param`, over a new variable
    /// object and domain object, with the objects of `known`.
    fn fresh<'py>(
        py: Python<'py>,
        param: Param,
        known: &[(Constraint, Bound<'py, PyAny>)],
    ) -> PyResult<Bound<'py, PyAny>> {
        let domain = super::objects::domain_to_python(py, param.domain())?;
        let variable = identifier_to_python(py, param.variable())?;
        build_param_object(py, param, &domain, &variable, known)
    }

    /// Return the normalized Python value of `value`, as the domain
    /// normalizes it.
    fn normalize<'py>(
        this: &Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = this.py();
        match this.get().core.domain() {
            ParamDomain::Permutation(_) => {
                if value.is_instance_of::<PyString>() || value.try_iter().is_err() {
                    Ok(value.clone())
                } else {
                    to_tuple(value)
                }
            }
            ParamDomain::Custom(_) => this
                .get()
                .objects
                .domain
                .bind(py)
                .call_method1(intern!(py, "normalize_value"), (value,)),
            _ => Ok(value.clone()),
        }
    }

    /// Refuse `bindings` that bind this param's own variable.
    fn validate_bindings(
        this: &Bound<'_, Self>,
        bindings: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let py = this.py();
        let Some(bindings) = bindings.filter(|bindings| !bindings.is_none()) else {
            return Ok(());
        };
        if bindings.contains(this.get().objects.variable.bind(py))? {
            return Err(Self::error_to_py(
                this,
                ParamError::BindingsBindVariable(this.get().core.variable().clone()),
                &Site {
                    known: &[],
                    other: None,
                },
            ));
        }
        Ok(())
    }

    /// Return whether `value` lies in the domain's value set.
    fn admits(this: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = this.py();
        let candidate = read_candidate(value)?;
        with_pending_errors(|| {
            this.get()
                .core
                .domain()
                .is_value_admissible(&candidate)
                .map_err(|error| param_error_to_py(py, error, None))
        })
    }

    /// Evaluate the constraints with the normalized `value` bound to the
    /// variable, then `bindings`, and return the outcome and its member.
    fn evaluate(
        this: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        bindings: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Outcome, Option<usize>)> {
        let py = this.py();
        Self::validate_bindings(this, bindings)?;
        let normalized = Self::normalize(this, value)?;
        let environment = PyDict::new(py);
        environment.set_item(this.get().objects.variable.bind(py), &normalized)?;
        if let Some(bindings) = bindings.filter(|bindings| !bindings.is_none()) {
            for item in bindings.call_method0(intern!(py, "items"))?.try_iter()? {
                let (key, bound) = item?.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()?;
                environment.set_item(key, bound)?;
            }
        }
        let read = read_scoped_bindings(environment.as_any(), None)?;
        let core_bindings = read
            .core
            .clone()
            .with_source(Arc::new(PythonBindings(environment.clone().unbind())));
        let core = this.get().core.clone();
        run_with_context(
            py,
            false,
            |context| {
                core.evaluate_constraints(&core_bindings, context)
                    .map(|evaluation| (evaluation.outcome(), evaluation.deciding_member()))
            },
            |error| evaluation_error(py, error, &read),
        )
    }

    /// Raise unless `value` is a valid assignment under `bindings`.
    fn validate(
        this: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        bindings: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        Self::validate_bindings(this, bindings)?;
        if !Self::admits(this, value)? {
            return Err(Self::value_error(this, value, ValueCheck::Inadmissible));
        }
        match Self::evaluate(this, value, bindings)? {
            (Outcome::Violated, Some(member)) => Err(Self::value_error(
                this,
                value,
                ValueCheck::Violated { member },
            )),
            (Outcome::Undecided, Some(member)) => Err(Self::value_error(
                this,
                value,
                ValueCheck::Undecided { member },
            )),
            _ => Ok(()),
        }
    }

    /// Return the operand of a Python value of interval arithmetic: a
    /// param, or an `int` that is no `bool`; `None` for any other value.
    fn read_operand(other: &Bound<'_, PyAny>) -> PyResult<Option<Operand>> {
        if let Ok(param) = other.cast::<Self>() {
            return Ok(Some(Operand::Param(param.get().core.clone())));
        }
        if other.is_instance_of::<pyo3::types::PyInt>()
            && !other.is_instance_of::<pyo3::types::PyBool>()
        {
            return Ok(Some(Operand::Integer(read_big_int(other)?)));
        }
        Ok(None)
    }

    /// Run the arithmetic `operation` with `other`, answering
    /// `NotImplemented` where it declines: for a value that is no operand,
    /// unless this param is an interval operand, which refuses it.
    fn arithmetic<'py>(
        this: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
        operation: impl FnOnce(&Param, &Operand, &ParamContext<'_>) -> Result<Option<Param>, ParamError>
        + Send,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = this.py();
        let Some(operand) = Self::read_operand(other)? else {
            let is_operand = this
                .get()
                .core
                .domain()
                .interval_profile()
                .map_err(|error| param_error_to_py(py, error, None))?
                .is_some_and(|profile| profile.admits_only_bounds);
            if is_operand {
                return Err(PyTypeError::new_err(format!(
                    "Unsupported operand type: {}",
                    repr_text(other.get_type().as_any())
                )));
            }
            return Ok(py.NotImplemented().into_bound(py));
        };
        let core = this.get().core.clone();
        let result = run_with_context(
            py,
            false,
            |context| operation(&core, &operand, context),
            |error| {
                Self::error_to_py(
                    this,
                    error,
                    &Site {
                        known: &[],
                        other: Some(other),
                    },
                )
            },
        )?;
        match result {
            Some(param) => Self::fresh(py, param, &[]),
            None => Ok(py.NotImplemented().into_bound(py)),
        }
    }
}

/// Return the pairs of this param's core constraints and their objects.
fn this_known<'py>(this: &Bound<'py, PyParam>) -> Vec<(Constraint, Bound<'py, PyAny>)> {
    let py = this.py();
    let objects = this.get().objects.constraints.bind(py);
    this.get()
        .core
        .constraints()
        .iter()
        .cloned()
        .zip(objects.iter())
        .collect()
}

/// Return the `ParamError` of a native constant used as the variable.
fn native_constant_error(variable: &Bound<'_, PyAny>) -> PyErr {
    let py = variable.py();
    let name = try_get_native_constant_for_identifier(variable)
        .ok()
        .flatten()
        .and_then(|constant| constant.getattr(intern!(py, "name")).ok())
        .map_or_else(|| "?".to_owned(), |name| repr_text(&name));
    param_error(
        py,
        format!(
            "Parameter variable {} is the canonical identifier of the native constant {name}; \
             it names a value, not a variable.",
            repr_text(variable)
        ),
    )
}

/// Return the `ParamError` of a constraint whose scope lacks the variable.
fn out_of_scope_error(
    py: Python<'_>,
    variable: &Bound<'_, PyAny>,
    constraint: &Constraint,
    known: &[(Constraint, Bound<'_, PyAny>)],
) -> PyErr {
    let object = known
        .iter()
        .find(|(core, _)| Constraint::ptr_eq(core, constraint))
        .map(|(_, object)| object.clone())
        .or_else(|| constraint_to_python(py, constraint).ok());
    let (rendered, scope) = object.map_or_else(
        || ("?".to_owned(), "?".to_owned()),
        |object| {
            let scope = object
                .call_method0(intern!(py, "get_free_identifiers"))
                .map_or_else(|_| "?".to_owned(), |scope| repr_text(&scope));
            (repr_text(&object), scope)
        },
    );
    param_error(
        py,
        format!(
            "Constraint scope must include the parameter's variable {}, but got constraint \
             {rendered} with scope {scope}.",
            repr_text(variable)
        ),
    )
}

/// Return the exception of an evaluation's error, naming the Python objects
/// of the binding an unusable-binding error concerns.
fn evaluation_error(py: Python<'_>, error: ParamError, read: &ReadBindings<'_>) -> PyErr {
    match error {
        ParamError::Constraint(error) => {
            let binding = match &error {
                ConstraintError::UnusableBinding { identifier, .. } => read.objects(identifier),
                _ => None,
            };
            constraint_error_to_py(py, error, binding)
        }
        other => param_error_to_py(py, other, None),
    }
}

/// Return the core constraints of the Python constraints `constraints`,
/// with their objects.
fn read_known<'py>(
    constraints: &Bound<'py, PyAny>,
) -> PyResult<Vec<(Constraint, Bound<'py, PyAny>)>> {
    constraints
        .try_iter()?
        .map(|object| {
            let object = object?;
            Ok((read_constraint(&object)?, object))
        })
        .collect()
}

#[pymethods]
impl PyParam {
    /// Create the param over `domain` whose variable is `variable` (a new
    /// `Identifier("param")` by default), narrowed by the members of
    /// `constraint_system`.
    ///
    /// Raises `ParamError` for a native constant's identifier as the
    /// variable, a constraint whose scope lacks the variable, and what the
    /// domain refuses.
    #[new]
    #[pyo3(signature = (domain, variable = OptionalArgument::Omitted, constraint_system = OptionalArgument::Omitted, **kwargs))]
    fn new(
        domain: &Bound<'_, PyAny>,
        variable: OptionalArgument<'_>,
        constraint_system: OptionalArgument<'_>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let py = domain.py();
        if let Some(seed) = read_seed::<PyParamSeed>(kwargs)? {
            let (core, objects) = seed
                .get()
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take()
                .ok_or_else(|| PyRuntimeError::new_err("a param seed is used once"))?;
            return Ok(Self { core, objects });
        }
        let variable = match variable {
            OptionalArgument::Given(variable) if !variable.is_none() => variable,
            _ => new_python_identifier(py, "param")?,
        };
        let known = match constraint_system {
            OptionalArgument::Given(system) if !system.is_none() => {
                read_known(&system.getattr(intern!(py, "constraints"))?)?
            }
            _ => Vec::new(),
        };
        let core_domain = read_domain(domain);
        let identifier = restore_identifier(&variable, "Param", "variable")?;
        let constraints: Vec<Constraint> = known.iter().map(|(core, _)| core.clone()).collect();
        let result = run_with_context(
            py,
            false,
            |context| Param::new(core_domain, identifier, constraints, context),
            |error| match error {
                ParamError::NativeConstantVariable(_) => native_constant_error(&variable),
                ParamError::OutOfScope { constraint, .. } => {
                    out_of_scope_error(py, &variable, &constraint, &known)
                }
                other => param_error_to_py(py, other, None),
            },
        )?;
        let system = system_object(py, &result, &known)?;
        let constraints = system
            .getattr(intern!(py, "constraints"))?
            .cast_into::<PyTuple>()?
            .unbind();
        Ok(Self {
            core: result,
            objects: ParamObjects {
                domain: domain.clone().unbind(),
                variable: variable.unbind(),
                system: system.unbind(),
                constraints,
            },
        })
    }

    /// The domain.
    #[getter]
    fn domain(&self, py: Python<'_>) -> Py<PyAny> {
        self.objects.domain.clone_ref(py)
    }

    /// The variable.
    #[getter]
    fn variable(&self, py: Python<'_>) -> Py<PyAny> {
        self.objects.variable.clone_ref(py)
    }

    /// The constraints, as their `ConstraintSystem`.
    #[getter]
    fn constraint_system(&self, py: Python<'_>) -> Py<PyAny> {
        self.objects.system.clone_ref(py)
    }

    /// The constraints, in canonical order.
    #[getter]
    fn constraints(&self, py: Python<'_>) -> Py<PyTuple> {
        self.objects.constraints.clone_ref(py)
    }

    /// The variable, as an identifier expression.
    #[getter]
    fn variable_expression<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        coerce_to_expression(self.objects.variable.bind(py))
    }

    /// The domain's numeric symbol type, or `None` if non-numeric.
    #[getter]
    fn symbol_type<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.objects
            .domain
            .bind(py)
            .getattr(intern!(py, "symbol_type"))
    }

    /// Return a param with the same domain and variable and `constraints`.
    fn replace_constraints<'py>(
        slf: &Bound<'py, Self>,
        constraints: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let known = read_known(constraints)?;
        let core_constraints: Vec<Constraint> =
            known.iter().map(|(core, _)| core.clone()).collect();
        let core = slf.get().core.clone();
        let result = run_with_context(
            py,
            false,
            |context| core.with_constraints(core_constraints, context),
            |error| {
                Self::error_to_py(
                    slf,
                    error,
                    &Site {
                        known: &known,
                        other: None,
                    },
                )
            },
        )?;
        let objects = &slf.get().objects;
        build_param_object(
            py,
            result,
            objects.domain.bind(py),
            objects.variable.bind(py),
            &known,
        )
    }

    /// Return whether `value` is admissible and provably satisfies every
    /// constraint, under `bindings` of the other identifiers.
    #[pyo3(signature = (value, *, bindings = None))]
    fn is_value_valid(
        slf: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        bindings: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        Self::validate_bindings(slf, bindings)?;
        Ok(Self::admits(slf, value)?
            && Self::evaluate(slf, value, bindings)?.0 == Outcome::Satisfied)
    }

    /// Return whether `value` lies in the domain's value set.
    fn is_value_admissible(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        Self::admits(slf, value)
    }

    /// Return whether `value` provably satisfies every constraint.
    #[pyo3(signature = (value, *, bindings = None))]
    fn is_constraints_satisfied(
        slf: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        bindings: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        Ok(Self::evaluate(slf, value, bindings)?.0 == Outcome::Satisfied)
    }

    /// Raise `ParamError` unless `value` is a valid assignment.
    #[pyo3(signature = (value, *, bindings = None))]
    fn validate_value(
        slf: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        bindings: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        Self::validate(slf, value, bindings)
    }

    /// Return the assignment of `value`, normalized, checked under
    /// `bindings`.
    #[pyo3(signature = (value, *, bindings = None))]
    fn assign<'py>(
        slf: &Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
        bindings: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::validate(slf, value, bindings)?;
        let normalized = Self::normalize(slf, value)?;
        build_assignment(slf, &normalized)
    }

    /// Return whether this param's value set is a subset of `other`'s.
    fn is_value_set_subset(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let other = read_param(other)?;
        with_pending_errors(|| {
            slf.get()
                .core
                .is_value_set_subset(&other)
                .map_err(|error| param_error_to_py(py, error, None))
        })
    }

    /// Decide whether this param's feasible set is a subset of `other`'s.
    fn check_subset<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let other_param = read_param(other)?;
        let core = slf.get().core.clone();
        let outcome = run_with_context(
            py,
            true,
            |context| core.check_subset(&other_param, context),
            |error| param_error_to_py(py, error, Some(other)),
        )?;
        outcome_to_python(py, outcome)
    }

    /// Return whether this param's feasible set is proven within `other`'s.
    fn is_subset(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        Ok(Self::check_subset(slf, other)?.is(&outcome_to_python(py, Outcome::Satisfied)?))
    }

    /// Decide whether some value satisfies the domain and every constraint.
    fn check_feasibility<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let outcome = feasibility(slf)?;
        outcome_to_python(py, outcome)
    }

    /// Return whether this param is proven to admit some value.
    fn is_feasible(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(feasibility(slf)? == Outcome::Satisfied)
    }

    /// Return whether this param is proven to admit no value.
    fn is_empty(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(feasibility(slf)? == Outcome::Violated)
    }

    /// Return a param with `constraint` added, or this param when an
    /// equivalent constraint is present.
    fn add_constraint<'py>(
        slf: &Bound<'py, Self>,
        constraint: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let known = vec![(read_constraint(constraint)?, constraint.clone())];
        let core_constraint = known[0].0.clone();
        let core = slf.get().core.clone();
        let result = run_with_context(
            py,
            false,
            |context| core.with_constraint(core_constraint, context),
            |error| {
                Self::error_to_py(
                    slf,
                    error,
                    &Site {
                        known: &known,
                        other: None,
                    },
                )
            },
        )?;
        Self::rebuilt(slf, result, &known)
    }

    /// Return a param with each of `constraints` added, in order.
    fn add_constraints<'py>(
        slf: &Bound<'py, Self>,
        constraints: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let known = read_known(constraints)?;
        let core_constraints: Vec<Constraint> =
            known.iter().map(|(core, _)| core.clone()).collect();
        let core = slf.get().core.clone();
        let result = run_with_context(
            py,
            false,
            |context| {
                core_constraints
                    .into_iter()
                    .try_fold(core.clone(), |param, constraint| {
                        param.with_constraint(constraint, context)
                    })
            },
            |error| {
                Self::error_to_py(
                    slf,
                    error,
                    &Site {
                        known: &known,
                        other: None,
                    },
                )
            },
        )?;
        Self::rebuilt(slf, result, &known)
    }

    /// Raise unless `constraint` can be added: its scope must hold the
    /// variable, and the domain must allow it.
    fn validate_constraint(slf: &Bound<'_, Self>, constraint: &Bound<'_, PyAny>) -> PyResult<()> {
        let known = vec![(read_constraint(constraint)?, constraint.clone())];
        slf.get()
            .core
            .validate_constraint(&known[0].0)
            .map_err(|error| {
                Self::error_to_py(
                    slf,
                    error,
                    &Site {
                        known: &known,
                        other: None,
                    },
                )
            })
    }

    /// Return a param with the lower bound `lower_bound` added.
    #[pyo3(signature = (lower_bound, *, is_inclusive = true))]
    fn add_lower_bound_constraint<'py>(
        slf: &Bound<'py, Self>,
        lower_bound: &Bound<'py, PyAny>,
        is_inclusive: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        add_bound(slf, lower_bound, BoundSide::Lower, is_inclusive)
    }

    /// Return a param with the upper bound `upper_bound` added.
    #[pyo3(signature = (upper_bound, *, is_inclusive = true))]
    fn add_upper_bound_constraint<'py>(
        slf: &Bound<'py, Self>,
        upper_bound: &Bound<'py, PyAny>,
        is_inclusive: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        add_bound(slf, upper_bound, BoundSide::Upper, is_inclusive)
    }

    fn __add__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::arithmetic(slf, other, |param, operand, context| {
            param.checked_add(operand, context)
        })
    }

    fn __radd__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::__add__(slf, other)
    }

    fn __sub__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::arithmetic(slf, other, |param, operand, context| {
            param.checked_sub(operand, context)
        })
    }

    fn __rsub__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::arithmetic(slf, other, |param, operand, context| {
            param.checked_reverse_sub(operand, context)
        })
    }

    fn __mul__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::arithmetic(slf, other, |param, operand, context| {
            param.checked_mul(operand, context)
        })
    }

    fn __rmul__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::__mul__(slf, other)
    }

    fn __neg__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let core = slf.get().core.clone();
        let result = run_with_context(
            py,
            false,
            |context| core.checked_neg(context),
            |error| {
                Self::error_to_py(
                    slf,
                    error,
                    &Site {
                        known: &[],
                        other: None,
                    },
                )
            },
        )?;
        Self::fresh(py, result, &[])
    }

    fn __or__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if other.cast::<Self>().is_err() {
            return Ok(py.NotImplemented().into_bound(py));
        }
        Self::union(slf, other, None)
    }

    fn __and__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if other.cast::<Self>().is_err() {
            return Ok(py.NotImplemented().into_bound(py));
        }
        Self::intersection(slf, other, None)
    }

    /// Return the param over `name` (a new `Identifier("param")` by
    /// default) admitting exactly the values valid for either param.
    #[pyo3(signature = (other, name = None))]
    fn union<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
        name: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        set_operation(slf, other, name, true)
    }

    /// Return the param over `name` (a new `Identifier("param")` by
    /// default) admitting exactly the values valid for both params.
    #[pyo3(signature = (other, name = None))]
    fn intersection<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
        name: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        set_operation(slf, other, name, false)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let objects = &slf.get().objects;
        let set_repr = objects
            .domain
            .bind(py)
            .call_method0(intern!(py, "render_set_repr"))?
            .str()?
            .to_string();
        let set_repr = if set_repr.is_empty() {
            set_repr
        } else {
            format!("{set_repr}, ")
        };
        Ok(format!(
            "{}({}, {set_repr}constraints={})",
            slf.get_type().name()?,
            repr_text(objects.variable.bind(py)),
            repr_text(objects.constraints.bind(py).as_any())
        ))
    }

    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let objects = &slf.get().objects;
        let set_string = objects
            .domain
            .bind(py)
            .call_method0(intern!(py, "render_set_string"))?
            .str()?
            .to_string();
        let constraints = objects
            .constraints
            .bind(py)
            .iter()
            .map(|constraint| constraint.str().map(|text| text.to_string()))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(format!(
            "{{{} in {set_string} | {}}}",
            objects.variable.bind(py).str()?,
            constraints.join(" /\\ ")
        ))
    }

    /// Return whether `other` is a param of the same class with an
    /// equivalent domain, the same variable and equivalent constraints.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if !slf.get_type().is(other.get_type()) {
            return Ok(false);
        }
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        with_pending_errors(|| Ok(slf.get().core.is_structurally_equivalent(&other.get().core)))
    }

    /// Return whether `other` is a param of the same class equivalent up to
    /// the renaming of the variables, under `renaming`.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let renaming = read_renaming(renaming)?;
        if !slf.get_type().is(other.get_type()) {
            return Ok(false);
        }
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        with_pending_errors(|| {
            Ok(slf
                .get()
                .core
                .is_alpha_equivalent_under(&other.get().core, renaming.get().value().renaming()))
        })
    }

    /// Return whether `other` is alpha-equivalent under no renaming.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if !slf.get_type().is(other.get_type()) {
            return Ok(false);
        }
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        with_pending_errors(|| Ok(slf.get().core.is_alpha_equivalent(&other.get().core)))
    }

    /// Always true: params are immutable.
    #[getter]
    fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: params are always frozen.
    fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: params are always frozen, and mutating one raises.
    fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    /// Pickle as a constructor call of the class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let objects = &slf.get().objects;
        Ok((
            slf.get_type(),
            PyTuple::new(
                py,
                [
                    objects.domain.bind(py),
                    objects.variable.bind(py),
                    objects.system.bind(py),
                ],
            )?,
        ))
    }

    /// Return the payload `{"domain": .., "variable": .., "constraint_system": ..}`.
    fn serialize_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let payload = PyDict::new(py);
        payload.set_item(
            intern!(py, "domain"),
            self.objects
                .domain
                .bind(py)
                .call_method0(intern!(py, "serialize_to_dict"))?,
        )?;
        payload.set_item(
            intern!(py, "variable"),
            serialize_identifier(py, self.core.variable())?,
        )?;
        payload.set_item(
            intern!(py, "constraint_system"),
            self.objects
                .system
                .bind(py)
                .call_method0(intern!(py, "serialize_to_dict"))?,
        )?;
        Ok(payload)
    }

    /// Return the param of a payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        check_structure(
            cls,
            data,
            &[
                ("domain", true),
                ("variable", true),
                ("constraint_system", true),
            ],
        )?;
        let fields = PyDict::new(py);
        fields.set_item(
            intern!(py, "domain"),
            domain_class(py)?.call_method1(
                intern!(py, "deserialize_from_dict"),
                (data.get_item("domain")?,),
            )?,
        )?;
        fields.set_item(
            intern!(py, "variable"),
            deserialize_identifier(&data.get_item("variable")?)?,
        )?;
        fields.set_item(
            intern!(py, "constraint_system"),
            system_class(py)?.call_method1(
                intern!(py, "deserialize_from_dict"),
                (data.get_item("constraint_system")?,),
            )?,
        )?;
        construct_from_decoded_fields(cls, &fields)
    }

    /// Build the param of the decoded fields.
    #[classmethod]
    fn construct_from_fields<'py>(
        cls: &Bound<'py, PyType>,
        fields: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let keywords = PyDict::new(cls.py());
        keywords.update(fields.cast::<pyo3::types::PyMapping>()?)?;
        cls.call((), Some(&keywords))
    }
}

/// Return the feasibility of `this`, asked detached.
fn feasibility(this: &Bound<'_, PyParam>) -> PyResult<Outcome> {
    let py = this.py();
    let core = this.get().core.clone();
    run_with_context(
        py,
        true,
        |context| core.check_feasibility(context),
        |error| param_error_to_py(py, error, None),
    )
}

/// Return the core param of the Python param `other`.
fn read_param(other: &Bound<'_, PyAny>) -> PyResult<Param> {
    other
        .cast::<PyParam>()
        .map(|param| param.get().core.clone())
        .map_err(|_not_a_param| {
            PyTypeError::new_err(format!("expected a Param, got {}.", type_name(other)))
        })
}

/// Return `this` with the bound `bound` of `side` added.
fn add_bound<'py>(
    this: &Bound<'py, PyParam>,
    bound: &Bound<'py, PyAny>,
    side: BoundSide,
    is_inclusive: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let py = this.py();
    let literal = read_bound_literal(bound)?;
    let core = this.get().core.clone();
    let result = run_with_context(
        py,
        false,
        |context| core.with_bound(&literal, side, is_inclusive, context),
        |error| {
            PyParam::error_to_py(
                this,
                error,
                &Site {
                    known: &[],
                    other: None,
                },
            )
        },
    )?;
    PyParam::rebuilt(this, result, &[])
}

/// Return the union or intersection of `this` and `other` over `name`.
fn set_operation<'py>(
    this: &Bound<'py, PyParam>,
    other: &Bound<'py, PyAny>,
    name: Option<&Bound<'py, PyAny>>,
    is_union: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let py = this.py();
    let other_param = read_param(other)?;
    let variable = match name.filter(|name| !name.is_none()) {
        Some(name) => name.clone(),
        None => new_python_identifier(py, "param")?,
    };
    let identifier = restore_identifier(&variable, "Param", "variable")?;
    let core = this.get().core.clone();
    let result = run_with_context(
        py,
        !is_union,
        |context| {
            if is_union {
                core.union(&other_param, identifier, context)
            } else {
                core.intersection(&other_param, identifier, context)
            }
        },
        |error| {
            PyParam::error_to_py(
                this,
                error,
                &Site {
                    known: &[],
                    other: Some(other),
                },
            )
        },
    )?;
    let domain = super::objects::domain_to_python(py, result.domain())?;
    build_param_object(py, result, &domain, &variable, &[])
}

/// Return an assignment of `value`, already checked and normalized, to
/// `param`.
fn build_assignment<'py>(
    param: &Bound<'py, PyParam>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = param.py();
    let core_value = read_assignment_value(value)?;
    let core = ParamAssignment::new_unchecked(param.get().core.clone(), core_value);
    let seed = Py::new(
        py,
        PyAssignmentSeed {
            state: Mutex::new(Some((
                core,
                param.clone().into_any().unbind(),
                value.clone().unbind(),
            ))),
        },
    )?;
    let class = assignment_class(py)?;
    let keywords = PyDict::new(py);
    keywords.set_item(intern!(py, "_seed"), seed)?;
    let instance = class.call_method(
        intern!(py, "__new__"),
        (class, py.None(), py.None()),
        Some(&keywords),
    )?;
    instance.call_method1(intern!(py, "__init__"), (param, value))?;
    Ok(instance)
}

/// Return the core value of an assignment's Python value.
fn read_assignment_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    Ok(match read_binding(value)? {
        Binding::Value(value) => value,
        Binding::Expression(_) => read_candidate(value)?,
    })
}

/// Check that `data` is a mapping holding exactly `fields`, each a payload
/// dict when marked so, as the derived deserialization checks it.
fn check_structure(
    cls: &Bound<'_, PyType>,
    data: &Bound<'_, PyAny>,
    fields: &[(&str, bool)],
) -> PyResult<()> {
    static STRUCTURE_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    let py = cls.py();
    let is_well_formed = match data.cast::<pyo3::types::PyMapping>() {
        Ok(mapping) => {
            mapping.len()? == fields.len()
                && fields.iter().try_fold(
                    true,
                    |is_well_formed, (name, is_payload)| -> PyResult<bool> {
                        if !is_well_formed || !mapping.contains(*name)? {
                            return Ok(false);
                        }
                        Ok(!is_payload || is_serialized_dict(&mapping.get_item(*name)?)?)
                    },
                )?
        }
        Err(_not_a_mapping) => false,
    };
    if is_well_formed {
        return Ok(());
    }
    let expected = PyDict::new(py);
    for (name, is_payload) in fields {
        if *is_payload {
            expected.set_item(*name, py.get_type::<PyDict>())?;
        } else {
            expected.set_item(*name, py.get_type::<PyAny>())?;
        }
    }
    let error = STRUCTURE_ERROR
        .import(
            py,
            "fhy_core.serialization",
            "DeserializationDictStructureError",
        )?
        .call1((cls, expected, data))?;
    Err(PyErr::from_value(error))
}

/// An assignment and the Python objects of its param and value.
type AssignmentState = (ParamAssignment, Py<PyAny>, Py<PyAny>);

/// The state an assignment object is built from, handed to `__new__` as
/// the private keyword `_seed`.
#[pyclass(frozen, module = "fhy_core._rs", name = "_ParamAssignmentSeed")]
pub(crate) struct PyAssignmentSeed {
    state: Mutex<Option<AssignmentState>>,
}

/// Return the seed of type `T` the private keyword `_seed` of `kwargs`
/// holds, if any.
///
/// # Errors
///
/// Raises `TypeError` for any other keyword, or a seed of another type.
fn read_seed<'py, T: pyo3::PyClass>(
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Option<Bound<'py, T>>> {
    let Some(kwargs) = kwargs else {
        return Ok(None);
    };
    let mut seed = None;
    for (key, value) in kwargs.iter() {
        if key.eq("_seed")? {
            seed = Some(value.cast_into::<T>()?);
        } else {
            return Err(PyTypeError::new_err(format!(
                "unexpected keyword argument {}",
                repr_text(&key)
            )));
        }
    }
    Ok(seed)
}

/// A param bound to one value, backed by the core's [`ParamAssignment`].
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "ParamAssignment")]
pub(crate) struct PyParamAssignment {
    core: ParamAssignment,
    param: Py<PyAny>,
    value: Py<PyAny>,
}

#[pymethods]
impl PyParamAssignment {
    /// Create the assignment of `value` to `param`, checked without
    /// bindings and normalized.
    ///
    /// Raises what `Param.validate_value` raises.
    #[new]
    #[pyo3(signature = (param, value, **kwargs))]
    fn new(
        param: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if let Some(seed) = read_seed::<PyAssignmentSeed>(kwargs)? {
            let (core, param, value) = seed
                .get()
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take()
                .ok_or_else(|| PyRuntimeError::new_err("an assignment seed is used once"))?;
            return Ok(Self { core, param, value });
        }
        let param_object = param.cast::<PyParam>().map_err(|_not_a_param| {
            PyTypeError::new_err(format!(
                "ParamAssignment param must be a Param, got {}.",
                type_name(param)
            ))
        })?;
        PyParam::validate(param_object, value, None)?;
        let normalized = PyParam::normalize(param_object, value)?;
        Ok(Self {
            core: ParamAssignment::new_unchecked(
                param_object.get().core.clone(),
                read_assignment_value(&normalized)?,
            ),
            param: param.clone().unbind(),
            value: normalized.unbind(),
        })
    }

    /// The param.
    #[getter]
    fn param(&self, py: Python<'_>) -> Py<PyAny> {
        self.param.clone_ref(py)
    }

    /// The value, normalized.
    #[getter]
    fn value(&self, py: Python<'_>) -> Py<PyAny> {
        self.value.clone_ref(py)
    }

    /// Return whether this assignment has a value: always true.
    fn is_value_set(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Return whether `other` assigns an equivalent param an equal value,
    /// type-strictly.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if !slf.get_type().is(other.get_type()) {
            return Ok(false);
        }
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        with_pending_errors(|| Ok(slf.get().core.is_structurally_equivalent(&other.get().core)))
    }

    /// Return whether `other` assigns a param alpha-equivalent under
    /// `renaming` an equal value.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let renaming = read_renaming(renaming)?;
        if !slf.get_type().is(other.get_type()) {
            return Ok(false);
        }
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        with_pending_errors(|| {
            Ok(slf
                .get()
                .core
                .is_alpha_equivalent_under(&other.get().core, renaming.get().value().renaming()))
        })
    }

    /// Return whether `other` is alpha-equivalent under no renaming.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if !slf.get_type().is(other.get_type()) {
            return Ok(false);
        }
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        with_pending_errors(|| Ok(slf.get().core.is_alpha_equivalent(&other.get().core)))
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        crate::dataclass::format_dataclass_repr(
            &slf.get_type(),
            &[
                ("param", this.param.bind(py)),
                ("value", this.value.bind(py)),
            ],
        )
    }

    /// Always true: assignments are immutable.
    #[getter]
    fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: assignments are always frozen.
    fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: assignments are always frozen, and mutating one raises.
    fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    /// Pickle as a call of `_restore`, which does not check the value
    /// again, as bindings that proved it are not part of the assignment.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        Ok((
            slf.get_type().getattr(intern!(py, "_restore"))?,
            PyTuple::new(py, [this.param.bind(py), this.value.bind(py)])?,
        ))
    }

    /// Return the assignment of the checked `value` to `param`, without a
    /// check.
    #[classmethod]
    fn _restore<'py>(
        cls: &Bound<'py, PyType>,
        param: &Bound<'py, PyAny>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = cls;
        let param = param.cast::<PyParam>()?;
        build_assignment(param, value)
    }

    /// Return the payload `{"param": .., "value": ..}`.
    fn serialize_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        static SERIALIZE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
        let serialize = SERIALIZE.import(
            py,
            "fhy_core.serialization",
            "serialize_registry_wrapped_value",
        )?;
        let payload = PyDict::new(py);
        payload.set_item(
            intern!(py, "param"),
            self.param
                .bind(py)
                .call_method0(intern!(py, "serialize_to_dict"))?,
        )?;
        payload.set_item(
            intern!(py, "value"),
            serialize.call1((self.value.bind(py),))?,
        )?;
        Ok(payload)
    }

    /// Return the assignment of a payload, rejecting only a value that is
    /// provably invalid.
    ///
    /// Raises the serialization framework's errors for a malformed payload.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        static DESERIALIZE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
        let py = cls.py();
        check_structure(cls, data, &[("param", true), ("value", false)])?;
        let fields = PyDict::new(py);
        fields.set_item(
            intern!(py, "param"),
            param_class(py)?.call_method1(
                intern!(py, "deserialize_from_dict"),
                (data.get_item("param")?,),
            )?,
        )?;
        let payload = data.get_item("value")?;
        let deserialize = DESERIALIZE.import(
            py,
            "fhy_core.serialization",
            "deserialize_registry_wrapped_value",
        )?;
        let value = match deserialize.call1((&payload,)) {
            Ok(value) => value,
            Err(error)
                if error.is_instance_of::<pyo3::exceptions::PyValueError>(py)
                    || error.is_instance_of::<PyTypeError>(py) =>
            {
                let wrapped = PyErr::from_value(deserialization_value_error_class(py)?.call1((
                    cls,
                    "value",
                    "a decodable value",
                    &payload,
                ))?);
                wrapped.set_cause(py, Some(error));
                return Err(wrapped);
            }
            Err(error) => return Err(error),
        };
        fields.set_item(intern!(py, "value"), value)?;
        construct_from_decoded_fields(cls, &fields)
    }

    /// Rebuild an assignment of decoded fields, rejecting only a value that
    /// is provably invalid: inadmissible, or violating a constraint.
    #[classmethod]
    fn construct_from_fields<'py>(
        cls: &Bound<'py, PyType>,
        fields: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = cls;
        let py = fields.py();
        let param = fields.get_item("param")?;
        let value = fields.get_item("value")?;
        let param = param.cast::<PyParam>()?;
        if !PyParam::admits(param, &value)? {
            return Err(PyParam::value_error(
                param,
                &value,
                ValueCheck::Inadmissible,
            ));
        }
        if let (Outcome::Violated, Some(member)) = PyParam::evaluate(param, &value, None)? {
            return Err(PyParam::value_error(
                param,
                &value,
                ValueCheck::Violated { member },
            ));
        }
        let normalized = PyParam::normalize(param, &value)?;
        let _ = py;
        build_assignment(param, &normalized)
    }
}
