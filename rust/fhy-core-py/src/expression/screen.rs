//! The Boolean-position screen: `validate_logical_operands` and
//! `validate_predicate` over the Rust [`BooleanScreen`] (D-S4-4).
//!
//! The screen reads the sorts of user constants and of named user functions
//! from a snapshot of the one function registry, the core's
//! [`FunctionRegistry`](fhy_core::expression::registry::FunctionRegistry)
//! the binding keeps (D-S7-6), with no call into Python. Built-in functions
//! and constants are judged by the core's catalogue, since their names and
//! identifiers are reserved (D-9, D-S7-4).

use std::collections::HashMap;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyMapping, PyString, PyType};

use fhy_core::expression::{BooleanScreen, Expression, NonBooleanLogicalOperandError, SymbolType};
use fhy_core::identifier::Identifier;

use crate::error::IntoPyErr;
use crate::identifier::{read_identifier_id, restore_identifier};

use super::node::PyExpression;
use super::text::render_kind_repr;

/// Return `fhy_core.symbolic.expression.errors.NonBooleanLogicalOperandError`.
fn non_boolean_operand_error_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    crate::python::cached_attr!(py, "fhy_core.symbolic.expression.errors", "NonBooleanLogicalOperandError" => PyType)
}

/// Return `NonBooleanLogicalOperandError` with `message`.
pub(crate) fn non_boolean_operand_error(py: Python<'_>, message: String) -> PyErr {
    match non_boolean_operand_error_class(py).and_then(|class| class.call1((message,))) {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

/// Raises `NonBooleanLogicalOperandError` (a `TypeError`) with the core's
/// text, followed by the `repr` of the operand and of the node taking it:
/// `operand 0 of a logical and provably denotes a number but sits in a
/// boolean position: LiteralExpression(2) in LogicalExpression((and 2 4))`.
impl IntoPyErr for NonBooleanLogicalOperandError {
    fn into_py_err(self) -> PyErr {
        let message = match self.parent() {
            Some((parent, _)) => format!(
                "{self}: {} in {}",
                render_kind_repr(self.operand()),
                render_kind_repr(parent)
            ),
            None => format!("{self}: {}", render_kind_repr(self.operand())),
        };
        Python::attach(|py| match non_boolean_operand_error_class(py) {
            Ok(class) => match class.call1((message,)) {
                Ok(error) => PyErr::from_value(error),
                Err(error) => error,
            },
            Err(error) => error,
        })
    }
}

/// The arguments of a screen, converted.
struct ScreenArguments {
    expression: Expression,
    environment: HashMap<Identifier, Expression>,
    symbol_types: HashMap<Identifier, SymbolType>,
}

/// Return the screen's arguments, converted.
///
/// # Errors
///
/// Raises `TypeError` for an expression, or a bound value, that is not an
/// `Expression`, and for a declared sort that is not a `SymbolType`.
fn read_screen_arguments<'py>(
    expression: &Bound<'py, PyAny>,
    environment: Option<&Bound<'py, PyAny>>,
    symbol_types: Option<&Bound<'py, PyAny>>,
) -> PyResult<ScreenArguments> {
    let root = expression
        .cast::<PyExpression>()
        .map_err(|_not_an_expression| {
            PyTypeError::new_err(format!(
                "expression must be an Expression, got {}.",
                expression
                    .get_type()
                    .name()
                    .map_or_else(|_| "?".to_owned(), |name| name.to_string())
            ))
        })?;
    let mut rust_environment = HashMap::new();
    if let Some(environment) = environment.filter(|environment| !environment.is_none()) {
        for item in environment.cast::<PyMapping>()?.items()?.iter() {
            let (key, value) = item.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()?;
            if read_identifier_id(&key)?.is_none() {
                continue;
            }
            let bound = value.cast::<PyExpression>().map_err(|_not_an_expression| {
                PyTypeError::new_err(format!(
                    "environment values must be Expressions, got {} for {}.",
                    value
                        .get_type()
                        .name()
                        .map_or_else(|_| "?".to_owned(), |name| name.to_string()),
                    key.repr()
                        .map_or_else(|_| "?".to_owned(), |text| text.to_string())
                ))
            })?;
            rust_environment.insert(
                restore_identifier(&key, "environment", "key")?,
                bound.get().expression().clone(),
            );
        }
    }
    let mut rust_symbol_types = HashMap::new();
    if let Some(symbol_types) = symbol_types.filter(|symbol_types| !symbol_types.is_none()) {
        for item in symbol_types.cast::<PyMapping>()?.items()?.iter() {
            let (key, value) = item.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()?;
            if read_identifier_id(&key)?.is_none() {
                continue;
            }
            let symbol_type = value
                .cast::<PyString>()
                .ok()
                .and_then(|text| text.to_str().ok()?.parse::<SymbolType>().ok())
                .ok_or_else(|| {
                    PyTypeError::new_err(format!(
                        "symbol_types values must be SymbolTypes, got {}.",
                        value
                            .repr()
                            .map_or_else(|_| "?".to_owned(), |text| text.to_string())
                    ))
                })?;
            rust_symbol_types.insert(
                restore_identifier(&key, "symbol_types", "key")?,
                symbol_type,
            );
        }
    }
    Ok(ScreenArguments {
        expression: root.get().expression().clone(),
        environment: rust_environment,
        symbol_types: rust_symbol_types,
    })
}

/// Screen `arguments` over a snapshot of the registry, as a predicate when
/// `is_predicate` is set, raising the refusal.
fn run_screen(arguments: &ScreenArguments, is_predicate: bool) -> PyResult<()> {
    let registry = super::registry::snapshot();
    let screen = BooleanScreen::new()
        .with_sorts(registry.registry())
        .with_environment(&arguments.environment)
        .with_symbol_types(&arguments.symbol_types);
    let result = if is_predicate {
        screen.check_predicate(&arguments.expression)
    } else {
        screen.check_logical_operands(&arguments.expression)
    };
    result.map_err(IntoPyErr::into_py_err)
}

/// Raise unless every Boolean position in `expression` holds a Boolean: an
/// operand of a logical not, and, or, a piecewise case condition, and the
/// case values and otherwise branch of a piecewise in a Boolean position.
///
/// `environment` maps identifiers to the expressions bound to them, each
/// screened in the identifier's place without applying another binding;
/// `symbol_types` declares the sorts of identifiers left free. A native
/// constant's canonical identifier has its registered sort, whatever it is
/// bound to, and a call its callee's result sort.
///
/// Raises `NonBooleanLogicalOperandError` for an operand that provably
/// denotes a number, naming its position, the operand and the node taking
/// it.
#[pyfunction]
#[pyo3(signature = (expression, environment = None, *, symbol_types = None))]
pub(crate) fn validate_logical_operands(
    expression: &Bound<'_, PyAny>,
    environment: Option<&Bound<'_, PyAny>>,
    symbol_types: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let arguments = read_screen_arguments(expression, environment, symbol_types)?;
    run_screen(&arguments, false)
}

/// Raise unless `expression` can be used as a predicate: its root holds a
/// Boolean, and every Boolean position in it does, as
/// `validate_logical_operands` screens them with the root in a Boolean
/// position.
///
/// Raises `NonBooleanLogicalOperandError` for a root, or an operand, that
/// provably denotes a number.
#[pyfunction]
#[pyo3(signature = (expression, environment = None, *, symbol_types = None))]
pub(crate) fn validate_predicate(
    expression: &Bound<'_, PyAny>,
    environment: Option<&Bound<'_, PyAny>>,
    symbol_types: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let arguments = read_screen_arguments(expression, environment, symbol_types)?;
    run_screen(&arguments, true)
}
