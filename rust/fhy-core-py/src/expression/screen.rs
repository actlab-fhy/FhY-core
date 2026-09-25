//! The Boolean-position screen: `validate_logical_operands` and
//! `validate_predicate` over the Rust [`BooleanScreen`] (D-S4-4).
//!
//! The function registry stays Python (`fhy_core.symbolic.expression
//! .registry`), with runtime registration, so the screen learns the sorts
//! of native constants and of named user functions through
//! [`RegistrySorts`], a [`SortLookup`] adapter that asks the registry's
//! public lookups once per identifier and per name. Built-in functions are
//! judged by the core's catalogue, since their names are reserved (D-9).

use std::cell::RefCell;
use std::collections::HashMap;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyMapping, PyString, PyType};

use fhy_core::expression::{
    BooleanScreen, Expression, ExpressionKind, FunctionName, FunctionSort,
    NonBooleanLogicalOperandError, SortLookup, SymbolType,
};
use fhy_core::identifier::Identifier;

use crate::error::IntoPyErr;
use crate::identifier::{identifier_to_python, read_identifier_id, restore_identifier};

use super::node::{PyExpression, PyIdentifierExpression};
use super::text::render_kind_repr;

/// The Python module of the function registry.
const REGISTRY_MODULE: &str = "fhy_core.symbolic.expression.registry";

/// Return `fhy_core.symbolic.expression.errors.NonBooleanLogicalOperandError`.
fn non_boolean_operand_error_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    CLASS.import(
        py,
        "fhy_core.symbolic.expression.errors",
        "NonBooleanLogicalOperandError",
    )
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

/// The sorts the Python registry declares, for the screen.
struct RegistrySorts<'py> {
    py: Python<'py>,
    /// The Python identifiers of the trees screened, by id, for asking the
    /// registry about their Rust identifiers.
    identifiers: HashMap<u64, Bound<'py, PyAny>>,
    constant_sorts: RefCell<HashMap<u64, Option<FunctionSort>>>,
    call_sorts: RefCell<HashMap<String, Option<FunctionSort>>>,
    /// The first error a registry lookup raised, raised after the screen.
    error: RefCell<Option<PyErr>>,
}

impl<'py> RegistrySorts<'py> {
    fn new(py: Python<'py>) -> Self {
        Self {
            py,
            identifiers: HashMap::new(),
            constant_sorts: RefCell::new(HashMap::new()),
            call_sorts: RefCell::new(HashMap::new()),
            error: RefCell::new(None),
        }
    }

    /// Record the Python identifier of every reference in `root`.
    fn collect_identifiers(&mut self, root: &Bound<'py, PyExpression>) -> PyResult<()> {
        let mut pending = vec![root.clone()];
        let mut visited = std::collections::HashSet::new();
        while let Some(node) = pending.pop() {
            if let ExpressionKind::Identifier(identifier) = node.get().expression().kind() {
                if !self.identifiers.contains_key(&identifier.id()) {
                    let leaf = node.cast::<PyIdentifierExpression>()?;
                    self.identifiers
                        .insert(identifier.id(), leaf.get().identifier(self.py).clone());
                }
                continue;
            }
            for child in node.get().children(self.py) {
                if visited.insert(child.as_ptr() as usize) {
                    pending.push(child.cast_into::<PyExpression>()?);
                }
            }
        }
        Ok(())
    }

    /// Record the error `error` unless one is recorded.
    fn record(&self, error: PyErr) {
        let mut slot = self.error.borrow_mut();
        if slot.is_none() {
            *slot = Some(error);
        }
    }

    /// Return the Rust sort of the Python `FunctionSort` member `sort`.
    fn read_sort(sort: &Bound<'_, PyAny>) -> PyResult<Option<FunctionSort>> {
        if sort.is_none() {
            return Ok(None);
        }
        let value = sort.getattr(intern!(sort.py(), "value"))?;
        Ok(value.cast::<PyString>()?.to_str()?.parse().ok())
    }

    /// Return the sort the registry declares for the native constant
    /// `identifier` denotes, or `None` if it denotes none.
    fn look_up_constant(&self, identifier: &Identifier) -> PyResult<Option<FunctionSort>> {
        static FUNCTION: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
        let py = self.py;
        let python_identifier = match self.identifiers.get(&identifier.id()) {
            Some(object) => object.clone(),
            None => identifier_to_python(py, identifier)?,
        };
        let constant = FUNCTION
            .import(
                py,
                REGISTRY_MODULE,
                "try_get_native_constant_for_identifier",
            )?
            .call1((python_identifier,))?;
        if constant.is_none() {
            return Ok(None);
        }
        Self::read_sort(&constant.getattr(intern!(py, "sort"))?)
    }

    /// Return the result sort the registry declares for the entry `name`,
    /// or `None` if none is registered.
    fn look_up_call(&self, name: &str) -> PyResult<Option<FunctionSort>> {
        static FUNCTION: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
        let sort = FUNCTION
            .import(self.py, REGISTRY_MODULE, "try_get_registered_result_sort")?
            .call1((name,))?;
        Self::read_sort(&sort)
    }
}

impl SortLookup for RegistrySorts<'_> {
    fn native_constant_sort(&self, identifier: &Identifier) -> Option<FunctionSort> {
        if let Some(cached) = self.constant_sorts.borrow().get(&identifier.id()) {
            return *cached;
        }
        let sort = self.look_up_constant(identifier).unwrap_or_else(|error| {
            self.record(error);
            None
        });
        self.constant_sorts
            .borrow_mut()
            .insert(identifier.id(), sort);
        sort
    }

    fn call_result_sort(&self, name: &FunctionName) -> Option<FunctionSort> {
        if let Some(cached) = self.call_sorts.borrow().get(name.as_str()) {
            return *cached;
        }
        let sort = self.look_up_call(name.as_str()).unwrap_or_else(|error| {
            self.record(error);
            None
        });
        self.call_sorts
            .borrow_mut()
            .insert(name.as_str().to_owned(), sort);
        sort
    }
}

/// The arguments of a screen, converted.
struct ScreenArguments<'py> {
    expression: Expression,
    environment: HashMap<Identifier, Expression>,
    symbol_types: HashMap<Identifier, SymbolType>,
    sorts: RegistrySorts<'py>,
}

/// Return the screen's arguments, converted, with the Python identifiers
/// of every tree recorded for the registry lookups.
///
/// # Errors
///
/// Raises `TypeError` for an expression, or a bound value, that is not an
/// `Expression`, and for a declared sort that is not a `SymbolType`.
fn read_screen_arguments<'py>(
    expression: &Bound<'py, PyAny>,
    environment: Option<&Bound<'py, PyAny>>,
    symbol_types: Option<&Bound<'py, PyAny>>,
) -> PyResult<ScreenArguments<'py>> {
    let py = expression.py();
    let mut sorts = RegistrySorts::new(py);
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
    sorts.collect_identifiers(root)?;
    let mut rust_environment = HashMap::new();
    if let Some(environment) = environment.filter(|environment| !environment.is_none()) {
        for item in environment.cast::<PyMapping>()?.items()?.iter() {
            let (key, value) = item.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()?;
            let Some(id) = read_identifier_id(&key)? else {
                continue;
            };
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
            sorts.collect_identifiers(bound)?;
            sorts.identifiers.entry(id).or_insert_with(|| key.clone());
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
        sorts,
    })
}

/// Screen `arguments`, as a predicate when `is_predicate` is set, raising
/// the refusal, or the first error a registry lookup raised.
fn run_screen(arguments: &ScreenArguments<'_>, is_predicate: bool) -> PyResult<()> {
    let screen = BooleanScreen::new()
        .with_sorts(&arguments.sorts)
        .with_environment(&arguments.environment)
        .with_symbol_types(&arguments.symbol_types);
    let result = if is_predicate {
        screen.check_predicate(&arguments.expression)
    } else {
        screen.check_logical_operands(&arguments.expression)
    };
    if let Some(error) = arguments.sorts.error.borrow_mut().take() {
        return Err(error);
    }
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
