//! The values a backend exchanges with a solver: `fhy_core._rs.SmtScript`
//! and `fhy_core._rs.SatResult` (D-S8-11), and the conversions of symbol
//! types and query kinds.

use std::collections::HashMap;
use std::sync::OnceLock;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyMapping, PyString, PyTuple, PyType};

use fhy_core::expression::SymbolType;
use fhy_core::identifier::Identifier;
use fhy_core::solver::{QueryKind, SatResult, SmtScript};

use crate::expression::PyExpression;
use crate::identifier::{identifier_to_python, read_identifier_id, restore_identifier};

use super::error::lowering_error_to_py;

/// Return `fhy_core.symbolic.symbol_type.SymbolType`.
fn symbol_type_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    crate::python::cached_attr!(py, "fhy_core.symbolic.symbol_type", "SymbolType" => PyType)
}

/// Return the Python `SymbolType` member of `symbol_type`.
pub(crate) fn symbol_type_to_python(
    py: Python<'_>,
    symbol_type: SymbolType,
) -> PyResult<Bound<'_, PyAny>> {
    symbol_type_class(py)?.call1((symbol_type.as_str(),))
}

/// Return the name of the Python `SymbolType` member of `symbol_type`, such
/// as `INT`.
pub(super) fn symbol_type_name(symbol_type: SymbolType) -> &'static str {
    match symbol_type {
        SymbolType::Real => "REAL",
        SymbolType::Int => "INT",
        SymbolType::Bool => "BOOL",
    }
}

/// Return the symbol types of the mapping `symbol_types`, an identifier to
/// a `SymbolType` each; a key that is no `Identifier` is ignored.
///
/// # Errors
///
/// Raises `TypeError` for a value that is not a `SymbolType`.
pub(crate) fn read_symbol_types(
    symbol_types: Option<&Bound<'_, PyAny>>,
) -> PyResult<HashMap<Identifier, SymbolType>> {
    let mut read = HashMap::new();
    let Some(symbol_types) = symbol_types.filter(|symbol_types| !symbol_types.is_none()) else {
        return Ok(read);
    };
    for item in symbol_types.cast::<PyMapping>()?.items()?.iter() {
        let (key, value) = item.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()?;
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
        read.insert(
            restore_identifier(&key, "symbol_types", "key")?,
            symbol_type,
        );
    }
    Ok(read)
}

/// Return the query kind `kind` names: a `SolverQueryKind` member or its
/// value.
///
/// # Errors
///
/// Raises `ValueError` for any other value.
pub(super) fn read_query_kind(kind: &Bound<'_, PyAny>) -> PyResult<QueryKind> {
    let text = kind
        .cast::<PyString>()
        .ok()
        .and_then(|text| text.to_str().ok());
    match text {
        Some("simplification") => Ok(QueryKind::Simplification),
        Some("satisfiability") => Ok(QueryKind::Satisfiability),
        Some("implication") => Ok(QueryKind::Implication),
        Some("universal_validity") => Ok(QueryKind::UniversalValidity),
        _ => Err(PyValueError::new_err(format!(
            "{} is not a SolverQueryKind.",
            kind.repr()?
        ))),
    }
}

// ---------------------------------------------------------------------------
// SmtScript
// ---------------------------------------------------------------------------

/// An SMT-LIB2 script: a logic, the declarations of constants, and
/// assertions over them.
///
/// `text` is the whole script, one command per line, without
/// `check-sat`; a backend hands it to its solver. `declarations` pairs each
/// identifier the script declares with its symbol and `SymbolType`.
#[pyclass(frozen, module = "fhy_core._rs", name = "SmtScript")]
pub(crate) struct PySmtScript {
    script: SmtScript,
    text: OnceLock<String>,
    declarations: PyOnceLock<Py<PyTuple>>,
}

impl PySmtScript {
    /// Return the Python script of `script`.
    pub(super) fn new(script: SmtScript) -> Self {
        Self {
            script,
            text: OnceLock::new(),
            declarations: PyOnceLock::new(),
        }
    }

    /// Return the core script.
    pub(super) fn script(&self) -> &SmtScript {
        &self.script
    }

    fn text(&self) -> &str {
        self.text.get_or_init(|| self.script.to_string())
    }
}

#[pymethods]
impl PySmtScript {
    /// Lower `expression` to a script, with the sorts `symbol_types`
    /// declares: a Boolean expression is asserted, and any other is named by
    /// the constant `value`, which `value_sort` then gives the sort of.
    ///
    /// Raises `KeyError` for identifiers `symbol_types` misses,
    /// `NonBooleanLogicalOperandError` for a number in a Boolean position,
    /// `NativeConstantLoweringError` for a native constant, and `TypeError`
    /// for a node with no SMT-LIB2 term: a call, a Boolean meeting a number,
    /// a non-finite float, or a power without an integer literal exponent of
    /// at least one.
    #[staticmethod]
    #[pyo3(signature = (expression, symbol_types = None))]
    fn lower(
        expression: &Bound<'_, PyAny>,
        symbol_types: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let py = expression.py();
        let expression = expression
            .cast::<PyExpression>()
            .map_err(|_not_an_expression| {
                PyTypeError::new_err(format!(
                    "SmtScript.lower expression must be an Expression, got {}.",
                    expression
                        .get_type()
                        .name()
                        .map_or_else(|_| "?".to_owned(), |name| name.to_string())
                ))
            })?
            .get()
            .expression()
            .clone();
        let symbol_types = read_symbol_types(symbol_types)?;
        let registry = crate::expression::registry_snapshot();
        let script = SmtScript::lower(&expression, &symbol_types, registry.registry())
            .map_err(|error| lowering_error_to_py(py, error))?;
        Ok(Self::new(script))
    }

    /// The script's text.
    #[getter(text)]
    fn get_text(&self) -> &str {
        self.text()
    }

    /// The SMT-LIB2 name of the script's logic, such as `QF_LIA`.
    #[getter]
    fn logic(&self) -> &'static str {
        self.script.logic().as_str()
    }

    /// Each declared constant as `(identifier, symbol, sort)`, ordered by
    /// the identifier's id; the script writes the symbol quoted, as
    /// `|x_7|`.
    #[getter]
    fn declarations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.declarations
            .get_or_try_init(py, || {
                let items: Vec<Bound<'py, PyTuple>> = self
                    .script
                    .declarations()
                    .iter()
                    .map(|declaration| {
                        PyTuple::new(
                            py,
                            [
                                identifier_to_python(py, declaration.identifier())?,
                                PyString::new(py, declaration.symbol()).into_any(),
                                symbol_type_to_python(py, declaration.sort())?,
                            ],
                        )
                    })
                    .collect::<PyResult<_>>()?;
                PyTuple::new(py, items).map(Bound::unbind)
            })
            .map(|tuple| tuple.bind(py).clone())
    }

    /// The sort of the expression the script names as `value`, or `None`
    /// when its assertions are predicates.
    #[getter]
    fn value_sort<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.script
            .value_sort()
            .map(|sort| symbol_type_to_python(py, sort))
            .transpose()
    }

    fn __str__(&self) -> &str {
        self.text()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "SmtScript(logic={}, text={})",
            PyString::new(py, self.logic()).repr()?,
            PyString::new(py, self.text()).repr()?
        ))
    }
}

// ---------------------------------------------------------------------------
// SatResult
// ---------------------------------------------------------------------------

/// Return `fhy_core.symbolic.solver.SatStatus`.
fn sat_status_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    crate::python::cached_attr!(py, "fhy_core.symbolic.solver", "SatStatus" => PyType)
}

/// The answer of a `check-sat`: `SatResult.SAT`, `SatResult.UNSAT`, or
/// `SatResult.unknown(reason)`.
///
/// Compares and hashes by value, and pickles as the attribute or the call
/// that builds it.
#[pyclass(frozen, eq, hash, module = "fhy_core._rs", name = "SatResult")]
#[derive(PartialEq, Eq, Hash)]
pub(crate) struct PySatResult {
    result: SatResult,
}

impl From<SatResult> for PySatResult {
    fn from(result: SatResult) -> Self {
        Self { result }
    }
}

impl PySatResult {
    /// Return the core result.
    pub(super) fn result(&self) -> &SatResult {
        &self.result
    }
}

#[pymethods]
impl PySatResult {
    /// The assertions can hold together.
    #[classattr]
    #[pyo3(name = "SAT")]
    fn sat() -> Self {
        Self {
            result: SatResult::Sat,
        }
    }

    /// The assertions cannot hold together.
    #[classattr]
    #[pyo3(name = "UNSAT")]
    fn unsat() -> Self {
        Self {
            result: SatResult::Unsat,
        }
    }

    /// Return the result of a backend that could not decide, for
    /// `reason`, such as `"timeout"`.
    ///
    /// Raises `TypeError` if `reason` is not a `str`.
    #[staticmethod]
    fn unknown(reason: &Bound<'_, PyAny>) -> PyResult<Self> {
        let reason = reason.cast::<PyString>().map_err(|_not_a_str| {
            PyTypeError::new_err(format!(
                "SatResult.unknown reason must be a str, got {}.",
                reason
                    .get_type()
                    .name()
                    .map_or_else(|_| "?".to_owned(), |name| name.to_string())
            ))
        })?;
        Ok(Self {
            result: SatResult::Unknown {
                reason: reason.to_str()?.to_owned(),
            },
        })
    }

    /// The `SatStatus`: `sat`, `unsat` or `unknown`.
    #[getter]
    fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = match self.result {
            SatResult::Sat => "sat",
            SatResult::Unsat => "unsat",
            SatResult::Unknown { .. } => "unknown",
        };
        sat_status_class(py)?.call1((value,))
    }

    /// Why the backend could not decide, or `None` when it decided.
    #[getter]
    fn reason(&self) -> Option<&str> {
        match &self.result {
            SatResult::Unknown { reason } => Some(reason),
            SatResult::Sat | SatResult::Unsat => None,
        }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(match &self.result {
            SatResult::Sat => "SatResult.SAT".to_owned(),
            SatResult::Unsat => "SatResult.UNSAT".to_owned(),
            SatResult::Unknown { reason } => {
                format!("SatResult.unknown({})", PyString::new(py, reason).repr()?)
            }
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let class = slf.get_type();
        match &slf.get().result {
            SatResult::Unknown { reason } => PyTuple::new(
                py,
                [
                    class.getattr(intern!(py, "unknown"))?,
                    PyTuple::new(py, [reason])?.into_any(),
                ],
            ),
            decided => {
                let name = if *decided == SatResult::Sat {
                    "SAT"
                } else {
                    "UNSAT"
                };
                let getattr = py
                    .import(intern!(py, "builtins"))?
                    .getattr(intern!(py, "getattr"))?;
                PyTuple::new(
                    py,
                    [
                        getattr,
                        PyTuple::new(py, [class.into_any(), PyString::new(py, name).into_any()])?
                            .into_any(),
                    ],
                )
            }
        }
    }
}
