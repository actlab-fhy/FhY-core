//! `PyO3` class and function for [`fhy_core::expression::AffineForm`]: the
//! exact affine form of an expression, which
//! `fhy_core.symbolic.expression.passes.affine` exposes.
//!
//! Coefficients and the constant reach Python as `fractions.Fraction`s.

#![expect(
    dead_code,
    unused_variables,
    clippy::todo,
    reason = "interface stub; bodies are todo!() until implementation"
)]

use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::PyDict;

use fhy_core::expression::AffineForm;

/// An expression as an exact linear combination of its free identifiers
/// plus a constant, backed by the core [`AffineForm`].
///
/// It has no constructor; `affine_form` builds it. It is hashable and
/// compares structurally.
#[pyclass(frozen, module = "fhy_core._rs", name = "AffineForm")]
pub(crate) struct PyAffineForm {
    form: AffineForm,
}

#[pymethods]
impl PyAffineForm {
    /// Return the coefficient of the identifier `identifier`, a
    /// `fractions.Fraction`: zero for one the form does not hold.
    ///
    /// Raises `TypeError` for an argument that is no `Identifier`.
    fn coefficient<'py>(&self, identifier: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// The constant term, a `fractions.Fraction`.
    #[getter]
    fn constant<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// The terms, a `dict` from each `Identifier` with a non-zero
    /// coefficient to its `fractions.Fraction`, in order of the
    /// identifiers' ids.
    #[getter]
    fn terms<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        todo!()
    }

    /// Return whether the form has no term.
    fn is_constant(&self) -> bool {
        todo!()
    }

    /// Return the form's canonical `Expression`.
    fn to_expression<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Compare structurally with another form; another type is
    /// `NotImplemented`.
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// Return the form's hash, consistent with `==`.
    fn __hash__(&self) -> u64 {
        todo!()
    }

    /// Return the canonical expression's text.
    fn __str__(&self) -> String {
        todo!()
    }

    /// Return `AffineForm(<text>)`.
    fn __repr__(&self) -> String {
        todo!()
    }
}

/// Return the exact affine form of the `Expression` `expression` over its
/// free identifiers, or `None` when the analysis cannot prove it affine.
///
/// Raises `TypeError` for an argument that is no `Expression`.
#[pyfunction]
pub(crate) fn affine_form<'py>(
    expression: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    todo!()
}
