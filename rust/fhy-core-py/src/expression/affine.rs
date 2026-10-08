//! `PyO3` class and function for [`fhy_core::expression::AffineForm`]: the
//! exact affine form of an expression, which
//! `fhy_core.symbolic.expression.passes.affine` exposes.
//!
//! Coefficients and the constant reach Python as `fractions.Fraction`s.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyBool, PyDict, PyType};

use fhy_core::expression::{AffineForm, Rational};

use crate::identifier::{identifier_to_python, restore_identifier};
use crate::util::dataclass::hash_value;
use crate::util::python::read_type_name;

use super::literal::big_int_to_python;
use super::materialize::materialize_expression;
use super::node::PyExpression;

/// Return `fractions.Fraction`.
fn fraction_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    crate::util::python::cached_attr!(py, "fractions", "Fraction" => PyType)
}

/// Return the `fractions.Fraction` of `rational`.
fn rational_to_fraction<'py>(py: Python<'py>, rational: &Rational) -> PyResult<Bound<'py, PyAny>> {
    fraction_class(py)?.call1((
        big_int_to_python(py, rational.numerator())?,
        big_int_to_python(py, rational.denominator())?,
    ))
}

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
        let held = restore_identifier(identifier, "AffineForm", "identifier")?;
        rational_to_fraction(identifier.py(), &self.form.coefficient(&held))
    }

    /// The constant term, a `fractions.Fraction`.
    #[getter]
    fn constant<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        rational_to_fraction(py, self.form.constant())
    }

    /// The terms, a `dict` from each `Identifier` with a non-zero
    /// coefficient to its `fractions.Fraction`, in order of the
    /// identifiers' ids.
    #[getter]
    fn terms<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let terms = PyDict::new(py);
        for (identifier, coefficient) in self.form.terms() {
            terms.set_item(
                identifier_to_python(py, identifier)?,
                rational_to_fraction(py, coefficient)?,
            )?;
        }
        Ok(terms)
    }

    /// Return whether the form has no term.
    fn is_constant(&self) -> bool {
        self.form.is_constant()
    }

    /// Return the form's canonical `Expression`.
    fn to_expression<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        materialize_expression(py, &self.form.to_expression())
    }

    /// Compare structurally with another form; another type is
    /// `NotImplemented`.
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> Py<PyAny> {
        let py = other.py();
        let Ok(other) = other.cast::<Self>() else {
            return py.NotImplemented();
        };
        let equal = self.form == other.get().form;
        match op {
            CompareOp::Eq => PyBool::new(py, equal).to_owned().into_any().unbind(),
            CompareOp::Ne => PyBool::new(py, !equal).to_owned().into_any().unbind(),
            _ => py.NotImplemented(),
        }
    }

    /// Return the form's hash, consistent with `==`.
    fn __hash__(&self) -> u64 {
        hash_value(&self.form)
    }

    /// Return the canonical expression's text.
    fn __str__(&self) -> String {
        self.form.to_string()
    }

    /// Return `AffineForm(<text>)`.
    fn __repr__(&self) -> String {
        format!("AffineForm({})", self.form)
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
    let py = expression.py();
    let root = expression
        .cast::<PyExpression>()
        .map_err(|_not_an_expression| {
            PyTypeError::new_err(format!(
                "expression must be an Expression, got {}.",
                read_type_name(expression)
            ))
        })?;
    root.get()
        .expression()
        .affine_form()
        .map(|form| Ok(Bound::new(py, PyAffineForm { form })?.into_any()))
        .transpose()
}
