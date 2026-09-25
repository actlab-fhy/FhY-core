//! `fhy_core._rs.MatchBindings`: the base of the public `MatchBindings`
//! class, backed by the Rust [`MatchBindings`] (D-S5-3).

use std::collections::HashMap;

use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyTuple, PyType};

use fhy_core::expression::pattern::{Capture, MatchBindings};

use crate::dataclass::hash_value;
use crate::frozen::build_frozen_mutation_error;
use crate::public_class::PublicClass;

use super::capture::PyCapture;
use super::objects::current_bindings_object;

/// The Python capture object of each Rust capture of a pattern.
pub(super) type CaptureObjects = HashMap<Capture, Py<PyCapture>>;

/// The contents of a bindings object the binding builds, handed to the
/// public class's constructor. Not exported, so only the binding builds
/// bindings that bind anything.
#[pyclass(frozen, module = "fhy_core._rs", name = "_MatchBindingsSeed")]
struct MatchBindingsSeed {
    bindings: MatchBindings,
    entries: Py<PyTuple>,
}

/// The captures of a successful match, backed by the Rust
/// [`MatchBindings`]: each bound capture with the expression it matched,
/// in binding order.
///
/// Only a match produces bindings that bind a capture; `MatchBindings()`
/// and `empty()` bind none. The expressions are the node objects the match
/// saw. Two bindings are equal when they bind the same captures, in any
/// order, each to structurally equal expressions; hashing agrees.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "MatchBindings")]
pub(crate) struct PyMatchBindings {
    bindings: MatchBindings,
    /// The `(Capture, Expression)` pairs, in binding order.
    entries: Py<PyTuple>,
}

impl PyMatchBindings {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("MatchBindings");
        &PUBLIC_CLASS
    }

    /// Return the object of the public class for `bindings`, a match of a
    /// pattern whose captures' objects are `captures`, with each bound
    /// node's object from the current table.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` outside a match or walk, or for a capture
    /// missing from `captures`, and what building an object raises.
    pub(super) fn build<'py>(
        py: Python<'py>,
        bindings: &MatchBindings,
        captures: &CaptureObjects,
    ) -> PyResult<Bound<'py, PyAny>> {
        current_bindings_object(py, bindings, |entries| {
            let pairs = entries
                .into_iter()
                .map(|(capture, node)| {
                    let capture = captures.get(capture).ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "capture `{capture}` is bound by no capture pattern of the match"
                        ))
                    })?;
                    PyTuple::new(py, [capture.bind(py).clone().into_any(), node])
                })
                .collect::<PyResult<Vec<_>>>()?;
            let seed = MatchBindingsSeed {
                bindings: bindings.clone(),
                entries: PyTuple::new(py, pairs)?.unbind(),
            };
            Self::public_class().get(py)?.call1((seed,))
        })
    }

    /// Return the pair bound to `capture`, a capture object, if any.
    fn find<'py>(&self, py: Python<'py>, capture: &Bound<'py, PyAny>) -> Option<Bound<'py, PyAny>> {
        self.entries.bind(py).iter().find_map(|pair| {
            let pair = pair.cast_into::<PyTuple>().ok()?;
            let bound = pair.get_item(0).ok()?;
            bound.is(capture).then(|| pair.get_item(1).ok())?
        })
    }
}

#[pymethods]
impl PyMatchBindings {
    /// Create bindings that bind no capture.
    ///
    /// Raises `TypeError` if an argument is given: only a match produces
    /// bindings that bind a capture.
    #[new]
    #[pyo3(signature = (seed = None))]
    fn new(py: Python<'_>, seed: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        match seed {
            None => Ok(Self {
                bindings: MatchBindings::new(),
                entries: PyTuple::empty(py).unbind(),
            }),
            Some(seed) => match seed.cast::<MatchBindingsSeed>() {
                Ok(seed) => {
                    let seed = seed.get();
                    Ok(Self {
                        bindings: seed.bindings.clone(),
                        entries: seed.entries.clone_ref(py),
                    })
                }
                Err(_not_a_seed) => Err(PyTypeError::new_err(
                    "MatchBindings() takes no arguments: only a match produces bindings \
                     that bind a capture",
                )),
            },
        }
    }

    /// Return bindings that bind no capture.
    #[classmethod]
    fn empty<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyAny>> {
        cls.call0()
    }

    /// Return whether no capture is bound.
    fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Return the expression bound to `capture`, or `None` if it is not
    /// bound, or not a capture.
    fn get<'py>(&self, py: Python<'py>, capture: &Bound<'py, PyAny>) -> Option<Bound<'py, PyAny>> {
        self.find(py, capture)
    }

    /// Return whether `capture` is bound.
    fn has(&self, py: Python<'_>, capture: &Bound<'_, PyAny>) -> bool {
        self.find(py, capture).is_some()
    }

    fn __contains__(&self, py: Python<'_>, capture: &Bound<'_, PyAny>) -> bool {
        self.find(py, capture).is_some()
    }

    /// Return the expression bound to `capture`.
    ///
    /// Raises `KeyError` if it is not bound, with the core's text,
    /// ``capture `x` is not bound``.
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        capture: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.find(py, capture).ok_or_else(|| {
            let name = capture
                .str()
                .map_or_else(|_| "?".to_owned(), |name| name.to_string());
            PyKeyError::new_err(format!("capture `{name}` is not bound"))
        })
    }

    /// Return the number of bound captures.
    fn __len__(&self) -> usize {
        self.bindings.len()
    }

    /// Iterate over the bound captures, in binding order.
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let captures = self
            .entries
            .bind(py)
            .iter()
            .map(|pair| pair.get_item(0))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, captures)?
            .as_any()
            .try_iter()
            .map(Bound::into_any)
    }

    /// Return the `(capture, expression)` pairs, in binding order.
    fn items<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.entries.bind(py).clone()
    }

    /// Return whether `other` binds the same captures, in any order, each
    /// to structurally equal expressions; `NotImplemented` for anything
    /// but bindings.
    fn __eq__<'py>(&self, other: &Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        let py = other.py();
        match other.cast::<Self>() {
            Ok(other) => PyBool::new(py, self.bindings == other.get().bindings)
                .to_owned()
                .into_any(),
            Err(_not_bindings) => py.NotImplemented().into_bound(py),
        }
    }

    fn __hash__(&self) -> u64 {
        hash_value(&self.bindings)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let mut text = slf.get_type().qualname()?.to_string();
        text.push_str("({");
        for (index, pair) in slf.get().entries.bind(py).iter().enumerate() {
            if index > 0 {
                text.push_str(", ");
            }
            text.push_str(pair.get_item(0)?.repr()?.to_str()?);
            text.push_str(": ");
            text.push_str(pair.get_item(1)?.repr()?.to_str()?);
        }
        text.push_str("})");
        Ok(text)
    }

    /// Refuse to pickle: only a match produces bindings.
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<()> {
        Err(PyTypeError::new_err(format!(
            "cannot pickle {} objects: only a match produces bindings",
            slf.get_type().qualname()?
        )))
    }

    /// Always true: bindings are immutable.
    #[getter]
    fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: bindings are always frozen.
    fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: bindings are always frozen, and mutating them raises.
    fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    /// Register `cls` as the public class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }
}
