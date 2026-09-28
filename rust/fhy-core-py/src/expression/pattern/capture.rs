//! `fhy_core._rs.Capture`: the base of the public `Capture` class, backed by
//! the Rust [`Capture`].

use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyString, PyTuple, PyType};

use fhy_core::expression::pattern::Capture;

use crate::dataclass::read_str;
use crate::frozen::build_frozen_mutation_error;
use crate::public_class::PublicClass;

/// A handle a capture pattern binds a matched expression to, backed by the
/// Rust [`Capture`].
///
/// Identity decides equality: a capture equals only itself, and hashes by
/// identity, so two captures made with one name are independent. The name
/// serves only `str`, `repr` and messages, and may be any `str`, the empty
/// one included.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Capture")]
pub(crate) struct PyCapture {
    capture: Capture,
    /// The name, as given.
    #[pyo3(get)]
    name: Py<PyString>,
}

impl PyCapture {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Capture");
        &PUBLIC_CLASS
    }

    /// Return the Rust capture.
    pub(super) const fn capture(&self) -> &Capture {
        &self.capture
    }
}

#[pymethods]
impl PyCapture {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.name)?;
        Ok(())
    }

    /// Create a capture named `name`, distinct from every other capture.
    ///
    /// Raises `TypeError` for a name that is not a `str`.
    #[new]
    fn new(name: &Bound<'_, PyAny>) -> PyResult<Self> {
        let name = read_str(name, "Capture", "name")?;
        Ok(Self {
            capture: Capture::new(name.to_str()?),
            name: name.clone().unbind(),
        })
    }

    /// Return the name.
    fn __str__(&self, py: Python<'_>) -> Py<PyString> {
        self.name.clone_ref(py)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        Ok(format!(
            "{}({})",
            slf.get_type().qualname()?,
            slf.get().name.bind(py).repr()?
        ))
    }

    /// Pickle as a call of the class with the name. A capture shared within
    /// one pickle stays shared, through the pickle memo; each load makes a
    /// new capture.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        Ok((slf.get_type(), PyTuple::new(py, [slf.get().name.bind(py)])?))
    }

    /// Always true: captures are immutable.
    #[getter]
    const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: captures are always frozen.
    const fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: captures are always frozen, and mutating one raises.
    const fn assert_frozen(_slf: &Bound<'_, Self>) {}

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
