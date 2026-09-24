//! Conversion of the core crate's errors into the Python exceptions the
//! binding raises.

use pyo3::{PyErr, PyResult};

/// Conversion of a core error into the Python exception the binding raises.
///
/// A local trait, because the orphan rule forbids `From<CoreError> for
/// PyErr`. Each binding module implements it next to the functions that
/// raise those errors.
pub(crate) trait IntoPyErr {
    /// Return the Python exception for this error.
    fn into_py_err(self) -> PyErr;
}

/// `map_err` through [`IntoPyErr`] for any result whose error implements it.
pub(crate) trait IntoPyResult<T> {
    /// Return the value, or the error converted with [`IntoPyErr`].
    fn into_py_result(self) -> PyResult<T>;
}

impl<T, E: IntoPyErr> IntoPyResult<T> for Result<T, E> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(IntoPyErr::into_py_err)
    }
}
