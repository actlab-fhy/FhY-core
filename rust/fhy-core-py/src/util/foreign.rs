//! A Python-defined part as a core [`Foreign`].
//!
//! The core serializes a part it does not ship as the type id the part is
//! registered under and its own payload as text. [`foreign_of`] asks the
//! framework, `fhy_core.serialization._foreign_payload`, for both, for a
//! `to_foreign` of a core part trait that a Python object stands behind.
//! A Python exception a hook raises while it does is kept as the pending
//! exception ([`pending`](super::pending)), so the entry point that started
//! the serialization raises it as itself, and the core only sees a
//! [`ForeignError::Failed`] with its message.

use std::error::Error;
use std::fmt;

use fhy_core::foreign::{Foreign, ForeignError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::pending::record_pending_error;
use super::python::{ImportedAttr, type_name};

/// The framework function that returns a part's type id and payload text.
static FOREIGN_PAYLOAD: ImportedAttr =
    ImportedAttr::new("fhy_core.serialization", "_foreign_payload");

/// The message of a Python exception, the source of a failed foreign part or
/// of a refusal a Python-defined part raised.
///
/// It carries the text only: the exception itself is kept pending, so that
/// the entry point raises it as itself.
#[derive(Debug)]
pub struct RaisedError(String);

impl RaisedError {
    /// Return the error that displays as `message`.
    #[must_use]
    pub const fn new(message: String) -> Self {
        Self(message)
    }
}

impl fmt::Display for RaisedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for RaisedError {}

/// Keep `error` as the pending exception, and return the foreign error of
/// the part named `type_id` it stands for.
///
/// The returned [`ForeignError::Failed`] holds a [`RaisedError`] with the
/// exception's message.
pub fn foreign_failure(py: Python<'_>, type_id: &str, error: PyErr) -> ForeignError {
    let message = error.value(py).to_string();
    record_pending_error(error);
    ForeignError::Failed {
        type_id: type_id.to_owned(),
        source: Box::new(RaisedError::new(message)),
    }
}

/// Return the foreign part of the Python-defined part `object`: its type
/// id and the canonical text of its data, as a family member when
/// `family`, and its whole payload otherwise.
///
/// Attaches to the interpreter if the calling thread is not attached.
///
/// # Errors
///
/// Returns [`ForeignError::Failed`], keeping the Python exception pending,
/// when the object's hooks raise or the framework's answer is not a pair of
/// `str`s.
///
/// # Panics
///
/// Panics if the Python interpreter is not initialized, as
/// [`Python::attach`] does.
pub fn foreign_of(object: &Py<PyAny>, family: bool) -> Result<Foreign, ForeignError> {
    Python::attach(|py| {
        let object = object.bind(py);
        let result = (|| -> PyResult<(String, String)> {
            let keywords = PyDict::new(py);
            keywords.set_item(intern!(py, "family"), family)?;
            FOREIGN_PAYLOAD
                .get(py)?
                .call((object,), Some(&keywords))?
                .extract()
        })();
        match result {
            Ok((type_id, data)) => Ok(Foreign::new(type_id, data)),
            Err(error) => Err(foreign_failure(py, &type_name(object), error)),
        }
    })
}

#[cfg(test)]
mod tests;
