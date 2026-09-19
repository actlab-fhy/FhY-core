//! `PyO3` wrapper around the pure-Rust [`crate::identifier::Identifier`].
//!
//! This module owns all Python-specific concerns (argument extraction, the
//! dict protocol, exception types). The identity, equality, hashing, and
//! counter semantics all live in the pure Rust core and are only delegated
//! to here. The class is immutable and not picklable on its own: the public
//! `fhy_core.identifier.Identifier` wraps it and pickles plain data.

use pyo3::import_exception;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBool, PyDict, PyInt, PyString, PyType};

use crate::identifier::Identifier as RustIdentifier;

import_exception!(fhy_core.serialization, DeserializationDictStructureError);
import_exception!(fhy_core.serialization, DeserializationValueError);

/// Modulus `CPython` reduces a non-negative `int` by when hashing it
/// (`sys.hash_info.modulus`).
#[cfg(target_pointer_width = "64")]
const PY_HASH_MODULUS: u64 = (1 << 61) - 1;
#[cfg(target_pointer_width = "32")]
const PY_HASH_MODULUS: u64 = (1 << 31) - 1;

/// Process-globally unique, named compiler symbol.
///
/// See ``fhy_core.identifier.Identifier`` for the full contract: two
/// identifiers are equal iff they share the same ``id``; ``name_hint`` is a
/// debugging aid only.
#[pyclass(frozen, name = "Identifier", module = "fhy_core._rs")]
pub(crate) struct Identifier {
    inner: RustIdentifier,
}

#[pymethods]
impl Identifier {
    #[new]
    fn new(name_hint: &str) -> Self {
        Self {
            inner: RustIdentifier::new(name_hint),
        }
    }

    #[getter]
    fn id(&self) -> u64 {
        self.inner.id()
    }

    #[getter]
    fn name_hint(&self) -> &str {
        self.inner.name_hint()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<PyRef<'_, Identifier>>() {
            Ok(other) => self.inner == other.inner,
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> isize {
        compute_python_int_hash(self.inner.id())
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn serialize_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("id", self.inner.id())?;
        dict.set_item("name_hint", self.inner.name_hint())?;
        Ok(dict)
    }

    #[classmethod]
    fn deserialize_from_dict(cls: &Bound<'_, PyType>, data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let (id, name_hint) = extract_identifier_data(cls, data)?;
        Ok(Self {
            inner: RustIdentifier::deserialize(id, name_hint),
        })
    }
}

/// Return `hash(id)` as Python computes it for a non-negative `int`, so an
/// identifier hashes exactly like its id.
#[expect(
    clippy::cast_possible_wrap,
    reason = "the remainder is below PY_HASH_MODULUS, which fits in isize \
              on every supported pointer width"
)]
fn compute_python_int_hash(id: u64) -> isize {
    (id % PY_HASH_MODULUS) as isize
}

/// Validate and extract `{"id": int, "name_hint": str}` from a Python dict.
///
/// Raises `DeserializationDictStructureError` for missing or extra keys, a
/// non-`int` id (including `bool`, a subtype of `int` in Python), or a
/// non-`str` name hint, and `DeserializationValueError` for a negative id or
/// one that does not fit in 64 bits.
fn extract_identifier_data(
    cls: &Bound<'_, PyType>,
    data: &Bound<'_, PyDict>,
) -> PyResult<(u64, String)> {
    let id_obj = data.get_item("id")?;
    let name_hint_obj = data.get_item("name_hint")?;
    let (Some(id_obj), Some(name_hint_obj)) = (id_obj, name_hint_obj) else {
        return Err(create_structure_error(cls, data)?);
    };
    if data.len() != 2
        || id_obj.is_instance_of::<PyBool>()
        || !id_obj.is_instance_of::<PyInt>()
        || !name_hint_obj.is_instance_of::<PyString>()
    {
        return Err(create_structure_error(cls, data)?);
    }

    if id_obj.lt(0)? {
        return Err(create_value_error(cls, "a non-negative integer", &id_obj));
    }
    let id = id_obj.extract::<u64>().map_err(|_source| {
        create_value_error(cls, "a non-negative integer below 2**64", &id_obj)
    })?;
    Ok((id, name_hint_obj.extract()?))
}

fn create_structure_error(cls: &Bound<'_, PyType>, data: &Bound<'_, PyDict>) -> PyResult<PyErr> {
    let py = cls.py();
    let expected_structure = [
        ("id", py.get_type::<PyInt>()),
        ("name_hint", py.get_type::<PyString>()),
    ]
    .into_py_dict(py)?;
    Ok(DeserializationDictStructureError::new_err((
        cls.clone().unbind(),
        expected_structure.unbind(),
        data.clone().unbind(),
    )))
}

fn create_value_error(
    cls: &Bound<'_, PyType>,
    expected_description_phrase: &'static str,
    actual_value: &Bound<'_, PyAny>,
) -> PyErr {
    DeserializationValueError::new_err((
        cls.clone().unbind(),
        "id",
        expected_description_phrase,
        actual_value.clone().unbind(),
    ))
}
