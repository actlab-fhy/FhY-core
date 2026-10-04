//! The V2 form of a member value, for `fhy_core.serialization`'s
//! `serialize_value` and `deserialize_value`.

use pyo3::prelude::*;
use pyo3::types::PyType;

use fhy_core::constraint::wire::ValueData;
use fhy_core::constraint::{Member, MemberError, Value};

use crate::constraint::{read_bound_value, value_to_python};

use super::{PyResolver, build, parse_dict, to_dict};

/// Return the V2 form of the Python value `value`: a value that can be a
/// member in the canonical member order, and any other as it is.
///
/// Raises what writing an opaque part raises.
#[pyfunction]
pub(crate) fn serialize_wire_value<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = value.py();
    let core = read_bound_value(value)?;
    if core.is_member_shaped() {
        match Member::try_from(core.clone()) {
            Ok(member) => return to_dict(py, &member),
            // An opaque value whose key raised: raise its exception.
            Err(MemberError::OrderingKey { source, .. }) => {
                return Err(crate::kit::exceptions::boxed_error_to_py(source));
            }
            Err(_not_a_member) => {}
        }
    }
    to_dict(py, &core)
}

/// Return the Python value of the V2 form `data`.
///
/// Raises `DeserializationValueError` for data of another shape, and what
/// decoding an opaque part raises.
#[pyfunction]
pub(crate) fn deserialize_wire_value<'py>(data: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = data.py();
    let owner = py.get_type::<PyType>();
    let wire: ValueData = parse_dict(&owner, data)?;
    let value: Value = build(&owner, || wire.build(&PyResolver))?;
    value_to_python(py, &value)
}
