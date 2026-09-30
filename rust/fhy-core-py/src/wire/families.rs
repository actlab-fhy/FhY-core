//! The V2 wire format of the families whose base is a Python class: the
//! constraints, the domains, the frames and the constraint system.
//!
//! Their public classes inherit `WrappedFamilySerializable`, whose V2 path
//! names the family in its `_WIRE_FAMILY` and calls these functions; a
//! Python-defined member of an open family reaches the core through its
//! adapter, so it writes its foreign part under the family's `custom`
//! variant.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyType};

use fhy_core::constraint::wire::{ConstraintData, ConstraintSystemData};
use fhy_core::param::wire::ParamDomainData;
use fhy_core::symbol_table::wire::SymbolFrameData;

use crate::constraint::{PyConstraintSystem, read_constraint};
use crate::param::{constraint_to_python, domain_to_python, read_domain_object};
use crate::symbol_table::{frame_from_wire, frame_wire_data};

use super::{PyResolver, build, check_instance, parse, parse_dict, read_json, to_dict, to_json};

/// A family whose base is a Python class.
#[derive(Debug, Clone, Copy)]
enum Family {
    Constraint,
    ConstraintSystem,
    ParamDomain,
    SymbolFrame,
}

impl Family {
    /// Return the family `name` names.
    fn of(name: &str) -> PyResult<Self> {
        match name {
            "constraint" => Ok(Self::Constraint),
            "constraint_system" => Ok(Self::ConstraintSystem),
            "param_domain" => Ok(Self::ParamDomain),
            "symbol_frame" => Ok(Self::SymbolFrame),
            other => Err(PyValueError::new_err(format!(
                "unknown wire family {other:?}"
            ))),
        }
    }
}

/// Return the V2 dict of `object`, a member of `family`.
fn encode_dict<'py>(family: Family, object: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = object.py();
    match family {
        Family::Constraint => to_dict(py, &read_constraint(object)?),
        Family::ConstraintSystem => to_dict(py, object.cast::<PyConstraintSystem>()?.get().core()),
        Family::ParamDomain => to_dict(py, &read_domain_object(object)?),
        Family::SymbolFrame => to_dict(py, &frame_wire_data(object)?),
    }
}

/// Return the canonical V2 text of `object`, a member of `family`.
fn encode(family: Family, object: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = object.py();
    match family {
        Family::Constraint => to_json(py, &read_constraint(object)?),
        Family::ConstraintSystem => to_json(py, object.cast::<PyConstraintSystem>()?.get().core()),
        Family::ParamDomain => to_json(py, &read_domain_object(object)?),
        Family::SymbolFrame => to_json(py, &frame_wire_data(object)?),
    }
}

/// Return the object of the wire form of a member of `family`, read by
/// `read`, an instance of `cls`.
fn decode<'py>(
    family: Family,
    cls: &Bound<'py, PyType>,
    read: &impl ReadWire,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cls.py();
    let object = match family {
        Family::Constraint => {
            let data: ConstraintData = read.read(cls)?;
            let constraint = build(cls, || data.build(&PyResolver))?;
            constraint_to_python(py, &constraint)?
        }
        Family::ConstraintSystem => {
            let data: ConstraintSystemData = read.read(cls)?;
            let members = data
                .into_constraints()
                .into_iter()
                .map(|data| {
                    let constraint = build(cls, || data.build(&PyResolver))?;
                    constraint_to_python(py, &constraint)
                })
                .collect::<PyResult<Vec<_>>>()?;
            cls.call1((PyTuple::new(py, members)?,))?
        }
        Family::ParamDomain => {
            let data: ParamDomainData = read.read(cls)?;
            let domain = build(cls, || data.build(&PyResolver))?;
            domain_to_python(py, &domain)?
        }
        Family::SymbolFrame => {
            let data: SymbolFrameData = read.read(cls)?;
            frame_from_wire(cls, data)?
        }
    };
    check_instance(cls, object)
}

/// A source of a V2 payload: a dict or a text.
trait ReadWire {
    /// Return the wire form `D` of the payload, a payload of `cls`.
    fn read<D: serde::de::DeserializeOwned>(&self, cls: &Bound<'_, PyType>) -> PyResult<D>;
}

/// A V2 payload dict.
struct Dict<'a, 'py>(&'a Bound<'py, PyAny>);

impl ReadWire for Dict<'_, '_> {
    fn read<D: serde::de::DeserializeOwned>(&self, cls: &Bound<'_, PyType>) -> PyResult<D> {
        parse_dict(cls, self.0)
    }
}

/// A V2 payload text.
struct Text<'a>(&'a str);

impl ReadWire for Text<'_> {
    fn read<D: serde::de::DeserializeOwned>(&self, cls: &Bound<'_, PyType>) -> PyResult<D> {
        parse(cls, self.0)
    }
}

/// Return the V2 dict of `object`, a member of the family `family`.
///
/// Raises what writing its parts raises.
#[pyfunction]
pub(crate) fn encode_wire_dict<'py>(
    family: &str,
    object: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    encode_dict(Family::of(family)?, object)
}

/// Return the canonical V2 text of `object`, a member of the family
/// `family`.
///
/// Raises what writing its parts raises.
#[pyfunction]
pub(crate) fn encode_wire_json(family: &str, object: &Bound<'_, PyAny>) -> PyResult<String> {
    encode(Family::of(family)?, object)
}

/// Return the member of the family `family` the V2 dict `data` encodes, an
/// instance of `cls`.
///
/// Raises `DeserializationValueError` for a payload of another shape, and
/// what decoding its parts raises.
#[pyfunction]
pub(crate) fn decode_wire_family<'py>(
    family: &str,
    cls: &Bound<'py, PyType>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    decode(Family::of(family)?, cls, &Dict(data))
}

/// Return the member of the family `family` the JSON payload `payload`
/// encodes, an instance of `cls`: a V2 text, or through the framework's
/// `from_json` a text that may hold a V1 envelope.
///
/// Raises what [`decode_wire_family`] raises.
#[pyfunction]
pub(crate) fn decode_wire_family_json<'py>(
    family: &str,
    cls: &Bound<'py, PyType>,
    payload: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let family = Family::of(family)?;
    read_json(cls, payload, |text| decode(family, cls, &Text(text)))
}
