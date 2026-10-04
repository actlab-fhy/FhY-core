//! `CoreDataType` and `TypeQualifier`, which stay Python `StrEnum`s and
//! convert by value at the boundary.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

use fhy_core::types::{CoreDataType, TypeQualifier};

use crate::kit::dataclass::build_argument_type_error;

/// The module that defines both enums.
const MODULE: &str = "fhy_core.types.core";

/// A Python enum and its member for each Rust value.
struct EnumTable<T> {
    class: Py<PyType>,
    members: Vec<(T, Py<PyAny>)>,
}

impl<T: Copy + PartialEq> EnumTable<T> {
    /// Build the table of the Python enum `name`, whose member for each of
    /// `values` has the value `text(value)`.
    fn build(
        py: Python<'_>,
        name: &str,
        values: impl IntoIterator<Item = T>,
        text: impl Fn(T) -> &'static str,
    ) -> PyResult<Self> {
        let class = py.import(MODULE)?.getattr(name)?.cast_into::<PyType>()?;
        let members = values
            .into_iter()
            .map(|value| Ok((value, class.call1((text(value),))?.unbind())))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            class: class.unbind(),
            members,
        })
    }

    /// Return the Rust value of `object`, a member of the enum, or `None`
    /// for any other object.
    fn read(
        &self,
        object: &Bound<'_, PyAny>,
        text: impl Fn(T) -> &'static str,
    ) -> PyResult<Option<T>> {
        if let Some((value, _)) = self
            .members
            .iter()
            .find(|(_, member)| member.bind(object.py()).is(object))
        {
            return Ok(Some(*value));
        }
        if !object.is_instance(self.class.bind(object.py()))? {
            return Ok(None);
        }
        let name: String = object.getattr("value")?.extract()?;
        Ok(self
            .members
            .iter()
            .map(|(value, _)| *value)
            .find(|&value| text(value) == name))
    }

    /// Return the member of `value`.
    fn member<'py>(&self, py: Python<'py>, value: T) -> Bound<'py, PyAny> {
        self.members
            .iter()
            .find(|(candidate, _)| *candidate == value)
            .map_or_else(
                || unreachable!("the table has a member per value"),
                |(_, member)| member.bind(py).clone(),
            )
    }
}

fn core_data_type_table(py: Python<'_>) -> PyResult<&EnumTable<CoreDataType>> {
    static TABLE: PyOnceLock<EnumTable<CoreDataType>> = PyOnceLock::new();
    TABLE.get_or_try_init(py, || {
        EnumTable::build(
            py,
            "CoreDataType",
            CoreDataType::all(),
            CoreDataType::as_str,
        )
    })
}

fn type_qualifier_table(py: Python<'_>) -> PyResult<&EnumTable<TypeQualifier>> {
    static TABLE: PyOnceLock<EnumTable<TypeQualifier>> = PyOnceLock::new();
    TABLE.get_or_try_init(py, || {
        EnumTable::build(
            py,
            "TypeQualifier",
            [
                TypeQualifier::Input,
                TypeQualifier::Output,
                TypeQualifier::State,
                TypeQualifier::Param,
                TypeQualifier::Temp,
            ],
            TypeQualifier::as_str,
        )
    })
}

/// Return the Rust value of the `CoreDataType` member `object`, or `None`
/// for any other object.
pub(crate) fn try_read_core_data_type(object: &Bound<'_, PyAny>) -> PyResult<Option<CoreDataType>> {
    core_data_type_table(object.py())?.read(object, CoreDataType::as_str)
}

/// Return the Rust value of the `CoreDataType` member `object`, or raise the
/// `TypeError` naming `owner` and `field`.
pub(crate) fn read_core_data_type(
    object: &Bound<'_, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<CoreDataType> {
    match try_read_core_data_type(object)? {
        Some(value) => Ok(value),
        None => Err(build_argument_type_error(
            owner,
            field,
            "a CoreDataType",
            object,
        )?),
    }
}

/// Return the `CoreDataType` member of `value`.
pub(crate) fn core_data_type_to_python(
    py: Python<'_>,
    value: CoreDataType,
) -> PyResult<Bound<'_, PyAny>> {
    Ok(core_data_type_table(py)?.member(py, value))
}

/// Return the Rust value of the `TypeQualifier` member `object`, or raise
/// the `TypeError` naming `owner` and `field`.
pub(crate) fn read_type_qualifier(
    object: &Bound<'_, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<TypeQualifier> {
    match type_qualifier_table(object.py())?.read(object, TypeQualifier::as_str)? {
        Some(value) => Ok(value),
        None => Err(build_argument_type_error(
            owner,
            field,
            "a TypeQualifier",
            object,
        )?),
    }
}

/// Return the `TypeQualifier` member of `value`.
pub(crate) fn type_qualifier_to_python(
    py: Python<'_>,
    value: TypeQualifier,
) -> PyResult<Bound<'_, PyAny>> {
    Ok(type_qualifier_table(py)?.member(py, value))
}
