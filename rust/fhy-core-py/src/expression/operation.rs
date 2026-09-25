//! The operation enums of expressions: the Rust operations and the members
//! of the Python `StrEnum`s of the same meaning.
//!
//! `UnaryOperation`, `BinaryOperation` and `LogicalOperation` stay Python
//! enums (pattern P1) in `fhy_core.symbolic.expression.core`. Each member's
//! value is the Rust operation's name (`as_str`), so a member converts by
//! value in both directions, and a payload holds the same text on both
//! sides. The binding keeps each enum's members in the order of the Rust
//! table below, so converting a Rust operation to its member is an index,
//! and converting a member back is an identity scan.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

use fhy_core::expression::{BinaryOperation, LogicalOperation, UnaryOperation};

/// The Python module that defines the operation enums.
const MODULE: &str = "fhy_core.symbolic.expression.core";

/// The members of one Python operation enum and the enum class itself.
pub(super) struct OperationMembers {
    class: Py<PyType>,
    /// The members, in the order of the Rust operation table.
    members: Vec<Py<PyAny>>,
}

/// A Rust operation enum and the Python enum of its members.
pub(super) trait PythonOperation: Copy + PartialEq + 'static {
    /// The name of the Python enum class in [`MODULE`].
    const CLASS_NAME: &'static str;

    /// Every operation, in a fixed order.
    fn all() -> &'static [Self];

    /// The operation's name, the value of its Python member.
    fn name(self) -> &'static str;

    /// The cache of the Python members.
    fn cache() -> &'static PyOnceLock<OperationMembers>;
}

impl PythonOperation for UnaryOperation {
    const CLASS_NAME: &'static str = "UnaryOperation";

    fn all() -> &'static [Self] {
        &[Self::Negate, Self::Positive, Self::LogicalNot]
    }

    fn name(self) -> &'static str {
        self.as_str()
    }

    fn cache() -> &'static PyOnceLock<OperationMembers> {
        static CACHE: PyOnceLock<OperationMembers> = PyOnceLock::new();
        &CACHE
    }
}

impl PythonOperation for BinaryOperation {
    const CLASS_NAME: &'static str = "BinaryOperation";

    fn all() -> &'static [Self] {
        &[
            Self::Add,
            Self::Subtract,
            Self::Multiply,
            Self::Divide,
            Self::FloorDivide,
            Self::FloorMod,
            Self::Power,
            Self::Equal,
            Self::NotEqual,
            Self::Less,
            Self::LessEqual,
            Self::Greater,
            Self::GreaterEqual,
        ]
    }

    fn name(self) -> &'static str {
        self.as_str()
    }

    fn cache() -> &'static PyOnceLock<OperationMembers> {
        static CACHE: PyOnceLock<OperationMembers> = PyOnceLock::new();
        &CACHE
    }
}

impl PythonOperation for LogicalOperation {
    const CLASS_NAME: &'static str = "LogicalOperation";

    fn all() -> &'static [Self] {
        &[Self::And, Self::Or]
    }

    fn name(self) -> &'static str {
        self.as_str()
    }

    fn cache() -> &'static PyOnceLock<OperationMembers> {
        static CACHE: PyOnceLock<OperationMembers> = PyOnceLock::new();
        &CACHE
    }
}

/// Return the members of the Python enum of `T`, building the table on
/// first use.
///
/// # Errors
///
/// Raises what importing the enum or looking up a member raises, such as a
/// `ValueError` if the enum lacks a member for a Rust operation.
fn members<T: PythonOperation>(py: Python<'_>) -> PyResult<&OperationMembers> {
    T::cache().get_or_try_init(py, || {
        let class = py
            .import(MODULE)?
            .getattr(T::CLASS_NAME)?
            .cast_into::<PyType>()?;
        let members = T::all()
            .iter()
            .map(|operation| class.call1((operation.name(),)).map(Bound::unbind))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(OperationMembers {
            class: class.unbind(),
            members,
        })
    })
}

/// Return the Python member of `operation`.
///
/// # Errors
///
/// Raises what building the member table raises.
pub(super) fn operation_to_python<T: PythonOperation>(
    py: Python<'_>,
    operation: T,
) -> PyResult<Bound<'_, PyAny>> {
    let table = members::<T>(py)?;
    let index = T::all()
        .iter()
        .position(|candidate| *candidate == operation)
        .unwrap_or_else(|| unreachable!("the table lists every operation"));
    Ok(table.members[index].bind(py).clone())
}

/// Return the Rust operation of `value` and its Python member: a member
/// itself, or any value the enum class maps to one, such as its name.
///
/// # Errors
///
/// Raises the enum's own `ValueError` for a value that names no member.
pub(super) fn operation_from_python<'py, T: PythonOperation>(
    value: &Bound<'py, PyAny>,
) -> PyResult<(T, Bound<'py, PyAny>)> {
    let py = value.py();
    let table = members::<T>(py)?;
    if let Some(found) = find_member::<T>(table, value) {
        return Ok((found, value.clone()));
    }
    let member = table.class.bind(py).call1((value,))?;
    find_member::<T>(table, &member).map_or_else(
        || unreachable!("the enum class returns one of its members"),
        |found| Ok((found, member)),
    )
}

/// Return the Rust operation whose member is `value`, by identity.
fn find_member<T: PythonOperation>(
    table: &OperationMembers,
    value: &Bound<'_, PyAny>,
) -> Option<T> {
    table
        .members
        .iter()
        .position(|member| member.is(value))
        .map(|index| T::all()[index])
}
