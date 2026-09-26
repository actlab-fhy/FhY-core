//! The type classes: the bases `fhy_core._rs.Type` and `DataType`, and the
//! four built-in classes `PrimitiveDataType`, `TemplateDataType`,
//! `NumericalType` and `IndexType` (pattern P2, D-S11-11).
//!
//! Each built-in class holds its Rust value and the Python objects it was
//! built from, which its properties return. A Python-defined `Type` or
//! `DataType` subclass has the base alone, holds no Rust value, and reaches
//! the core as an extension (D-S11-9).

use std::sync::OnceLock;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyEllipsis, PyInt, PyList, PyString, PyTuple, PyType};

use fhy_core::expression::Expression;
use fhy_core::types::{
    CoreDataType, DataType, Dimension, IndexType, NumericalType, TemplateDataType, Type,
};

use crate::dataclass::{build_argument_type_error, collect_tuple, hash_value};
use crate::expression::PyExpression;
use crate::frozen::build_frozen_mutation_error;
use crate::identifier::{deserialize_identifier, restore_identifier};
use crate::public_class::PublicClass;
use crate::serialization::{FieldShape, deserialization_value_error_class, read_payload_fields};

use super::adapter::run_in_context;
use super::convert::{read_data_type_value, read_type_value};
use super::enums::{core_data_type_to_python, read_core_data_type};

/// The `__type__` of the sentinel payload of a wildcard shape dimension.
const ELLIPSIS_TYPE_ID: &str = "__numerical_type_shape_ellipsis__";

/// Implement `public_class` for a built-in class named `$name`.
macro_rules! impl_public_class {
    ($class:ty, $name:literal) => {
        impl $class {
            /// Return the public Python class registered for this class.
            pub(crate) fn public_class() -> &'static PublicClass {
                static PUBLIC_CLASS: PublicClass = PublicClass::new($name);
                &PUBLIC_CLASS
            }
        }
    };
}

/// Return `value` compared with `other` by `is_equal`, or `NotImplemented`
/// when `other` is no value of the tier `read` reads.
fn compare<'py, T>(
    slf: &Bound<'py, PyAny>,
    other: &Bound<'py, PyAny>,
    value: &T,
    read: impl Fn(&Bound<'py, PyAny>) -> Option<T>,
    expected: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    T: PartialEq,
{
    let py = slf.py();
    run_in_context(py, None, |_context| match read(other) {
        Some(other) => Ok(PyBool::new(py, (*value == other) == expected)
            .to_owned()
            .into_any()),
        None => Ok(py.NotImplemented().into_bound(py)),
    })
}

/// Return the hash of `value`, computed once into `cache`.
fn cached_hash<T: std::hash::Hash>(
    py: Python<'_>,
    cache: &OnceLock<u64>,
    value: &T,
) -> PyResult<u64> {
    if let Some(hash) = cache.get() {
        return Ok(*hash);
    }
    let hash = run_in_context(py, None, |_context| Ok(hash_value(value)))?;
    Ok(*cache.get_or_init(|| hash))
}

/// Return the payload of the serializable `value`.
fn serialize_nested<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    value.call_method0(intern!(value.py(), "serialize_to_dict"))
}

/// Return the expression a payload encodes.
fn deserialize_expression<'py>(payload: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = payload.py();
    PyExpression::public_class()
        .get(py)?
        .call_method1(intern!(py, "deserialize_from_dict"), (payload,))
}

/// Return the expression handle of `value`, an argument `field` of
/// `owner`, or raise the `TypeError` of a value that is no expression.
fn read_expression_argument(
    value: &Bound<'_, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<Expression> {
    match value.cast::<PyExpression>() {
        Ok(expression) => Ok(expression.get().expression().clone()),
        Err(_not_an_expression) => Err(build_argument_type_error(
            owner,
            field,
            "an Expression",
            value,
        )?),
    }
}

// ---------------------------------------------------------------------------
// The bases
// ---------------------------------------------------------------------------

/// The base of every type class: the built-in ones, which extend it, and
/// the Python-defined `Type` subclasses, which reach the core as
/// extensions.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Type")]
pub(crate) struct PyTypeBase;

#[pymethods]
impl PyTypeBase {
    /// Accept and ignore every argument, so a subclass with its own
    /// `__init__` constructs.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self
    }

    /// Register `cls` as the public `Type` class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }
}

impl PyTypeBase {
    /// Return the public Python class registered for this class.
    pub(crate) fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Type");
        &PUBLIC_CLASS
    }
}

/// The base of every data-type class: the built-in ones, which extend it,
/// and the Python-defined `DataType` subclasses, which reach the core as
/// extensions.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "DataType")]
pub(crate) struct PyDataTypeBase;

#[pymethods]
impl PyDataTypeBase {
    /// Accept and ignore every argument, so a subclass with its own
    /// `__init__` constructs.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self
    }

    /// Register `cls` as the public `DataType` class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }
}

impl PyDataTypeBase {
    /// Return the public Python class registered for this class.
    pub(crate) fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("DataType");
        &PUBLIC_CLASS
    }
}

// ---------------------------------------------------------------------------
// PrimitiveDataType
// ---------------------------------------------------------------------------

/// A core data type as a data type, backed by [`DataType::Primitive`].
#[pyclass(
    extends = PyDataTypeBase,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "PrimitiveDataType"
)]
pub(crate) struct PyPrimitiveDataType {
    /// The `CoreDataType` member.
    #[pyo3(get)]
    core_data_type: Py<PyAny>,
    value: CoreDataType,
}

impl_public_class!(PyPrimitiveDataType, "PrimitiveDataType");

impl PyPrimitiveDataType {
    /// Return the Rust data type.
    pub(crate) fn value(&self) -> DataType {
        DataType::Primitive(self.value)
    }
}

#[pymethods]
impl PyPrimitiveDataType {
    /// Always true: the built-in types are immutable.
    #[getter]
    fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: the built-in types are always frozen.
    fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: the built-in types are always frozen, and mutating one
    /// raises.
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

    /// Create the data type of the `CoreDataType` member `core_data_type`.
    ///
    /// Raises `TypeError` for anything else.
    #[new]
    fn new(core_data_type: &Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let value = read_core_data_type(core_data_type, "PrimitiveDataType", "core_data_type")?;
        let member = core_data_type_to_python(core_data_type.py(), value)?;
        Ok(PyClassInitializer::from(PyDataTypeBase).add_subclass(Self {
            core_data_type: member.unbind(),
            value,
        }))
    }

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare(
            slf.as_any(),
            other,
            &slf.get().value(),
            read_data_type_value,
            true,
        )
    }

    fn __ne__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare(
            slf.as_any(),
            other,
            &slf.get().value(),
            read_data_type_value,
            false,
        )
    }

    fn __hash__(&self) -> u64 {
        hash_value(&self.value())
    }

    /// Return whether `other` is the same primitive data type.
    fn is_structurally_equivalent(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let value = self.value();
        run_in_context(other.py(), None, |_context| {
            Ok(read_data_type_value(other)
                .is_some_and(|other| value.is_structurally_equivalent(&other)))
        })
    }

    fn __str__(&self) -> &'static str {
        self.value.as_str()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(format!(
            "{}({})",
            slf.get_type().name()?,
            slf.get().core_data_type.bind(slf.py()).repr()?
        ))
    }

    /// Pickle as a constructor call of the class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        Ok((
            slf.get_type(),
            PyTuple::new(py, [slf.get().core_data_type.bind(py)])?,
        ))
    }

    /// Return the data payload `{"core_data_type": ..}`.
    fn serialize_data_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let payload = PyDict::new(py);
        payload.set_item(intern!(py, "core_data_type"), self.value.as_str())?;
        Ok(payload)
    }

    /// Return the data type of a data payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [name] = read_payload_fields(cls, data, [("core_data_type", FieldShape::Str)])?;
        let Ok(value) = name.cast::<PyString>()?.to_str()?.parse::<CoreDataType>() else {
            let error = deserialization_value_error_class(py)?.call1((
                cls,
                "core_data_type",
                "a valid core data type",
                name,
            ))?;
            return Err(PyErr::from_value(error));
        };
        cls.call1((core_data_type_to_python(py, value)?,))
    }
}

// ---------------------------------------------------------------------------
// TemplateDataType
// ---------------------------------------------------------------------------

/// A placeholder data type, backed by [`DataType::Template`].
#[pyclass(
    extends = PyDataTypeBase,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "TemplateDataType"
)]
pub(crate) struct PyTemplateDataType {
    /// The placeholder's `Identifier`.
    #[pyo3(get)]
    data_type: Py<PyAny>,
    value: TemplateDataType,
}

impl_public_class!(PyTemplateDataType, "TemplateDataType");

impl PyTemplateDataType {
    /// Return the Rust data type.
    pub(crate) fn value(&self) -> DataType {
        DataType::Template(self.value.clone())
    }

    /// Return the Python list of the widths, or `None`.
    fn widths_object<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.value.widths() {
            Some(widths) => Ok(PyList::new(py, widths)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }
}

/// Return the widths `widths` holds, each a positive `int`, or `None`.
///
/// Raises `TypeError` for an argument that is no iterable of `int`s, and
/// `ValueError` for a width that is not positive or above `2**32 - 1`.
fn read_widths(widths: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<u32>>> {
    let Some(widths) = widths.filter(|widths| !widths.is_none()) else {
        return Ok(None);
    };
    let mut values = Vec::new();
    for width in widths.try_iter()? {
        let width = width?;
        if !width.is_instance_of::<PyInt>() || width.is_instance_of::<PyBool>() {
            return Err(build_argument_type_error(
                "TemplateDataType",
                "width",
                "an int",
                &width,
            )?);
        }
        match width.extract::<u32>() {
            Ok(value) if value > 0 => values.push(value),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "template data type widths must be positive integers below 2**32, but got {}",
                    width.repr()?
                )));
            }
        }
    }
    Ok(Some(values))
}

#[pymethods]
impl PyTemplateDataType {
    /// Always true: the built-in types are immutable.
    #[getter]
    fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: the built-in types are always frozen.
    fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: the built-in types are always frozen, and mutating one
    /// raises.
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

    /// Create the placeholder of the `Identifier` `data_type`, which only
    /// data types of one of the bit `widths` bind, or any when `widths` is
    /// `None`.
    ///
    /// Raises `TypeError` for an identifier that is no `Identifier` or a
    /// width that is no `int`, and `ValueError` for a width that is not
    /// positive.
    #[new]
    #[pyo3(signature = (data_type, widths = None))]
    fn new(
        data_type: &Bound<'_, PyAny>,
        widths: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let identifier = restore_identifier(data_type, "TemplateDataType", "data_type")?;
        let value = match read_widths(widths)? {
            Some(widths) => TemplateDataType::with_widths(identifier, widths)
                .unwrap_or_else(|_zero| unreachable!("every width is positive")),
            None => TemplateDataType::new(identifier),
        };
        Ok(PyClassInitializer::from(PyDataTypeBase).add_subclass(Self {
            data_type: data_type.clone().unbind(),
            value,
        }))
    }

    /// Return a new list of the width constraint, or `None`.
    #[getter]
    fn widths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.widths_object(py)
    }

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare(
            slf.as_any(),
            other,
            &slf.get().value(),
            read_data_type_value,
            true,
        )
    }

    fn __ne__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare(
            slf.as_any(),
            other,
            &slf.get().value(),
            read_data_type_value,
            false,
        )
    }

    fn __hash__(&self) -> u64 {
        hash_value(&self.value)
    }

    /// Return whether `other` is the same template: the same identifier
    /// and the same widths.
    fn is_structurally_equivalent(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let value = self.value();
        run_in_context(other.py(), None, |_context| {
            Ok(read_data_type_value(other)
                .is_some_and(|other| value.is_structurally_equivalent(&other)))
        })
    }

    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(slf.get().data_type.bind(slf.py()).str()?.to_string())
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        let identifier = this.data_type.bind(py).repr()?;
        let class = slf.get_type().name()?;
        match this.value.widths() {
            Some(_) => Ok(format!(
                "{class}({identifier}, widths={})",
                this.widths_object(py)?.repr()?
            )),
            None => Ok(format!("{class}({identifier})")),
        }
    }

    /// Pickle as a constructor call of the class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(
            py,
            [this.data_type.bind(py).clone(), this.widths_object(py)?],
        )?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the data payload `{"data_type": .., "widths": ..}`.
    fn serialize_data_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let payload = PyDict::new(py);
        payload.set_item(
            intern!(py, "data_type"),
            serialize_nested(self.data_type.bind(py))?,
        )?;
        payload.set_item(intern!(py, "widths"), self.widths_object(py)?)?;
        Ok(payload)
    }

    /// Return the template of a data payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload,
    /// and `DeserializationValueError` for a width that is not positive.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [identifier, widths] = read_payload_fields(
            cls,
            data,
            [
                ("data_type", FieldShape::Payload),
                ("widths", FieldShape::OptionalIntList),
            ],
        )?;
        if !widths.is_none() {
            for width in widths.try_iter()? {
                if width?.extract::<i64>().is_ok_and(|width| width <= 0) {
                    let error = deserialization_value_error_class(py)?.call1((
                        cls,
                        "widths",
                        "a list of positive integers or None",
                        &widths,
                    ))?;
                    return Err(PyErr::from_value(error));
                }
            }
        }
        cls.call1((deserialize_identifier(&identifier)?, widths))
    }
}

// ---------------------------------------------------------------------------
// NumericalType
// ---------------------------------------------------------------------------

/// A numerical array type, backed by [`Type::Numerical`].
#[pyclass(
    extends = PyTypeBase,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "NumericalType"
)]
pub(crate) struct PyNumericalType {
    /// The data type object.
    #[pyo3(get)]
    data_type: Py<PyAny>,
    /// The shape's objects: expressions and `Ellipsis`.
    shape: Py<PyTuple>,
    value: NumericalType,
    hash: OnceLock<u64>,
}

impl_public_class!(PyNumericalType, "NumericalType");

impl PyNumericalType {
    /// Return the Rust type.
    pub(crate) fn value(&self) -> Type {
        Type::Numerical(self.value.clone())
    }

    /// Return the data type object.
    pub(crate) fn data_type_object(&self) -> &Py<PyAny> {
        &self.data_type
    }

    /// Return the shape's objects.
    pub(crate) fn shape_objects<'py>(&self, py: Python<'py>) -> &Bound<'py, PyTuple> {
        self.shape.bind(py)
    }
}

#[pymethods]
impl PyNumericalType {
    /// Always true: the built-in types are immutable.
    #[getter]
    fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: the built-in types are always frozen.
    fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: the built-in types are always frozen, and mutating one
    /// raises.
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

    /// Create the array of the data type `data_type` over `shape`, a
    /// sequence of expressions and `Ellipsis`, or the scalar when `shape` is
    /// `None` or empty.
    ///
    /// Raises `TypeError` for a data type that is no `DataType` and a
    /// dimension that is neither an `Expression` nor `Ellipsis`.
    #[new]
    #[pyo3(signature = (data_type, shape = None))]
    fn new(
        data_type: &Bound<'_, PyAny>,
        shape: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = data_type.py();
        let Some(rust_data_type) = read_data_type_value(data_type) else {
            return Err(build_argument_type_error(
                "NumericalType",
                "data_type",
                "a DataType",
                data_type,
            )?);
        };
        let shape = match shape.filter(|shape| !shape.is_none()) {
            Some(shape) => collect_tuple(shape)?,
            None => PyTuple::empty(py),
        };
        let mut dimensions = Vec::with_capacity(shape.len());
        for dimension in shape.iter() {
            if dimension.is_instance_of::<PyEllipsis>() {
                dimensions.push(Dimension::Wildcard);
            } else if let Ok(expression) = dimension.cast::<PyExpression>() {
                dimensions.push(Dimension::Expression(expression.get().expression().clone()));
            } else {
                return Err(build_argument_type_error(
                    "NumericalType",
                    "shape dimension",
                    "an Expression or Ellipsis",
                    &dimension,
                )?);
            }
        }
        Ok(PyClassInitializer::from(PyTypeBase).add_subclass(Self {
            data_type: data_type.clone().unbind(),
            shape: shape.unbind(),
            value: NumericalType::new(rust_data_type, dimensions),
            hash: OnceLock::new(),
        }))
    }

    /// Return a new list of the shape's dimensions.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.shape.bind(py))
    }

    /// Return whether the type is a scalar: its shape is empty.
    fn is_scalar(&self) -> bool {
        self.value.is_scalar()
    }

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare(
            slf.as_any(),
            other,
            &slf.get().value(),
            read_type_value,
            true,
        )
    }

    fn __ne__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare(
            slf.as_any(),
            other,
            &slf.get().value(),
            read_type_value,
            false,
        )
    }

    fn __hash__(slf: &Bound<'_, Self>) -> PyResult<u64> {
        let this = slf.get();
        cached_hash(slf.py(), &this.hash, &this.value())
    }

    /// Return whether `other` is structurally equivalent: an array of an
    /// equivalent data type over equal dimensions.
    fn is_structurally_equivalent(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let value = self.value();
        run_in_context(other.py(), None, |_context| {
            Ok(
                read_type_value(other)
                    .is_some_and(|other| value.is_structurally_equivalent(&other)),
            )
        })
    }

    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let value = slf.get().value();
        run_in_context(slf.py(), None, |_context| Ok(value.to_string()))
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        Ok(format!(
            "{}({}, {})",
            slf.get_type().name()?,
            this.data_type.bind(py).repr()?,
            this.shape.bind(py).repr()?
        ))
    }

    /// Pickle as a constructor call of the class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(
            py,
            [
                this.data_type.bind(py).clone(),
                this.shape.bind(py).clone().into_any(),
            ],
        )?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the data payload `{"data_type": .., "shape": [..]}`, with the
    /// sentinel payload for each `Ellipsis`.
    fn serialize_data_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dimensions = PyList::empty(py);
        for dimension in self.shape.bind(py) {
            if dimension.is_instance_of::<PyEllipsis>() {
                let sentinel = PyDict::new(py);
                sentinel.set_item(intern!(py, "__type__"), ELLIPSIS_TYPE_ID)?;
                sentinel.set_item(intern!(py, "__data__"), PyDict::new(py))?;
                dimensions.append(sentinel)?;
            } else {
                dimensions.append(serialize_nested(&dimension)?)?;
            }
        }
        let payload = PyDict::new(py);
        payload.set_item(
            intern!(py, "data_type"),
            serialize_nested(self.data_type.bind(py))?,
        )?;
        payload.set_item(intern!(py, "shape"), dimensions)?;
        Ok(payload)
    }

    /// Return the type of a data payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [data_type, shape] = read_payload_fields(
            cls,
            data,
            [
                ("data_type", FieldShape::Payload),
                ("shape", FieldShape::PayloadList),
            ],
        )?;
        let mut dimensions = Vec::new();
        for payload in shape.try_iter()? {
            let payload = payload?;
            let is_sentinel = payload
                .get_item(intern!(py, "__type__"))
                .is_ok_and(|type_id| type_id.eq(ELLIPSIS_TYPE_ID).unwrap_or(false));
            if is_sentinel {
                dimensions.push(py.Ellipsis().into_bound(py));
            } else {
                dimensions.push(deserialize_expression(&payload)?);
            }
        }
        let data_type = PyDataTypeBase::public_class()
            .get(py)?
            .call_method1(intern!(py, "deserialize_from_dict"), (data_type,))?;
        cls.call1((data_type, PyList::new(py, dimensions)?))
    }
}

// ---------------------------------------------------------------------------
// IndexType
// ---------------------------------------------------------------------------

/// An index range type, backed by [`Type::Index`].
#[pyclass(
    extends = PyTypeBase,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "IndexType"
)]
pub(crate) struct PyIndexType {
    /// The lower bound expression.
    #[pyo3(get)]
    lower_bound: Py<PyAny>,
    /// The upper bound expression.
    #[pyo3(get)]
    upper_bound: Py<PyAny>,
    /// The stride expression.
    #[pyo3(get)]
    stride: Py<PyAny>,
    value: IndexType,
    hash: OnceLock<u64>,
}

impl_public_class!(PyIndexType, "IndexType");

impl PyIndexType {
    /// Return the Rust type.
    pub(crate) fn value(&self) -> Type {
        Type::Index(self.value.clone())
    }

    /// Return the bound and stride objects.
    pub(crate) fn expression_objects(&self) -> [&Py<PyAny>; 3] {
        [&self.lower_bound, &self.upper_bound, &self.stride]
    }
}

/// Return the literal expression `1`, the default stride.
fn unit_stride(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    crate::expression::PyLiteralExpression::public_class()
        .get(py)?
        .call1((1,))
}

#[pymethods]
impl PyIndexType {
    /// Always true: the built-in types are immutable.
    #[getter]
    fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: the built-in types are always frozen.
    fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: the built-in types are always frozen, and mutating one
    /// raises.
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

    /// Create the range from `lower_bound` to `upper_bound` in steps of
    /// `stride`, the literal `1` when `stride` is `None`.
    ///
    /// Raises `TypeError` for a bound or a stride that is no `Expression`.
    #[new]
    #[pyo3(signature = (lower_bound, upper_bound, stride = None))]
    fn new(
        lower_bound: &Bound<'_, PyAny>,
        upper_bound: &Bound<'_, PyAny>,
        stride: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = lower_bound.py();
        let stride = match stride.filter(|stride| !stride.is_none()) {
            Some(stride) => stride.clone(),
            None => unit_stride(py)?,
        };
        let value = IndexType::new(
            read_expression_argument(lower_bound, "IndexType", "lower_bound")?,
            read_expression_argument(upper_bound, "IndexType", "upper_bound")?,
            read_expression_argument(&stride, "IndexType", "stride")?,
        );
        Ok(PyClassInitializer::from(PyTypeBase).add_subclass(Self {
            lower_bound: lower_bound.clone().unbind(),
            upper_bound: upper_bound.clone().unbind(),
            stride: stride.unbind(),
            value,
            hash: OnceLock::new(),
        }))
    }

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare(
            slf.as_any(),
            other,
            &slf.get().value(),
            read_type_value,
            true,
        )
    }

    fn __ne__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare(
            slf.as_any(),
            other,
            &slf.get().value(),
            read_type_value,
            false,
        )
    }

    fn __hash__(slf: &Bound<'_, Self>) -> PyResult<u64> {
        let this = slf.get();
        cached_hash(slf.py(), &this.hash, &this.value())
    }

    /// Return whether `other` is an index type with equal bounds and
    /// stride.
    fn is_structurally_equivalent(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let value = self.value();
        run_in_context(other.py(), None, |_context| {
            Ok(
                read_type_value(other)
                    .is_some_and(|other| value.is_structurally_equivalent(&other)),
            )
        })
    }

    fn __str__(&self) -> String {
        self.value.to_string()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        Ok(format!(
            "{}({}, {}, {})",
            slf.get_type().name()?,
            this.lower_bound.bind(py).repr()?,
            this.upper_bound.bind(py).repr()?,
            this.stride.bind(py).repr()?
        ))
    }

    /// Pickle as a constructor call of the class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(
            py,
            [
                this.lower_bound.bind(py),
                this.upper_bound.bind(py),
                this.stride.bind(py),
            ],
        )?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the data payload of the bounds and the stride.
    fn serialize_data_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let payload = PyDict::new(py);
        payload.set_item(
            intern!(py, "lower_bound"),
            serialize_nested(self.lower_bound.bind(py))?,
        )?;
        payload.set_item(
            intern!(py, "upper_bound"),
            serialize_nested(self.upper_bound.bind(py))?,
        )?;
        payload.set_item(
            intern!(py, "stride"),
            serialize_nested(self.stride.bind(py))?,
        )?;
        Ok(payload)
    }

    /// Return the type of a data payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let [lower, upper, stride] = read_payload_fields(
            cls,
            data,
            [
                ("lower_bound", FieldShape::Payload),
                ("upper_bound", FieldShape::Payload),
                ("stride", FieldShape::Payload),
            ],
        )?;
        cls.call1((
            deserialize_expression(&lower)?,
            deserialize_expression(&upper)?,
            deserialize_expression(&stride)?,
        ))
    }
}

/// Raise the `TypeError` of a value that is no `Type` or `DataType`, as the
/// substitution defaults do.
pub(crate) fn build_not_a_type_error(
    function: &str,
    expected: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    Ok(PyTypeError::new_err(format!(
        "{function} received {}, which is not a {expected}.",
        value.get_type().name()?
    )))
}
