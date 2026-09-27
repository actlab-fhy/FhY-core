//! Conversion of types, data types, expressions and environments between
//! their Python objects and the core's values.
//!
//! Reading a value in a [`Context`] remembers its object, and the objects
//! of its parts, so a value the core hands back unchanged becomes the
//! object it came from; any other value is built through its public class.

use fhy_core::foreign::Part;

use pyo3::prelude::*;
use pyo3::types::{PyEllipsis, PyList};

use fhy_core::expression::{Expression, ExpressionKind};
use fhy_core::identifier::Identifier;
use fhy_core::tree::NodeHandle;
use fhy_core::types::{DataType, Dimension, Type, TypeUnificationEnvironment};

use crate::expression::{PyExpression, PyIdentifierExpression, materialize_with_known};
use crate::identifier;

use super::adapter::{Context, PyDataTypeAdapter, PyTypeAdapter};
use super::classes::{
    PyDataTypeBase, PyIndexType, PyNumericalType, PyPrimitiveDataType, PyTemplateDataType,
    PyTypeBase,
};
use super::enums::core_data_type_to_python;
use super::environment::PyTypeUnificationEnvironment;

/// Return the core type of `object`: a built-in type's value, a
/// Python-defined `Type` as an extension, or `None` for anything else.
pub(crate) fn read_type_value(object: &Bound<'_, PyAny>) -> Option<Type> {
    if let Ok(numerical) = object.cast::<PyNumericalType>() {
        return Some(numerical.get().value());
    }
    if let Ok(index) = object.cast::<PyIndexType>() {
        return Some(index.get().value());
    }
    if object.is_instance_of::<PyTypeBase>() {
        return Some(Type::Extension(Part::new(PyTypeAdapter::new(
            object.clone().unbind(),
        ))));
    }
    None
}

/// Return the core data type of `object`: a built-in data type's value, a
/// Python-defined `DataType` as an extension, or `None` for anything else.
pub(crate) fn read_data_type_value(object: &Bound<'_, PyAny>) -> Option<DataType> {
    if let Ok(primitive) = object.cast::<PyPrimitiveDataType>() {
        return Some(primitive.get().value());
    }
    if let Ok(template) = object.cast::<PyTemplateDataType>() {
        return Some(template.get().value());
    }
    if object.is_instance_of::<PyDataTypeBase>() {
        return Some(DataType::Extension(Part::new(PyDataTypeAdapter::new(
            object.clone().unbind(),
        ))));
    }
    None
}

/// Remember the object of the expression `object`, and the identifier
/// object of an identifier reference.
fn remember_expression(context: &Context, object: &Bound<'_, PyAny>) {
    let Ok(expression) = object.cast::<PyExpression>() else {
        return;
    };
    let mut known = context.known.borrow_mut();
    known.expressions.insert(
        expression.get().expression().identity(),
        object.clone().unbind(),
    );
    if let Ok(reference) = object.cast::<PyIdentifierExpression>() {
        if let ExpressionKind::Identifier(identifier) = expression.get().expression().kind() {
            known.identifiers.insert(
                identifier.id(),
                reference.get().identifier_object().clone_ref(object.py()),
            );
        }
    }
}

/// Return the core type of `object`, remembering its objects in `context`.
pub(crate) fn read_type(context: &Context, object: &Bound<'_, PyAny>) -> PyResult<Option<Type>> {
    let Some(value) = read_type_value(object) else {
        return Ok(None);
    };
    let py = object.py();
    if let Ok(numerical) = object.cast::<PyNumericalType>() {
        let this = numerical.get();
        read_data_type(context, this.data_type_object().bind(py))?;
        for dimension in this.shape_objects(py) {
            remember_expression(context, &dimension);
        }
    } else if let Ok(index) = object.cast::<PyIndexType>() {
        for expression in index.get().expression_objects() {
            remember_expression(context, expression.bind(py));
        }
    }
    context
        .known
        .borrow_mut()
        .types
        .push((value.clone(), object.clone().unbind()));
    Ok(Some(value))
}

/// Return the core data type of `object`, remembering its objects in
/// `context`.
pub(crate) fn read_data_type(
    context: &Context,
    object: &Bound<'_, PyAny>,
) -> PyResult<Option<DataType>> {
    let Some(value) = read_data_type_value(object) else {
        return Ok(None);
    };
    if let Ok(template) = object.cast::<PyTemplateDataType>() {
        if let DataType::Template(rust_template) = &value {
            context.known.borrow_mut().identifiers.insert(
                rust_template.identifier().id(),
                template.getattr("data_type")?.unbind(),
            );
        }
    }
    context
        .known
        .borrow_mut()
        .data_types
        .push((value.clone(), object.clone().unbind()));
    Ok(Some(value))
}

/// Return the expression handle of `object`, remembering its object, or
/// `None` if it is no expression.
pub(crate) fn read_expression(context: &Context, object: &Bound<'_, PyAny>) -> Option<Expression> {
    let expression = object.cast::<PyExpression>().ok()?;
    remember_expression(context, object);
    Some(expression.get().expression().clone())
}

/// Return the core environment of `object`, remembering its objects, or
/// `None` if it is no environment.
pub(crate) fn read_environment(
    context: &Context,
    object: &Bound<'_, PyAny>,
) -> Option<TypeUnificationEnvironment> {
    let environment = object.cast::<PyTypeUnificationEnvironment>().ok()?;
    environment.get().remember_objects(context, object.py());
    Some(environment.get().value().clone())
}

/// Return the Python object of the identifier `identifier`.
pub(crate) fn identifier_to_python<'py>(
    py: Python<'py>,
    context: &Context,
    identifier: &Identifier,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(object) = context.known.borrow().identifiers.get(&identifier.id()) {
        return Ok(object.bind(py).clone());
    }
    let object = identifier::identifier_to_python(py, identifier)?;
    context
        .known
        .borrow_mut()
        .identifiers
        .insert(identifier.id(), object.clone().unbind());
    Ok(object)
}

/// Return the Python object of the expression `expression`.
pub(crate) fn expression_to_python<'py>(
    py: Python<'py>,
    context: &Context,
    expression: &Expression,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(object) = context
        .known
        .borrow()
        .expressions
        .get(&expression.identity())
    {
        return Ok(object.bind(py).clone());
    }
    let known = context
        .known
        .borrow()
        .expressions
        .iter()
        .map(|(identity, object)| (*identity, object.bind(py).clone()))
        .collect();
    let object = materialize_with_known(py, expression, known)?;
    context
        .known
        .borrow_mut()
        .expressions
        .insert(expression.identity(), object.clone().unbind());
    Ok(object)
}

/// Return the known object of a value equal to `value` in `entries`, by
/// `is_same`.
fn find_known<'py, T>(
    py: Python<'py>,
    entries: &[(T, Py<PyAny>)],
    is_same: impl Fn(&T) -> bool,
) -> Option<Bound<'py, PyAny>> {
    entries
        .iter()
        .rev()
        .find(|(candidate, _)| is_same(candidate))
        .map(|(_, object)| object.bind(py).clone())
}

/// Return the Python object of the data type `data_type`.
pub(crate) fn data_type_to_python<'py>(
    py: Python<'py>,
    context: &Context,
    data_type: &DataType,
) -> PyResult<Bound<'py, PyAny>> {
    if let DataType::Extension(extension) = data_type {
        if let Some(adapter) = extension.get().as_any().downcast_ref::<PyDataTypeAdapter>() {
            return Ok(adapter.object(py));
        }
    }
    let found = find_known(py, &context.known.borrow().data_types, |candidate| {
        matches!(
            (candidate, data_type),
            (DataType::Primitive(_), DataType::Primitive(_))
                | (DataType::Template(_), DataType::Template(_))
        ) && candidate == data_type
    });
    if let Some(object) = found {
        return Ok(object);
    }
    let object = match data_type {
        DataType::Primitive(core_data_type) => PyPrimitiveDataType::public_class()
            .get(py)?
            .call1((core_data_type_to_python(py, *core_data_type)?,))?,
        DataType::Template(template) => {
            let identifier = identifier_to_python(py, context, template.identifier())?;
            let widths = match template.widths() {
                Some(widths) => PyList::new(py, widths)?.into_any(),
                None => py.None().into_bound(py),
            };
            PyTemplateDataType::public_class()
                .get(py)?
                .call1((identifier, widths))?
        }
        _ => {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "no Python object for the data type {data_type}"
            )));
        }
    };
    context
        .known
        .borrow_mut()
        .data_types
        .push((data_type.clone(), object.clone().unbind()));
    Ok(object)
}

/// Return the Python object of the type `value`.
pub(crate) fn type_to_python<'py>(
    py: Python<'py>,
    context: &Context,
    value: &Type,
) -> PyResult<Bound<'py, PyAny>> {
    if let Type::Extension(extension) = value {
        if let Some(adapter) = extension.get().as_any().downcast_ref::<PyTypeAdapter>() {
            return Ok(adapter.object(py));
        }
    }
    let found = find_known(py, &context.known.borrow().types, |candidate| {
        Type::ptr_eq(candidate, value)
    });
    if let Some(object) = found {
        return Ok(object);
    }
    let object = match value {
        Type::Numerical(numerical) => {
            let data_type = data_type_to_python(py, context, numerical.data_type())?;
            let shape = numerical
                .shape()
                .iter()
                .map(|dimension| match dimension {
                    Dimension::Expression(expression) => {
                        expression_to_python(py, context, expression)
                    }
                    Dimension::Wildcard => Ok(PyEllipsis::get(py).to_owned().into_any()),
                })
                .collect::<PyResult<Vec<_>>>()?;
            PyNumericalType::public_class()
                .get(py)?
                .call1((data_type, PyList::new(py, shape)?))?
        }
        Type::Index(index) => PyIndexType::public_class().get(py)?.call1((
            expression_to_python(py, context, index.lower_bound())?,
            expression_to_python(py, context, index.upper_bound())?,
            expression_to_python(py, context, index.stride())?,
        ))?,
        _ => {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "no Python object for the type {value}"
            )));
        }
    };
    context
        .known
        .borrow_mut()
        .types
        .push((value.clone(), object.clone().unbind()));
    Ok(object)
}

/// Return the Python object of the environment `environment`, of the class
/// of the context's template, with the objects the context knows.
pub(crate) fn environment_to_python<'py>(
    py: Python<'py>,
    context: &Context,
    environment: &TypeUnificationEnvironment,
) -> PyResult<Bound<'py, PyAny>> {
    PyTypeUnificationEnvironment::build(py, context, environment)
}
