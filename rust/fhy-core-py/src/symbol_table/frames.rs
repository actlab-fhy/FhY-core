//! The three built-in frames, `ImportSymbolTableFrame`,
//! `VariableSymbolTableFrame` and `FunctionSymbolTableFrame`, and
//! `FunctionKeyword`, which stays a Python enum.
//!
//! Each frame holds its core [`SymbolFrame`] and the Python objects it was
//! built from, which its properties return. It stands in for the frozen
//! dataclass it replaces: `==`, `hash` and `repr` follow its fields, it is
//! always frozen, a pickle is a call of its class, and it writes and reads
//! the dataclass's payload. Its equivalence methods have the derived plan's
//! meaning.

use std::sync::OnceLock;

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PySequence, PyString, PyTuple, PyType};

use fhy_core::symbol_table::{
    Frame, FunctionFrame, FunctionKeyword, ImportFrame, SymbolFrame, VariableFrame,
};
use fhy_core::types::{Type, TypeQualifier};

use crate::dataclass::{
    build_argument_type_error, collect_tuple, format_dataclass_repr, hash_value,
};
use crate::error::IntoPyErr;
use crate::frozen::build_frozen_mutation_error;
use crate::identifier::{deserialize_identifier, restore_identifier, serialize_identifier};
use crate::serialization::{FieldShape, is_serialized_dict, read_payload_fields};
use crate::term::read_renaming;
use crate::types::{
    MayCallPython, read_type_qualifier, read_type_value, run_in_context, type_qualifier_to_python,
};

/// The module that defines `FunctionKeyword` and `SymbolTableFrame`.
pub(super) const MODULE: &str = "fhy_core.symbol_table";

// ---------------------------------------------------------------------------
// FunctionKeyword
// ---------------------------------------------------------------------------

/// Every keyword, in the Python enum's order.
const KEYWORDS: [FunctionKeyword; 3] = [
    FunctionKeyword::Procedure,
    FunctionKeyword::Operation,
    FunctionKeyword::Native,
];

/// The Python `FunctionKeyword` enum and its member per keyword.
struct KeywordTable {
    class: Py<PyType>,
    members: [Py<PyAny>; 3],
}

fn keyword_table(py: Python<'_>) -> PyResult<&KeywordTable> {
    static TABLE: PyOnceLock<KeywordTable> = PyOnceLock::new();
    TABLE.get_or_try_init(py, || {
        let class = py
            .import(MODULE)?
            .getattr("FunctionKeyword")?
            .cast_into::<PyType>()?;
        let member = |keyword: FunctionKeyword| -> PyResult<Py<PyAny>> {
            Ok(class.call1((keyword.as_str(),))?.unbind())
        };
        let members = [
            member(KEYWORDS[0])?,
            member(KEYWORDS[1])?,
            member(KEYWORDS[2])?,
        ];
        Ok(KeywordTable {
            class: class.unbind(),
            members,
        })
    })
}

/// Return the keyword of the `FunctionKeyword` member `object`, or `None`
/// for any other object.
fn try_read_keyword(object: &Bound<'_, PyAny>) -> PyResult<Option<FunctionKeyword>> {
    let py = object.py();
    let table = keyword_table(py)?;
    if let Some(position) = table
        .members
        .iter()
        .position(|member| member.bind(py).is(object))
    {
        return Ok(Some(KEYWORDS[position]));
    }
    if !object.is_instance(table.class.bind(py))? {
        return Ok(None);
    }
    let value: String = object.getattr(intern!(py, "value"))?.extract()?;
    Ok(value.parse().ok())
}

/// Return the `FunctionKeyword` member of `keyword`.
fn keyword_to_python(py: Python<'_>, keyword: FunctionKeyword) -> PyResult<Bound<'_, PyAny>> {
    let table = keyword_table(py)?;
    let position = KEYWORDS
        .iter()
        .position(|candidate| *candidate == keyword)
        .unwrap_or_else(|| unreachable!("the table has a member per keyword"));
    Ok(table.members[position].bind(py).clone())
}

// ---------------------------------------------------------------------------
// What the three classes share
// ---------------------------------------------------------------------------

impl MayCallPython for SymbolFrame {
    fn may_call_python(&self) -> bool {
        match self {
            SymbolFrame::Variable(frame) => frame.ty().may_call_python(),
            SymbolFrame::Function(frame) => {
                frame.signature().iter().any(|(_, ty)| ty.may_call_python())
            }
            _ => false,
        }
    }
}

/// Return whether `left` and `right` are equal frames, `==` (`structural`
/// false) or structurally equivalent; a comparison that may call Python's
/// `==` or a handler runs in a context, which raises its first exception.
fn compare_values(
    py: Python<'_>,
    left: &SymbolFrame,
    right: &SymbolFrame,
    structural: bool,
) -> PyResult<bool> {
    let answer = || -> PyResult<bool> {
        if structural {
            left.is_structurally_equivalent(right)
                .map_err(IntoPyErr::into_py_err)
        } else {
            Ok(left == right)
        }
    };
    if !left.may_call_python() && !right.may_call_python() {
        return answer();
    }
    run_in_context(py, None, |_context| answer())
}

/// Return whether two built-in frames of one class are structurally
/// equivalent, in a context when that may call Python.
pub(super) fn compare_natives(
    py: Python<'_>,
    left: &SymbolFrame,
    right: &SymbolFrame,
) -> PyResult<bool> {
    compare_values(py, left, right, true)
}

/// Return the core frame of a built-in frame object, or `None` for any
/// other object.
pub(super) fn read_frame_value(object: &Bound<'_, PyAny>) -> Option<SymbolFrame> {
    if let Ok(frame) = object.cast::<PyImportSymbolTableFrame>() {
        return Some(frame.get().value.clone());
    }
    if let Ok(frame) = object.cast::<PyVariableSymbolTableFrame>() {
        return Some(frame.get().value.clone());
    }
    if let Ok(frame) = object.cast::<PyFunctionSymbolTableFrame>() {
        return Some(frame.get().value.clone());
    }
    None
}

/// Return whether `other` is exactly `slf`'s class and its frame compares
/// with `value` as [`compare_values`] does.
fn compare_same_class(
    slf: &Bound<'_, PyAny>,
    value: &SymbolFrame,
    other: &Bound<'_, PyAny>,
    structural: bool,
) -> PyResult<bool> {
    if !slf.get_type().is(other.get_type()) {
        return Ok(false);
    }
    let Some(other_value) = read_frame_value(other) else {
        return Ok(false);
    };
    compare_values(slf.py(), value, &other_value, structural)
}

/// Return `slf == other` as the dataclass's `__eq__` does: `NotImplemented`
/// unless `other` is exactly `slf`'s class, and otherwise whether the
/// frames are equal.
fn dataclass_eq<'py>(
    slf: &Bound<'py, PyAny>,
    value: &SymbolFrame,
    other: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    if !slf.get_type().is(other.get_type()) {
        return Ok(py.NotImplemented().into_bound(py));
    }
    let is_equal = slf.is(other) || compare_same_class(slf, value, other, false)?;
    Ok(pyo3::types::PyBool::new(py, is_equal).to_owned().into_any())
}

/// Return the hash of `value`, computed once into `cache`.
fn cached_hash(py: Python<'_>, cache: &OnceLock<u64>, value: &SymbolFrame) -> PyResult<u64> {
    if let Some(hash) = cache.get() {
        return Ok(*hash);
    }
    let hash = if value.may_call_python() {
        run_in_context(py, None, |_context| Ok(hash_value(value)))?
    } else {
        hash_value(value)
    };
    Ok(*cache.get_or_init(|| hash))
}

/// Return the payload of the serializable `value`.
fn serialize_nested<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    value.call_method0(intern!(value.py(), "serialize_to_dict"))
}

/// Return the type a payload encodes, through `Type`'s family dispatch.
fn deserialize_type<'py>(payload: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = payload.py();
    crate::python::cached_attr!(py, "fhy_core.types", "Type" => PyType)?
        .call_method1(intern!(py, "deserialize_from_dict"), (payload,))
}

/// Return the core type of `object`, an argument `field` of `owner`, or
/// raise the `TypeError` of a value that is no `Type`.
fn read_type_argument(object: &Bound<'_, PyAny>, owner: &str, field: &str) -> PyResult<Type> {
    match read_type_value(object) {
        Some(ty) => Ok(ty),
        None => Err(build_argument_type_error(owner, field, "a Type", object)?),
    }
}

/// Return the `DeserializationDictStructureError` of `data` for `cls`,
/// whose payload holds the fields `expected`, each a name and a type.
pub(super) fn structure_error(
    cls: &Bound<'_, PyType>,
    expected: &[(&str, Bound<'_, PyAny>)],
    data: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    let py = cls.py();
    let fields = PyDict::new(py);
    for (name, ty) in expected {
        fields.set_item(name, ty)?;
    }
    Ok(crate::exceptions::DESERIALIZATION_DICT_STRUCTURE_ERROR.err(py, (cls, fields, data)))
}

// ---------------------------------------------------------------------------
// ImportSymbolTableFrame
// ---------------------------------------------------------------------------

/// The frame of an imported symbol, backed by [`ImportFrame`].
#[pyclass(
    frozen,
    subclass,
    module = "fhy_core._rs",
    name = "ImportSymbolTableFrame"
)]
pub(crate) struct PyImportSymbolTableFrame {
    /// The symbol's `Identifier`.
    #[pyo3(get)]
    name: Py<PyAny>,
    value: SymbolFrame,
}

#[pymethods]
impl PyImportSymbolTableFrame {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.name)?;
        Ok(())
    }

    /// Create the frame of the imported symbol `name`.
    ///
    /// Raises `TypeError` if `name` is no `Identifier`.
    #[new]
    fn new(name: &Bound<'_, PyAny>) -> PyResult<Self> {
        let identifier = restore_identifier(name, "ImportSymbolTableFrame", "name")?;
        Ok(Self {
            name: name.clone().unbind(),
            value: SymbolFrame::from(ImportFrame::new(identifier)),
        })
    }

    /// Always true: the built-in frames are immutable.
    #[getter]
    const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: the built-in frames are always frozen.
    const fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: the built-in frames are always frozen.
    const fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        dataclass_eq(slf.as_any(), &slf.get().value, other)
    }

    fn __hash__(&self) -> u64 {
        hash_value(&self.value)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        format_dataclass_repr(&slf.get_type(), &[("name", slf.get().name.bind(py))])
    }

    /// Return whether `other` is a frame of exactly this class with the
    /// same name.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        compare_same_class(slf.as_any(), &slf.get().value, other, true)
    }

    /// Return whether `other` is structurally equivalent: the frame binds
    /// nothing, so no renaming applies.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        compare_same_class(slf.as_any(), &slf.get().value, other, true)
    }

    /// Return whether `other` is structurally equivalent; `renaming` must
    /// be an `AlphaRenaming`, and is not consulted.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        read_renaming(renaming)?;
        compare_same_class(slf.as_any(), &slf.get().value, other, true)
    }

    /// Pickle as a constructor call of the class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        Ok((slf.get_type(), PyTuple::new(py, [slf.get().name.bind(py)])?))
    }

    /// Return the data payload `{"name": ..}`.
    fn serialize_data_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let payload = PyDict::new(py);
        payload.set_item(
            intern!(py, "name"),
            serialize_identifier(py, self.value.name())?,
        )?;
        Ok(payload)
    }

    /// Return the frame of a data payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let [name] = read_payload_fields(cls, data, [("name", FieldShape::Payload)])?;
        cls.call1((deserialize_identifier(&name)?,))
    }
}

// ---------------------------------------------------------------------------
// VariableSymbolTableFrame
// ---------------------------------------------------------------------------

/// The frame of a variable, backed by [`VariableFrame`].
#[pyclass(
    frozen,
    subclass,
    module = "fhy_core._rs",
    name = "VariableSymbolTableFrame"
)]
pub(crate) struct PyVariableSymbolTableFrame {
    /// The variable's `Identifier`.
    #[pyo3(get)]
    name: Py<PyAny>,
    /// The variable's `Type`.
    #[pyo3(get, name = "type")]
    ty: Py<PyAny>,
    /// The variable's `TypeQualifier`.
    #[pyo3(get)]
    type_qualifier: Py<PyAny>,
    value: SymbolFrame,
    hash: OnceLock<u64>,
    /// The slot of a Python-defined type's adapter, which the frame owns.
    slots: crate::gc::Slots,
}

#[pymethods]
impl PyVariableSymbolTableFrame {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.name)?;
        visit.call(&self.ty)?;
        visit.call(&self.type_qualifier)?;
        self.slots.traverse(&visit)
    }

    /// Create the frame of the variable `name` of type `type`, qualified
    /// `type_qualifier`.
    ///
    /// Raises `TypeError` for a name that is no `Identifier`, a type that is
    /// no `Type`, or a qualifier that is no `TypeQualifier`.
    #[new]
    #[pyo3(signature = (name, r#type, type_qualifier))]
    fn new(
        name: &Bound<'_, PyAny>,
        r#type: &Bound<'_, PyAny>,
        type_qualifier: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        const OWNER: &str = "VariableSymbolTableFrame";
        let identifier = restore_identifier(name, OWNER, "name")?;
        let (ty, slots) = crate::gc::collect_slots(|| read_type_argument(r#type, OWNER, "type"));
        let ty = ty?;
        let qualifier = read_type_qualifier(type_qualifier, OWNER, "type_qualifier")?;
        Ok(Self {
            name: name.clone().unbind(),
            ty: r#type.clone().unbind(),
            type_qualifier: type_qualifier.clone().unbind(),
            value: SymbolFrame::from(VariableFrame::new(identifier, ty, qualifier)),
            hash: OnceLock::new(),
            slots,
        })
    }

    /// Always true: the built-in frames are immutable.
    #[getter]
    const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: the built-in frames are always frozen.
    const fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: the built-in frames are always frozen.
    const fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        dataclass_eq(slf.as_any(), &slf.get().value, other)
    }

    fn __hash__(slf: &Bound<'_, Self>) -> PyResult<u64> {
        let this = slf.get();
        cached_hash(slf.py(), &this.hash, &this.value)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        format_dataclass_repr(
            &slf.get_type(),
            &[
                ("name", this.name.bind(py)),
                ("type", this.ty.bind(py)),
                ("type_qualifier", this.type_qualifier.bind(py)),
            ],
        )
    }

    /// Return whether `other` is a frame of exactly this class with the
    /// same name and qualifier and a structurally equivalent type.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        compare_same_class(slf.as_any(), &slf.get().value, other, true)
    }

    /// Return whether `other` is structurally equivalent: the frame binds
    /// nothing, so no renaming applies.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        compare_same_class(slf.as_any(), &slf.get().value, other, true)
    }

    /// Return whether `other` is structurally equivalent; `renaming` must
    /// be an `AlphaRenaming`, and is not consulted.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        read_renaming(renaming)?;
        compare_same_class(slf.as_any(), &slf.get().value, other, true)
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
                this.name.bind(py),
                this.ty.bind(py),
                this.type_qualifier.bind(py),
            ],
        )?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the data payload `{"name": .., "type": .., "type_qualifier":
    /// ..}`.
    fn serialize_data_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let SymbolFrame::Variable(frame) = &self.value else {
            unreachable!("a variable frame holds a variable");
        };
        let payload = PyDict::new(py);
        payload.set_item(intern!(py, "name"), serialize_identifier(py, frame.name())?)?;
        payload.set_item(intern!(py, "type"), serialize_nested(self.ty.bind(py))?)?;
        payload.set_item(intern!(py, "type_qualifier"), frame.qualifier().as_str())?;
        Ok(payload)
    }

    /// Return the frame of a data payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload,
    /// and `DeserializationValueError` for a qualifier that is no
    /// `TypeQualifier` value.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [name, ty, qualifier] = read_payload_fields(
            cls,
            data,
            [
                ("name", FieldShape::Payload),
                ("type", FieldShape::Payload),
                ("type_qualifier", FieldShape::Str),
            ],
        )?;
        let Ok(value) = qualifier
            .cast::<PyString>()?
            .to_str()?
            .parse::<TypeQualifier>()
        else {
            return Err(crate::exceptions::DESERIALIZATION_VALUE_ERROR.err(
                py,
                (
                    cls,
                    "type_qualifier",
                    "a valid TypeQualifier value",
                    qualifier,
                ),
            ));
        };
        cls.call1((
            deserialize_identifier(&name)?,
            deserialize_type(&ty)?,
            type_qualifier_to_python(py, value)?,
        ))
    }
}

// ---------------------------------------------------------------------------
// FunctionSymbolTableFrame
// ---------------------------------------------------------------------------

/// The frame of a function, backed by [`FunctionFrame`].
#[pyclass(
    frozen,
    subclass,
    module = "fhy_core._rs",
    name = "FunctionSymbolTableFrame"
)]
pub(crate) struct PyFunctionSymbolTableFrame {
    /// The function's `Identifier`.
    #[pyo3(get)]
    name: Py<PyAny>,
    /// The `FunctionKeyword` member.
    #[pyo3(get)]
    keyword: Py<PyAny>,
    /// The `(TypeQualifier, Type)` pair of each parameter.
    #[pyo3(get)]
    signature: Py<PyTuple>,
    value: SymbolFrame,
    hash: OnceLock<u64>,
    /// The slots of the Python-defined parameter types' adapters, which the
    /// frame owns.
    slots: crate::gc::Slots,
}

/// The core values of a function frame's signature.
type Signature = Vec<(TypeQualifier, Type)>;

/// Return the pairs of `signature`, each a tuple of its qualifier and type
/// objects, and their core values.
///
/// Raises `TypeError` for an entry that is no `(TypeQualifier, Type)` pair.
fn read_signature<'py>(
    signature: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyTuple>, Signature)> {
    const OWNER: &str = "FunctionSymbolTableFrame";
    let py = signature.py();
    let entries = collect_tuple(signature)?;
    let mut pairs = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());
    for entry in entries.iter() {
        let pair = entry
            .cast::<PySequence>()
            .ok()
            .filter(|_| !entry.is_instance_of::<PyString>())
            .map(|sequence| collect_tuple(sequence.as_any()))
            .transpose()?
            .filter(|pair| pair.len() == 2);
        let Some(pair) = pair else {
            return Err(build_argument_type_error(
                OWNER,
                "signature entry",
                "a (TypeQualifier, Type) pair",
                &entry,
            )?);
        };
        let qualifier_object = pair.get_item(0)?;
        let type_object = pair.get_item(1)?;
        let qualifier = read_type_qualifier(&qualifier_object, OWNER, "signature qualifier")?;
        let ty = read_type_argument(&type_object, OWNER, "signature type")?;
        pairs.push(PyTuple::new(py, [qualifier_object, type_object])?);
        values.push((qualifier, ty));
    }
    Ok((PyTuple::new(py, pairs)?, values))
}

#[pymethods]
impl PyFunctionSymbolTableFrame {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.name)?;
        visit.call(&self.keyword)?;
        visit.call(&self.signature)?;
        self.slots.traverse(&visit)
    }

    /// Create the frame of the function `name`, declared with `keyword`,
    /// whose parameters have the `(TypeQualifier, Type)` pairs of
    /// `signature`, in order.
    ///
    /// Raises `TypeError` for a name that is no `Identifier`, a keyword that
    /// is no `FunctionKeyword`, or an entry that is no such pair.
    #[new]
    #[pyo3(signature = (name, keyword, signature = None))]
    fn new(
        name: &Bound<'_, PyAny>,
        keyword: &Bound<'_, PyAny>,
        signature: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        const OWNER: &str = "FunctionSymbolTableFrame";
        let py = name.py();
        let identifier = restore_identifier(name, OWNER, "name")?;
        let Some(keyword_value) = try_read_keyword(keyword)? else {
            return Err(build_argument_type_error(
                OWNER,
                "keyword",
                "a FunctionKeyword",
                keyword,
            )?);
        };
        let (read, slots) = crate::gc::collect_slots(|| match signature {
            Some(signature) => read_signature(signature),
            None => Ok((PyTuple::empty(py), Vec::new())),
        });
        let (pairs, values) = read?;
        Ok(Self {
            name: name.clone().unbind(),
            keyword: keyword.clone().unbind(),
            signature: pairs.unbind(),
            value: SymbolFrame::from(FunctionFrame::new(identifier, keyword_value, values)),
            hash: OnceLock::new(),
            slots,
        })
    }

    /// Always true: the built-in frames are immutable.
    #[getter]
    const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: the built-in frames are always frozen.
    const fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: the built-in frames are always frozen.
    const fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        dataclass_eq(slf.as_any(), &slf.get().value, other)
    }

    fn __hash__(slf: &Bound<'_, Self>) -> PyResult<u64> {
        let this = slf.get();
        cached_hash(slf.py(), &this.hash, &this.value)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        format_dataclass_repr(
            &slf.get_type(),
            &[
                ("name", this.name.bind(py)),
                ("keyword", this.keyword.bind(py)),
                ("signature", this.signature.bind(py).as_any()),
            ],
        )
    }

    /// Return whether `other` is a frame of exactly this class with the
    /// same name, keyword and qualifiers, and structurally equivalent
    /// types.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        compare_same_class(slf.as_any(), &slf.get().value, other, true)
    }

    /// Return whether `other` is structurally equivalent: the frame binds
    /// nothing, so no renaming applies.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        compare_same_class(slf.as_any(), &slf.get().value, other, true)
    }

    /// Return whether `other` is structurally equivalent; `renaming` must
    /// be an `AlphaRenaming`, and is not consulted.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        read_renaming(renaming)?;
        compare_same_class(slf.as_any(), &slf.get().value, other, true)
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
                this.name.bind(py).clone(),
                this.keyword.bind(py).clone(),
                this.signature.bind(py).clone().into_any(),
            ],
        )?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the data payload `{"name": .., "keyword": .., "signature":
    /// [{"type_qualifier": .., "type": ..}, ..]}`.
    fn serialize_data_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let SymbolFrame::Function(frame) = &self.value else {
            unreachable!("a function frame holds a function");
        };
        let entries = PyList::empty(py);
        for (pair, (qualifier, _)) in self.signature.bind(py).iter().zip(frame.signature()) {
            let entry = PyDict::new(py);
            entry.set_item(intern!(py, "type_qualifier"), qualifier.as_str())?;
            entry.set_item(intern!(py, "type"), serialize_nested(&pair.get_item(1)?)?)?;
            entries.append(entry)?;
        }
        let payload = PyDict::new(py);
        payload.set_item(intern!(py, "name"), serialize_identifier(py, frame.name())?)?;
        payload.set_item(intern!(py, "keyword"), frame.keyword().as_str())?;
        payload.set_item(intern!(py, "signature"), entries)?;
        Ok(payload)
    }

    /// Return the frame of a data payload.
    ///
    /// Raises `DeserializationDictStructureError` for a malformed payload,
    /// and `DeserializationValueError` for a keyword or a qualifier that is
    /// not a member's value, or a type whose payload holds a bad value.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let expected = || -> PyResult<Vec<(&str, Bound<'py, PyAny>)>> {
            Ok(vec![
                ("name", py.get_type::<PyDict>().into_any()),
                ("keyword", py.get_type::<PyString>().into_any()),
                ("signature", py.get_type::<PyList>().into_any()),
            ])
        };
        let malformed = || -> PyResult<PyErr> { structure_error(cls, &expected()?, data) };
        let Ok(mapping) = data.cast::<PyDict>() else {
            return Err(malformed()?);
        };
        let field = |name: &str| mapping.get_item(name);
        let (Some(name), Some(keyword), Some(signature)) =
            (field("name")?, field("keyword")?, field("signature")?)
        else {
            return Err(malformed()?);
        };
        let Ok(entries) = signature.cast::<PyList>() else {
            return Err(malformed()?);
        };
        if !is_serialized_dict(&name)? || !keyword.is_instance_of::<PyString>() {
            return Err(malformed()?);
        }
        let mut parts = Vec::with_capacity(entries.len());
        for entry in entries.iter() {
            let part = if is_serialized_dict(&entry)? {
                let entry = entry.cast::<PyDict>()?;
                match (entry.get_item("type_qualifier")?, entry.get_item("type")?) {
                    (Some(qualifier), Some(ty))
                        if qualifier.is_instance_of::<PyString>() && is_serialized_dict(&ty)? =>
                    {
                        Some((qualifier, ty))
                    }
                    _ => None,
                }
            } else {
                None
            };
            let Some(part) = part else {
                return Err(malformed()?);
            };
            parts.push(part);
        }
        let invalid = |text: String| -> PyResult<PyErr> {
            Ok(crate::exceptions::DESERIALIZATION_VALUE_ERROR
                .err(py, (format!("Invalid function frame values: {text}"),)))
        };
        let keyword_text = keyword.cast::<PyString>()?.to_str()?;
        let Ok(keyword_value) = keyword_text.parse::<FunctionKeyword>() else {
            return Err(invalid(format!(
                "{} is not a valid FunctionKeyword",
                keyword.repr()?
            ))?);
        };
        let pairs = PyList::empty(py);
        for (qualifier, ty) in parts {
            let Ok(value) = qualifier
                .cast::<PyString>()?
                .to_str()?
                .parse::<TypeQualifier>()
            else {
                return Err(invalid(format!(
                    "{} is not a valid TypeQualifier",
                    qualifier.repr()?
                ))?);
            };
            let ty = match deserialize_type(&ty) {
                Ok(ty) => ty,
                Err(error) if error.is_instance_of::<PyValueError>(py) => {
                    return Err(invalid(error.value(py).str()?.to_string())?);
                }
                Err(error) => return Err(error),
            };
            pairs.append(PyTuple::new(
                py,
                [type_qualifier_to_python(py, value)?, ty],
            )?)?;
        }
        cls.call1((
            deserialize_identifier(&name)?,
            keyword_to_python(py, keyword_value)?,
            PyTuple::new(py, pairs)?,
        ))
    }
}

// ---------------------------------------------------------------------------
// The V2 wire format
// ---------------------------------------------------------------------------

/// Return the wire form of the frame object `object`: a built-in frame's
/// core frame, or a Python-defined frame's foreign part.
///
/// # Errors
///
/// Raises the exception a Python-defined frame's hooks raise.
pub(crate) fn frame_wire_data(
    object: &Bound<'_, PyAny>,
) -> PyResult<fhy_core::symbol_table::wire::SymbolFrameData> {
    use fhy_core::symbol_table::wire::SymbolFrameData;
    crate::constraint::with_pending_errors(|| {
        match read_frame_value(object) {
            Some(frame) => SymbolFrameData::of(&frame),
            None => {
                crate::wire::foreign_of(&object.clone().unbind(), true).map(SymbolFrameData::custom)
            }
        }
        .map_err(|error| crate::wire::foreign_error(object.py(), &error))
    })
}

/// Return a new Python object of the built-in frame `frame`, built through
/// its public class, as a V2 payload decodes.
///
/// # Errors
///
/// Raises what building the object raises.
pub(crate) fn frame_to_python<'py>(
    py: Python<'py>,
    frame: &SymbolFrame,
) -> PyResult<Bound<'py, PyAny>> {
    let type_object = |value: &Type| {
        crate::types::run_in_context(py, None, |context| {
            crate::types::type_to_python(py, context, value)
        })
    };
    match frame {
        SymbolFrame::Import(frame) => {
            crate::python::cached_attr!(py, MODULE, "ImportSymbolTableFrame" => PyType)?
                .call1((crate::identifier::identifier_to_python(py, frame.name())?,))
        }
        SymbolFrame::Variable(frame) => {
            crate::python::cached_attr!(py, MODULE, "VariableSymbolTableFrame" => PyType)?.call1((
                crate::identifier::identifier_to_python(py, frame.name())?,
                type_object(frame.ty())?,
                crate::types::type_qualifier_to_python(py, frame.qualifier())?,
            ))
        }
        SymbolFrame::Function(frame) => {
            let signature = frame
                .signature()
                .iter()
                .map(|(qualifier, value)| {
                    PyTuple::new(
                        py,
                        [
                            crate::types::type_qualifier_to_python(py, *qualifier)?,
                            type_object(value)?,
                        ],
                    )
                })
                .collect::<PyResult<Vec<_>>>()?;
            crate::python::cached_attr!(py, MODULE, "FunctionSymbolTableFrame" => PyType)?.call1((
                crate::identifier::identifier_to_python(py, frame.name())?,
                keyword_to_python(py, frame.keyword())?,
                PyTuple::new(py, signature)?,
            ))
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "a frame of an unknown kind",
        )),
    }
}

/// Return the Python object of the frame wire form `data`: a built-in
/// frame built through its public class, or a Python-defined frame its
/// registered class decodes.
///
/// # Errors
///
/// Raises `DeserializationValueError` for a type extension the registry
/// cannot resolve, and what decoding a Python-defined frame raises.
pub(crate) fn frame_from_wire<'py>(
    cls: &Bound<'py, PyType>,
    data: fhy_core::symbol_table::wire::SymbolFrameData,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cls.py();
    if let Some(foreign) = data.foreign() {
        return crate::wire::resolve_frame(py, foreign);
    }
    let frame = crate::wire::build(cls, || data.build(&crate::wire::PyResolver))?;
    frame_to_python(py, &frame)
}
