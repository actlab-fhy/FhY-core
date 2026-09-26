//! The three entry classes: `RegisteredFunction`, `NativeFunction` and
//! `NativeConstant` (D-S7-9).
//!
//! Each is a frozen pyclass holding the Rust value, or for a built-in the
//! catalogue item, next to the Python objects its fields return, so
//! `entry.body is body` and `native.implementation is implementation` hold.
//! The public classes are thin subclasses that register themselves at
//! import. Equality, hashing and `repr` follow the fields, as the
//! dataclasses' did; a user entry pickles as a call of its class with its
//! fields, and a built-in one as a lookup of the built-in by name.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString, PyTuple, PyType};

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction, ComposedFunction};
use fhy_core::expression::registry::{
    ConstantValueError, FunctionDefinition, FunctionDefinitionError, NativeConstant, NativeFunction,
};
use fhy_core::expression::{Expression, FunctionName, FunctionSort, LiteralValue};
use fhy_core::identifier::Identifier;
use fhy_core::term::AlphaRenaming;
use fhy_core::types::checking::CallTarget;

use crate::dataclass::{
    OptionalArgument, build_argument_type_error, collect_tuple, format_dataclass_repr, hash_value,
    read_str,
};
use crate::error::{IntoPyErr, IntoPyResult};
use crate::frozen::build_frozen_mutation_error;
use crate::identifier::{identifier_to_python, restore_identifier};
use crate::public_class::PublicClass;

use super::super::literal::read_big_int;
use super::super::materialize::materialize_expression;
use super::super::node::read_expression;
use super::state;
use crate::term::read_renaming;

/// Raises `ValueError` with the core's text.
impl IntoPyErr for FunctionDefinitionError {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

/// Raises `ValueError` with the core's text.
impl IntoPyErr for ConstantValueError {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

// ---------------------------------------------------------------------------
// Sorts
// ---------------------------------------------------------------------------

/// Return `fhy_core.symbolic.expression.sort.FunctionSort`.
fn sort_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    CLASS.import(py, "fhy_core.symbolic.expression.sort", "FunctionSort")
}

/// Every sort, in the order of [`sort_index`].
const SORTS: [FunctionSort; 4] = [
    FunctionSort::Bool,
    FunctionSort::Nat,
    FunctionSort::Int,
    FunctionSort::Real,
];

/// Return the position of `sort` in [`SORTS`].
fn sort_index(sort: FunctionSort) -> usize {
    match sort {
        FunctionSort::Bool => 0,
        FunctionSort::Nat => 1,
        FunctionSort::Int => 2,
        FunctionSort::Real => 3,
    }
}

/// Return the Python `FunctionSort` member of `sort`.
pub(crate) fn sort_to_python(py: Python<'_>, sort: FunctionSort) -> PyResult<Bound<'_, PyAny>> {
    static MEMBERS: PyOnceLock<[Py<PyAny>; 4]> = PyOnceLock::new();
    let members = MEMBERS.get_or_try_init(py, || -> PyResult<[Py<PyAny>; 4]> {
        let class = sort_class(py)?;
        let [bool_sort, nat, int, real] = SORTS.map(|sort| class.call1((sort.as_str(),)));
        Ok([
            bool_sort?.unbind(),
            nat?.unbind(),
            int?.unbind(),
            real?.unbind(),
        ])
    })?;
    Ok(members[sort_index(sort)].bind(py).clone())
}

/// Return the Rust sort of the Python `FunctionSort` member `value`, the
/// argument `field` of `owner`.
///
/// # Errors
///
/// Raises `TypeError` if `value` is not a `FunctionSort`.
pub(crate) fn read_sort(
    value: &Bound<'_, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<FunctionSort> {
    let py = value.py();
    if value.is_instance(sort_class(py)?)? {
        let text = value.getattr(intern!(py, "value"))?;
        if let Ok(sort) = text.cast::<PyString>()?.to_str()?.parse::<FunctionSort>() {
            return Ok(sort);
        }
    }
    Err(build_argument_type_error(
        owner,
        field,
        "a FunctionSort",
        value,
    )?)
}

/// Return the tuple of the iterable of sorts `values` and their Rust sorts.
fn read_sorts<'py>(
    values: &Bound<'py, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<(Bound<'py, PyTuple>, Vec<FunctionSort>)> {
    let values = collect_tuple(values)?;
    let sorts = values
        .iter()
        .map(|value| read_sort(&value, owner, field))
        .collect::<PyResult<Vec<_>>>()?;
    Ok((values, sorts))
}

/// Return the tuple of the Python members of `sorts`.
fn sorts_to_python<'py>(py: Python<'py>, sorts: &[FunctionSort]) -> PyResult<Bound<'py, PyTuple>> {
    let members = sorts
        .iter()
        .map(|sort| sort_to_python(py, *sort))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, members)
}

// ---------------------------------------------------------------------------
// Shared members
// ---------------------------------------------------------------------------

/// Return the name `value`, the `name` argument of `owner`, as a `str`
/// and a function name.
///
/// # Errors
///
/// Raises `TypeError` for a value that is not a `str`, and `ValueError`
/// with the core's text for an empty name or a built-in function's name.
fn read_name<'py>(
    value: &Bound<'py, PyAny>,
    owner: &str,
) -> PyResult<(Bound<'py, PyString>, FunctionName)> {
    let name = read_str(value, owner, "name")?;
    let function_name = FunctionName::try_new(name.to_str()?).into_py_result()?;
    Ok((name.clone(), function_name))
}

/// Return the given argument, or raise the `TypeError` for the missing
/// argument `field` of `owner`'s constructor.
fn require<'py>(
    argument: OptionalArgument<'py>,
    owner: &str,
    field: &str,
) -> PyResult<Bound<'py, PyAny>> {
    match argument {
        OptionalArgument::Given(value) => Ok(value),
        OptionalArgument::Omitted => Err(PyTypeError::new_err(format!(
            "{owner}() missing required argument: '{field}'"
        ))),
    }
}

/// Implement the `#[pymethods]` of an entry class: its own `methods`, and
/// the frozen members, equality, hashing, `repr`, pickling and the public
/// class slot, from its `fields` method returning the tuple of its dataclass
/// fields in order and its `FIELD_NAMES`.
macro_rules! impl_entry_protocols {
    ($class:ty, $name:literal, { $($methods:tt)* }) => {
        impl $class {
            /// Return the public Python class registered for this class.
            pub(super) fn public_class() -> &'static PublicClass {
                static PUBLIC_CLASS: PublicClass = PublicClass::new($name);
                &PUBLIC_CLASS
            }
        }

        #[pymethods]
        impl $class {
            $($methods)*

            /// Return `True`: entries are always frozen.
            #[getter]
            fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
                true
            }

            /// Do nothing: entries are always frozen.
            fn freeze(_slf: &Bound<'_, Self>) {}

            /// Do nothing: entries are always frozen, and mutating one raises.
            fn assert_frozen(_slf: &Bound<'_, Self>) {}

            fn __setattr__(
                slf: &Bound<'_, Self>,
                name: &str,
                value: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                let _ = value;
                Err(build_frozen_mutation_error(slf, "modify", name)?)
            }

            fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
                Err(build_frozen_mutation_error(slf, "delete", name)?)
            }

            /// Return whether `other` has exactly this class and equal fields.
            fn __eq__<'py>(
                slf: &Bound<'py, Self>,
                other: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let py = slf.py();
                if !slf.get_type().is(other.get_type()) {
                    return Ok(py.NotImplemented().into_bound(py));
                }
                let is_equal = slf.get().has_equal_fields(other.cast::<Self>()?.get(), py)?;
                Ok(PyBool::new(py, is_equal).to_owned().into_any())
            }

            /// Return the hash of the fields, as a frozen dataclass does.
            fn __hash__(slf: &Bound<'_, Self>) -> PyResult<u64> {
                slf.get().hash_fields(slf.py())
            }

            fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                let fields = slf.get().fields(slf.py())?;
                let values: Vec<Bound<'_, PyAny>> = fields.iter().collect();
                let named: Vec<(&str, &Bound<'_, PyAny>)> =
                    Self::FIELD_NAMES.iter().copied().zip(values.iter()).collect();
                format_dataclass_repr(&slf.get_type(), &named)
            }

            /// Pickle a user entry as a call of its class with its fields,
            /// and a built-in one as the built-in entry of its name.
            fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
                let py = slf.py();
                let this = slf.get();
                if this.is_builtin() {
                    let lookup = slf.get_type().getattr(intern!(py, "_builtin"))?;
                    let arguments = PyTuple::new(py, [this.name.bind(py)])?;
                    return PyTuple::new(py, [lookup, arguments.into_any()]);
                }
                PyTuple::new(
                    py,
                    [slf.get_type().into_any(), this.fields(py)?.into_any()],
                )
            }

            /// Return the built-in entry named `name`, the unpickling of one.
            ///
            /// Raises `KeyError` if no built-in has the name.
            #[classmethod]
            fn _builtin<'py>(cls: &Bound<'py, PyType>, name: &str) -> PyResult<Bound<'py, PyAny>> {
                state::find_builtin(cls.py(), name)?.ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(name.to_owned())
                })
            }

            /// Register `cls` as the public class.
            ///
            /// Raises `RuntimeError` if another public class is registered.
            #[classmethod]
            fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
                Self::public_class().register(cls)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// RegisteredFunction
// ---------------------------------------------------------------------------

/// What a `RegisteredFunction` defines.
enum FunctionSource {
    /// A user function.
    User(FunctionDefinition),
    /// A composed built-in of the catalogue.
    Builtin(&'static ComposedFunction),
}

/// The contents of a function entry the binding builds, handed to the public
/// class's constructor. Not exported.
#[pyclass(frozen, module = "fhy_core._rs", name = "_RegisteredFunctionSeed")]
struct RegisteredFunctionSeed {
    entry: std::sync::Mutex<Option<PyRegisteredFunction>>,
}

/// A named function whose body is an expression over its parameters,
/// backed by the Rust `FunctionDefinition`, or for a built-in by its
/// catalogue definition.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "RegisteredFunction")]
pub(crate) struct PyRegisteredFunction {
    source: FunctionSource,
    /// The name.
    #[pyo3(get)]
    name: Py<PyString>,
    /// The parameter identifiers, in order.
    #[pyo3(get)]
    parameters: Py<PyTuple>,
    /// The parameters' sorts, in order.
    #[pyo3(get)]
    parameter_sorts: Py<PyTuple>,
    /// The result's sort.
    #[pyo3(get)]
    result_sort: Py<PyAny>,
    /// The body expression.
    #[pyo3(get)]
    body: Py<PyAny>,
}

impl PyRegisteredFunction {
    const FIELD_NAMES: [&'static str; 5] = [
        "name",
        "parameters",
        "parameter_sorts",
        "result_sort",
        "body",
    ];

    /// Return the dataclass fields, in order.
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                self.name.bind(py).clone().into_any(),
                self.parameters.bind(py).clone().into_any(),
                self.parameter_sorts.bind(py).clone().into_any(),
                self.result_sort.bind(py).clone(),
                self.body.bind(py).clone(),
            ],
        )
    }

    /// Return whether `other`'s fields equal this entry's: the same name,
    /// parameters, sorts and a structurally equal body. The Rust values
    /// answer as the field objects' `==` would, without calling Python.
    fn has_equal_fields(&self, other: &Self, py: Python<'_>) -> PyResult<bool> {
        Ok(
            self.name.bind(py).to_str()? == other.name.bind(py).to_str()?
                && self.rust_parameters() == other.rust_parameters()
                && self.rust_parameter_sorts() == other.rust_parameter_sorts()
                && self.rust_result_sort() == other.rust_result_sort()
                && self.rust_body() == other.rust_body(),
        )
    }

    /// Return the hash of the fields: the name, the parameters' ids, the
    /// sorts and the body's structural hash.
    fn hash_fields(&self, py: Python<'_>) -> PyResult<u64> {
        let parameter_ids: Vec<u64> = self.rust_parameters().iter().map(Identifier::id).collect();
        let hash = hash_value(&(
            self.name.bind(py).to_str()?,
            parameter_ids,
            self.rust_parameter_sorts(),
            self.rust_result_sort(),
            self.body.bind(py).hash()?,
        ));
        Ok(hash)
    }

    /// Return whether this is a built-in's entry.
    fn is_builtin(&self) -> bool {
        matches!(self.source, FunctionSource::Builtin(_))
    }

    /// Return the user definition, or `None` for a built-in.
    pub(super) fn definition(&self) -> Option<&FunctionDefinition> {
        match &self.source {
            FunctionSource::User(definition) => Some(definition),
            FunctionSource::Builtin(_) => None,
        }
    }

    /// Return the Rust parameters.
    fn rust_parameters(&self) -> &[Identifier] {
        match &self.source {
            FunctionSource::User(definition) => definition.parameters(),
            FunctionSource::Builtin(composed) => composed.parameters(),
        }
    }

    /// Return the Rust body.
    fn rust_body(&self) -> &Expression {
        match &self.source {
            FunctionSource::User(definition) => definition.body(),
            FunctionSource::Builtin(composed) => composed.body(),
        }
    }

    /// Return the Rust parameter sorts.
    fn rust_parameter_sorts(&self) -> &[FunctionSort] {
        match &self.source {
            FunctionSource::User(definition) => definition.parameter_sorts(),
            FunctionSource::Builtin(composed) => composed.function().parameter_sorts(),
        }
    }

    /// Return the Rust result sort.
    fn rust_result_sort(&self) -> FunctionSort {
        match &self.source {
            FunctionSource::User(definition) => definition.result_sort(),
            FunctionSource::Builtin(composed) => composed.function().result_sort(),
        }
    }

    /// Return the entry of the composed built-in `composed`, an instance of
    /// the public class, with its parameters and body built as Python
    /// objects.
    pub(super) fn build_builtin<'py>(
        py: Python<'py>,
        composed: &'static ComposedFunction,
    ) -> PyResult<Bound<'py, PyAny>> {
        let function = composed.function();
        let parameters = composed
            .parameters()
            .iter()
            .map(|parameter| identifier_to_python(py, parameter))
            .collect::<PyResult<Vec<_>>>()?;
        let entry = Self {
            source: FunctionSource::Builtin(composed),
            name: PyString::new(py, function.name()).unbind(),
            parameters: PyTuple::new(py, parameters)?.unbind(),
            parameter_sorts: sorts_to_python(py, function.parameter_sorts())?.unbind(),
            result_sort: sort_to_python(py, function.result_sort())?.unbind(),
            body: materialize_expression(py, composed.body())?.unbind(),
        };
        let seed = RegisteredFunctionSeed {
            entry: std::sync::Mutex::new(Some(entry)),
        };
        Self::public_class().get(py)?.call1((seed,))
    }

    /// Return whether the binder terms `self` and `other` are
    /// alpha-equivalent under `renaming`, extended by the frame pairing
    /// their parameters.
    fn is_alpha_equivalent_under_renaming(&self, other: &Self, renaming: &AlphaRenaming) -> bool {
        let (parameters, other_parameters) = (self.rust_parameters(), other.rust_parameters());
        if parameters.len() != other_parameters.len()
            || self.rust_parameter_sorts() != other.rust_parameter_sorts()
            || self.rust_result_sort() != other.rust_result_sort()
        {
            return false;
        }
        let mut renaming = renaming.clone();
        if renaming
            .enter_binders(parameters, other_parameters)
            .is_err()
        {
            return false;
        }
        self.rust_body()
            .is_alpha_equivalent_under(other.rust_body(), &renaming)
    }
}

impl_entry_protocols!(PyRegisteredFunction, "RegisteredFunction", {
    /// Create the function `name` of the identifiers `parameters`, whose
    /// sorts are `parameter_sorts` in order, returning `result_sort`, as the
    /// expression `body`.
    ///
    /// The body is not checked: which identifiers it may refer to depends
    /// on the registry, which checks them when the function is registered.
    ///
    /// Raises `TypeError` for an argument of the wrong type, and
    /// `ValueError` with the core's text for an empty name, a built-in
    /// function's name, a sort count other than the parameter count, or a
    /// repeated parameter.
    #[new]
    #[pyo3(signature = (
        name,
        parameters = OptionalArgument::Omitted,
        parameter_sorts = OptionalArgument::Omitted,
        result_sort = OptionalArgument::Omitted,
        body = OptionalArgument::Omitted,
    ))]
    fn new(
        name: &Bound<'_, PyAny>,
        parameters: OptionalArgument<'_>,
        parameter_sorts: OptionalArgument<'_>,
        result_sort: OptionalArgument<'_>,
        body: OptionalArgument<'_>,
    ) -> PyResult<Self> {
        const OWNER: &str = "RegisteredFunction";
        let py = name.py();
        if let Ok(seed) = name.cast::<RegisteredFunctionSeed>() {
            if let Some(entry) = seed
                .get()
                .entry
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
            {
                return Ok(entry);
            }
        }
        let (name, function_name) = read_name(name, OWNER)?;
        let parameters = collect_tuple(&require(parameters, OWNER, "parameters")?)?;
        let rust_parameters = parameters
            .iter()
            .map(|parameter| restore_identifier(&parameter, OWNER, "parameters"))
            .collect::<PyResult<Vec<_>>>()?;
        let (parameter_sorts, rust_sorts) = read_sorts(
            &require(parameter_sorts, OWNER, "parameter_sorts")?,
            OWNER,
            "parameter_sorts",
        )?;
        let result_sort = require(result_sort, OWNER, "result_sort")?;
        let rust_result_sort = read_sort(&result_sort, OWNER, "result_sort")?;
        let body = require(body, OWNER, "body")?;
        let rust_body = read_expression(&body, OWNER, "body")?
            .get()
            .expression()
            .clone();
        let definition = FunctionDefinition::try_new(
            function_name,
            rust_parameters,
            rust_sorts,
            rust_result_sort,
            rust_body,
        )
        .into_py_result()?;
        let _ = py;
        Ok(Self {
            source: FunctionSource::User(definition),
            name: name.unbind(),
            parameters: parameters.unbind(),
            parameter_sorts: parameter_sorts.unbind(),
            result_sort: result_sort.unbind(),
            body: body.unbind(),
        })
    }

    /// Return whether `other` is a function of exactly this class with the
    /// same parameters, sorts and structurally equal body; the name is not
    /// compared.
    fn is_structurally_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> bool {
        if !slf.get_type().is(other.get_type()) {
            return false;
        }
        let Ok(other) = other.cast::<Self>() else {
            return false;
        };
        let (this, other) = (slf.get(), other.get());
        this.rust_parameters() == other.rust_parameters()
            && this.rust_parameter_sorts() == other.rust_parameter_sorts()
            && this.rust_result_sort() == other.rust_result_sort()
            && this.rust_body() == other.rust_body()
    }

    /// Return whether `other` is this function with its parameters renamed:
    /// a function of exactly this class with as many parameters, the same
    /// sorts, and a body equal under the pairing of the parameters. The
    /// name is not compared, and a pairing that is not injective is no
    /// renaming.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> bool {
        if !slf.get_type().is(other.get_type()) {
            return false;
        }
        let Ok(other) = other.cast::<Self>() else {
            return false;
        };
        slf.get()
            .is_alpha_equivalent_under_renaming(other.get(), &AlphaRenaming::default())
    }

    /// Return whether `other` is this function with its parameters renamed,
    /// as `is_alpha_equivalent` compares them, with the free identifiers of
    /// the bodies compared under the `AlphaRenaming` `renaming`.
    ///
    /// Raises `TypeError` if `renaming` is not an `AlphaRenaming`.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let renaming = read_renaming(renaming)?;
        if !slf.get_type().is(other.get_type()) {
            return Ok(false);
        }
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        Ok(slf
            .get()
            .is_alpha_equivalent_under_renaming(other.get(), renaming.get().value().renaming()))
    }
});

// ---------------------------------------------------------------------------
// NativeFunction
// ---------------------------------------------------------------------------

/// What a `NativeFunction` declares.
enum NativeSource {
    /// A user function.
    User(NativeFunction),
    /// A native built-in of the catalogue, whose name the entry holds.
    Builtin,
}

/// The contents of a native entry the binding builds. Not exported.
#[pyclass(frozen, module = "fhy_core._rs", name = "_NativeFunctionSeed")]
struct NativeFunctionSeed {
    entry: std::sync::Mutex<Option<PyNativeFunction>>,
}

/// Return the Python function checking a native implementation's arity,
/// `_check_native_implementation_arity(name, count, implementation)`.
fn arity_check(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static FUNCTION: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    FUNCTION.import(
        py,
        "fhy_core.symbolic.expression.registry.entries",
        "_check_native_implementation_arity",
    )
}

/// A named function computed by a Python callable, backed by the Rust
/// `NativeFunction`, or for a built-in by its catalogue item.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "NativeFunction")]
pub(crate) struct PyNativeFunction {
    source: NativeSource,
    /// The name.
    #[pyo3(get)]
    name: Py<PyString>,
    /// The parameters' sorts, in order.
    #[pyo3(get)]
    parameter_sorts: Py<PyTuple>,
    /// The result's sort.
    #[pyo3(get)]
    result_sort: Py<PyAny>,
    /// The Python callable computing the function.
    #[pyo3(get)]
    implementation: Py<PyAny>,
}

impl PyNativeFunction {
    const FIELD_NAMES: [&'static str; 4] =
        ["name", "parameter_sorts", "result_sort", "implementation"];

    /// Return the dataclass fields, in order.
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                self.name.bind(py).clone().into_any(),
                self.parameter_sorts.bind(py).clone().into_any(),
                self.result_sort.bind(py).clone(),
                self.implementation.bind(py).clone(),
            ],
        )
    }

    /// Return whether `other`'s fields equal this entry's, by Python `==`.
    fn has_equal_fields(&self, other: &Self, py: Python<'_>) -> PyResult<bool> {
        self.fields(py)?.eq(other.fields(py)?)
    }

    /// Return the hash of the field tuple.
    fn hash_fields(&self, py: Python<'_>) -> PyResult<u64> {
        let hash = i64::try_from(self.fields(py)?.hash()?)?;
        Ok(u64::from_ne_bytes(hash.to_ne_bytes()))
    }

    /// Return whether this is a built-in's entry.
    fn is_builtin(&self) -> bool {
        matches!(self.source, NativeSource::Builtin)
    }

    /// Return the Python callable computing the function.
    pub(super) fn implementation<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.implementation.bind(py).clone()
    }

    /// Return the user declaration, or `None` for a built-in.
    pub(super) fn declaration(&self) -> Option<&NativeFunction> {
        match &self.source {
            NativeSource::User(function) => Some(function),
            NativeSource::Builtin => None,
        }
    }

    /// Return the entry of the native built-in `function` computed by
    /// `implementation`, an instance of the public class.
    pub(super) fn build_builtin<'py>(
        py: Python<'py>,
        function: BuiltinFunction,
        implementation: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let entry = Self {
            source: NativeSource::Builtin,
            name: PyString::new(py, function.name()).unbind(),
            parameter_sorts: sorts_to_python(py, function.parameter_sorts())?.unbind(),
            result_sort: sort_to_python(py, function.result_sort())?.unbind(),
            implementation: implementation.clone().unbind(),
        };
        let seed = NativeFunctionSeed {
            entry: std::sync::Mutex::new(Some(entry)),
        };
        Self::public_class().get(py)?.call1((seed,))
    }
}

impl_entry_protocols!(PyNativeFunction, "NativeFunction", {
    /// Create the function `name` taking arguments of `parameter_sorts`, in
    /// order, returning `result_sort`, computed by the callable
    /// `implementation`.
    ///
    /// Raises `TypeError` for an argument of the wrong type, and
    /// `ValueError` for an empty name, a built-in function's name (the
    /// core's text), or an implementation whose inspectable signature
    /// cannot take as many positional arguments as there are sorts.
    #[new]
    #[pyo3(signature = (
        name,
        parameter_sorts = OptionalArgument::Omitted,
        result_sort = OptionalArgument::Omitted,
        implementation = OptionalArgument::Omitted,
    ))]
    fn new(
        name: &Bound<'_, PyAny>,
        parameter_sorts: OptionalArgument<'_>,
        result_sort: OptionalArgument<'_>,
        implementation: OptionalArgument<'_>,
    ) -> PyResult<Self> {
        const OWNER: &str = "NativeFunction";
        let py = name.py();
        if let Ok(seed) = name.cast::<NativeFunctionSeed>() {
            if let Some(entry) = seed
                .get()
                .entry
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
            {
                return Ok(entry);
            }
        }
        let (name, function_name) = read_name(name, OWNER)?;
        let (parameter_sorts, rust_sorts) = read_sorts(
            &require(parameter_sorts, OWNER, "parameter_sorts")?,
            OWNER,
            "parameter_sorts",
        )?;
        let result_sort = require(result_sort, OWNER, "result_sort")?;
        let rust_result_sort = read_sort(&result_sort, OWNER, "result_sort")?;
        let implementation = require(implementation, OWNER, "implementation")?;
        if !implementation.is_callable() {
            return Err(build_argument_type_error(
                OWNER,
                "implementation",
                "callable",
                &implementation,
            )?);
        }
        arity_check(py)?.call1((&name, rust_sorts.len(), &implementation))?;
        Ok(Self {
            source: NativeSource::User(NativeFunction::new(
                function_name,
                rust_sorts,
                rust_result_sort,
            )),
            name: name.unbind(),
            parameter_sorts: parameter_sorts.unbind(),
            result_sort: result_sort.unbind(),
            implementation: implementation.unbind(),
        })
    }

    /// Build the entries of the built-ins once, each native built-in
    /// computed by its `BuiltinNativeImplementation`, the core's kernel
    /// (D-S9-9). `builtins.py` calls it at import; a later call does
    /// nothing.
    #[classmethod]
    fn _install_builtins(cls: &Bound<'_, PyType>) -> PyResult<()> {
        state::install_builtins(cls.py())
    }
});

// ---------------------------------------------------------------------------
// NativeConstant
// ---------------------------------------------------------------------------

/// What a `NativeConstant` declares.
enum ConstantSource {
    /// A user constant.
    User(NativeConstant),
    /// A built-in constant of the catalogue, whose name the entry holds.
    Builtin,
}

/// The contents of a constant entry the binding builds. Not exported.
#[pyclass(frozen, module = "fhy_core._rs", name = "_NativeConstantSeed")]
struct NativeConstantSeed {
    entry: std::sync::Mutex<Option<PyNativeConstant>>,
}

/// Return the literal value of the Python `value`, the `value` argument of
/// `owner`.
///
/// # Errors
///
/// Raises `TypeError` unless `value` is a `bool`, an `int` or a `float`.
fn read_constant_value(value: &Bound<'_, PyAny>, owner: &str) -> PyResult<LiteralValue> {
    if let Ok(boolean) = value.cast::<PyBool>() {
        return Ok(LiteralValue::Bool(boolean.is_true()));
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(LiteralValue::Int(read_big_int(value)?));
    }
    if let Ok(float) = value.cast::<PyFloat>() {
        return Ok(LiteralValue::Float(float.value()));
    }
    Err(build_argument_type_error(
        owner,
        "value",
        "a bool, an int or a float",
        value,
    )?)
}

/// A named constant, backed by the Rust `NativeConstant`, or for a
/// built-in by its catalogue item.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "NativeConstant")]
pub(crate) struct PyNativeConstant {
    source: ConstantSource,
    /// The name.
    #[pyo3(get)]
    name: Py<PyString>,
    /// The sort.
    #[pyo3(get)]
    sort: Py<PyAny>,
    /// The value: a `bool`, an `int` or a `float`.
    #[pyo3(get)]
    value: Py<PyAny>,
}

impl PyNativeConstant {
    const FIELD_NAMES: [&'static str; 3] = ["name", "sort", "value"];

    /// Return the dataclass fields, in order.
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                self.name.bind(py).clone().into_any(),
                self.sort.bind(py).clone(),
                self.value.bind(py).clone(),
            ],
        )
    }

    /// Return whether `other`'s fields equal this entry's, by Python `==`.
    fn has_equal_fields(&self, other: &Self, py: Python<'_>) -> PyResult<bool> {
        self.fields(py)?.eq(other.fields(py)?)
    }

    /// Return the hash of the field tuple.
    fn hash_fields(&self, py: Python<'_>) -> PyResult<u64> {
        let hash = i64::try_from(self.fields(py)?.hash()?)?;
        Ok(u64::from_ne_bytes(hash.to_ne_bytes()))
    }

    /// Return whether this is a built-in's entry.
    fn is_builtin(&self) -> bool {
        matches!(self.source, ConstantSource::Builtin)
    }

    /// Return the user declaration, or `None` for a built-in.
    pub(super) fn declaration(&self) -> Option<&NativeConstant> {
        match &self.source {
            ConstantSource::User(constant) => Some(constant),
            ConstantSource::Builtin => None,
        }
    }

    /// Return the entry of the built-in `constant`, an instance of the
    /// public class.
    pub(super) fn build_builtin(
        py: Python<'_>,
        constant: BuiltinConstant,
    ) -> PyResult<Bound<'_, PyAny>> {
        let entry = Self {
            source: ConstantSource::Builtin,
            name: PyString::new(py, constant.name()).unbind(),
            sort: sort_to_python(py, constant.sort())?.unbind(),
            value: PyFloat::new(py, constant.value()).into_any().unbind(),
        };
        let seed = NativeConstantSeed {
            entry: std::sync::Mutex::new(Some(entry)),
        };
        Self::public_class().get(py)?.call1((seed,))
    }
}

impl_entry_protocols!(PyNativeConstant, "NativeConstant", {
    /// Create the constant `name` of the sort `sort` holding `value`.
    ///
    /// Raises `TypeError` for an argument of the wrong type, a value that is
    /// not a `bool`, an `int` or a `float` included, and `ValueError` with
    /// the core's text for an empty name, a built-in function's name, or a
    /// value the sort does not accept: a `bool` only `BOOL`, a non-negative
    /// `int` `NAT`, an `int` `INT`, and an `int` or a `float` `REAL`.
    #[new]
    #[pyo3(signature = (
        name,
        sort = OptionalArgument::Omitted,
        value = OptionalArgument::Omitted,
    ))]
    fn new(
        name: &Bound<'_, PyAny>,
        sort: OptionalArgument<'_>,
        value: OptionalArgument<'_>,
    ) -> PyResult<Self> {
        const OWNER: &str = "NativeConstant";
        if let Ok(seed) = name.cast::<NativeConstantSeed>() {
            if let Some(entry) = seed
                .get()
                .entry
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
            {
                return Ok(entry);
            }
        }
        let (name, constant_name) = read_name(name, OWNER)?;
        let sort = require(sort, OWNER, "sort")?;
        let rust_sort = read_sort(&sort, OWNER, "sort")?;
        let value = require(value, OWNER, "value")?;
        let rust_value = read_constant_value(&value, OWNER)?;
        let constant =
            NativeConstant::try_new(constant_name, rust_sort, rust_value).into_py_result()?;
        Ok(Self {
            source: ConstantSource::User(constant),
            name: name.unbind(),
            sort: sort.unbind(),
            value: value.unbind(),
        })
    }
});

// ---------------------------------------------------------------------------
// Call targets
// ---------------------------------------------------------------------------

/// Return what the entry `object` is as a type checker's call target, or
/// `None` if it is no entry: a function with its Rust sorts, or a
/// constant.
///
/// # Errors
///
/// Raises `TypeError` if a native built-in's sorts are no `FunctionSort`s,
/// which the class never builds.
pub(crate) fn read_call_target(object: &Bound<'_, PyAny>) -> PyResult<Option<CallTarget>> {
    let py = object.py();
    if let Ok(function) = object.cast::<PyRegisteredFunction>() {
        let function = function.get();
        return Ok(Some(CallTarget::Function {
            name: function.name.bind(py).to_str()?.to_owned(),
            parameter_sorts: function.rust_parameter_sorts().to_vec(),
            result_sort: function.rust_result_sort(),
        }));
    }
    if let Ok(function) = object.cast::<PyNativeFunction>() {
        let function = function.get();
        let name = function.name.bind(py).to_str()?.to_owned();
        return Ok(Some(match function.declaration() {
            Some(declaration) => CallTarget::Function {
                name,
                parameter_sorts: declaration.parameter_sorts().to_vec(),
                result_sort: declaration.result_sort(),
            },
            None => CallTarget::Function {
                parameter_sorts: function
                    .parameter_sorts
                    .bind(py)
                    .iter()
                    .map(|sort| read_sort(&sort, "NativeFunction", "parameter_sorts"))
                    .collect::<PyResult<Vec<_>>>()?,
                result_sort: read_sort(
                    function.result_sort.bind(py),
                    "NativeFunction",
                    "result_sort",
                )?,
                name,
            },
        }));
    }
    if let Ok(constant) = object.cast::<PyNativeConstant>() {
        return Ok(Some(CallTarget::Constant {
            name: constant.get().name.bind(py).to_str()?.to_owned(),
        }));
    }
    Ok(None)
}
