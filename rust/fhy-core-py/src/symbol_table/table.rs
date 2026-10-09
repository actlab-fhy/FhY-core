//! `SymbolTable`, over the core's [`SymbolTable`].
//!
//! Each symbol's entry keeps the symbol's `Identifier` object, the frame
//! object, the frame's name, and, for a built-in frame, its core
//! [`SymbolFrame`], so two built-in frames of one class compare in Rust and
//! any other pair through the left frame's own Python method.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyString, PyTuple, PyType};

use fhy_core::identifier::Identifier;
use fhy_core::symbol_table::{Frame, SymbolFrame, SymbolTable, SymbolTableError};

use crate::diagnostic::diagnostic_class;
use crate::error::IntoPyErr;
use crate::identifier::{
    deserialize_identifier, identifier_to_python, restore_identifier, serialize_identifier,
};
use crate::util::dataclass::build_argument_type_error;
use crate::util::serialization::is_serialized_dict;

use super::frames::{MODULE, read_frame_value, structure_error};

/// The source of `verify`'s diagnostics.
const VERIFY_SOURCE: &str = "fhy_core.symbol_table.SymbolTable.verify";

/// The owner the argument errors name.
const OWNER: &str = "SymbolTable";

/// Raises `fhy_core.symbol_table.SymbolTableError` with the core's text.
impl IntoPyErr for SymbolTableError {
    fn into_py_err(self) -> PyErr {
        Python::attach(|py| {
            crate::util::exceptions::SYMBOL_TABLE_ERROR.err(py, (self.to_string(),))
        })
    }
}

/// Return the Python `SymbolTableError` of `error`.
fn raise(error: SymbolTableError) -> PyErr {
    error.into_py_err()
}

/// Return `fhy_core.symbol_table.SymbolTableFrame`.
fn frame_base_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    crate::util::python::cached_attr!(py, MODULE, "SymbolTableFrame" => PyType)
}

/// Return the module's logger, which the DEBUG lines go to.
fn logger(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static LOGGER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    LOGGER
        .get_or_try_init(py, || {
            Ok::<_, PyErr>(py.import(MODULE)?.getattr("_LOGGER")?.unbind())
        })
        .map(|logger| logger.bind(py))
}

/// Log `message` with `arguments` at DEBUG on the module's logger.
fn log_debug<'py>(
    py: Python<'py>,
    message: &str,
    arguments: impl IntoIterator<Item = Bound<'py, PyAny>>,
) -> PyResult<()> {
    let mut items = vec![PyString::new(py, message).into_any()];
    items.extend(arguments);
    logger(py)?.call_method1(intern!(py, "debug"), PyTuple::new(py, items)?)?;
    Ok(())
}

/// What the table holds for one symbol.
struct EntryData {
    /// The symbol's `Identifier` object.
    symbol: Py<PyAny>,
    /// The frame object.
    frame: Py<PyAny>,
    /// The frame's name, read when it was added.
    name: Identifier,
    /// The core frame of a built-in frame.
    native: Option<SymbolFrame>,
}

/// One symbol's entry, shared between clones of the table.
#[derive(Clone)]
struct Entry(Arc<EntryData>);

impl Frame for Entry {
    fn name(&self) -> &Identifier {
        &self.0.name
    }
}

impl Entry {
    /// Return the entry of `symbol` described by the frame object `frame`.
    ///
    /// Raises `TypeError` if `frame` is no `SymbolTableFrame`, or a frame
    /// Python defines whose `name` is no `Identifier`.
    fn new(symbol: &Bound<'_, PyAny>, frame: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = frame.py();
        let (name, native) = if let Some(native) = read_frame_value(frame) {
            (native.name().clone(), Some(native))
        } else {
            if !frame.is_instance(frame_base_class(py)?)? {
                return Err(build_argument_type_error(
                    OWNER,
                    "frame",
                    "a SymbolTableFrame",
                    frame,
                )?);
            }
            let name = frame.getattr(intern!(py, "name"))?;
            (restore_identifier(&name, "SymbolTableFrame", "name")?, None)
        };
        Ok(Self(Arc::new(EntryData {
            symbol: symbol.clone().unbind(),
            frame: frame.clone().unbind(),
            name,
            native,
        })))
    }

    /// Visit the symbol and the frame, if this handle is the entry's only
    /// one — an entry shared with another table, or with a copy a
    /// call is working on, is visited by none, since a reference must be
    /// visited at most once.
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        if Arc::strong_count(&self.0) != 1 {
            return Ok(());
        }
        visit.call(&self.0.symbol)?;
        visit.call(&self.0.frame)
    }

    /// Return the frame object.
    fn frame<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.0.frame.bind(py)
    }

    /// Return whether this entry's frame is structurally equivalent to
    /// `other`'s, as the frame's own `is_structurally_equivalent` answers.
    fn is_structurally_equivalent(&self, py: Python<'_>, other: &Self) -> PyResult<bool> {
        let (frame, other_frame) = (self.frame(py), other.frame(py));
        if let Some(native) = &self.0.native {
            if !frame.get_type().is(other_frame.get_type()) {
                return Ok(false);
            }
            if let Some(other_native) = &other.0.native {
                return super::frames::compare_natives(py, native, other_native);
            }
        }
        frame
            .call_method1(intern!(py, "is_structurally_equivalent"), (other_frame,))?
            .is_truthy()
    }
}

/// Return the Rust identifier of `object`, an argument `field`.
fn read_identifier(object: &Bound<'_, PyAny>, field: &str) -> PyResult<Identifier> {
    restore_identifier(object, OWNER, field)
}

/// A symbol table: namespaces in insertion order, each with an optional
/// parent and its symbols' frames in insertion order, backed by the core's
/// [`SymbolTable`].
#[pyclass(subclass, module = "fhy_core._rs", name = "SymbolTable")]
pub(crate) struct PySymbolTable {
    table: SymbolTable<Entry>,
    /// The dict `get_namespace` built of each namespace, by the namespace's
    /// id, until the namespace changes; each call returns a copy.
    namespace_dicts: HashMap<u64, Py<PyDict>>,
}

impl PySymbolTable {
    /// Add `namespace_name` under `parent_namespace_name`, without logging.
    ///
    /// Both are read before the table is borrowed, so a Python read they
    /// run may read the table.
    fn insert_namespace_checked(
        slf: &Bound<'_, Self>,
        namespace_name: &Bound<'_, PyAny>,
        parent_namespace_name: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let namespace = read_identifier(namespace_name, "namespace_name")?;
        let parent = parent_namespace_name
            .filter(|parent| !parent.is_none())
            .map(|parent| read_identifier(parent, "parent_namespace_name"))
            .transpose()?;
        let mut this = slf.borrow_mut();
        this.namespace_dicts.remove(&namespace.id());
        this.table.add_namespace(namespace, parent).map_err(raise)
    }

    /// Add `symbol_name` to `namespace_name`, without logging.
    ///
    /// The identifiers and the entry, whose frame's `name` a Python-defined
    /// frame answers, are read before the table is borrowed, so that read
    /// may read or change the table.
    fn insert_symbol_checked(
        slf: &Bound<'_, Self>,
        namespace_name: &Bound<'_, PyAny>,
        symbol_name: &Bound<'_, PyAny>,
        frame: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let namespace = read_identifier(namespace_name, "namespace_name")?;
        let symbol = read_identifier(symbol_name, "symbol_name")?;
        let entry = Entry::new(symbol_name, frame)?;
        let mut this = slf.borrow_mut();
        this.namespace_dicts.remove(&namespace.id());
        this.table
            .add_symbol(&namespace, symbol, entry)
            .map_err(raise)
    }

    /// Return the state a pickle keeps: each namespace, in order, as its
    /// `Identifier`, its parent's or `None`, and its `(symbol, frame)`
    /// pairs.
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let namespaces = PyList::empty(py);
        for namespace in self.table.namespaces() {
            let parent = match namespace.parent() {
                Some(parent) => identifier_to_python(py, parent)?,
                None => py.None().into_bound(py),
            };
            let symbols = PyList::empty(py);
            for (_, entry) in namespace.iter() {
                symbols.append((entry.0.symbol.bind(py), entry.frame(py)))?;
            }
            namespaces.append((identifier_to_python(py, namespace.name())?, parent, symbols))?;
        }
        Ok(namespaces)
    }
}

/// The error of a pickle state that `__setstate__` cannot read.
fn bad_state() -> PyErr {
    PyTypeError::new_err("SymbolTable state must be the one __reduce__ returns.")
}

#[pymethods]
impl PySymbolTable {
    /// Visit the entries this table alone holds and its cached namespace
    /// dicts, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for namespace in self.table.namespaces() {
            for (_, entry) in namespace.iter() {
                entry.traverse(&visit)?;
            }
        }
        crate::util::gc::traverse_all(&visit, self.namespace_dicts.values())
    }

    /// Drop the table's entries and dicts, for the cycle collector: the
    /// table is left empty, a consistent state.
    fn __clear__(&mut self) {
        self.table = SymbolTable::new();
        self.namespace_dicts.clear();
    }

    /// Create an empty table; arguments are accepted and ignored, so a
    /// subclass with its own `__init__` constructs.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self {
            table: SymbolTable::new(),
            namespace_dicts: HashMap::new(),
        }
    }

    /// Add the namespace `namespace_name`, whose parent is
    /// `parent_namespace_name`; the parent is not checked.
    ///
    /// Raises `SymbolTableError` if the namespace is defined.
    #[pyo3(signature = (namespace_name, parent_namespace_name = None))]
    fn add_namespace(
        slf: &Bound<'_, Self>,
        namespace_name: &Bound<'_, PyAny>,
        parent_namespace_name: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let py = slf.py();
        Self::insert_namespace_checked(slf, namespace_name, parent_namespace_name)?;
        let parent = parent_namespace_name.map_or_else(|| py.None().into_bound(py), Clone::clone);
        log_debug(
            py,
            "added namespace %s (parent=%s)",
            [namespace_name.clone(), parent],
        )
    }

    /// Return whether the namespace `namespace_name` is defined.
    fn is_namespace_defined(&self, namespace_name: &Bound<'_, PyAny>) -> PyResult<bool> {
        let namespace = read_identifier(namespace_name, "namespace_name")?;
        Ok(self.table.contains_namespace(&namespace))
    }

    /// Return the number of namespaces.
    fn get_number_of_namespaces(&self) -> usize {
        self.table.len()
    }

    /// Return a new dict of the namespace's symbols to their frames, in
    /// insertion order.
    ///
    /// Raises `SymbolTableError` if the namespace is not defined.
    fn get_namespace<'py>(
        slf: &Bound<'py, Self>,
        namespace_name: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let namespace = read_identifier(namespace_name, "namespace_name")?;
        let symbols = {
            let this = slf.borrow();
            if let Some(symbols) = this.namespace_dicts.get(&namespace.id()) {
                return symbols.bind(py).copy();
            }
            let Some(view) = this.table.namespace(&namespace) else {
                return Err(raise(SymbolTableError::NamespaceNotFound { namespace }));
            };
            let symbols = PyDict::new(py);
            for (_, entry) in view.iter() {
                symbols.set_item(entry.0.symbol.bind(py), entry.frame(py))?;
            }
            symbols
        };
        let copy = symbols.copy()?;
        slf.borrow_mut()
            .namespace_dicts
            .insert(namespace.id(), symbols.unbind());
        Ok(copy)
    }

    /// Remove the namespace `namespace_name` and its symbols.
    ///
    /// Raises `SymbolTableError` if it is not defined, or if another
    /// namespace names it as its parent.
    fn remove_namespace(slf: &Bound<'_, Self>, namespace_name: &Bound<'_, PyAny>) -> PyResult<()> {
        let namespace = read_identifier(namespace_name, "namespace_name")?;
        {
            let mut this = slf.borrow_mut();
            this.namespace_dicts.remove(&namespace.id());
            this.table.remove_namespace(&namespace).map_err(raise)?;
        }
        log_debug(slf.py(), "removed namespace %s", [namespace_name.clone()])
    }

    /// Remove `symbol_name` from `namespace_name`, which must hold it itself.
    ///
    /// Raises `SymbolTableError` if the namespace is not defined or does not
    /// itself hold the symbol.
    fn remove_symbol(
        slf: &Bound<'_, Self>,
        namespace_name: &Bound<'_, PyAny>,
        symbol_name: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let namespace = read_identifier(namespace_name, "namespace_name")?;
        let symbol = read_identifier(symbol_name, "symbol_name")?;
        {
            let mut this = slf.borrow_mut();
            this.namespace_dicts.remove(&namespace.id());
            this.table
                .remove_symbol(&namespace, &symbol)
                .map_err(raise)?;
        }
        log_debug(
            slf.py(),
            "removed symbol %s from namespace %s",
            [symbol_name.clone(), namespace_name.clone()],
        )
    }

    /// Copy every namespace of `other_symbol_table` into the table: a
    /// defined namespace keeps its position and has its symbols replaced,
    /// and takes the other's parent when the other names one.
    ///
    /// Raises `TypeError` if the argument is no `SymbolTable`.
    fn update_namespaces(
        slf: &Bound<'_, Self>,
        other_symbol_table: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let Ok(other) = other_symbol_table.cast::<Self>() else {
            return Err(build_argument_type_error(
                OWNER,
                "other_symbol_table",
                "a SymbolTable",
                other_symbol_table,
            )?);
        };
        if slf.is(other) {
            return Ok(());
        }
        let other = other.borrow();
        let mut this = slf.borrow_mut();
        this.namespace_dicts.clear();
        this.table.update_namespaces(&other.table);
        Ok(())
    }

    /// Add `symbol_name`, described by `frame`, to `namespace_name`.
    ///
    /// Raises `SymbolTableError` if the namespace or one of its ancestors
    /// defines the symbol, or the namespace is not defined, or the walk up
    /// its parents meets a cycle or a missing parent; `TypeError` if
    /// `frame` is no `SymbolTableFrame`.
    fn add_symbol(
        slf: &Bound<'_, Self>,
        namespace_name: &Bound<'_, PyAny>,
        symbol_name: &Bound<'_, PyAny>,
        frame: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let py = slf.py();
        Self::insert_symbol_checked(slf, namespace_name, symbol_name, frame)?;
        log_debug(
            py,
            "added symbol %s to namespace %s (frame=%s)",
            [
                symbol_name.clone(),
                namespace_name.clone(),
                frame.get_type().name()?.into_any(),
            ],
        )
    }

    /// Return whether any namespace holds `symbol_name`.
    fn is_symbol_defined(&self, symbol_name: &Bound<'_, PyAny>) -> PyResult<bool> {
        let symbol = read_identifier(symbol_name, "symbol_name")?;
        Ok(self.table.find(&symbol).is_some())
    }

    /// Return whether `namespace_name` or one of its ancestors holds
    /// `symbol_name`.
    ///
    /// Raises `SymbolTableError` if the namespace is not defined, or the
    /// walk up its parents meets a cycle or a missing parent.
    fn is_symbol_defined_in_namespace(
        &self,
        namespace_name: &Bound<'_, PyAny>,
        symbol_name: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let namespace = read_identifier(namespace_name, "namespace_name")?;
        let symbol = read_identifier(symbol_name, "symbol_name")?;
        Ok(self
            .table
            .lookup(&namespace, &symbol)
            .map_err(raise)?
            .is_some())
    }

    /// Return the frame of `symbol_name` in the first namespace that holds
    /// it, in insertion order.
    ///
    /// Raises `SymbolTableError` if no namespace holds it.
    fn get_frame<'py>(
        &self,
        py: Python<'py>,
        symbol_name: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let symbol = read_identifier(symbol_name, "symbol_name")?;
        match self.table.find(&symbol) {
            Some(entry) => Ok(entry.frame(py).clone()),
            None => Err(raise(SymbolTableError::SymbolNotFound {
                namespace: None,
                symbol,
            })),
        }
    }

    /// Return the frame of `symbol_name` in `namespace_name` or its nearest
    /// ancestor that holds it.
    ///
    /// Raises `SymbolTableError` if none does, the namespace is not
    /// defined, or the walk up its parents meets a cycle or a missing
    /// parent.
    fn get_frame_from_namespace<'py>(
        &self,
        py: Python<'py>,
        namespace_name: &Bound<'py, PyAny>,
        symbol_name: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let namespace = read_identifier(namespace_name, "namespace_name")?;
        let symbol = read_identifier(symbol_name, "symbol_name")?;
        match self.table.lookup(&namespace, &symbol).map_err(raise)? {
            Some(entry) => Ok(entry.frame(py).clone()),
            None => Err(raise(SymbolTableError::SymbolNotFound {
                namespace: Some(namespace),
                symbol,
            })),
        }
    }

    /// Sort the namespaces, and each namespace's symbols, by identifier, in
    /// place.
    fn canonicalize(&mut self) {
        self.namespace_dicts.clear();
        self.table.canonicalize();
    }

    /// Return the report of the table's broken invariants, one ERROR
    /// diagnostic each: missing and self parents, cyclic parent chains, and
    /// frames naming another symbol; the report is empty for a well-formed
    /// table.
    fn verify<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let error_level = diagnostic_class(py, "DiagnosticLevel")?.getattr("ERROR")?;
        let note = diagnostic_class(py, "Note")?;
        let diagnostic = diagnostic_class(py, "Diagnostic")?;
        let mut diagnostics = Vec::new();
        for violation in self.table.violations() {
            let kwargs = PyDict::new(py);
            kwargs.set_item("level", &error_level)?;
            kwargs.set_item("message", note.call1((violation.to_string(),))?)?;
            kwargs.set_item("source", VERIFY_SOURCE)?;
            diagnostics.push(diagnostic.call((), Some(&kwargs))?);
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("diagnostics", PyTuple::new(py, diagnostics)?)?;
        diagnostic_class(py, "ValidationReport")?.call((), Some(&kwargs))
    }

    /// Return whether `other` is a table with the same namespaces, parents
    /// and symbols, whose frames are structurally equivalent; the orders do
    /// not count.
    ///
    /// The frames' own `is_structurally_equivalent` runs over copies of the
    /// two tables, under no borrow, so it may read or change either table.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let py = other.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        let this = slf.borrow().table.clone();
        let other = other.try_borrow()?.table.clone();
        this.is_equivalent_by(&other, |left, right| {
            left.is_structurally_equivalent(py, right)
        })
    }

    /// Return the payload `{"namespaces": [..]}`, in insertion order, each
    /// frame in its own form: the core's serde under V2, written in one
    /// pass, and each frame's own V1 payload inside
    /// `wire_version(WireVersion.V1)`.
    ///
    /// A Python-defined frame's hooks run over a copy of the table, under no
    /// borrow.
    fn serialize_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if !crate::wire::is_writing_v1(py)? {
            let table = slf.borrow().table.clone();
            let data = fhy_core::symbol_table::wire::SymbolTableData::of(&table, |entry| {
                match &entry.0.native {
                    Some(frame) => fhy_core::symbol_table::wire::SymbolFrameData::of(frame)
                        .map_err(|error| crate::wire::foreign_error(py, &error)),
                    None => super::frames::frame_wire_data(entry.frame(py)),
                }
            })?;
            return crate::wire::to_dict(py, &data);
        }
        Ok(Self::serialize_v1(slf)?.into_any())
    }

    /// Return the table a payload describes: every namespace is added, then
    /// every symbol, through the class's own methods.
    ///
    /// Raises `DeserializationDictStructureError` for a malformed payload,
    /// and `SymbolTableError` for one that cannot be rebuilt.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let Some(namespaces) = read_table_payload(data)? else {
            let expected = [("namespaces", py.get_type::<PyList>().into_any())];
            return Err(structure_error(cls, &expected, data)?);
        };
        let table = cls.call0()?;
        let mut names = Vec::with_capacity(namespaces.len());
        for namespace in &namespaces {
            let name = deserialize_identifier(&namespace.name)?;
            let parent = match &namespace.parent {
                Some(parent) => deserialize_identifier(parent)?,
                None => py.None().into_bound(py),
            };
            table.call_method1(intern!(py, "add_namespace"), (&name, parent))?;
            names.push(name);
        }
        let frame_base = frame_base_class(py)?;
        for (namespace, name) in namespaces.iter().zip(&names) {
            for (symbol, frame) in &namespace.symbols {
                let symbol = deserialize_identifier(symbol)?;
                let frame =
                    frame_base.call_method1(intern!(py, "deserialize_from_dict"), (frame,))?;
                table.call_method1(intern!(py, "add_symbol"), (name, symbol, frame))?;
            }
        }
        Ok(table)
    }

    /// Pickle as a new instance of the class, made without calling its
    /// `__init__`, as `copyreg.__newobj__` makes one, and a state of the
    /// namespaces and the instance dictionary of a subclass.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let instance_dict = match slf.getattr(intern!(py, "__dict__")) {
            Ok(instance_dict) => instance_dict,
            Err(_no_dict) => py.None().into_bound(py),
        };
        let state = PyTuple::new(py, [slf.borrow().state(py)?.into_any(), instance_dict])?;
        let new_object =
            crate::util::python::cached_attr!(py, "copyreg", "__newobj__" => PyAny)?.clone();
        Ok((new_object, PyTuple::new(py, [slf.get_type()])?, state))
    }

    /// Restore the table from the state `__reduce__` returned, without
    /// checking it, so any table a pickle holds restores.
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let state = state
            .cast::<PyTuple>()
            .map_err(|_not_a_tuple| bad_state())?;
        if state.len() != 2 {
            return Err(bad_state());
        }
        let mut table = SymbolTable::new();
        for namespace in state.get_item(0)?.try_iter()? {
            let namespace = namespace?;
            let namespace = namespace
                .cast::<PyTuple>()
                .map_err(|_not_a_tuple| bad_state())?;
            if namespace.len() != 3 {
                return Err(bad_state());
            }
            let name = read_identifier(&namespace.get_item(0)?, "namespace_name")?;
            let parent_object = namespace.get_item(1)?;
            let parent = if parent_object.is_none() {
                None
            } else {
                Some(read_identifier(&parent_object, "parent_namespace_name")?)
            };
            let mut symbols = Vec::new();
            for pair in namespace.get_item(2)?.try_iter()? {
                let pair = pair?;
                let pair = pair.cast::<PyTuple>().map_err(|_not_a_tuple| bad_state())?;
                if pair.len() != 2 {
                    return Err(bad_state());
                }
                let symbol_object = pair.get_item(0)?;
                let symbol = read_identifier(&symbol_object, "symbol_name")?;
                symbols.push((symbol, Entry::new(&symbol_object, &pair.get_item(1)?)?));
            }
            table.insert_namespace(name, parent, symbols);
        }
        {
            let mut this = slf.borrow_mut();
            this.namespace_dicts.clear();
            this.table = table;
        }
        let instance_dict = state.get_item(1)?;
        if !instance_dict.is_none() {
            let own = slf.getattr(intern!(py, "__dict__"))?;
            own.call_method1(intern!(py, "update"), (instance_dict,))?;
        }
        Ok(())
    }
}

/// One namespace of a table payload.
struct NamespacePayload<'py> {
    name: Bound<'py, PyAny>,
    parent: Option<Bound<'py, PyAny>>,
    symbols: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
}

/// Return the namespaces of a table payload, or `None` if it is malformed.
///
/// Matches the Python implementation: the payload holds a `namespaces`
/// list, each entry holds a `namespace_name` payload, a
/// `parent_namespace_name` payload or `None`, and a `symbols` list, and each
/// symbol entry holds a `symbol_name` and a `frame` payload; other keys are
/// ignored.
fn read_table_payload<'py>(
    data: &Bound<'py, PyAny>,
) -> PyResult<Option<Vec<NamespacePayload<'py>>>> {
    let Ok(data) = data.cast::<PyDict>() else {
        return Ok(None);
    };
    let Some(namespaces) = data.get_item("namespaces")? else {
        return Ok(None);
    };
    let Ok(namespaces) = namespaces.cast::<PyList>() else {
        return Ok(None);
    };
    let mut entries = Vec::with_capacity(namespaces.len());
    for namespace in namespaces.iter() {
        if !is_serialized_dict(&namespace)? {
            return Ok(None);
        }
        let namespace = namespace.cast::<PyDict>()?;
        let (Some(name), Some(parent), Some(symbols)) = (
            namespace.get_item("namespace_name")?,
            namespace.get_item("parent_namespace_name")?,
            namespace.get_item("symbols")?,
        ) else {
            return Ok(None);
        };
        if !is_serialized_dict(&name)? || !(parent.is_none() || is_serialized_dict(&parent)?) {
            return Ok(None);
        }
        let Ok(symbols) = symbols.cast::<PyList>() else {
            return Ok(None);
        };
        let mut pairs = Vec::with_capacity(symbols.len());
        for symbol in symbols.iter() {
            if !is_serialized_dict(&symbol)? {
                return Ok(None);
            }
            let symbol = symbol.cast::<PyDict>()?;
            let (Some(symbol_name), Some(frame)) =
                (symbol.get_item("symbol_name")?, symbol.get_item("frame")?)
            else {
                return Ok(None);
            };
            if !is_serialized_dict(&symbol_name)? || !is_serialized_dict(&frame)? {
                return Ok(None);
            }
            pairs.push((symbol_name, frame));
        }
        entries.push(NamespacePayload {
            name,
            parent: (!parent.is_none()).then_some(parent),
            symbols: pairs,
        });
    }
    Ok(Some(entries))
}

impl PySymbolTable {
    /// Return the V1 payload `{"namespaces": [..]}`, in insertion order.
    ///
    /// V1: removed with the V1 wire format; its shape is V2's but for the
    /// frames' envelopes.
    fn serialize_v1<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let table = slf.borrow().table.clone();
        let namespaces = PyList::empty(py);
        for namespace in table.namespaces() {
            let symbols = PyList::empty(py);
            for (symbol, entry) in namespace.iter() {
                let item = PyDict::new(py);
                item.set_item(
                    intern!(py, "symbol_name"),
                    serialize_identifier(py, symbol)?,
                )?;
                item.set_item(
                    intern!(py, "frame"),
                    entry
                        .frame(py)
                        .call_method0(intern!(py, "serialize_to_dict"))?,
                )?;
                symbols.append(item)?;
            }
            let item = PyDict::new(py);
            item.set_item(
                intern!(py, "namespace_name"),
                serialize_identifier(py, namespace.name())?,
            )?;
            match namespace.parent() {
                Some(parent) => item.set_item(
                    intern!(py, "parent_namespace_name"),
                    serialize_identifier(py, parent)?,
                )?,
                None => item.set_item(intern!(py, "parent_namespace_name"), py.None())?,
            }
            item.set_item(intern!(py, "symbols"), symbols)?;
            namespaces.append(item)?;
        }
        let payload = PyDict::new(py);
        payload.set_item(intern!(py, "namespaces"), namespaces)?;
        Ok(payload)
    }
}
