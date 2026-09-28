//! `fhy_core._rs.TypeUnificationEnvironment`, the base of the class of the
//! same name in `fhy_core.types.dispatch` (pattern P2, D-S11-13).
//!
//! It holds the core's environment and, for each binding, the Python
//! objects of its identifier and its value, so a lookup returns the object
//! that was bound. It is subclassable: every environment derived from one,
//! by a `with_*` method or by a dispatcher, is an instance of its class,
//! with a copy of its instance attributes.

use std::collections::HashMap;
use std::sync::OnceLock;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyMapping, PyTuple, PyType};

use fhy_core::identifier::Identifier;
use fhy_core::types::TypeUnificationEnvironment;

use crate::dataclass::{build_argument_type_error, hash_value};
use crate::error::IntoPyErr;
use crate::gc::{Slots, collect_slots};
use crate::identifier::{read_identifier_id, restore_identifier};
use crate::public_class::PublicClass;
use crate::python::Seed;

use super::adapter::{Context, environment_class, run_in_context};
use super::convert::{
    data_type_to_python, expression_to_python, identifier_to_python, read_data_type_value,
    read_type_value, type_to_python,
};
use crate::expression::PyExpression;

/// The owner the argument checks name.
const OWNER: &str = "TypeUnificationEnvironment";

/// The identifier and value objects of one table's bindings, by the
/// identifier's id.
type ObjectTable = HashMap<u64, ObjectPair>;

/// The identifier object and the value object of one binding.
type ObjectPair = (Py<PyAny>, Py<PyAny>);

/// The objects of the three tables.
struct Objects {
    data_types: ObjectTable,
    types: ObjectTable,
    expressions: ObjectTable,
    /// The slots of the Python-defined parts' adapters this environment's
    /// construction made, which it owns (R2-003); a derived environment
    /// starts with none, since its parent owns the parent's.
    slots: Slots,
}

/// Return a copy of `table`.
fn copy_table(py: Python<'_>, table: &ObjectTable) -> ObjectTable {
    table
        .iter()
        .map(|(id, (key, value))| (*id, (key.clone_ref(py), value.clone_ref(py))))
        .collect()
}

impl Objects {
    fn copy(&self, py: Python<'_>) -> Self {
        Self {
            data_types: copy_table(py, &self.data_types),
            types: copy_table(py, &self.types),
            expressions: copy_table(py, &self.expressions),
            slots: Slots::default(),
        }
    }
}

/// The state a new environment is built from, handed to `__new__` as the
/// private keyword `_state`.
#[pyclass(frozen, module = "fhy_core._rs", name = "_EnvironmentState")]
pub(crate) struct PyEnvironmentState {
    state: Seed<(TypeUnificationEnvironment, Objects)>,
}

/// The binding environment of template binding, substitution and
/// unification, backed by the core's [`TypeUnificationEnvironment`].
#[pyclass(
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "TypeUnificationEnvironment"
)]
pub(crate) struct PyTypeUnificationEnvironment {
    value: TypeUnificationEnvironment,
    objects: Objects,
    /// The `immutabledict` views of the three tables, built on first
    /// access.
    views: [OnceLock<Py<PyAny>>; 3],
    hash: OnceLock<u64>,
}

/// Return `immutabledict.immutabledict`.
fn immutabledict_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    CLASS.import(py, "immutabledict", "immutabledict")
}

/// Which table a binding belongs to.
#[derive(Clone, Copy)]
enum Table {
    DataTypes,
    Types,
    Expressions,
}

impl Table {
    const ALL: [Self; 3] = [Self::DataTypes, Self::Types, Self::Expressions];

    /// Return the name of the table's attribute and keyword.
    fn name(self) -> &'static str {
        match self {
            Self::DataTypes => "data_type_bindings",
            Self::Types => "type_bindings",
            Self::Expressions => "expression_bindings",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::DataTypes => 0,
            Self::Types => 1,
            Self::Expressions => 2,
        }
    }
}

impl PyTypeUnificationEnvironment {
    /// Return the public Python class registered for this class.
    pub(crate) fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("TypeUnificationEnvironment");
        &PUBLIC_CLASS
    }

    /// Return the core environment.
    pub(crate) fn value(&self) -> &TypeUnificationEnvironment {
        &self.value
    }

    /// Return the identifier and value objects of `table`, ordered by the
    /// identifier's id, so views and pickles list them in one order.
    fn ordered(&self, table: Table) -> Vec<&ObjectPair> {
        let mut entries: Vec<(&u64, &ObjectPair)> = self.table(table).iter().collect();
        entries.sort_by_key(|(id, _)| **id);
        entries.into_iter().map(|(_, entry)| entry).collect()
    }

    fn table(&self, table: Table) -> &ObjectTable {
        match table {
            Table::DataTypes => &self.objects.data_types,
            Table::Types => &self.objects.types,
            Table::Expressions => &self.objects.expressions,
        }
    }

    /// Remember the objects of every binding in `context`.
    pub(crate) fn remember_objects(&self, context: &Context, py: Python<'_>) {
        let mut known = context.known.borrow_mut();
        for (id, (key, _)) in self
            .objects
            .data_types
            .iter()
            .chain(&self.objects.types)
            .chain(&self.objects.expressions)
        {
            known.identifiers.insert(*id, key.clone_ref(py));
        }
        for (identifier, value) in self.value.data_type_bindings() {
            if let Some((_, object)) = self.objects.data_types.get(&identifier.id()) {
                known.data_types.push((value.clone(), object.clone_ref(py)));
            }
        }
        for (identifier, value) in self.value.type_bindings() {
            if let Some((_, object)) = self.objects.types.get(&identifier.id()) {
                known.types.push((value.clone(), object.clone_ref(py)));
            }
        }
        for (identifier, value) in self.value.expression_bindings() {
            if let Some((_, object)) = self.objects.expressions.get(&identifier.id()) {
                known.expressions.insert(
                    fhy_core::tree::NodeHandle::identity(value),
                    object.clone_ref(py),
                );
            }
        }
    }

    /// Return a new instance of `cls` holding `value` and `objects`, with a
    /// copy of the instance attributes of `template`.
    fn instantiate<'py>(
        cls: &Bound<'py, PyType>,
        value: TypeUnificationEnvironment,
        objects: Objects,
        template: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let state = Py::new(
            py,
            PyEnvironmentState {
                state: Seed::new((value, objects)),
            },
        )?;
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "_state"), state)?;
        let instance = cls.call_method(intern!(py, "__new__"), (cls,), Some(&kwargs))?;
        if let Some(template) = template {
            if let Ok(attributes) = template.getattr(intern!(py, "__dict__")) {
                if attributes.len()? > 0 {
                    instance
                        .getattr(intern!(py, "__dict__"))?
                        .call_method1(intern!(py, "update"), (attributes,))?;
                }
            }
        }
        Ok(instance)
    }

    /// Return the Python object of the core environment `environment`, an
    /// instance of the class of `context`'s template, with the objects
    /// `context` knows; the template itself when `environment` equals it.
    pub(crate) fn build<'py>(
        py: Python<'py>,
        context: &Context,
        environment: &TypeUnificationEnvironment,
    ) -> PyResult<Bound<'py, PyAny>> {
        let template = context.template.as_ref().map(|template| template.bind(py));
        if let Some(template) = template {
            if template.get().value == *environment {
                return Ok(template.clone().into_any());
            }
        }
        let kept = |table: Table, identifier: &Identifier| -> Option<(Py<PyAny>, Py<PyAny>)> {
            let template = template?.get();
            let (key, object) = template.table(table).get(&identifier.id())?;
            let unchanged = match table {
                Table::DataTypes => {
                    template.value.data_type_binding(identifier)
                        == environment.data_type_binding(identifier)
                }
                Table::Types => {
                    template.value.type_binding(identifier) == environment.type_binding(identifier)
                }
                Table::Expressions => {
                    template.value.expression_binding(identifier)
                        == environment.expression_binding(identifier)
                }
            };
            unchanged.then(|| (key.clone_ref(py), object.clone_ref(py)))
        };
        let mut objects = Objects {
            data_types: HashMap::new(),
            types: HashMap::new(),
            expressions: HashMap::new(),
            slots: Slots::default(),
        };
        for (identifier, value) in environment.data_type_bindings() {
            let entry = match kept(Table::DataTypes, identifier) {
                Some(entry) => entry,
                None => (
                    identifier_to_python(py, context, identifier)?.unbind(),
                    data_type_to_python(py, context, value)?.unbind(),
                ),
            };
            objects.data_types.insert(identifier.id(), entry);
        }
        for (identifier, value) in environment.type_bindings() {
            let entry = match kept(Table::Types, identifier) {
                Some(entry) => entry,
                None => (
                    identifier_to_python(py, context, identifier)?.unbind(),
                    type_to_python(py, context, value)?.unbind(),
                ),
            };
            objects.types.insert(identifier.id(), entry);
        }
        for (identifier, value) in environment.expression_bindings() {
            let entry = match kept(Table::Expressions, identifier) {
                Some(entry) => entry,
                None => (
                    identifier_to_python(py, context, identifier)?.unbind(),
                    expression_to_python(py, context, value)?.unbind(),
                ),
            };
            objects.expressions.insert(identifier.id(), entry);
        }
        let cls = environment_class(py, context)?;
        Self::instantiate(
            &cls,
            environment.clone(),
            objects,
            template.map(Bound::as_any),
        )
    }

    /// Return the environment extended by the binding of `name` to `value`
    /// in `table`.
    fn extended<'py>(
        slf: &Bound<'py, Self>,
        table: Table,
        name: &Bound<'py, PyAny>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.get();
        let method = match table {
            Table::DataTypes => "with_data_type_binding",
            Table::Types => "with_type_binding",
            Table::Expressions => "with_expression_binding",
        };
        let identifier = restore_identifier(name, method, "name")?;
        let mut objects = this.objects.copy(py);
        let (next, slots) = collect_slots(|| -> PyResult<TypeUnificationEnvironment> {
            Ok(match table {
                Table::DataTypes => {
                    let Some(data_type) = read_data_type_value(value) else {
                        return Err(build_argument_type_error(
                            method,
                            "value",
                            "a DataType",
                            value,
                        )?);
                    };
                    this.value
                        .with_data_type_binding(identifier.clone(), data_type)
                }
                Table::Types => {
                    let Some(bound) = read_type_value(value) else {
                        return Err(build_argument_type_error(method, "value", "a Type", value)?);
                    };
                    this.value.with_type_binding(identifier.clone(), bound)
                }
                Table::Expressions => {
                    let Ok(expression) = value.cast::<PyExpression>() else {
                        return Err(build_argument_type_error(
                            method,
                            "value",
                            "an Expression",
                            value,
                        )?);
                    };
                    this.value.with_expression_binding(
                        identifier.clone(),
                        expression.get().expression().clone(),
                    )
                }
            })
        });
        let next = next?;
        objects.slots = slots;
        let entry = (name.clone().unbind(), value.clone().unbind());
        match table {
            Table::DataTypes => objects.data_types.insert(identifier.id(), entry),
            Table::Types => objects.types.insert(identifier.id(), entry),
            Table::Expressions => objects.expressions.insert(identifier.id(), entry),
        };
        Self::instantiate(&slf.get_type(), next, objects, Some(slf.as_any()))
    }

    /// Return the object `name` is bound to in `table`, or `None`.
    fn lookup<'py>(
        &self,
        py: Python<'py>,
        table: Table,
        name: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(id) = read_identifier_id(name)? else {
            return Ok(None);
        };
        Ok(self
            .table(table)
            .get(&id)
            .map(|(_, value)| value.bind(py).clone()))
    }

    /// Return the `immutabledict` view of `table`.
    fn view<'py>(&self, py: Python<'py>, table: Table) -> PyResult<Bound<'py, PyAny>> {
        let slot = &self.views[table.index()];
        if let Some(view) = slot.get() {
            return Ok(view.bind(py).clone());
        }
        let entries = PyDict::new(py);
        for (key, value) in self.ordered(table) {
            entries.set_item(key.bind(py), value.bind(py))?;
        }
        let view = immutabledict_class(py)?.call1((entries,))?;
        Ok(slot.get_or_init(|| view.unbind()).bind(py).clone())
    }
}

/// Read the bindings of the mapping `mapping`, the argument `table` of the
/// constructor, into `value` and `table_objects`.
fn read_table(
    mapping: &Bound<'_, PyAny>,
    table: Table,
    value: &mut TypeUnificationEnvironment,
    table_objects: &mut ObjectTable,
) -> PyResult<()> {
    let Ok(mapping) = mapping.cast::<PyMapping>() else {
        return Err(build_argument_type_error(
            OWNER,
            table.name(),
            "a mapping",
            mapping,
        )?);
    };
    for item in mapping.items()?.iter() {
        let (key, bound) = item.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()?;
        let identifier = restore_identifier(&key, OWNER, &format!("{} key", table.name()))?;
        *value = match table {
            Table::DataTypes => {
                let Some(data_type) = read_data_type_value(&bound) else {
                    return Err(build_argument_type_error(
                        OWNER,
                        "data_type_bindings value",
                        "a DataType",
                        &bound,
                    )?);
                };
                value.with_data_type_binding(identifier.clone(), data_type)
            }
            Table::Types => {
                let Some(bound_type) = read_type_value(&bound) else {
                    return Err(build_argument_type_error(
                        OWNER,
                        "type_bindings value",
                        "a Type",
                        &bound,
                    )?);
                };
                value.with_type_binding(identifier.clone(), bound_type)
            }
            Table::Expressions => {
                let Ok(expression) = bound.cast::<PyExpression>() else {
                    return Err(build_argument_type_error(
                        OWNER,
                        "expression_bindings value",
                        "an Expression",
                        &bound,
                    )?);
                };
                value.with_expression_binding(
                    identifier.clone(),
                    expression.get().expression().clone(),
                )
            }
        };
        table_objects.insert(identifier.id(), (key.unbind(), bound.unbind()));
    }
    Ok(())
}

#[pymethods]
impl PyTypeUnificationEnvironment {
    /// Visit the Python objects the object holds, for the cycle collector (R2-003).
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for table in [
            &self.objects.data_types,
            &self.objects.types,
            &self.objects.expressions,
        ] {
            for (identifier, value) in table.values() {
                visit.call(identifier)?;
                visit.call(value)?;
            }
        }
        self.views
            .iter()
            .try_for_each(|view| visit.call(view.get()))?;
        self.objects.slots.traverse(&visit)
    }

    /// Create the environment of the mappings `data_type_bindings`,
    /// `type_bindings` and `expression_bindings`, each from `Identifier`s
    /// to data types, types and expressions; a table not given is empty.
    /// Other keyword arguments are left to a subclass.
    ///
    /// Raises `TypeError` for a table that is no mapping, a key that is no
    /// `Identifier`, or a value of the wrong kind.
    ///
    /// A subclass that defines its own `__init__` receives its constructor
    /// arguments there: the base then reads none of them and starts empty.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        cls: &Bound<'_, PyType>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if let Some(state) = kwargs.and_then(|kwargs| kwargs.get_item("_state").ok().flatten()) {
            if let Ok(state) = state.cast::<PyEnvironmentState>() {
                // A reused state raises, where it built an empty
                // environment (R2-033).
                let (value, objects) = state.get().state.take("an environment")?;
                return Ok(Self::from_parts(value, objects));
            }
        }
        let mut value = TypeUnificationEnvironment::new();
        let mut objects = Objects {
            data_types: HashMap::new(),
            types: HashMap::new(),
            expressions: HashMap::new(),
            slots: Slots::default(),
        };
        let py = cls.py();
        let object_init = py.get_type::<PyAny>().getattr(intern!(py, "__init__"))?;
        if !cls.getattr(intern!(py, "__init__"))?.is(&object_init) {
            return Ok(Self::from_parts(value, objects));
        }
        let (read, slots) = collect_slots(|| -> PyResult<()> {
            for table in Table::ALL {
                let given = match args.get_item(table.index()) {
                    Ok(given) => Some(given),
                    Err(_absent) => {
                        kwargs.and_then(|kwargs| kwargs.get_item(table.name()).ok().flatten())
                    }
                };
                if let Some(given) = given {
                    let table_objects = match table {
                        Table::DataTypes => &mut objects.data_types,
                        Table::Types => &mut objects.types,
                        Table::Expressions => &mut objects.expressions,
                    };
                    read_table(&given, table, &mut value, table_objects)?;
                }
            }
            Ok(())
        });
        read?;
        objects.slots = slots;
        Ok(Self::from_parts(value, objects))
    }

    /// Return a new environment of this class with no bindings.
    #[classmethod]
    fn empty<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyAny>> {
        cls.call0()
    }

    /// Return this environment with `name` bound to the data type `value`.
    fn with_data_type_binding<'py>(
        slf: &Bound<'py, Self>,
        name: &Bound<'py, PyAny>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::extended(slf, Table::DataTypes, name, value)
    }

    /// Return this environment with `name` bound to the type `value`.
    fn with_type_binding<'py>(
        slf: &Bound<'py, Self>,
        name: &Bound<'py, PyAny>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::extended(slf, Table::Types, name, value)
    }

    /// Return this environment with `name` bound to the expression `value`.
    fn with_expression_binding<'py>(
        slf: &Bound<'py, Self>,
        name: &Bound<'py, PyAny>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::extended(slf, Table::Expressions, name, value)
    }

    /// Return the data type `name` is bound to, or `None`.
    fn get_data_type_binding<'py>(
        &self,
        py: Python<'py>,
        name: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.lookup(py, Table::DataTypes, name)
    }

    /// Return the type `name` is bound to, or `None`.
    fn get_type_binding<'py>(
        &self,
        py: Python<'py>,
        name: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.lookup(py, Table::Types, name)
    }

    /// Return the expression `name` is bound to, or `None`.
    fn get_expression_binding<'py>(
        &self,
        py: Python<'py>,
        name: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.lookup(py, Table::Expressions, name)
    }

    /// The data-type bindings, as an `immutabledict`.
    #[getter]
    fn data_type_bindings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.view(py, Table::DataTypes)
    }

    /// The full-type bindings, as an `immutabledict`.
    #[getter]
    fn type_bindings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.view(py, Table::Types)
    }

    /// The expression bindings, as an `immutabledict`.
    #[getter]
    fn expression_bindings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.view(py, Table::Expressions)
    }

    /// Return whether `other` is an environment binding the same
    /// identifiers to structurally equivalent values.
    fn is_structurally_equivalent(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        run_in_context(other.py(), None, |_context| {
            self.value
                .is_structurally_equivalent(&other.get().value)
                .map_err(IntoPyErr::into_py_err)
        })
    }

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if !slf.get_type().is(other.get_type()) {
            return Ok(py.NotImplemented().into_bound(py));
        }
        let other = other.cast::<Self>()?;
        let is_equal =
            run_in_context(
                py,
                None,
                |_context| Ok(slf.get().value == other.get().value),
            )?;
        Ok(PyBool::new(py, is_equal).to_owned().into_any())
    }

    fn __hash__(slf: &Bound<'_, Self>) -> PyResult<u64> {
        let this = slf.get();
        if let Some(hash) = this.hash.get() {
            return Ok(*hash);
        }
        let hash = run_in_context(slf.py(), None, |_context| Ok(hash_value(&this.value)))?;
        Ok(*this.hash.get_or_init(|| hash))
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        Ok(format!(
            "{}(data_type_bindings={}, type_bindings={}, expression_bindings={})",
            slf.get_type().qualname()?,
            this.view(py, Table::DataTypes)?.repr()?,
            this.view(py, Table::Types)?.repr()?,
            this.view(py, Table::Expressions)?.repr()?,
        ))
    }

    /// Pickle as a call of `_from_tables` of the class with the three
    /// tables, and the instance attributes of a subclass as the state.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.get();
        let tables = Table::ALL
            .into_iter()
            .map(|table| {
                let entries = PyDict::new(py);
                for (key, value) in this.ordered(table) {
                    entries.set_item(key.bind(py), value.bind(py))?;
                }
                Ok(entries.into_any())
            })
            .collect::<PyResult<Vec<_>>>()?;
        let state = match slf.getattr(intern!(py, "__dict__")) {
            Ok(attributes) if attributes.len()? > 0 => attributes,
            _ => py.None().into_bound(py),
        };
        PyTuple::new(
            py,
            [
                slf.get_type().getattr(intern!(py, "_from_tables"))?,
                PyTuple::new(py, tables)?.into_any(),
                state,
            ],
        )
    }

    /// Return an instance of `cls` holding the three tables, without calling
    /// a subclass's `__init__`; the pickles' constructor.
    ///
    /// Raises `TypeError` as the constructor does.
    #[classmethod]
    fn _from_tables<'py>(
        cls: &Bound<'py, PyType>,
        data_type_bindings: &Bound<'py, PyAny>,
        type_bindings: &Bound<'py, PyAny>,
        expression_bindings: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut value = TypeUnificationEnvironment::new();
        let mut objects = Objects {
            data_types: HashMap::new(),
            types: HashMap::new(),
            expressions: HashMap::new(),
            slots: Slots::default(),
        };
        let (read, slots) = collect_slots(|| -> PyResult<()> {
            read_table(
                data_type_bindings,
                Table::DataTypes,
                &mut value,
                &mut objects.data_types,
            )?;
            read_table(type_bindings, Table::Types, &mut value, &mut objects.types)?;
            read_table(
                expression_bindings,
                Table::Expressions,
                &mut value,
                &mut objects.expressions,
            )
        });
        read?;
        objects.slots = slots;
        Self::instantiate(cls, value, objects, None)
    }

    /// Restore the instance attributes of a subclass.
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        if state.is_none() {
            return Ok(());
        }
        slf.getattr(intern!(slf.py(), "__dict__"))?
            .call_method1(intern!(slf.py(), "update"), (state,))?;
        Ok(())
    }

    /// Register `cls` as the public class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }
}

impl PyTypeUnificationEnvironment {
    fn from_parts(value: TypeUnificationEnvironment, objects: Objects) -> Self {
        Self {
            value,
            objects,
            views: [OnceLock::new(), OnceLock::new(), OnceLock::new()],
            hash: OnceLock::new(),
        }
    }
}
