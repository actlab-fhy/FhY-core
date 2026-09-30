//! The engine of `DerivedEquivalenceMixin`: structural and alpha equivalence
//! derived from a dataclass's fields.
//!
//! A class's plan is built on its first comparison from
//! `dataclasses.fields`, and kept in the Python module's `_PLAN_CACHE`
//! dict, keyed by the class, for the life of the process. A
//! comparison walks the two objects' fields on its own stack, with the
//! dispatch order and error texts of the Python implementation. It compares
//! `None`, identifiers, expressions, exact scalars and nested derived
//! values without calling their Python comparison methods; everything else
//! goes through Python: field reads, user comparators, `key` normalizers,
//! hand-written methods, and `==`.

use std::collections::HashMap;
use std::rc::Rc;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyType};

use crate::expression::PyExpression;
use crate::identifier::restore_identifier;

use super::renaming::{IdentifierList, PyAlphaRenaming, RenamingValue, read_renaming};

// ---------------------------------------------------------------------------
// Python names
// ---------------------------------------------------------------------------

const MODULE: &str = "fhy_core.term.derived_equivalence";

fn plan_cache(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    Ok(crate::python::cached_attr!(py, MODULE, "_PLAN_CACHE" => PyDict)?.clone())
}

fn metadata_key(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    Ok(crate::python::cached_attr!(py, MODULE, "EQUIVALENCE_METADATA_KEY")?.clone())
}

fn derivation_error(py: Python<'_>, message: String) -> PyErr {
    crate::exceptions::EQUIVALENCE_DERIVATION_ERROR.err(py, (message,))
}

fn mixin_method<'py>(py: Python<'py>, name: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyAny>> {
    crate::python::cached_attr!(py, MODULE, "DerivedEquivalenceMixin")?.getattr(name)
}

fn dataclasses_function<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("dataclasses")?.getattr(name)
}

fn enum_class(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    Ok(crate::python::cached_attr!(py, "enum", "Enum" => PyType)?.clone())
}

fn identifier_class(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    Ok(crate::python::cached_attr!(py, "fhy_core.identifier", "Identifier" => PyType)?.clone())
}

// ---------------------------------------------------------------------------
// Roles
// ---------------------------------------------------------------------------

enum Role {
    Value(Option<Py<PyAny>>),
    Reference,
    Binder(Py<PyTuple>),
    Excluded,
    Explicit(Py<PyAny>),
}

/// How one dataclass field takes part in derived equivalence, as the field
/// metadata `compared_as_value`, `compared_as_reference`,
/// `compared_as_binder`, `excluded_from_equivalence` and `compared_with`
/// build.
#[pyclass(frozen, module = "fhy_core._rs", name = "EquivalenceRole")]
pub(crate) struct PyEquivalenceRole {
    role: Role,
}

#[pymethods]
impl PyEquivalenceRole {
    /// Return the role comparing a field by `==`, of `key(value)` when a
    /// `key` is given.
    #[staticmethod]
    #[pyo3(signature = (key = None))]
    fn value(key: Option<Bound<'_, PyAny>>) -> Self {
        Self {
            role: Role::Value(key.filter(|key| !key.is_none()).map(Bound::unbind)),
        }
    }

    /// Return the role of a referenced identifier, compared by `==`
    /// structurally and through the renaming in alpha mode.
    #[staticmethod]
    const fn reference() -> Self {
        Self {
            role: Role::Reference,
        }
    }

    /// Return the role of the identifiers a node binds over the fields
    /// named in `scopes_over`.
    ///
    /// Raises `TypeError` if a name is not a `str`.
    #[staticmethod]
    fn binder(scopes_over: &Bound<'_, PyAny>) -> PyResult<Self> {
        let names = scopes_over
            .try_iter()?
            .map(|name| {
                let name = name?;
                match name.cast::<PyString>() {
                    Ok(name) => Ok(name.clone().into_any()),
                    Err(_not_a_str) => Err(PyTypeError::new_err(format!(
                        "compared_as_binder scopes_over must hold str names, got {}.",
                        name.get_type().name()?
                    ))),
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            role: Role::Binder(PyTuple::new(scopes_over.py(), names)?.unbind()),
        })
    }

    /// Return the role of a field that takes no part in equivalence.
    #[staticmethod]
    const fn excluded() -> Self {
        Self {
            role: Role::Excluded,
        }
    }

    /// Return the role of a field compared by `comparator`, a
    /// `FieldComparator`.
    #[staticmethod]
    fn explicit(comparator: Bound<'_, PyAny>) -> Self {
        Self {
            role: Role::Explicit(comparator.unbind()),
        }
    }

    /// The kind of role: `"value"`, `"reference"`, `"binder"`, `"excluded"`
    /// or `"explicit"`.
    #[getter]
    const fn kind(&self) -> &'static str {
        match self.role {
            Role::Value(_) => "value",
            Role::Reference => "reference",
            Role::Binder(_) => "binder",
            Role::Excluded => "excluded",
            Role::Explicit(_) => "explicit",
        }
    }

    /// The normalizer of a value role, or `None`.
    #[getter]
    fn key(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        match &self.role {
            Role::Value(Some(key)) => Some(key.clone_ref(py)),
            _ => None,
        }
    }

    /// The names a binder role scopes over, or an empty tuple.
    #[getter]
    fn scopes_over<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        match &self.role {
            Role::Binder(names) => names.bind(py).clone(),
            _ => PyTuple::empty(py),
        }
    }

    /// The comparator of an explicit role, or `None`.
    #[getter]
    fn comparator(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        match &self.role {
            Role::Explicit(comparator) => Some(comparator.clone_ref(py)),
            _ => None,
        }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(match &self.role {
            Role::Value(None) => "EquivalenceRole.value()".to_owned(),
            Role::Value(Some(key)) => {
                format!("EquivalenceRole.value(key={})", key.bind(py).repr()?)
            }
            Role::Reference => "EquivalenceRole.reference()".to_owned(),
            Role::Binder(names) => format!("EquivalenceRole.binder({})", names.bind(py).repr()?),
            Role::Excluded => "EquivalenceRole.excluded()".to_owned(),
            Role::Explicit(comparator) => {
                format!("EquivalenceRole.explicit({})", comparator.bind(py).repr()?)
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Plans
// ---------------------------------------------------------------------------

enum Comparator {
    Default,
    Value(Option<Py<PyAny>>),
    Reference,
    Explicit(Py<PyAny>),
}

struct PlannedField {
    name: Py<PyString>,
    comparator: Comparator,
}

struct PlannedBinder {
    name: Py<PyString>,
    /// The indexes, in `fields`, of the fields the binder scopes over.
    scopes: Vec<usize>,
}

/// The comparison plan of one class, kept in `_PLAN_CACHE`.
#[pyclass(frozen, module = "fhy_core._rs", name = "EquivalencePlan")]
struct PyEquivalencePlan {
    class_name: String,
    fields: Vec<PlannedField>,
    binders: Vec<PlannedBinder>,
}

/// Return the plan of `class`, building it on its first comparison.
///
/// # Errors
///
/// Raises `EquivalenceDerivationError` if `class` is not a dataclass or a
/// binder scopes over a name that is no field.
fn find_plan<'py>(class: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyEquivalencePlan>> {
    let py = class.py();
    let cache = plan_cache(py)?;
    if let Some(plan) = cache.get_item(class)? {
        if let Ok(plan) = plan.cast::<PyEquivalencePlan>() {
            return Ok(plan.clone());
        }
    }
    let plan = Bound::new(py, build_plan(class)?)?;
    cache.set_item(class, &plan)?;
    Ok(plan)
}

fn build_plan(class: &Bound<'_, PyType>) -> PyResult<PyEquivalencePlan> {
    let py = class.py();
    let class_name = class.name()?.to_string();
    if !dataclasses_function(py, "is_dataclass")?
        .call1((class,))?
        .is_truthy()?
    {
        return Err(derivation_error(
            py,
            format!(
                "Cannot derive equivalence for \"{class_name}\": it is not a dataclass. \
                 Implement is_structurally_equivalent / is_alpha_equivalent_under by hand."
            ),
        ));
    }
    let dataclass_fields = dataclasses_function(py, "fields")?.call1((class,))?;
    let key = metadata_key(py)?;
    let mut field_names = Vec::new();
    for field in dataclass_fields.try_iter()? {
        field_names.push(
            field?
                .getattr(intern!(py, "name"))?
                .cast::<PyString>()?
                .to_string(),
        );
    }
    let mut fields = Vec::new();
    let mut binders: Vec<(Py<PyString>, Vec<String>)> = Vec::new();
    for field in dataclass_fields.try_iter()? {
        let field = field?;
        let name = field
            .getattr(intern!(py, "name"))?
            .cast::<PyString>()?
            .clone();
        let role = field
            .getattr(intern!(py, "metadata"))?
            .call_method1(intern!(py, "get"), (&key,))?;
        let role = role
            .cast::<PyEquivalenceRole>()
            .ok()
            .map(|role| &role.get().role);
        if matches!(role, Some(Role::Excluded))
            || !field.getattr(intern!(py, "compare"))?.is_truthy()?
        {
            continue;
        }
        let comparator = match role {
            Some(Role::Binder(scopes)) => {
                let mut scope_names = Vec::new();
                for scope in scopes.bind(py).iter() {
                    let scope = scope.cast::<PyString>()?.to_string();
                    if !field_names.contains(&scope) {
                        return Err(derivation_error(
                            py,
                            format!(
                                "Cannot derive equivalence for \"{class_name}\": binder field \
                                 \"{name}\" declares scopes_over=(\"{scope}\", ...), but \
                                 \"{scope}\" is not a field of \"{class_name}\"."
                            ),
                        ));
                    }
                    scope_names.push(scope);
                }
                binders.push((name.unbind(), scope_names));
                continue;
            }
            Some(Role::Value(key)) => Comparator::Value(key.as_ref().map(|key| key.clone_ref(py))),
            Some(Role::Reference) => Comparator::Reference,
            Some(Role::Explicit(comparator)) => Comparator::Explicit(comparator.clone_ref(py)),
            Some(Role::Excluded) | None => Comparator::Default,
        };
        fields.push(PlannedField {
            name: name.unbind(),
            comparator,
        });
    }
    let binders = binders
        .into_iter()
        .map(|(name, scoped)| {
            let scopes = fields
                .iter()
                .enumerate()
                .filter(|(_, field)| {
                    scoped
                        .iter()
                        .any(|scope| field.name.bind(py).to_str().is_ok_and(|name| name == scope))
                })
                .map(|(index, _)| index)
                .collect();
            PlannedBinder { name, scopes }
        })
        .collect();
    Ok(PyEquivalencePlan {
        class_name,
        fields,
        binders,
    })
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

/// What the default dispatch needs to know of a value. The protocol checks
/// ask the value, as a runtime-checkable protocol does; the answers are kept
/// per class for one walk.
#[derive(Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent facts about one class, each read on its own"
)]
struct Capabilities {
    /// `StructuralEquivalence`: it has `is_structurally_equivalent`.
    is_structural: bool,
    /// `AlphaEquivalence`: it has `is_alpha_equivalent` and
    /// `is_alpha_equivalent_under`.
    is_alpha: bool,
    /// An expression class keeping `_rs.Expression`'s structural method.
    native_structural: bool,
    /// An expression class keeping `_rs.Expression`'s alpha method.
    native_alpha: bool,
    /// A class keeping the mixin's `is_structurally_equivalent`, whose
    /// values are walked here.
    derives_structural: bool,
    /// A class keeping the mixin's `is_alpha_equivalent_under`.
    derives_alpha: bool,
    /// A `bool`, `int`, `float`, `str`, `Enum` or `PartialEqual`.
    is_value: bool,
    /// Exactly `Identifier`, whose `==` compares ids.
    is_identifier: bool,
    is_sequence: bool,
}

fn read_capabilities(value: &Bound<'_, PyAny>) -> PyResult<Capabilities> {
    let py = value.py();
    let class = value.get_type();
    let expression = py.get_type::<PyExpression>();
    let is_expression = class.is_subclass(expression.as_any())?;
    let keeps = |method: &Bound<'_, PyString>| -> PyResult<bool> {
        Ok(is_expression && class.getattr(method)?.is(&expression.getattr(method)?))
    };
    let derives = |method: &Bound<'_, PyString>| -> PyResult<bool> {
        Ok(class.hasattr(method)? && class.getattr(method)?.is(&mixin_method(py, method)?))
    };
    Ok(Capabilities {
        is_structural: value.hasattr(intern!(py, "is_structurally_equivalent"))?,
        is_alpha: value.hasattr(intern!(py, "is_alpha_equivalent"))?
            && value.hasattr(intern!(py, "is_alpha_equivalent_under"))?,
        native_structural: keeps(intern!(py, "is_structurally_equivalent"))?,
        native_alpha: keeps(intern!(py, "is_alpha_equivalent_under"))?,
        derives_structural: derives(intern!(py, "is_structurally_equivalent"))?,
        derives_alpha: derives(intern!(py, "is_alpha_equivalent_under"))?,
        is_value: class.is_subclass_of::<PyBool>()?
            || class.is_subclass_of::<PyInt>()?
            || class.is_subclass_of::<PyFloat>()?
            || class.is_subclass_of::<PyString>()?
            || class.is_subclass(enum_class(py)?.as_any())?
            || value.hasattr(intern!(py, "supports_partial_equality"))?,
        is_identifier: class.is(&identifier_class(py)?),
        is_sequence: class.is_subclass_of::<PyTuple>()? || class.is_subclass_of::<PyList>()?,
    })
}

// ---------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Structural,
    Alpha,
}

/// A renaming in scope, with its Python object once one is needed.
struct Scope<'py> {
    value: RenamingValue,
    object: std::cell::OnceCell<Bound<'py, PyAlphaRenaming>>,
}

impl<'py> Scope<'py> {
    fn new(value: RenamingValue) -> Rc<Self> {
        Rc::new(Self {
            value,
            object: std::cell::OnceCell::new(),
        })
    }

    fn object(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(object) = self.object.get() {
            return Ok(object.clone().into_any());
        }
        let object = self.value.clone().into_python(py)?;
        Ok(self.object.get_or_init(|| object).clone().into_any())
    }
}

/// Where a compared value sits, for the derivation error.
struct Site<'py> {
    plan: Bound<'py, PyEquivalencePlan>,
    field: usize,
}

enum Task<'py> {
    /// Compare two derived objects field by field.
    Node {
        left: Bound<'py, PyAny>,
        right: Bound<'py, PyAny>,
        scope: Option<Rc<Scope<'py>>>,
    },
    /// Compare one field of two derived objects.
    Field {
        plan: Bound<'py, PyEquivalencePlan>,
        field: usize,
        left: Bound<'py, PyAny>,
        right: Bound<'py, PyAny>,
        scope: Option<Rc<Scope<'py>>>,
    },
    /// Compare two values by the default dispatch.
    Value {
        site: Rc<Site<'py>>,
        left: Bound<'py, PyAny>,
        right: Bound<'py, PyAny>,
        scope: Option<Rc<Scope<'py>>>,
        in_sequence: bool,
    },
}

struct Walk<'py> {
    py: Python<'py>,
    mode: Mode,
    stack: Vec<Task<'py>>,
    capabilities: HashMap<usize, Capabilities>,
}

impl<'py> Walk<'py> {
    fn new(py: Python<'py>, mode: Mode) -> Self {
        Self {
            py,
            mode,
            stack: Vec::new(),
            capabilities: HashMap::new(),
        }
    }

    /// Return whether the root pair and everything it pushes compare equal.
    fn run(mut self, root: Task<'py>) -> PyResult<bool> {
        self.stack.push(root);
        while let Some(task) = self.stack.pop() {
            if !self.evaluate(task)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn capabilities_of(&mut self, value: &Bound<'py, PyAny>) -> PyResult<Capabilities> {
        let class = value.get_type();
        let key = class.as_ptr() as usize;
        if let Some(capabilities) = self.capabilities.get(&key) {
            return Ok(*capabilities);
        }
        let capabilities = read_capabilities(value)?;
        self.capabilities.insert(key, capabilities);
        Ok(capabilities)
    }

    /// Evaluate `task`: `false` when it decides the comparison is not
    /// equal, `true` when it holds or pushed the comparisons it needs.
    fn evaluate(&mut self, task: Task<'py>) -> PyResult<bool> {
        match task {
            Task::Node { left, right, scope } => self.expand_node(&left, &right, scope.as_ref()),
            Task::Field {
                plan,
                field,
                left,
                right,
                scope,
            } => self.compare_field(&plan, field, &left, &right, scope.as_ref()),
            Task::Value {
                site,
                left,
                right,
                scope,
                in_sequence,
            } => self.compare_value(&site, left, right, scope.as_ref(), in_sequence),
        }
    }

    fn expand_node(
        &mut self,
        left: &Bound<'py, PyAny>,
        right: &Bound<'py, PyAny>,
        scope: Option<&Rc<Scope<'py>>>,
    ) -> PyResult<bool> {
        let class = left.get_type();
        if !class.is(right.get_type()) {
            return Ok(false);
        }
        let plan = find_plan(&class)?;
        let planned = plan.get();
        let mut scopes: Vec<Option<Rc<Scope<'py>>>> = vec![None; planned.fields.len()];
        for binder in &planned.binders {
            let name = binder.name.bind(self.py);
            let left_bound = as_identifier_tuple(&left.getattr(name)?)?;
            let right_bound = as_identifier_tuple(&right.getattr(name)?)?;
            match (self.mode, scope) {
                (Mode::Alpha, Some(scope)) => {
                    let left_list =
                        IdentifierList::read(&left_bound, "compared_as_binder", "identifier")?;
                    let right_list =
                        IdentifierList::read(&right_bound, "compared_as_binder", "identifier")?;
                    if left_list.identifiers.len() != right_list.identifiers.len() {
                        return Ok(false);
                    }
                    // The pairing is checked even when the binder scopes
                    // over nothing: a list that repeats an identifier pairs
                    // with none.
                    let Some(paired) = scope.value.entered(&left_list, &right_list) else {
                        return Ok(false);
                    };
                    let paired = Scope::new(paired);
                    for &index in &binder.scopes {
                        scopes[index] = Some(match &scopes[index] {
                            None => Rc::clone(&paired),
                            Some(inner) => match inner.value.entered(&left_list, &right_list) {
                                Some(entered) => Scope::new(entered),
                                None => return Ok(false),
                            },
                        });
                    }
                }
                _ => {
                    if left_bound.ne(&right_bound)? {
                        return Ok(false);
                    }
                }
            }
        }
        for (index, field_scope) in scopes.into_iter().enumerate().rev() {
            self.stack.push(Task::Field {
                plan: plan.clone(),
                field: index,
                left: left.clone(),
                right: right.clone(),
                scope: field_scope.or_else(|| scope.cloned()),
            });
        }
        Ok(true)
    }

    fn compare_field(
        &mut self,
        plan: &Bound<'py, PyEquivalencePlan>,
        field: usize,
        left_owner: &Bound<'py, PyAny>,
        right_owner: &Bound<'py, PyAny>,
        scope: Option<&Rc<Scope<'py>>>,
    ) -> PyResult<bool> {
        let py = self.py;
        let planned = &plan.get().fields[field];
        let name = planned.name.bind(py);
        let left = left_owner.getattr(name)?;
        let right = right_owner.getattr(name)?;
        match &planned.comparator {
            Comparator::Value(None) => self.is_equal(&left, &right),
            Comparator::Value(Some(key)) => {
                let key = key.bind(py);
                self.is_equal(&key.call1((left,))?, &key.call1((right,))?)
            }
            Comparator::Reference => match (self.mode, scope) {
                (Mode::Alpha, Some(scope)) => {
                    let left = restore_identifier(&left, "compared_as_reference", "identifier")?;
                    let right = restore_identifier(&right, "compared_as_reference", "identifier")?;
                    Ok(scope.value.renaming().is_corresponding(&left, &right))
                }
                _ => self.is_equal(&left, &right),
            },
            Comparator::Explicit(comparator) => {
                let comparator = comparator.bind(py);
                match (self.mode, scope) {
                    (Mode::Alpha, Some(scope)) => comparator
                        .call_method1(
                            intern!(py, "is_alpha_equivalent_under"),
                            (left, right, scope.object(py)?),
                        )?
                        .is_truthy(),
                    _ => comparator
                        .call_method1(intern!(py, "is_structurally_equivalent"), (left, right))?
                        .is_truthy(),
                }
            }
            Comparator::Default => {
                let site = Rc::new(Site {
                    plan: plan.clone(),
                    field,
                });
                self.compare_value(&site, left, right, scope, false)
            }
        }
    }

    /// Return the truth of `left == right`, comparing two identifiers by id.
    fn is_equal(&mut self, left: &Bound<'py, PyAny>, right: &Bound<'py, PyAny>) -> PyResult<bool> {
        if self.capabilities_of(left)?.is_identifier && self.capabilities_of(right)?.is_identifier {
            let id = intern!(self.py, "id");
            return left.getattr(id)?.eq(right.getattr(id)?);
        }
        left.eq(right)
    }

    fn compare_value(
        &mut self,
        site: &Rc<Site<'py>>,
        left: Bound<'py, PyAny>,
        right: Bound<'py, PyAny>,
        scope: Option<&Rc<Scope<'py>>>,
        in_sequence: bool,
    ) -> PyResult<bool> {
        let py = self.py;
        if left.is_none() || right.is_none() {
            return Ok(left.is(&right));
        }
        let capabilities = self.capabilities_of(&left)?;
        match (self.mode, scope) {
            (Mode::Alpha, Some(scope)) if capabilities.is_alpha => {
                if capabilities.native_alpha {
                    let expression = left.cast::<PyExpression>()?.get();
                    return Ok(right.cast::<PyExpression>().is_ok_and(|right| {
                        let renaming = scope.value.renaming();
                        if renaming.is_empty() {
                            expression.is_structurally_equal(right.get())
                        } else {
                            expression
                                .expression()
                                .is_alpha_equivalent_under(right.get().expression(), renaming)
                        }
                    }));
                }
                if capabilities.derives_alpha {
                    self.stack.push(Task::Node {
                        left,
                        right,
                        scope: Some(Rc::clone(scope)),
                    });
                    return Ok(true);
                }
                return left
                    .call_method1(
                        intern!(py, "is_alpha_equivalent_under"),
                        (right, scope.object(py)?),
                    )?
                    .is_truthy();
            }
            _ => {}
        }
        if capabilities.is_structural {
            if capabilities.native_structural {
                let expression = left.cast::<PyExpression>()?.get();
                return Ok(right
                    .cast::<PyExpression>()
                    .is_ok_and(|right| expression.is_structurally_equal(right.get())));
            }
            if self.mode == Mode::Structural && capabilities.derives_structural {
                self.stack.push(Task::Node {
                    left,
                    right,
                    scope: None,
                });
                return Ok(true);
            }
            return left
                .call_method1(intern!(py, "is_structurally_equivalent"), (right,))?
                .is_truthy();
        }
        if capabilities.is_sequence && self.capabilities_of(&right)?.is_sequence {
            if left.len()? != right.len()? {
                return Ok(false);
            }
            let left_items: Vec<_> = left.try_iter()?.collect::<PyResult<_>>()?;
            let right_items: Vec<_> = right.try_iter()?.collect::<PyResult<_>>()?;
            for (left, right) in left_items.into_iter().zip(right_items).rev() {
                self.stack.push(Task::Value {
                    site: Rc::clone(site),
                    left,
                    right,
                    scope: scope.cloned(),
                    in_sequence: true,
                });
            }
            return Ok(true);
        }
        if capabilities.is_value {
            return self.is_equal(&left, &right);
        }
        Err(self.inference_error(site, &left, in_sequence)?)
    }

    fn inference_error(
        &self,
        site: &Site<'py>,
        value: &Bound<'py, PyAny>,
        in_sequence: bool,
    ) -> PyResult<PyErr> {
        let py = self.py;
        let plan = site.plan.get();
        let field = plan.fields[site.field].name.bind(py).to_string();
        let class = &plan.class_name;
        let value_type = value.get_type().name()?;
        let message = if in_sequence {
            format!(
                "Cannot derive equivalence for field \"{class}.{field}\": contains a sequence \
                 element of type {value_type} that is not comparable. Supply compared_with(...) \
                 or hand-write the comparison."
            )
        } else {
            format!(
                "Cannot derive equivalence for field \"{class}.{field}\" of type {value_type}. \
                 Tag it with compared_as_value(), supply compared_with(...), exclude it with \
                 excluded_from_equivalence(), or hand-write the comparison."
            )
        };
        Ok(derivation_error(py, message))
    }
}

/// Return `value` as a tuple of bound identifiers: a tuple itself, a list
/// as a tuple, and anything else as a one-item tuple.
fn as_identifier_tuple<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return Ok(tuple.clone());
    }
    if let Ok(list) = value.cast::<PyList>() {
        return Ok(list.to_tuple());
    }
    PyTuple::new(value.py(), [value])
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Return whether `obj` and `other` are structurally equivalent by the
/// field schema of `obj`'s dataclass: the same class, and every
/// participating field equal by its role, bound identifiers compared by
/// `==`.
///
/// Raises `EquivalenceDerivationError` if the class is not a dataclass, a
/// binder scopes over no field, or a field's value is of no comparable
/// kind, and whatever a comparison raises.
#[pyfunction]
pub(crate) fn derived_is_structurally_equivalent(
    obj: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    Walk::new(obj.py(), Mode::Structural).run(Task::Node {
        left: obj.clone(),
        right: other.clone(),
        scope: None,
    })
}

/// Return whether `obj` and `other` are alpha-equivalent under `renaming`
/// by the field schema of `obj`'s dataclass: as structural equivalence, but
/// with sub-objects compared under the renaming, references through it, and
/// the fields a binder scopes over under one more frame pairing the bound
/// identifiers by position. A pairing the core refuses, such as a list that
/// repeats an identifier, is not equivalent.
///
/// Raises `TypeError` if `renaming` is not an `AlphaRenaming` or a bound or
/// referenced identifier is not an `Identifier`, and otherwise as
/// `derived_is_structurally_equivalent` does.
#[pyfunction]
pub(crate) fn derived_is_alpha_equivalent_under(
    obj: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
    renaming: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let renaming = read_renaming(renaming)?;
    let scope = Rc::new(Scope {
        value: renaming.get().value().clone(),
        object: std::cell::OnceCell::from(renaming.clone()),
    });
    Walk::new(obj.py(), Mode::Alpha).run(Task::Node {
        left: obj.clone(),
        right: other.clone(),
        scope: Some(scope),
    })
}
