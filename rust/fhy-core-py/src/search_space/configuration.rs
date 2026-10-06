//! `fhy_core._rs.Configuration` and `ConfigurationKey`.

use std::collections::HashMap;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{CompareOp, PyTraverseError, PyVisit};
use pyo3::types::{PyBool, PyDict, PyMapping, PyTuple, PyType};

use fhy_core::constraint::Value;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::wire::{ConfigurationData, ConfigurationKeyData};
use fhy_core::search_space::{Activity, Configuration, ConfigurationErrors, ConfigurationKey};
use fhy_core::term::AlphaEquivalence;

use crate::constraint::{read_bound_value, value_to_python};
use crate::convert::param::run_with_context;
use crate::identifier::restore_identifier;
use crate::term::read_renaming;
use crate::util::dataclass::hash_value;
use crate::util::exceptions::DESERIALIZATION_VALUE_ERROR;
use crate::util::frozen::{refuse_attribute_assignment, refuse_attribute_deletion};
use crate::util::pending::with_pending_errors;
use crate::util::public_class::PublicClass;
use crate::wire::PyResolver;

use super::arguments::{Seeded, instantiate, take_seed, wrong_argument, wrong_seed};
use super::choice::PyChoice;
use super::errors::{configuration_errors_to_py, equivalence_error_to_py};
use super::space::PySpace;
use super::wire::{Family, decode_part, refuse_v1, write_part, write_part_json};

/// The entries of a configuration as given: the core entries and the value
/// objects by decision name.
type Entries = (Vec<(Identifier, Value)>, HashMap<Identifier, Py<PyAny>>);

/// Return the entries `entries` gives, a mapping or an iterable of
/// `(name, value)` pairs.
///
/// # Errors
///
/// Raises `TypeError` for entries of another shape and a name that is no
/// `Identifier`.
fn read_entries(entries: Option<&Bound<'_, PyAny>>) -> PyResult<Entries> {
    let mut core = Vec::new();
    let mut objects = HashMap::new();
    let Some(entries) = entries.filter(|entries| !entries.is_none()) else {
        return Ok((core, objects));
    };
    let expected = "a mapping or (name, value) pairs";
    let pairs = match entries.cast::<PyMapping>() {
        Ok(mapping) => mapping.items()?.into_any(),
        Err(_not_a_mapping) => entries.clone(),
    };
    let items = pairs
        .try_iter()
        .map_err(|_not_iterable| wrong_argument("Configuration", "entries", expected, entries))?;
    for item in items {
        let item = item?;
        let (name, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item
            .extract()
            .map_err(|_not_a_pair| wrong_argument("Configuration", "entries", expected, &item))?;
        let identifier = restore_identifier(&name, "Configuration", "entry name")?;
        core.push((identifier.clone(), read_bound_value(&value)?));
        objects.insert(identifier, value.unbind());
    }
    Ok((core, objects))
}

/// A point of a space, checked against it, backed by the core
/// [`Configuration`]; the base of the public `Configuration`.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Configuration")]
pub(crate) struct PyConfiguration {
    configuration: Configuration,
    /// The `Space` the configuration is a point of.
    space: Py<PySpace>,
    /// The value objects given, by decision name.
    values: HashMap<Identifier, Py<PyAny>>,
}

impl PyConfiguration {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Configuration");
        &PUBLIC_CLASS
    }

    /// Return the configuration `check` builds under the default solver's
    /// context, raising its problems.
    fn check(
        py: Python<'_>,
        check: impl FnOnce(
            &fhy_core::param::ParamContext<'_>,
        ) -> Result<Configuration, ConfigurationErrors>
        + Send,
    ) -> PyResult<Configuration> {
        run_with_context(py, true, check, |errors| {
            configuration_errors_to_py(py, &errors)
        })
    }

    /// Return a new instance of the public class of `self`'s values.
    fn into_python(self, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        instantiate(
            Self::public_class().get(py)?,
            1,
            Seeded::Configuration(self),
        )
    }

    /// Return the configuration with `entries` given on top of its own.
    fn with_entries_given<'py>(
        slf: &Bound<'py, Self>,
        entries: Entries,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.get();
        let (core, objects) = entries;
        let current = this.configuration.clone();
        let configuration = Self::check(py, move |context| current.with_entries(core, context))?;
        let mut values: HashMap<Identifier, Py<PyAny>> = this
            .values
            .iter()
            .filter(|(name, _)| configuration.value(name).is_some())
            .map(|(name, value)| (name.clone(), value.clone_ref(py)))
            .collect();
        values.extend(objects);
        Self {
            configuration,
            space: this.space.clone_ref(py),
            values,
        }
        .into_python(py)
    }

    /// Return the Python object of the value of the decision `name`: the
    /// object given, or a new one.
    fn value_object<'py>(
        &self,
        py: Python<'py>,
        name: &Identifier,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(value) = self.configuration.value(name) else {
            return Ok(None);
        };
        match self.values.get(name) {
            Some(object) => Ok(Some(object.bind(py).clone())),
            None => value_to_python(py, value).map(Some),
        }
    }
}

#[pymethods]
impl PyConfiguration {
    /// Return the configuration of `space` holding `entries`, a mapping or
    /// an iterable of `(name, value)` pairs, a choice's value the chosen
    /// alternative's name, checked under the default solver's context.
    ///
    /// Raises `TypeError` for an argument of the wrong type,
    /// `ConfigurationError` carrying every problem of the entries, and the
    /// exception a Python-defined constraint raised.
    #[new]
    #[pyo3(signature = (space, entries = None, **kwargs))]
    fn new(
        space: &Bound<'_, PyAny>,
        entries: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if let Some(seeded) = take_seed(kwargs)? {
            return match seeded {
                Seeded::Configuration(configuration) => Ok(configuration),
                _ => Err(wrong_seed()),
            };
        }
        let py = space.py();
        let space = space
            .cast::<PySpace>()
            .map_err(|_not_a_space| wrong_argument("Configuration", "space", "a Space", space))?;
        let (core, values) = read_entries(entries)?;
        let core_space = space.get().core().clone();
        let configuration = Self::check(py, move |context| {
            Configuration::new(&core_space, core, context)
        })?;
        Ok(Self {
            configuration,
            space: space.clone().unbind(),
            values,
        })
    }

    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.space)?;
        for value in self.values.values() {
            visit.call(value)?;
        }
        Ok(())
    }

    /// Register `cls` as the public `Configuration` class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }

    /// The `Space` the configuration is a point of.
    #[getter]
    fn space<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.space.bind(py).clone().into_any()
    }

    /// The `(name, value)` entries, in canonical order of the decisions.
    #[getter]
    fn entries<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.get();
        let space = this.space.bind(py).get();
        let entries = this
            .configuration
            .entries()
            .map(|(name, _)| -> PyResult<Bound<'py, PyTuple>> {
                let decision = space
                    .position(name)
                    .map(|position| space.decision_at(py, position));
                let name_object = match decision {
                    Some(decision) => decision?.getattr(intern!(py, "name"))?,
                    None => crate::identifier::identifier_to_python(py, name)?,
                };
                let value = this.value_object(py, name)?;
                PyTuple::new(py, [Some(name_object), value])
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, entries)
    }

    /// Return the value of the decision `name`, or `None` when it has none.
    ///
    /// Raises `TypeError` if `name` is not an `Identifier`.
    fn value<'py>(
        slf: &Bound<'py, Self>,
        name: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let name = restore_identifier(name, "Configuration", "name")?;
        slf.get().value_object(slf.py(), &name)
    }

    /// Return the alternative the choice `choice` chose, or `None`.
    ///
    /// Raises `TypeError` if `choice` is not an `Identifier`.
    fn alternative<'py>(
        slf: &Bound<'py, Self>,
        choice: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = slf.py();
        let this = slf.get();
        let name = restore_identifier(choice, "Configuration", "choice")?;
        let Some(chosen) = this.configuration.alternative(&name) else {
            return Ok(None);
        };
        let space = this.space.bind(py).get();
        let Some(position) = space.position(&name) else {
            return Ok(None);
        };
        let choice = space.decision_at(py, position)?;
        let choice = choice.cast::<PyChoice>()?.get();
        let index = choice
            .core()
            .alternatives()
            .iter()
            .position(|alternative| fhy_core::foreign::Part::ptr_eq(alternative, chosen));
        index
            .map(|index| choice.alternative_objects(py).get_item(index))
            .transpose()
    }

    /// Return the `Activity` of the decision `name`, or `None` for a name
    /// the space lacks.
    ///
    /// Raises `TypeError` if `name` is not an `Identifier`.
    fn activity<'py>(
        &self,
        py: Python<'py>,
        name: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let name = restore_identifier(name, "Configuration", "name")?;
        let Some(activity) = self.configuration.activity(&name) else {
            return Ok(None);
        };
        let member = match activity {
            Activity::Active => "ACTIVE",
            Activity::Inactive => "INACTIVE",
            Activity::Pending => "PENDING",
        };
        crate::cached_attr!(py, "fhy_core.search_space.core", "Activity")?
            .getattr(member)
            .map(Some)
    }

    /// Return whether every decision is assigned or inactive.
    fn is_complete(&self) -> bool {
        self.configuration.is_complete()
    }

    /// Return the `Trace` of the assigned decisions, in decision order.
    ///
    /// Raises `NotEnumerableError` for an assigned variable with no finite
    /// domain.
    #[expect(
        unused_variables,
        reason = "interface stub: the body is todo!() until the implementation"
    )]
    fn trace<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the key that identifies the configuration within its space.
    fn key(&self) -> PyConfigurationKey {
        PyConfigurationKey {
            key: self.configuration.key(),
        }
    }

    /// Return the configuration with `value` for the decision `name`, in
    /// place of its value if it has one, checked as a whole.
    ///
    /// Raises as the constructor does.
    fn with_entry<'py>(
        slf: &Bound<'py, Self>,
        name: &Bound<'py, PyAny>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pair = PyTuple::new(slf.py(), [name, value])?;
        let entries = read_entries(Some(PyTuple::new(slf.py(), [pair])?.as_any()))?;
        Self::with_entries_given(slf, entries)
    }

    /// Return the configuration with `entries` in place of or beside its
    /// entries, checked as a whole.
    ///
    /// Raises as the constructor does.
    fn with_entries<'py>(
        slf: &Bound<'py, Self>,
        entries: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::with_entries_given(slf, read_entries(Some(entries))?)
    }

    /// Return whether `other` is a configuration of a structurally
    /// equivalent space with equal values.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let py = slf.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        with_pending_errors(|| {
            slf.get()
                .configuration
                .is_structurally_equivalent(&other.get().configuration)
                .map_err(|error| equivalence_error_to_py(py, error))
        })
    }

    /// Return whether `other` is a configuration of an alpha-equivalent
    /// space whose values correspond, under no renaming.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        with_pending_errors(|| {
            slf.get()
                .configuration
                .is_alpha_equivalent(&other.get().configuration)
                .map_err(|error| equivalence_error_to_py(py, error))
        })
    }

    /// Return whether `other` is a configuration of an alpha-equivalent
    /// space whose values correspond, under `renaming`.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let py = slf.py();
        let renaming = read_renaming(renaming)?;
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        with_pending_errors(|| {
            slf.get()
                .configuration
                .is_alpha_equivalent_under(
                    &other.get().configuration,
                    renaming.get().value().renaming(),
                )
                .map_err(|error| equivalence_error_to_py(py, error))
        })
    }

    /// Refuse to set an attribute: the configuration is frozen.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        refuse_attribute_assignment(slf.as_any(), name)
    }

    /// Refuse to delete an attribute: the configuration is frozen.
    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        refuse_attribute_deletion(slf.as_any(), name)
    }

    /// Return `Configuration(space=<space name>, entries=...)`.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let space_name = slf.get().space.bind(py).getattr(intern!(py, "name"))?;
        let entries = Self::entries(slf)?;
        Ok(format!(
            "{}(space={}, entries={})",
            slf.get_type().qualname()?,
            space_name.repr()?,
            entries.repr()?
        ))
    }

    /// Pickle as a call of the class with its space and entries.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let fields = PyTuple::new(
            py,
            [
                slf.get().space.bind(py).clone().into_any(),
                Self::entries(slf)?.into_any(),
            ],
        )?;
        PyTuple::new(py, [slf.get_type().into_any(), fields.into_any()])
    }

    /// Return the V2 payload `{"space", "entries"}`.
    ///
    /// Raises `SerializationError` inside `wire_version(WireVersion.V1)`.
    fn serialize_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        refuse_v1(slf.as_any())?;
        write_part(slf.py(), || ConfigurationData::of(&slf.get().configuration))
    }

    /// Return the canonical V2 text of the payload, re-formatted for
    /// `indent` or `sort_keys`.
    #[pyo3(signature = (*, indent = None, sort_keys = None))]
    fn to_json(
        slf: &Bound<'_, Self>,
        indent: Option<&Bound<'_, PyAny>>,
        sort_keys: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        refuse_v1(slf.as_any())?;
        write_part_json(slf.as_any(), indent, sort_keys, || {
            ConfigurationData::of(&slf.get().configuration)
        })
    }

    /// Return the configuration of the V2 payload `data`, its values
    /// checked as a restored assignment is.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        decode_part(cls, Family::Configuration, data, false)
    }

    /// Return the configuration of the JSON text `payload`.
    #[classmethod]
    fn from_json<'py>(
        cls: &Bound<'py, PyType>,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        decode_part(cls, Family::Configuration, payload, true)
    }
}

/// Return a new public `Configuration` of `configuration`, over the `Space`
/// object `space`, which holds `configuration`'s space.
///
/// # Errors
///
/// Raises `TypeError` if `space` is no `Space`, and what building the
/// object raises.
pub(super) fn configuration_to_python<'py>(
    space: &Bound<'py, PyAny>,
    configuration: Configuration,
) -> PyResult<Bound<'py, PyAny>> {
    let py = space.py();
    let space: &Bound<'py, PySpace> = space.cast()?;
    PyConfiguration {
        configuration,
        space: space.clone().unbind(),
        values: HashMap::new(),
    }
    .into_python(py)
}

/// The identity of a configuration within its space, backed by the core
/// [`ConfigurationKey`]: equal for configurations of alpha-equivalent
/// spaces whose values correspond.
///
/// It is hashable and compares structurally, so it keys a dict. It has no
/// constructor; it pickles through its wire form, which is self-contained:
/// each identifier the space binds is written as its position among the
/// space's names.
#[pyclass(frozen, module = "fhy_core._rs", name = "ConfigurationKey")]
pub(crate) struct PyConfigurationKey {
    key: ConfigurationKey,
}

#[pymethods]
impl PyConfigurationKey {
    /// Compare structurally with another key; another type is
    /// `NotImplemented`.
    fn __richcmp__<'py>(&self, other: &Bound<'py, PyAny>, op: CompareOp) -> Bound<'py, PyAny> {
        let py = other.py();
        let Ok(other) = other.cast::<Self>() else {
            return py.NotImplemented().into_bound(py);
        };
        let equal = self.key == other.get().key;
        match op {
            CompareOp::Eq => PyBool::new(py, equal).to_owned().into_any(),
            CompareOp::Ne => PyBool::new(py, !equal).to_owned().into_any(),
            _ => py.NotImplemented().into_bound(py),
        }
    }

    /// Return the key's hash, consistent with `==`.
    fn __hash__(&self) -> u64 {
        hash_value(&self.key)
    }

    /// Return `ConfigurationKey(...)`.
    fn __repr__(&self) -> String {
        format!("{:?}", self.key)
    }

    /// Pickle as a call of `_from_wire` with the key's V2 text.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let text = with_pending_errors(|| {
            let data = ConfigurationKeyData::of(&slf.get().key)
                .map_err(|error| crate::wire::foreign_error(py, &error))?;
            crate::wire::to_json(py, &data)
        })?;
        let restore = slf.get_type().getattr(intern!(py, "_from_wire"))?;
        PyTuple::new(py, [restore, PyTuple::new(py, [text])?.into_any()])
    }

    /// Return the key of its V2 text `text`, the inverse of the text
    /// `__reduce__` writes.
    ///
    /// Raises `DeserializationValueError` for a text of another shape.
    #[staticmethod]
    fn _from_wire(py: Python<'_>, text: &str) -> PyResult<Self> {
        let invalid = |reason: String| {
            DESERIALIZATION_VALUE_ERROR.err(
                py,
                (format!(
                    "Invalid V2 payload for \"ConfigurationKey\": {reason}"
                ),),
            )
        };
        let data: ConfigurationKeyData =
            serde_json::from_str(text).map_err(|error| invalid(error.to_string()))?;
        let key = with_pending_errors(|| {
            data.build(&PyResolver)
                .map_err(|error| invalid(error.to_string()))
        })?;
        Ok(Self { key })
    }
}
