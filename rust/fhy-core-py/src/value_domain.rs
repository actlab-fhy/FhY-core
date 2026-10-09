//! `PyO3` class for [`fhy_core::value_domain`]: `fhy_core._rs.ValueDomain`,
//! the base of `fhy_core.value_domain.ValueDomain`.
//!
//! The class wraps the canonical Rust domain and caches the Python objects
//! its attributes return, including its parent's single Python object. As
//! for the described tags, the public class's `__new__` is
//! `_new_canonical`, which returns the single Python object of the
//! canonical domain.
//!
//! Domains follow the Rust semantics: a name has one parent, so constructing
//! a registered name under another parent raises `ValueError` with the Rust
//! conflict's text, and domains compare by name.
//! Payloads keep the Python shape, `{"name": .., "description": ..,
//! "parent": <payload or None>}` with the parent nested; decoding a payload
//! that conflicts with the canonical domain raises the Python
//! implementation's `DeserializationValueError`.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyString, PyType};

use fhy_core::interned::{Canonical, Interned};
use fhy_core::value_domain::{ValueDomain, ValueDomainConflict};

use crate::error::{IntoPyErr, IntoPyResult};
use crate::identifier::{
    deserialize_identifier, identifier_to_python, read_identifier_id, restore_identifier,
    serialize_identifier,
};
use crate::util::interned::{
    IdentityCache, build_conflict_error, build_not_interned_error, raise_registry_append_only,
    warn_if_description_ignored,
};
use crate::util::python::Seed;
use crate::util::serialization::{FieldShape, read_constructor_fields, read_payload_fields};

/// The class name errors and restored identifiers name.
const CLASS_NAME: &str = "ValueDomain";

/// Raises `ValueError` with the conflict's own text: the value domain
/// semantics are the Rust ones, and constructing a domain has no Python
/// error to match.
impl IntoPyErr for ValueDomainConflict {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

/// The cache from each canonical domain's key to its Python object.
static IDENTITY_CACHE: IdentityCache = IdentityCache::new();

/// The canonical domain and cached Python objects an instance of
/// `ValueDomain` is built from.
///
/// Only the binding creates seeds, so only the binding can build an
/// instance, and each canonical domain gets one Python object.
#[pyclass(frozen, module = "fhy_core._rs")]
pub(crate) struct ValueDomainSeed(Seed<PyValueDomain>);

/// Return the canonical domain of the Python `ValueDomain` `object`.
///
/// # Errors
///
/// Raises `TypeError` if `object` is not a `ValueDomain`.
pub(crate) fn value_domain_from_python(
    object: &Bound<'_, PyAny>,
) -> PyResult<Canonical<ValueDomain>> {
    object
        .cast::<PyValueDomain>()
        .map(|domain| domain.get().domain.clone())
        .map_err(|_not_a_domain| {
            PyTypeError::new_err(format!(
                "expected a {CLASS_NAME}, got {}.",
                object
                    .get_type()
                    .name()
                    .map_or_else(|_| "?".to_owned(), |name| name.to_string())
            ))
        })
}

/// Return the single Python object of the canonical `domain`, an instance
/// of the public `ValueDomain` class.
///
/// # Errors
///
/// Raises what importing `fhy_core.value_domain` or building the object
/// raises.
pub(crate) fn value_domain_to_python(
    py: Python<'_>,
    domain: Canonical<ValueDomain>,
) -> PyResult<Bound<'_, PyAny>> {
    let class = py
        .import("fhy_core.value_domain")?
        .getattr("ValueDomain")?
        .cast_into::<PyType>()?;
    PyValueDomain::to_python(&class, domain, None)
}

/// Open classification of the kind of value an IR operation handles, backed
/// by the canonical Rust [`ValueDomain`].
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "ValueDomain")]
pub(crate) struct PyValueDomain {
    domain: Canonical<ValueDomain>,
    /// The Python `Identifier` of the domain's name.
    #[pyo3(get)]
    name: Py<PyAny>,
    /// The domain's description.
    #[pyo3(get)]
    description: Py<PyString>,
    /// The Python object of the domain's parent, or `None` for a root.
    #[pyo3(get)]
    parent: Option<Py<PyAny>>,
}

impl PyValueDomain {
    /// Return the Python object of the canonical `domain`, creating it, and
    /// any of its ancestors that have none yet, as instances of `cls`.
    ///
    /// The object holds `name` as its name when that is a Python identifier
    /// with the domain's name hint, and otherwise a Python identifier built
    /// from the domain's name.
    fn to_python<'py>(
        cls: &Bound<'py, PyType>,
        domain: Canonical<ValueDomain>,
        name: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let id = domain.name().id();
        if let Some(object) = IDENTITY_CACHE.get(py, &id) {
            return Ok(object);
        }
        let parent = domain
            .parent()
            .map(|parent| Self::to_python(cls, parent.clone(), None))
            .transpose()?;
        let name = match name {
            Some(name)
                if name
                    .getattr(intern!(py, "name_hint"))?
                    .eq(domain.name().name_hint())? =>
            {
                name.clone()
            }
            _ => identifier_to_python(py, domain.name())?,
        };
        let description = PyString::new(py, domain.description());
        let seed = Bound::new(
            py,
            ValueDomainSeed(Seed::new(Self {
                domain,
                name: name.unbind(),
                description: description.unbind(),
                parent: parent.map(Bound::unbind),
            })),
        )?;
        let object = py
            .get_type::<Self>()
            .call_method1(intern!(py, "__new__"), (cls, seed))?;
        Ok(IDENTITY_CACHE.insert(id, object))
    }

    /// Return the canonical domain of the Python `parent`, a `ValueDomain`
    /// or `None`.
    fn read_parent(parent: &Bound<'_, PyAny>) -> PyResult<Option<Canonical<ValueDomain>>> {
        if parent.is_none() {
            return Ok(None);
        }
        match parent.cast::<Self>() {
            Ok(parent) => Ok(Some(parent.get().domain.clone())),
            Err(_not_a_domain) => Err(PyTypeError::new_err(format!(
                "{CLASS_NAME} parent must be a {CLASS_NAME} or None, got {}.",
                parent.get_type().name()?
            ))),
        }
    }

    /// Register the domain `name` names under `parent`, unless it is
    /// registered, and return the canonical domain.
    fn register(
        name: &Bound<'_, PyAny>,
        description: &Bound<'_, PyString>,
        parent: Option<&Canonical<ValueDomain>>,
    ) -> PyResult<Result<Canonical<ValueDomain>, ValueDomainConflict>> {
        let name = restore_identifier(name, CLASS_NAME, "name")?;
        let description = description.to_str()?;
        Ok(match parent {
            None => ValueDomain::register_root(name, description),
            Some(parent) => ValueDomain::register_child(name, description, parent),
        })
    }

    /// Build the canonical domain from decoded fields, warning when the
    /// canonical domain keeps a different description.
    ///
    /// # Errors
    ///
    /// Raises the Python implementation's `DeserializationValueError` if
    /// the name is registered under another parent.
    fn construct<'py>(
        cls: &Bound<'py, PyType>,
        name: &Bound<'py, PyAny>,
        description: &Bound<'py, PyAny>,
        parent: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let description = description.cast::<PyString>()?;
        match Self::register(name, description, Self::read_parent(parent)?.as_ref())? {
            Ok(domain) => {
                let canonical_description = PyString::new(cls.py(), domain.description());
                let object = Self::to_python(cls, domain, Some(name))?;
                warn_if_description_ignored(cls, name, &canonical_description, description)?;
                Ok(object)
            }
            Err(conflict) => {
                let Some(canonical) = ValueDomain::intern_registry().get(conflict.name()) else {
                    return Err(conflict.into_py_err());
                };
                let canonical_parent = match canonical.parent() {
                    Some(canonical_parent) => Self::to_python(cls, canonical_parent.clone(), None)?,
                    None => cls.py().None().into_bound(cls.py()),
                };
                Err(build_conflict_error(
                    cls,
                    name,
                    "parent",
                    &canonical_parent,
                    parent,
                )?)
            }
        }
    }
}

#[pymethods]
impl PyValueDomain {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.name)?;
        visit.call(&self.description)?;
        visit.call(self.parent.as_ref())?;
        Ok(())
    }

    /// Build an instance from a seed the binding created.
    #[new]
    fn new(seed: &Bound<'_, ValueDomainSeed>) -> PyResult<Self> {
        seed.get().0.take("a value-domain")
    }

    /// Return the canonical domain named `name`, registering it with
    /// `description` under `parent` unless it is registered; the public
    /// class's `__new__`.
    ///
    /// Raises `ValueError` if `name` is registered under another parent.
    #[staticmethod]
    #[pyo3(signature = (target_class, name, description, parent = None))]
    fn _new_canonical<'py>(
        target_class: &Bound<'py, PyType>,
        name: &Bound<'py, PyAny>,
        description: &Bound<'py, PyString>,
        parent: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = target_class.py();
        let parent = match parent {
            Some(parent) => Self::read_parent(parent)?,
            None => None,
        };
        // A cached domain under the same parent is the answer, since the
        // first registration wins; any other case registers, which also
        // reports a conflicting parent.
        if let Some(id) = read_identifier_id(name)? {
            if let Some(object) = IDENTITY_CACHE.get(py, &id) {
                let cached_parent = object.cast::<Self>()?.get().domain.parent();
                if cached_parent.map(|domain| domain.name())
                    == parent.as_ref().map(|domain| domain.name())
                {
                    return Ok(object);
                }
            }
        }
        let domain = Self::register(name, description, parent.as_ref())?.into_py_result()?;
        Self::to_python(target_class, domain, Some(name))
    }

    /// Return the domain's name.
    fn get_identifier(&self, py: Python<'_>) -> Py<PyAny> {
        self.name.clone_ref(py)
    }

    /// Return the domain's name, the key it is interned under.
    fn get_intern_key(&self, py: Python<'_>) -> Py<PyAny> {
        self.name.clone_ref(py)
    }

    /// Return whether `other` is this domain or one of its ancestors.
    ///
    /// Returns `False` for anything but a `ValueDomain`.
    fn is_subdomain_of(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<Self>()
            .is_ok_and(|other| self.domain.is_subdomain_of(&other.get().domain))
    }

    /// Return whether `other` is a domain of the same class with the same
    /// name, which decides its parent too.
    fn is_structurally_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> bool {
        slf.get_type().is(other.get_type())
            && other
                .cast::<Self>()
                .is_ok_and(|other| other.get().domain == slf.get().domain)
    }

    /// Return whether `other` is a domain of the same class with the same
    /// name; a domain binds no names, so `renaming` is unused.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> bool {
        let _ = renaming;
        Self::is_structurally_equivalent(slf, other)
    }

    /// Always true: domains are immutable.
    #[getter]
    const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: domains are always frozen.
    const fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: domains are always frozen, and mutating one raises.
    const fn assert_frozen(_slf: &Bound<'_, Self>) {}

    /// Do nothing: a domain is registered when it is constructed.
    const fn register_interned_instance(_slf: &Bound<'_, Self>) {}

    fn __eq__(&self, other: &Bound<'_, Self>) -> bool {
        self.domain == other.get().domain
    }

    fn __hash__(&self) -> u64 {
        self.domain.name().id()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let this = slf.get();
        let py = slf.py();
        let parent = match &this.parent {
            Some(parent) => parent.bind(py).repr()?.to_string(),
            None => "None".to_owned(),
        };
        Ok(format!(
            "{}(name={}, description={}, parent={parent})",
            slf.get_type().qualname()?,
            this.name.bind(py).repr()?,
            this.description.bind(py).repr()?,
        ))
    }

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        crate::util::frozen::refuse_attribute_assignment(slf, name)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        crate::util::frozen::refuse_attribute_deletion(slf, name)
    }

    /// Pickle as the V2 payload, so unpickling returns the canonical domain.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyAny>,))> {
        let py = slf.py();
        Ok((
            slf.get_type()
                .getattr(intern!(py, "deserialize_from_dict"))?,
            (crate::wire::to_dict(py, &*slf.get().domain)?,),
        ))
    }

    /// Return the V2 payload `{"levels": [{"name", "description"}, ..]}`,
    /// the core's chain, root first; or the V1 payload inside
    /// `wire_version(WireVersion.V1)`.
    fn serialize_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        crate::wire::write_dict(
            slf.as_any(),
            || Ok(slf.get().serialize_v1(py)?.into_any()),
            || Ok(slf.get().domain.clone()),
        )
    }

    /// Return the JSON text of the payload: the canonical V2 text unless
    /// `indent` or `sort_keys` re-formats it or V1 is written.
    #[pyo3(signature = (*, indent = None, sort_keys = None))]
    fn to_json(
        slf: &Bound<'_, Self>,
        indent: Option<&Bound<'_, PyAny>>,
        sort_keys: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        crate::wire::write_json(slf.as_any(), indent, sort_keys, || {
            Ok(slf.get().domain.clone())
        })
    }

    /// Return the canonical domain for `key`, or `None` if none is
    /// registered.
    #[classmethod]
    fn get_interned<'py>(
        cls: &Bound<'py, PyType>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(id) = read_identifier_id(key)? else {
            // Only an identifier names a domain, but an unhashable key
            // raises, as a dict lookup does.
            key.hash()?;
            return Ok(None);
        };
        let object = if let Some(object) = IDENTITY_CACHE.get(cls.py(), &id) {
            object
        } else {
            let identifier = restore_identifier(key, CLASS_NAME, "key")?;
            let Some(domain) = ValueDomain::intern_registry().get(&identifier) else {
                return Ok(None);
            };
            Self::to_python(cls, domain, Some(key))?
        };
        Ok(object.is_instance(cls)?.then_some(object))
    }

    /// Return the canonical domain for `key`.
    ///
    /// Raises `KeyError` if none is registered.
    #[classmethod]
    fn require_interned<'py>(
        cls: &Bound<'py, PyType>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match Self::get_interned(cls, key)? {
            Some(object) => Ok(object),
            None => Err(build_not_interned_error(cls, key)?),
        }
    }

    /// Return the canonical domain for the decoded fields `name`,
    /// `description` and, optionally, `parent`, registering it unless it is
    /// registered.
    #[classmethod]
    fn construct_from_fields<'py>(
        cls: &Bound<'py, PyType>,
        fields: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let [name, description, parent] =
            read_constructor_fields(cls, fields, ["name", "description", "parent"], 1)?;
        Self::construct(cls, &name, &description, &parent)
    }

    /// Return the canonical domain for a payload, registering it, and its
    /// ancestors, unless they are registered: a V2 payload of its levels,
    /// root first, or a V1 payload, which nests the parent and warns.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let is_v1 = crate::wire::is_reading_v1(py)
            || data
                .cast::<PyDict>()
                .is_ok_and(|data| !data.contains(intern!(py, "levels")).unwrap_or(false));
        if is_v1 {
            return crate::wire::reading_v1(cls, || Self::deserialize_v1(cls, data));
        }
        let [levels] = read_payload_fields(cls, data, [("levels", FieldShape::PayloadList)])?;
        let mut parent = py.None().into_bound(py);
        let levels = levels.cast::<pyo3::types::PyList>()?;
        if levels.is_empty() {
            return Err(crate::util::exceptions::DESERIALIZATION_VALUE_ERROR
                .err(py, (cls, "levels", "a non-empty list of levels", levels)));
        }
        for level in levels.iter() {
            let [name, description] = read_payload_fields(
                cls,
                &level,
                [
                    ("name", FieldShape::Payload),
                    ("description", FieldShape::Str),
                ],
            )?;
            let name = deserialize_identifier(&name)?;
            parent = Self::construct(cls, &name, &description, &parent)?;
        }
        Ok(parent)
    }

    /// Raise `NotImplementedError`: the Rust registry is append-only.
    #[classmethod]
    fn clear_interned_registry(cls: &Bound<'_, PyType>) -> PyResult<()> {
        raise_registry_append_only(cls, "clear_interned_registry")
    }

    /// Raise `NotImplementedError`: the Rust registry is append-only, so
    /// the shipped domains are never unregistered.
    #[classmethod]
    fn register_default_instances(cls: &Bound<'_, PyType>) -> PyResult<()> {
        raise_registry_append_only(cls, "register_default_instances")
    }
}

impl PyValueDomain {
    /// Return the V1 payload `{"name": .., "description": .., "parent":
    /// ..}`, with the parent's payload nested, or `None` for a root.
    ///
    /// V1: removed with the V1 wire format.
    fn serialize_v1<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut chain = vec![&*self.domain];
        while let Some(parent) = chain[chain.len() - 1].parent() {
            chain.push(parent);
        }
        let mut parent_payload = py.None().into_bound(py);
        for domain in chain.into_iter().rev() {
            let payload = PyDict::new(py);
            payload.set_item(
                intern!(py, "name"),
                serialize_identifier(py, domain.name())?,
            )?;
            payload.set_item(intern!(py, "description"), domain.description())?;
            payload.set_item(intern!(py, "parent"), parent_payload)?;
            parent_payload = payload.into_any();
        }
        Ok(parent_payload.cast_into::<PyDict>()?)
    }

    /// Return the canonical domain for a V1 payload, registering it, and
    /// its ancestors, unless they are registered.
    ///
    /// V1: removed with the V1 wire format.
    fn deserialize_v1<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let [name, description, parent] = read_payload_fields(
            cls,
            data,
            [
                ("name", FieldShape::Payload),
                ("description", FieldShape::Str),
                ("parent", FieldShape::OptionalPayload),
            ],
        )?;
        let name = deserialize_identifier(&name)?;
        let parent = if parent.is_none() {
            parent
        } else {
            Self::deserialize_v1(cls, &parent)?
        };
        Self::construct(cls, &name, &description, &parent)
    }
}
