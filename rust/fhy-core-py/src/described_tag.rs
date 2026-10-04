//! `PyO3` classes for the [`fhy_core::described_tag`] vocabularies,
//! `OpAttribute` and `NoteKind`.
//!
//! A vocabulary's class wraps the canonical Rust tag and caches the Python
//! objects its attributes return. The public Python class is a thin
//! subclass that mixes in the stateless Python protocols and sets
//! `__new__` to the class's `_new_canonical`, which registers the tag in the
//! Rust registry and returns the single Python object of the canonical tag.
//! The class's own `__new__` only builds an instance from
//! a seed that `_new_canonical` creates, so no second Python object of a
//! canonical tag can exist. The public class registers itself with the
//! class's `_register_public_class` at import, so a canonical tag the
//! binding reaches from Rust, such as a note's kind, gets its Python object
//! as an instance of the public class.
//!
//! Payloads keep the Python shape `{"name": {"id": .., "name_hint": ..},
//! "description": ..}`, and every error matches the Python implementation's
//! class and message.

/// Define the `PyO3` class of one described-tag vocabulary.
///
/// `$class` becomes the class, exposed to Python as `$py_name`, over the
/// canonical tags of `$tag`. `$seed` becomes the private class its instances
/// are built from. `$extra` holds methods only this vocabulary has, which
/// join the class's single `#[pymethods]` block. The expansion names
/// `pyo3::prelude` items unqualified, so the calling module imports the
/// prelude.
macro_rules! define_described_tag_class {
    (
        $(#[$class_meta:meta])*
        class $class:ident as $py_name:literal;
        seed $seed:ident;
        tag $tag:ty;
        extra_methods { $($extra:tt)* }
    ) => {
        /// The canonical tag and cached Python objects an instance of the
        /// tag class is built from.
        ///
        /// Only the binding creates seeds, so only the binding can build an
        /// instance, and each canonical tag gets one Python object.
        #[pyclass(frozen, module = "fhy_core._rs")]
        pub(crate) struct $seed(
            $crate::kit::python::Seed<(
                ::fhy_core::interned::Canonical<$tag>,
                Py<PyAny>,
                Py<::pyo3::types::PyString>,
            )>,
        );

        $(#[$class_meta])*
        #[pyclass(subclass, frozen, module = "fhy_core._rs", name = $py_name)]
        pub(crate) struct $class {
            tag: ::fhy_core::interned::Canonical<$tag>,
            /// The Python `Identifier` of the tag's name.
            #[pyo3(get)]
            name: Py<PyAny>,
            /// The tag's description.
            #[pyo3(get)]
            description: Py<::pyo3::types::PyString>,
        }

        impl $class {
            /// Return the cache from each canonical tag's key to its Python
            /// object.
            fn identity_cache() -> &'static $crate::kit::interned::IdentityCache {
                static CACHE: $crate::kit::interned::IdentityCache =
                    $crate::kit::interned::IdentityCache::new();
                &CACHE
            }

            /// Return the public Python class registered for this class.
            fn public_class() -> &'static $crate::kit::public_class::PublicClass {
                static PUBLIC_CLASS: $crate::kit::public_class::PublicClass =
                    $crate::kit::public_class::PublicClass::new($py_name);
                &PUBLIC_CLASS
            }

            /// Return the Python object of the canonical `tag`, creating it
            /// if it has none yet as an instance of `cls`, or of the
            /// registered public class when `cls` is `None`.
            ///
            /// The object holds `name` as its name when that is a Python
            /// identifier with the tag's name hint, and otherwise a Python
            /// identifier built from the tag's name.
            fn to_python<'py>(
                py: Python<'py>,
                cls: Option<&Bound<'py, ::pyo3::types::PyType>>,
                tag: ::fhy_core::interned::Canonical<$tag>,
                name: Option<&Bound<'py, PyAny>>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let id = tag.name().id();
                if let Some(object) = Self::identity_cache().get(py, id) {
                    return Ok(object);
                }
                let cls = match cls {
                    Some(cls) => cls,
                    None => Self::public_class().get(py)?,
                };
                let name = match name {
                    Some(name)
                        if name
                            .getattr(::pyo3::intern!(py, "name_hint"))?
                            .eq(tag.name().name_hint())? =>
                    {
                        name.clone()
                    }
                    _ => $crate::identifier::identifier_to_python(py, tag.name())?,
                };
                let description = ::pyo3::types::PyString::new(py, tag.description());
                let seed = Bound::new(
                    py,
                    $seed($crate::kit::python::Seed::new((
                        tag,
                        name.unbind(),
                        description.unbind(),
                    ))),
                )?;
                let object = py
                    .get_type::<Self>()
                    .call_method1(::pyo3::intern!(py, "__new__"), (cls, seed))?;
                Ok(Self::identity_cache().insert(id, object))
            }

            /// Register the tag `name` names, unless it is registered, and
            /// return the Python object of the canonical tag.
            fn register<'py>(
                cls: &Bound<'py, ::pyo3::types::PyType>,
                name: &Bound<'py, PyAny>,
                description: &Bound<'py, ::pyo3::types::PyString>,
            ) -> PyResult<(Bound<'py, PyAny>, ::fhy_core::interned::Canonical<$tag>)> {
                let identifier = $crate::identifier::restore_identifier(name, $py_name, "name")?;
                let tag = <$tag>::register(identifier, description.to_str()?);
                let object = Self::to_python(cls.py(), Some(cls), tag.clone(), Some(name))?;
                Ok((object, tag))
            }

            /// Return the payload `{"name": .., "description": ..}` of `tag`.
            fn serialize_tag<'py>(
                py: Python<'py>,
                tag: &::fhy_core::interned::Canonical<$tag>,
            ) -> PyResult<Bound<'py, ::pyo3::types::PyDict>> {
                let payload = ::pyo3::types::PyDict::new(py);
                payload.set_item(
                    ::pyo3::intern!(py, "name"),
                    $crate::identifier::serialize_identifier(py, tag.name())?,
                )?;
                payload.set_item(::pyo3::intern!(py, "description"), tag.description())?;
                Ok(payload)
            }

            /// Build the canonical tag from decoded fields, warning when the
            /// canonical tag keeps a different description.
            fn construct<'py>(
                cls: &Bound<'py, ::pyo3::types::PyType>,
                name: &Bound<'py, PyAny>,
                description: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let description = description.cast::<::pyo3::types::PyString>()?;
                let (object, tag) = Self::register(cls, name, description)?;
                $crate::kit::interned::warn_if_description_ignored(
                    cls,
                    name,
                    &::pyo3::types::PyString::new(cls.py(), tag.description()),
                    description,
                )?;
                Ok(object)
            }
        }

        #[pymethods]
        impl $class {
            /// Visit the name and description, for the cycle collector.
            fn __traverse__(
                &self,
                visit: ::pyo3::pyclass::PyVisit<'_>,
            ) -> Result<(), ::pyo3::pyclass::PyTraverseError> {
                visit.call(&self.name)?;
                visit.call(&self.description)
            }

            /// Build an instance from a seed the binding created.
            #[new]
            fn new(seed: &Bound<'_, $seed>) -> PyResult<Self> {
                let (tag, name, description) = seed.get().0.take("a tag")?;
                Ok(Self {
                    tag,
                    name,
                    description,
                })
            }

            /// Return the canonical tag named `name`, registering it with
            /// `description` unless it is registered; the public class's
            /// `__new__`.
            #[staticmethod]
            fn _new_canonical<'py>(
                target_class: &Bound<'py, ::pyo3::types::PyType>,
                name: &Bound<'py, PyAny>,
                description: &Bound<'py, ::pyo3::types::PyString>,
            ) -> PyResult<Bound<'py, PyAny>> {
                // A cached key is registered, and the first registration
                // wins, so the cached object is the answer.
                if let Some(id) = $crate::identifier::read_identifier_id(name)? {
                    if let Some(object) = Self::identity_cache().get(target_class.py(), id) {
                        return Ok(object);
                    }
                }
                Self::register(target_class, name, description).map(|(object, _tag)| object)
            }

            /// Return the tag's name.
            fn get_identifier(&self, py: Python<'_>) -> Py<PyAny> {
                self.name.clone_ref(py)
            }

            /// Return the tag's name, the key it is interned under.
            fn get_intern_key(&self, py: Python<'_>) -> Py<PyAny> {
                self.name.clone_ref(py)
            }

            /// Return whether `other` is a tag of the same class with the
            /// same name.
            fn is_structurally_equivalent(
                slf: &Bound<'_, Self>,
                other: &Bound<'_, PyAny>,
            ) -> bool {
                slf.get_type().is(other.get_type())
                    && other
                        .cast::<Self>()
                        .is_ok_and(|other| other.get().tag == slf.get().tag)
            }

            /// Return whether `other` is a tag of the same class with the
            /// same name; a tag binds no names, so `renaming` is unused.
            fn is_alpha_equivalent_under(
                slf: &Bound<'_, Self>,
                other: &Bound<'_, PyAny>,
                renaming: &Bound<'_, PyAny>,
            ) -> bool {
                let _ = renaming;
                Self::is_structurally_equivalent(slf, other)
            }

            /// Always true: tags are immutable.
            #[getter]
            const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
                true
            }

            /// Do nothing: tags are always frozen.
            const fn freeze(_slf: &Bound<'_, Self>) {}

            /// Do nothing: tags are always frozen, and mutating one raises.
            const fn assert_frozen(_slf: &Bound<'_, Self>) {}

            /// Do nothing: a tag is registered when it is constructed.
            const fn register_interned_instance(_slf: &Bound<'_, Self>) {}

            fn __eq__(&self, other: &Bound<'_, Self>) -> bool {
                self.tag == other.get().tag
            }

            fn __hash__(&self) -> u64 {
                self.tag.name().id()
            }

            fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                let this = slf.get();
                let py = slf.py();
                Ok(format!(
                    "{}(name={}, description={})",
                    slf.get_type().qualname()?,
                    this.name.bind(py).repr()?,
                    this.description.bind(py).repr()?,
                ))
            }

            fn __setattr__(
                slf: &Bound<'_, Self>,
                name: &str,
                _value: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                $crate::kit::frozen::refuse_attribute_assignment(slf, name)
            }

            fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
                $crate::kit::frozen::refuse_attribute_deletion(slf, name)
            }

            /// Pickle as the payload, so unpickling returns the canonical
            /// tag.
            fn __reduce__<'py>(
                slf: &Bound<'py, Self>,
            ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, ::pyo3::types::PyDict>,))> {
                let py = slf.py();
                Ok((
                    slf.get_type()
                        .getattr(::pyo3::intern!(py, "deserialize_from_dict"))?,
                    (slf.get().serialize_to_dict(py)?,),
                ))
            }

            /// Return the payload `{"name": .., "description": ..}`.
            fn serialize_to_dict<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, ::pyo3::types::PyDict>> {
                Self::serialize_tag(py, &self.tag)
            }

            /// Return the canonical tag for `key`, or `None` if none is
            /// registered.
            #[classmethod]
            fn get_interned<'py>(
                cls: &Bound<'py, ::pyo3::types::PyType>,
                key: &Bound<'py, PyAny>,
            ) -> PyResult<Option<Bound<'py, PyAny>>> {
                let Some(id) = $crate::identifier::read_identifier_id(key)? else {
                    // Only an identifier names a tag, but an unhashable key
                    // raises, as a dict lookup does.
                    key.hash()?;
                    return Ok(None);
                };
                let object = if let Some(object) = Self::identity_cache().get(cls.py(), id) {
                    object
                } else {
                    let identifier =
                        $crate::identifier::restore_identifier(key, $py_name, "key")?;
                    let registry = <$tag as ::fhy_core::interned::Interned>::intern_registry();
                    let Some(tag) = registry.get(&identifier) else {
                        return Ok(None);
                    };
                    Self::to_python(cls.py(), Some(cls), tag, Some(key))?
                };
                Ok(object.is_instance(cls)?.then_some(object))
            }

            /// Return the canonical tag for `key`.
            ///
            /// Raises `KeyError` if none is registered.
            #[classmethod]
            fn require_interned<'py>(
                cls: &Bound<'py, ::pyo3::types::PyType>,
                key: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                match Self::get_interned(cls, key)? {
                    Some(object) => Ok(object),
                    None => Err($crate::kit::interned::build_not_interned_error(cls, key)?),
                }
            }

            /// Return the canonical tag for the decoded fields `name` and
            /// `description`, registering it unless it is registered.
            #[classmethod]
            fn construct_from_fields<'py>(
                cls: &Bound<'py, ::pyo3::types::PyType>,
                fields: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let [name, description] =
                    $crate::kit::serialization::read_constructor_fields(
                        cls,
                        fields,
                        ["name", "description"],
                        0,
                    )?;
                Self::construct(cls, &name, &description)
            }

            /// Return the canonical tag for a payload, registering it unless
            /// it is registered.
            #[classmethod]
            fn deserialize_from_dict<'py>(
                cls: &Bound<'py, ::pyo3::types::PyType>,
                data: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                use $crate::kit::serialization::FieldShape;
                let [name, description] = $crate::kit::serialization::read_payload_fields(
                    cls,
                    data,
                    [("name", FieldShape::Payload), ("description", FieldShape::Str)],
                )?;
                let name = $crate::identifier::deserialize_identifier(&name)?;
                Self::construct(cls, &name, &description)
            }

            /// Register `cls` as the public class, whose instances the
            /// binding builds for canonical tags it reaches from Rust.
            ///
            /// Raises `RuntimeError` if another public class is registered.
            #[classmethod]
            fn _register_public_class(cls: &Bound<'_, ::pyo3::types::PyType>) -> PyResult<()> {
                Self::public_class().register(cls)
            }

            /// Raise `NotImplementedError`: the Rust registry is
            /// append-only.
            #[classmethod]
            fn clear_interned_registry(cls: &Bound<'_, ::pyo3::types::PyType>) -> PyResult<()> {
                $crate::kit::interned::raise_registry_append_only(cls, "clear_interned_registry")
            }

            /// Raise `NotImplementedError`: the Rust registry is
            /// append-only, so the shipped tags are never unregistered.
            #[classmethod]
            fn register_default_instances(
                cls: &Bound<'_, ::pyo3::types::PyType>,
            ) -> PyResult<()> {
                $crate::kit::interned::raise_registry_append_only(cls, "register_default_instances")
            }

            $($extra)*
        }
    };
}

pub(crate) use define_described_tag_class;
