//! `fhy_core._rs.AlphaRenaming`, which is `fhy_core.term.AlphaRenaming`: the
//! Rust [`AlphaRenaming`] with the Python identifier objects it was built
//! from.
//!
//! The Rust renaming decides every lookup. The Python `Identifier` stays a
//! Python class, so the binding keeps each identifier object given for
//! a key or an image, by id, beside the Rust frames, and `resolve` returns
//! the object that was given. The objects live in a shared chain of tables,
//! one per extension, so extending a renaming copies no table, as the Rust
//! frames copy no map.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyMapping, PyTuple, PyType};

use fhy_core::identifier::Identifier;
use fhy_core::term::{AlphaRenaming, NonInjectiveRenamingError, RenamingMap};

use crate::error::{IntoPyErr, IntoPyResult};
use crate::identifier::{identifier_to_python, restore_identifier};
use crate::kit::dataclass::hash_value;
use crate::kit::frozen::{refuse_attribute_assignment, refuse_attribute_deletion};

/// Raises `ValueError` with the core's text.
impl IntoPyErr for NonInjectiveRenamingError {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

// ---------------------------------------------------------------------------
// The identifier objects
// ---------------------------------------------------------------------------

/// The Python identifier objects of a renaming, by id: a chain of tables,
/// each holding the objects one extension added, newest first.
#[derive(Clone, Default)]
pub(super) struct ObjectTable(Option<Arc<ObjectNode>>);

struct ObjectNode {
    parent: ObjectTable,
    objects: HashMap<u64, Py<PyAny>>,
}

impl ObjectTable {
    /// Return this table with `objects` added, or this table itself when
    /// there are none.
    pub(super) fn with(&self, objects: HashMap<u64, Py<PyAny>>) -> Self {
        if objects.is_empty() {
            self.clone()
        } else {
            Self(Some(Arc::new(ObjectNode {
                parent: self.clone(),
                objects,
            })))
        }
    }

    /// Return the newest object held for `id`.
    pub(super) fn get<'py>(&self, py: Python<'py>, id: u64) -> Option<Bound<'py, PyAny>> {
        let mut node = self.0.as_deref();
        while let Some(current) = node {
            if let Some(object) = current.objects.get(&id) {
                return Some(object.bind(py).clone());
            }
            node = current.parent.0.as_deref();
        }
        None
    }
}

impl ObjectTable {
    /// Visit the objects of the nodes this table alone holds.
    ///
    /// The chain is shared between renamings, and a reference must be
    /// visited at most once, so the walk stops at the first node another
    /// table or node holds too.
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        let mut node = self.0.as_ref();
        while let Some(current) = node {
            if Arc::strong_count(current) != 1 {
                break;
            }
            crate::kit::gc::traverse_all(visit, current.objects.values())?;
            node = current.parent.0.as_ref();
        }
        Ok(())
    }
}

/// Drops a long chain on the heap rather than by recursion.
impl Drop for ObjectTable {
    fn drop(&mut self) {
        let mut next = self.0.take();
        while let Some(node) = next {
            next = match Arc::try_unwrap(node) {
                Ok(mut node) => node.parent.0.take(),
                Err(_shared) => None,
            };
        }
    }
}

/// Python objects keyed by the id of the identifier each is.
type ObjectsById = HashMap<u64, Py<PyAny>>;

/// A list of Python identifiers, with the Rust identifiers of the same ids.
pub(super) struct IdentifierList<'py> {
    pub(super) identifiers: Vec<Identifier>,
    pub(super) objects: Vec<Bound<'py, PyAny>>,
}

impl<'py> IdentifierList<'py> {
    /// Return the list of the identifiers `values` yields.
    ///
    /// # Errors
    ///
    /// Raises `TypeError` naming `owner` and `field` for an item that is not
    /// an `Identifier`, and whatever iterating `values` raises.
    pub(super) fn read(values: &Bound<'py, PyAny>, owner: &str, field: &str) -> PyResult<Self> {
        let mut identifiers = Vec::new();
        let mut objects = Vec::new();
        for value in values.try_iter()? {
            let value = value?;
            identifiers.push(restore_identifier(&value, owner, field)?);
            objects.push(value);
        }
        Ok(Self {
            identifiers,
            objects,
        })
    }

    /// Return the objects by id.
    pub(super) fn object_map(&self) -> HashMap<u64, Py<PyAny>> {
        self.identifiers
            .iter()
            .zip(&self.objects)
            .map(|(identifier, object)| (identifier.id(), object.clone().unbind()))
            .collect()
    }
}

/// Return the identifier map of the Python mapping `mapping`, and its key
/// and value objects by id.
///
/// # Errors
///
/// Raises `TypeError` naming `owner` and `field` for a value that is not a
/// mapping, or a key or value that is not an `Identifier`.
fn read_identifier_map(
    mapping: &Bound<'_, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<(HashMap<Identifier, Identifier>, ObjectsById)> {
    let size = mapping.len().unwrap_or(0);
    let mut map = HashMap::with_capacity(size);
    let mut objects = HashMap::with_capacity(2 * size);
    let mut insert = |key: Bound<'_, PyAny>, value: Bound<'_, PyAny>| -> PyResult<()> {
        let rust_key = restore_identifier(&key, owner, "key")?;
        let rust_value = restore_identifier(&value, owner, "value")?;
        objects.insert(rust_key.id(), key.unbind());
        objects.insert(rust_value.id(), value.unbind());
        map.insert(rust_key, rust_value);
        Ok(())
    };
    if let Ok(dict) = mapping.cast::<PyDict>() {
        for (key, value) in dict.iter() {
            insert(key, value)?;
        }
    } else if let Ok(mapping) = mapping.cast::<PyMapping>() {
        for item in mapping.items()?.iter() {
            let (key, value) = item.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()?;
            insert(key, value)?;
        }
    } else {
        return Err(PyTypeError::new_err(format!(
            "{owner} {field} must be a mapping, got {}.",
            mapping.get_type().name()?
        )));
    }
    Ok((map, objects))
}

// ---------------------------------------------------------------------------
// The value
// ---------------------------------------------------------------------------

/// A Rust renaming with the Python objects of its identifiers.
#[derive(Clone, Default)]
pub(crate) struct RenamingValue {
    renaming: AlphaRenaming,
    objects: ObjectTable,
}

impl RenamingValue {
    /// Return the value of `renaming`, whose identifiers' objects `objects`
    /// holds.
    pub(super) const fn new(renaming: AlphaRenaming, objects: ObjectTable) -> Self {
        Self { renaming, objects }
    }

    /// Return the Rust renaming.
    pub(crate) const fn renaming(&self) -> &AlphaRenaming {
        &self.renaming
    }

    /// Return the identifier objects.
    pub(super) const fn objects(&self) -> &ObjectTable {
        &self.objects
    }

    /// Return this renaming with one more frame pairing `left` with
    /// `right` by position, or `None` if the core refuses the pairing.
    pub(super) fn entered(
        &self,
        left: &IdentifierList<'_>,
        right: &IdentifierList<'_>,
    ) -> Option<Self> {
        let mut renaming = self.renaming.clone();
        renaming
            .enter_binders(&left.identifiers, &right.identifiers)
            .ok()?;
        let mut objects = left.object_map();
        objects.extend(right.object_map());
        Some(Self {
            renaming,
            objects: self.objects.with(objects),
        })
    }

    /// Return the Python object of `identifier`: the one given for its id,
    /// or a new one with its id and name hint.
    ///
    /// # Errors
    ///
    /// Raises whatever building a new identifier raises.
    pub(super) fn object_of<'py>(
        &self,
        py: Python<'py>,
        identifier: &Identifier,
    ) -> PyResult<Bound<'py, PyAny>> {
        match self.objects.get(py, identifier.id()) {
            Some(object) => Ok(object),
            None => identifier_to_python(py, identifier),
        }
    }

    /// Return the Python object of this renaming.
    ///
    /// # Errors
    ///
    /// Raises whatever allocating the object raises.
    pub(super) fn into_python(self, py: Python<'_>) -> PyResult<Bound<'_, PyAlphaRenaming>> {
        Bound::new(py, PyAlphaRenaming { value: self })
    }

    /// Return the Python dict of the pairs of `map`.
    fn map_to_python<'py>(
        &self,
        py: Python<'py>,
        map: RenamingMap<'_>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (key, image) in sorted_pairs(map) {
            dict.set_item(self.object_of(py, key)?, self.object_of(py, image)?)?;
        }
        Ok(dict)
    }
}

/// Return the pairs of `map` in the order of their keys' ids.
fn sorted_pairs(map: RenamingMap<'_>) -> Vec<(&Identifier, &Identifier)> {
    let mut pairs: Vec<_> = map.iter().collect();
    pairs.sort_unstable_by_key(|(key, _)| key.id());
    pairs
}

/// Return the text of `map`: `{x::7: y::8, ...}` in the order of the keys'
/// ids.
fn format_map(map: RenamingMap<'_>) -> String {
    let pairs: Vec<String> = sorted_pairs(map)
        .into_iter()
        .map(|(key, image)| {
            format!(
                "{}::{}: {}::{}",
                key.name_hint(),
                key.id(),
                image.name_hint(),
                image.id()
            )
        })
        .collect();
    format!("{{{}}}", pairs.join(", "))
}

/// Return `renaming` as an `AlphaRenaming`.
///
/// # Errors
///
/// Raises `TypeError` if `renaming` is not an `AlphaRenaming`.
pub(crate) fn read_renaming<'a, 'py>(
    renaming: &'a Bound<'py, PyAny>,
) -> PyResult<&'a Bound<'py, PyAlphaRenaming>> {
    match renaming.cast::<PyAlphaRenaming>() {
        Ok(renaming) => Ok(renaming),
        Err(_not_a_renaming) => Err(PyTypeError::new_err(format!(
            "renaming must be an AlphaRenaming, got {}.",
            renaming.get_type().name()?
        ))),
    }
}

// ---------------------------------------------------------------------------
// The class
// ---------------------------------------------------------------------------

/// The correspondence between identifiers on two sides of an alpha
/// comparison: a stack of binder frames over a free-identifier renaming,
/// backed by the Rust `AlphaRenaming`.
///
/// Frames shadow outer frames and the free renaming; an identifier bound on
/// one side only corresponds to nothing on the other; each frame and the
/// free renaming must be injective. A renaming is immutable: `extend`
/// returns a new one, and mutating one raises `FrozenMutationError`. It is
/// equal by its frames, in order, and its free renaming, and hashable.
#[pyclass(frozen, module = "fhy_core._rs", name = "AlphaRenaming")]
pub(crate) struct PyAlphaRenaming {
    value: RenamingValue,
}

impl PyAlphaRenaming {
    /// Return the renaming and its identifier objects.
    pub(crate) const fn value(&self) -> &RenamingValue {
        &self.value
    }
}

#[pymethods]
impl PyAlphaRenaming {
    /// Visit the identifier objects the renaming alone holds, for the cycle
    /// collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.value.objects.traverse(&visit)
    }

    /// Return the renaming with no binder frame and no free renaming, one
    /// shared object.
    #[classmethod]
    fn empty(cls: &Bound<'_, PyType>) -> PyResult<Py<Self>> {
        static EMPTY: PyOnceLock<Py<PyAlphaRenaming>> = PyOnceLock::new();
        let py = cls.py();
        EMPTY
            .get_or_try_init(py, || {
                Ok::<_, PyErr>(RenamingValue::default().into_python(py)?.unbind())
            })
            .map(|empty| empty.clone_ref(py))
    }

    /// Return the renaming with no binder frame that maps each key of
    /// `free_renaming` to its value.
    ///
    /// Raises `TypeError` for a mapping of anything but `Identifier`s, and
    /// `ValueError` if two keys map to one value.
    #[classmethod]
    fn with_free_renaming(
        cls: &Bound<'_, PyType>,
        free_renaming: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let _ = cls;
        let (map, objects) = read_identifier_map(free_renaming, "AlphaRenaming", "free_renaming")?;
        let renaming = AlphaRenaming::new(map).into_py_result()?;
        Ok(Self {
            value: RenamingValue::new(renaming, ObjectTable::default().with(objects)),
        })
    }

    /// Return this renaming with one more innermost binder frame, pairing
    /// each key of `bindings`, bound on this side, with its value, bound on
    /// the other. This renaming is unchanged.
    ///
    /// Raises `TypeError` for a mapping of anything but `Identifier`s, and
    /// `ValueError` if two keys map to one value.
    fn extend(&self, bindings: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (map, objects) = read_identifier_map(bindings, "AlphaRenaming", "bindings")?;
        let renaming = self.value.renaming.extended(map).into_py_result()?;
        Ok(Self {
            value: RenamingValue::new(renaming, self.value.objects.with(objects)),
        })
    }

    /// Return the identifier on the other side that `self_identifier`
    /// resolves to: its image in the innermost frame binding it, else its
    /// image in the free renaming, else `self_identifier` itself. An image
    /// is the object given for it.
    ///
    /// Raises `TypeError` if `self_identifier` is not an `Identifier`.
    fn resolve<'py>(&self, self_identifier: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let identifier = restore_identifier(self_identifier, "AlphaRenaming", "identifier")?;
        let image = self.value.renaming.resolve(&identifier);
        if image.id() == identifier.id() {
            Ok(self_identifier.clone())
        } else {
            self.value.object_of(self_identifier.py(), image)
        }
    }

    /// Return whether `self_identifier` on this side corresponds to
    /// `other_identifier` on the other, refusing a capture.
    ///
    /// Raises `TypeError` if either is not an `Identifier`.
    fn are_identifiers_alpha_equivalent(
        &self,
        self_identifier: &Bound<'_, PyAny>,
        other_identifier: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let left = restore_identifier(self_identifier, "AlphaRenaming", "identifier")?;
        let right = restore_identifier(other_identifier, "AlphaRenaming", "identifier")?;
        Ok(self.value.renaming.is_corresponding(&left, &right))
    }

    fn __eq__<'py>(&self, other: &Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        let py = other.py();
        match other.cast::<Self>() {
            Ok(other) => {
                pyo3::types::PyBool::new(py, self.value.renaming == other.get().value.renaming)
                    .to_owned()
                    .into_any()
            }
            Err(_not_a_renaming) => py.NotImplemented().into_bound(py),
        }
    }

    fn __hash__(&self) -> u64 {
        hash_value(&self.value.renaming)
    }

    fn __repr__(&self) -> String {
        let frames: Vec<String> = self.value.renaming.frames().map(format_map).collect();
        format!(
            "AlphaRenaming(frames=[{}], free_renaming={})",
            frames.join(", "),
            format_map(self.value.renaming.free_renaming())
        )
    }

    /// Pickle as a call of `_from_parts` with the free renaming and the
    /// frames, outermost first.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let value = &slf.get().value;
        let free = value.map_to_python(py, value.renaming.free_renaming())?;
        let frames = PyList::empty(py);
        for frame in value.renaming.frames() {
            frames.append(value.map_to_python(py, frame)?)?;
        }
        let constructor = slf.get_type().getattr(intern!(py, "_from_parts"))?;
        Ok((
            constructor,
            PyTuple::new(py, [free.into_any(), frames.into_any()])?,
        ))
    }

    /// Return the renaming with the free renaming `free_renaming` and one
    /// frame per mapping of `frames`, outermost first: what a pickle holds.
    #[classmethod]
    fn _from_parts(
        cls: &Bound<'_, PyType>,
        free_renaming: &Bound<'_, PyAny>,
        frames: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let mut renaming = Self::with_free_renaming(cls, free_renaming)?;
        for frame in frames.try_iter()? {
            renaming = renaming.extend(&frame?)?;
        }
        Ok(renaming)
    }

    /// Always true: renamings are immutable.
    #[getter]
    const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: renamings are always frozen.
    const fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: renamings are always frozen, and mutating one raises.
    const fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        refuse_attribute_assignment(slf, name)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        refuse_attribute_deletion(slf, name)
    }
}
