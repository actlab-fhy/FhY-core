//! What the stories of the kit share: an embedded interpreter, and stand-ins
//! for the few modules of `fhy_core` the kit imports.
//!
//! `fhy_core` itself is not importable in the embedded interpreter, so the
//! stories that raise or read one of its classes install small stand-ins in
//! `sys.modules`, once, and keep them: an [`ImportedAttr`](super::python::ImportedAttr)
//! keeps the class it first imported for the life of the process, so the
//! stand-ins must not change between stories. The Python suite covers the
//! behavior against the real classes.

use std::ffi::CString;
use std::sync::{Mutex, PoisonError};

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// The stand-in for `fhy_core.serialization`.
const SERIALIZATION: &str = r#"
class SerializationError(Exception):
    pass


class DeserializationValueError(SerializationError):
    pass


class DeserializationDictStructureError(SerializationError):
    def __init__(self, cls, expected, data):
        super().__init__(cls, expected, data)
        self.cls = cls
        self.expected = expected
        self.data = data


class MalformedPayloadError(SerializationError):
    pass


class KeywordError(Exception):
    def __init__(self, *args, **keywords):
        super().__init__(*args)
        self.keywords = keywords


def is_serialized_dict(value):
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


class Foreign:
    def __init__(self, type_id, data, error=None):
        self.type_id = type_id
        self.data = data
        self.error = error


def _foreign_payload(part, *, family):
    if part.error is not None:
        raise part.error
    return (part.type_id + (":family" if family else ":whole"), part.data)
"#;

/// The stand-in for `fhy_core.traits.frozen`.
const FROZEN: &str = "class FrozenMutationError(Exception):\n    pass\n";

/// The stand-in for `fhy_core.term.derived_equivalence`.
const DERIVED_EQUIVALENCE: &str = "class EquivalenceDerivationError(Exception):\n    pass\n";

/// Install the module `name`, run from `source`, in `sys.modules` and as an
/// attribute of each of its parent packages, which are created when missing.
fn install(py: Python<'_>, name: &str, source: &str) -> PyResult<()> {
    let modules = py.import("sys")?.getattr("modules")?;
    let code = CString::new(source).expect("no nul");
    let file = CString::new(format!("{name}.py")).expect("no nul");
    let module_name = CString::new(name).expect("no nul");
    let module = PyModule::from_code(py, &code, &file, &module_name)?;
    modules.set_item(name, &module)?;
    let mut child = module.into_any();
    let mut path = name;
    while let Some((parent, leaf)) = path.rsplit_once('.') {
        let package = if let Ok(package) = modules.get_item(parent) {
            package
        } else {
            let package = PyModule::new(py, parent)?.into_any();
            modules.set_item(parent, &package)?;
            package
        };
        package.setattr(leaf, &child)?;
        child = package;
        path = parent;
    }
    Ok(())
}

/// Run `body` with the embedded interpreter, with the stand-ins installed.
pub(crate) fn with_framework<R>(body: impl FnOnce(Python<'_>) -> R) -> R {
    // Held while no thread is attached: a thread waiting for this lock with
    // the interpreter attached would keep the installer from running Python.
    static INSTALLED: Mutex<bool> = Mutex::new(false);
    Python::initialize();
    {
        let mut installed = INSTALLED.lock().unwrap_or_else(PoisonError::into_inner);
        if !*installed {
            Python::attach(|py| {
                install(py, "fhy_core.serialization", SERIALIZATION).expect("serialization");
                install(py, "fhy_core.traits.frozen", FROZEN).expect("frozen");
                install(py, "fhy_core.term.derived_equivalence", DERIVED_EQUIVALENCE)
                    .expect("derived_equivalence");
            });
            *installed = true;
        }
    }
    Python::attach(body)
}

/// Return the value of the Python expression `source`.
pub(crate) fn evaluate<'py>(py: Python<'py>, source: &str) -> Bound<'py, PyAny> {
    let code = CString::new(source).expect("no nul");
    py.eval(&code, None, None)
        .unwrap_or_else(|error| panic!("evaluating {source:?} failed: {error}"))
}

/// Run the Python `source` in a new namespace, and return the namespace.
pub(crate) fn define<'py>(py: Python<'py>, source: &str) -> Bound<'py, pyo3::types::PyDict> {
    let namespace = pyo3::types::PyDict::new(py);
    let code = CString::new(source).expect("no nul");
    py.run(&code, Some(&namespace), None)
        .unwrap_or_else(|error| panic!("running {source:?} failed: {error}"));
    namespace
}

/// Return the entry `name` of `namespace`.
pub(crate) fn entry<'py>(
    namespace: &Bound<'py, pyo3::types::PyDict>,
    name: &str,
) -> Bound<'py, PyAny> {
    namespace
        .get_item(name)
        .expect("a readable namespace")
        .unwrap_or_else(|| panic!("the namespace has no {name:?}"))
}
