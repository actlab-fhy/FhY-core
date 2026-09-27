//! The embedded interpreter, the SymPy backend and the expression builders
//! the SymPy backend's stories share.
//!
//! The stories need Python with SymPy (N-S12-1 of
//! `docs/design/python-switch.md`, resolved as required): without it,
//! [`backend`] fails every story that uses it with the setup recipe.

use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};

use fhy_core::expression::{Expression, LiteralValue};
use fhy_core::identifier::Identifier;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::SympySimplifier;

/// What a story needs to run, when SymPy cannot be loaded.
const RECIPE: &str = "the SymPy backend's stories need Python with SymPy. Build and run \
     them with PYO3_PYTHON naming a Python that has a shared libpython and the sympy \
     package, PYTHONPATH naming that Python's site-packages (an embedded interpreter \
     does not read a virtualenv's pyvenv.cfg), and LD_LIBRARY_PATH naming the \
     directory of its libpython when the loader does not find it; CONTRIBUTING's \
     \"Rust test layout\" records the recipe";

/// The backend the stories share, in an interpreter this test binary
/// embeds. The interpreter runs without signal handlers and is never
/// finalized.
static BACKEND: LazyLock<SympySimplifier> = LazyLock::new(|| {
    Python::initialize();
    let backend = SympySimplifier::new();
    if let Err(error) = Python::attach(|py| backend.load(py)) {
        let cause = std::error::Error::source(&error)
            .map_or_else(String::new, |source| format!(": {source}"));
        panic!("{RECIPE}. Loading SymPy failed: {error}{cause}");
    }
    backend
});

/// Mint an identifier named `name` and return it with a reference to it.
#[must_use]
pub(crate) fn build_identifier(name: &str) -> (Identifier, Expression) {
    let identifier = Identifier::new(name);
    let reference = Expression::from(identifier.clone());
    (identifier, reference)
}

/// Return a literal expression holding `value`.
#[must_use]
pub(crate) fn build_literal(value: impl Into<LiteralValue>) -> Expression {
    Expression::from(value.into())
}

/// Serializes the stories that replace a SymPy function for their own
/// thread, so each restores the function it replaced.
static PATCHES: Mutex<()> = Mutex::new(());

/// Return the shared backend, loaded.
///
/// # Panics
///
/// Panics with the setup recipe when SymPy cannot be loaded.
pub(crate) fn backend() -> &'static SympySimplifier {
    &BACKEND
}

/// Run `body` attached to the stories' interpreter, with SymPy loaded.
pub(crate) fn attached<R>(body: impl for<'py> FnOnce(Python<'py>) -> R) -> R {
    backend();
    Python::attach(body)
}

/// Return the value of the Python expression `source`, evaluated with
/// `sympy`, `S` and the backend's prelude `prelude` in scope.
///
/// # Panics
///
/// Panics if `source` fails to evaluate.
pub(crate) fn evaluate<'py>(py: Python<'py>, source: &str) -> Bound<'py, PyAny> {
    let scope = scope(py);
    let code = std::ffi::CString::new(source).expect("no nul");
    py.eval(&code, Some(&scope), None)
        .unwrap_or_else(|error| panic!("evaluating {source:?} failed: {error}"))
}

/// Run the Python statements `source` with the scope of [`evaluate`].
///
/// # Panics
///
/// Panics if `source` fails.
pub(crate) fn run(py: Python<'_>, source: &str) {
    let scope = scope(py);
    let code = std::ffi::CString::new(source).expect("no nul");
    py.run(&code, Some(&scope), None)
        .unwrap_or_else(|error| panic!("running {source:?} failed: {error}"));
}

/// Return the scope the stories' Python snippets run in.
fn scope(py: Python<'_>) -> Bound<'_, PyDict> {
    let scope = PyDict::new(py);
    let sympy = py.import("sympy").expect("sympy imports");
    scope
        .set_item("S", sympy.getattr("S").expect("sympy.S"))
        .expect("set");
    scope.set_item("sympy", sympy).expect("set");
    let prelude = py
        .import("sys")
        .and_then(|sys| sys.getattr("modules"))
        .and_then(|modules| modules.get_item("_fhy_core_sympy"))
        .expect("the prelude is published");
    scope.set_item("prelude", prelude).expect("set");
    scope
}

/// Return `sympy.srepr(object)`, the text that pins a SymPy object's
/// structure.
///
/// # Panics
///
/// Panics if `srepr` fails.
pub(crate) fn srepr(object: &Bound<'_, PyAny>) -> String {
    let py = object.py();
    py.import("sympy")
        .and_then(|sympy| sympy.getattr("srepr"))
        .and_then(|srepr| srepr.call1((object,)))
        .and_then(|text| text.extract())
        .expect("srepr")
}

/// Run `body` with `sympy.<name>` replaced, on this thread only, by the
/// Python callable `replacement` evaluates to; other threads keep reaching
/// the original. The original is restored afterwards.
///
/// `replacement` is evaluated with `original`, the function it replaces,
/// in scope.
pub(crate) fn with_patched_sympy<R>(name: &str, replacement: &str, body: impl FnOnce() -> R) -> R {
    let _serial: MutexGuard<'_, ()> = PATCHES.lock().unwrap_or_else(PoisonError::into_inner);
    attached(|py| {
        run(
            py,
            &format!(
                "import threading\n\
                 original = sympy.{name}\n\
                 replacement = (lambda original: {replacement})(original)\n\
                 owner = threading.get_ident()\n\
                 def patched(*args, **kwargs):\n\
                 \x20   if threading.get_ident() == owner:\n\
                 \x20       return replacement(*args, **kwargs)\n\
                 \x20   return original(*args, **kwargs)\n\
                 patched.fhy_original = original\n\
                 sympy.{name} = patched\n"
            ),
        );
    });
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
    attached(|py| run(py, &format!("sympy.{name} = sympy.{name}.fhy_original\n")));
    match result {
        Ok(result) => result,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}
