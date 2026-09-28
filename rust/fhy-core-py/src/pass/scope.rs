//! The state of one run the binding drives: a standalone `execute`, a
//! pipeline run or a validation, kept on a thread-local stack for the length
//! of the run.
//!
//! The adapters record into the innermost scope of their thread while a
//! hook runs: the diagnostics Python reported, the pass whose hook failed,
//! the nested run error a hook raised, and an exception that must not be
//! wrapped. The run's boundary takes the scope off the stack when the core
//! returns and uses it to convert what comes back to Python. A run started
//! inside a hook pushes a scope of its own, which it removes before the hook
//! returns.

use std::collections::{HashMap, VecDeque};
use std::mem;

use pyo3::prelude::*;

use fhy_core::diagnostic::Diagnostic;

use crate::dataclass::hash_value;
use crate::diagnostic::borrow_python_diagnostic;
use crate::scoped::{ScopedGuard, ScopedStack};

/// A nested run error that a hook raised, recorded when the adapter handed
/// its Rust error to the core, which nests it.
pub(super) struct NestedNote {
    /// The Python hook that raised it.
    pub(super) python_hook: &'static str,
    /// The exception the hook raised: the nested error's `__cause__`.
    pub(super) exception: Py<PyAny>,
    /// The inner error's rendered chain, as the outer failure's diagnostic
    /// ends with it.
    pub(super) chain: String,
}

/// What the adapters record during one run.
#[derive(Default)]
pub(super) struct RunScope {
    /// The diagnostic objects Python hooks reported, by the hash of their
    /// Rust diagnostic, in report order.
    reported: HashMap<u64, VecDeque<Py<PyAny>>>,
    /// The last nested run error a hook raised.
    nested: Option<NestedNote>,
    /// The pass whose hook failed last.
    failed_pass: Option<Py<PyAny>>,
    /// The first exception that is not an `Exception`, such as
    /// `KeyboardInterrupt`, which the run's boundary raises unchanged.
    interrupt: Option<PyErr>,
}

impl RunScope {
    /// Return the object Python reported for `diagnostic`, the first one
    /// not taken yet, removing it; or `None` for a diagnostic the core made.
    pub(super) fn take_reported<'py>(
        &mut self,
        py: Python<'py>,
        diagnostic: &Diagnostic,
    ) -> Option<Bound<'py, PyAny>> {
        let objects = self.reported.get_mut(&hash_value(diagnostic))?;
        let position = objects
            .iter()
            .position(|object| borrow_python_diagnostic(object.bind(py)) == Some(diagnostic))?;
        objects.remove(position).map(|object| object.into_bound(py))
    }

    /// Return the last nested run error a hook raised, removing it.
    pub(super) fn take_nested(&mut self) -> Option<NestedNote> {
        self.nested.take()
    }

    /// Return the pass whose hook failed last, removing it.
    pub(super) fn take_failed_pass(&mut self) -> Option<Py<PyAny>> {
        self.failed_pass.take()
    }

    /// Return the exception the run must raise unchanged, removing it.
    pub(super) fn take_interrupt(&mut self) -> Option<PyErr> {
        self.interrupt.take()
    }
}

thread_local! {
    /// The scopes of the runs in progress on this thread, innermost last.
    static SCOPES: ScopedStack<RunScope> = const { ScopedStack::new() };
}

/// Apply `update` to the innermost scope, if a run is in progress, and
/// return what it returns.
///
/// `update` must not call into Python: the stack is borrowed meanwhile. It
/// returns what it replaces, which is dropped after the borrow ends, since
/// dropping a Python object may run Python code.
fn with_innermost<R>(update: impl FnOnce(&mut RunScope) -> R) -> Option<R> {
    ScopedStack::with_top_mut(&SCOPES, |scope| scope.map(update))
}

/// Record `objects`, the diagnostic objects a hook reported, in the
/// innermost scope.
pub(super) fn record_reported(py: Python<'_>, objects: Vec<Py<PyAny>>) {
    let keyed: Vec<(u64, Py<PyAny>)> = objects
        .into_iter()
        .filter_map(|object| {
            let key = hash_value(borrow_python_diagnostic(object.bind(py))?);
            Some((key, object))
        })
        .collect();
    let mut keyed = Some(keyed);
    with_innermost(|scope| {
        for (key, object) in keyed.take().into_iter().flatten() {
            scope.reported.entry(key).or_default().push_back(object);
        }
    });
    drop(keyed);
}

/// Record `note`, the nested run error a hook raised, in the innermost
/// scope.
pub(super) fn record_nested(note: NestedNote) {
    let mut note = Some(note);
    let replaced = with_innermost(|scope| mem::replace(&mut scope.nested, note.take()));
    drop((replaced, note));
}

/// Record `pass` as the pass whose hook failed last.
pub(super) fn record_failed_pass(pass: Py<PyAny>) {
    let mut pass = Some(pass);
    let replaced = with_innermost(|scope| mem::replace(&mut scope.failed_pass, pass.take()));
    drop((replaced, pass));
}

/// Record `error`, an exception that is not an `Exception`, unless the
/// innermost scope has one already.
pub(super) fn record_interrupt(error: PyErr) {
    let mut error = Some(error);
    with_innermost(|scope| {
        if scope.interrupt.is_none() {
            scope.interrupt = error.take();
        }
    });
    drop(error);
}

/// Return whether the innermost run was interrupted by an exception that is
/// not an `Exception`, so no more Python hook may run.
pub(super) fn is_interrupted() -> bool {
    with_innermost(|scope| scope.interrupt.is_some()).unwrap_or(false)
}

/// The innermost scope of this thread while it is on the stack.
///
/// Dropping the guard without [`finish`](Self::finish), as unwinding does,
/// removes the scope too.
pub(super) struct ScopeGuard(ScopedGuard<RunScope>);

impl ScopeGuard {
    /// Push a new, empty scope.
    pub(super) fn enter() -> Self {
        Self(ScopedStack::push(&SCOPES, RunScope::default()))
    }

    /// Remove the scope from the stack and return it.
    pub(super) fn finish(self) -> RunScope {
        self.0.pop()
    }
}

#[cfg(test)]
mod scoped_stack_tests {
    use super::*;

    /// Test a panic inside a run's scope leaves no scope behind (R2-031).
    #[test]
    fn a_panic_inside_a_scope_leaves_the_stack_empty() {
        let unwound = std::panic::catch_unwind(|| {
            let _scope = ScopeGuard::enter();
            assert_eq!(ScopedStack::depth(&SCOPES), 1);
            panic!("inside a run");
        });

        let _panic = unwound.unwrap_err();
        assert_eq!(ScopedStack::depth(&SCOPES), 0);
        assert!(!is_interrupted());
    }
}
