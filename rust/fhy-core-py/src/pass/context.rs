//! The owned context of a Python hook: a frame on a thread-local stack that
//! `report`, `get_analysis` and `get_analysis_manager` find by the object
//! whose hook runs, and the `AnalysisManager` view of the run's analyses.
//!
//! A frame collects the diagnostics the hook reports, which the adapter
//! moves into the Rust `PassContext` when the hook returns, and holds a
//! detached handle to the run's analysis cache. It expires when the hook
//! returns, so an `AnalysisManager` kept past the hook raises instead of
//! reaching the run's cache. A hook the core runs without a context,
//! `did_change` or `get_preserved_analyses`, gets a frame that refuses both.

use std::cell::RefCell;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyType;

use fhy_core::pass::{AnalysisId, DetachedAnalyses};

use crate::dataclass::build_argument_type_error;
use crate::diagnostic::borrow_python_diagnostic;
use crate::identifier::restore_identifier;

use super::analysis::PyAnalysisBase;
use super::ir::PyIr;

/// The frame of one Python hook call.
pub(super) struct HookFrame {
    /// The address of the pass or validator whose hook runs.
    owner: usize,
    /// The Python hook's name.
    hook: &'static str,
    /// Whether the core gives the hook a context: `report` and
    /// `get_analysis` raise in a hook without one.
    has_context: bool,
    state: Mutex<FrameState>,
}

/// What a frame collects and holds, and whether it expired.
#[derive(Default)]
struct FrameState {
    reported: Vec<Py<PyAny>>,
    analyses: Option<DetachedAnalyses>,
    is_expired: bool,
}

impl HookFrame {
    fn lock(&self) -> MutexGuard<'_, FrameState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Return the `RuntimeError` for using `operation` in this frame's hook,
    /// which has no context.
    fn build_no_context_error(&self, operation: &str) -> PyErr {
        PyRuntimeError::new_err(format!(
            "{operation} is not available in {}, which runs without a pass context",
            self.hook
        ))
    }

    /// Return the frame's analyses, or raise if the frame expired.
    fn analyses(&self) -> PyResult<Option<DetachedAnalyses>> {
        let state = self.lock();
        if state.is_expired {
            return Err(PyRuntimeError::new_err(format!(
                "the analyses of {} expired when the hook returned",
                self.hook
            )));
        }
        Ok(state.analyses.clone())
    }
}

thread_local! {
    /// The frames of the hooks running on this thread, innermost last.
    static FRAMES: RefCell<Vec<Arc<HookFrame>>> = const { RefCell::new(Vec::new()) };
}

/// Return the innermost frame of a hook of `owner`, if one runs.
fn find_frame(owner: &Bound<'_, PyAny>) -> Option<Arc<HookFrame>> {
    let address = owner.as_ptr().addr();
    FRAMES.with_borrow(|frames| {
        frames
            .iter()
            .rev()
            .find(|frame| frame.owner == address)
            .cloned()
    })
}

/// A frame on this thread's stack for the length of one hook call.
pub(super) struct FrameGuard {
    frame: Arc<HookFrame>,
    is_finished: bool,
}

impl FrameGuard {
    /// Push the frame of `owner`'s hook `hook`, with `analyses` for a hook
    /// with a context, or `None` for one without.
    pub(super) fn push(
        owner: &Bound<'_, PyAny>,
        hook: &'static str,
        analyses: Option<DetachedAnalyses>,
        has_context: bool,
    ) -> Self {
        let frame = Arc::new(HookFrame {
            owner: owner.as_ptr().addr(),
            hook,
            has_context,
            state: Mutex::new(FrameState {
                analyses,
                ..FrameState::default()
            }),
        });
        FRAMES.with_borrow_mut(|frames| frames.push(Arc::clone(&frame)));
        Self {
            frame,
            is_finished: false,
        }
    }

    /// Pop the frame, expire it, and return the diagnostics its hook
    /// reported, in report order.
    pub(super) fn finish(mut self) -> Vec<Py<PyAny>> {
        self.is_finished = true;
        self.pop()
    }

    fn pop(&self) -> Vec<Py<PyAny>> {
        FRAMES.with_borrow_mut(|frames| {
            if let Some(position) = frames
                .iter()
                .rposition(|frame| Arc::ptr_eq(frame, &self.frame))
            {
                frames.remove(position);
            }
        });
        let mut state = self.frame.lock();
        state.is_expired = true;
        state.analyses = None;
        std::mem::take(&mut state.reported)
    }
}

impl Drop for FrameGuard {
    fn drop(&mut self) {
        if !self.is_finished {
            drop(self.pop());
        }
    }
}

/// Where a diagnostic an object reports goes.
pub(super) enum Recording {
    /// Into the frame of the hook that runs.
    InHook,
    /// Nowhere in Rust: no hook of the object runs.
    Standalone,
}

/// Record `diagnostic`, a `Diagnostic`, reported by `owner`: into the frame
/// of `owner`'s running hook, if any.
///
/// # Errors
///
/// Raises `TypeError` if `diagnostic` is not a `Diagnostic`, and
/// `RuntimeError` in a hook without a context.
pub(super) fn record_diagnostic(
    owner: &Bound<'_, PyAny>,
    diagnostic: &Bound<'_, PyAny>,
) -> PyResult<Recording> {
    if borrow_python_diagnostic(diagnostic).is_none() {
        return Err(build_argument_type_error(
            "report",
            "diagnostic",
            "a Diagnostic",
            diagnostic,
        )?);
    }
    let Some(frame) = find_frame(owner) else {
        return Ok(Recording::Standalone);
    };
    if !frame.has_context {
        return Err(frame.build_no_context_error("report"));
    }
    frame.lock().reported.push(diagnostic.clone().unbind());
    Ok(Recording::InHook)
}

/// Return whether analyses of `ir` may be cached: whether it is a `Frozen`
/// object that is frozen, as the Python cache required.
///
/// `Frozen` is a runtime protocol, so the check reads its members instead
/// of calling the protocol's `isinstance`.
fn is_cacheable(ir: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = ir.py();
    let ir_type = ir.get_type();
    if !ir_type.hasattr(intern!(py, "freeze"))? || !ir_type.hasattr(intern!(py, "assert_frozen"))? {
        return Ok(false);
    }
    match ir.getattr(intern!(py, "is_frozen")) {
        Ok(is_frozen) => is_frozen.is_truthy(),
        Err(_no_frozen_state) => Ok(false),
    }
}

/// Return `analysis_type` as an `Analysis` subclass.
///
/// # Errors
///
/// Raises `TypeError` naming `owner` for anything else.
fn read_analysis_type<'a, 'py>(
    analysis_type: &'a Bound<'py, PyAny>,
    owner: &str,
) -> PyResult<&'a Bound<'py, PyType>> {
    match analysis_type.cast::<PyType>() {
        Ok(class) if class.is_subclass_of::<PyAnalysisBase>()? => Ok(class),
        _ => Err(build_argument_type_error(
            owner,
            "analysis_type",
            "an Analysis subclass",
            analysis_type,
        )?),
    }
}

/// Return the result of the analysis `analysis_type` for `ir`, computed
/// afresh: a new instance's `run(ir)`.
fn compute<'py>(
    analysis_type: &Bound<'py, PyType>,
    ir: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    analysis_type
        .call0()?
        .call_method1(intern!(ir.py(), "run"), (ir,))
}

/// Return the result of `analysis_type` for `ir` through `analyses`: cached
/// per node under the analysis's identifier when `ir` may be cached.
///
/// The cache holds a cell per analysis and node, filled after the
/// computation, so an analysis that raises caches nothing and runs again on
/// the next request.
fn compute_through<'py>(
    analyses: &DetachedAnalyses,
    analysis_type: &Bound<'py, PyType>,
    ir: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ir.py();
    if !is_cacheable(ir)? {
        return compute(analysis_type, ir);
    }
    let name = analysis_type.call_method0(intern!(py, "get_analysis_name"))?;
    let id = AnalysisId::of_identifier(&restore_identifier(&name, "get_analysis_name", "result")?);
    let cell = analyses
        .analysis_by_id(&PyIr::new(ir), &id, |_| OnceLock::<Py<PyAny>>::new())
        .map_err(|expired| PyRuntimeError::new_err(expired.to_string()))?;
    if let Some(result) = cell.get() {
        return Ok(result.bind(py).clone());
    }
    let result = compute(analysis_type, ir)?;
    let stored = cell.get_or_init(|| result.clone().unbind());
    Ok(stored.bind(py).clone())
}

/// Return the result of `analysis_type` for `ir` as `owner`'s
/// `get_analysis` does: through the run's analyses in a hook of `owner`,
/// and computed afresh otherwise.
///
/// # Errors
///
/// Raises `TypeError` for an `analysis_type` that is not an `Analysis`
/// subclass, `RuntimeError` in a hook without a context, and whatever the
/// analysis raises.
pub(super) fn get_analysis<'py>(
    owner: &Bound<'py, PyAny>,
    analysis_type: &Bound<'py, PyAny>,
    ir: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let analysis_type = read_analysis_type(analysis_type, "get_analysis")?;
    let Some(frame) = find_frame(owner) else {
        return compute(analysis_type, ir);
    };
    if !frame.has_context {
        return Err(frame.build_no_context_error("get_analysis"));
    }
    match frame.analyses()? {
        Some(analyses) => compute_through(&analyses, analysis_type, ir),
        None => compute(analysis_type, ir),
    }
}

/// Return the `AnalysisManager` view of the run's analyses in a hook of
/// `owner`, or `None` outside one.
///
/// # Errors
///
/// Raises `RuntimeError` in a hook without a context.
pub(super) fn get_analysis_manager<'py>(
    owner: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAnalysisManager>>> {
    let Some(frame) = find_frame(owner) else {
        return Ok(None);
    };
    if !frame.has_context {
        return Err(frame.build_no_context_error("get_analysis_manager"));
    }
    Bound::new(owner.py(), PyAnalysisManager { frame }).map(Some)
}

/// The analyses of one pass run, as one hook sees them: a view that
/// computes an analysis through the run's cache.
///
/// Only a hook's `get_analysis_manager()` returns one; there is no
/// constructor. The view expires when its hook returns, and then raises
/// `RuntimeError`.
#[pyclass(frozen, module = "fhy_core._rs", name = "AnalysisManager")]
pub(crate) struct PyAnalysisManager {
    frame: Arc<HookFrame>,
}

#[pymethods]
impl PyAnalysisManager {
    /// Return the result of the analysis `analysis_type` for `ir`, cached
    /// per node for the run when `ir` is a frozen `Frozen` object.
    ///
    /// Raises `TypeError` for an `analysis_type` that is not an `Analysis`
    /// subclass, and `RuntimeError` after the hook returned.
    fn get<'py>(
        &self,
        analysis_type: &Bound<'py, PyAny>,
        ir: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let analysis_type = read_analysis_type(analysis_type, "AnalysisManager.get")?;
        match self.frame.analyses()? {
            Some(analyses) => compute_through(&analyses, analysis_type, ir),
            None => compute(analysis_type, ir),
        }
    }

    /// Return the class itself, so `AnalysisManager[IR]` works in
    /// annotations evaluated at run time.
    #[classmethod]
    fn __class_getitem__<'py>(
        cls: &Bound<'py, PyType>,
        item: &Bound<'py, PyAny>,
    ) -> Bound<'py, PyType> {
        let _ = item;
        cls.clone()
    }

    fn __reduce__(_slf: &Bound<'_, Self>) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "an AnalysisManager lives for one hook and cannot be pickled",
        ))
    }
}
