//! An owned handle to a pass run's analysis cache, lent out for the length
//! of one callback.

use std::error::Error;
use std::fmt;
use std::mem;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use super::analysis::AnalysisCache;
use super::preserved::AnalysisId;
use crate::tree::NodeHandle;

/// Where the requests through a [`DetachedAnalyses`] handle go.
#[derive(Debug)]
enum Slot {
    /// The run's cache, moved out of its context for the callback.
    Cached(AnalysisCache),
    /// The run has no cache, so every request computes afresh.
    Uncached,
    /// The callback returned, so every request fails.
    Expired,
}

/// An owned handle to the analyses of a pass run, valid while the callback
/// of [`PassContext::with_detached_analyses`](super::PassContext::with_detached_analyses)
/// runs.
///
/// A [`PassContext`](super::PassContext) is borrowed for each hook, so code
/// that must hold on to the run's analyses by value, such as an object of
/// another language's runtime handed to a hook written in that language,
/// takes this handle instead. It is `Send`, `Sync` and `'static`, and its
/// clones share one cache. Under a [`PassManager`](super::PassManager) it
/// serves the pipeline's cache, which moves into the handle for the
/// callback and back into the context when the callback returns or
/// unwinds; outside one, every request computes afresh. After that, the
/// handle and every clone of it are expired: a request fails with
/// [`DetachedAnalysesExpired`], and the handle holds no cached result or
/// node.
///
/// # Examples
///
/// ```
/// use std::sync::{Arc, Mutex};
///
/// use fhy_core::identifier::Identifier;
/// use fhy_core::pass::{AnalysisId, CompilerPass, DetachedAnalyses, ExecutePass, PassContext};
/// use fhy_core::foreign::BoxError;
/// use fhy_core::tree::{NodeHandle, NodeIdentity};
///
/// #[derive(Clone)]
/// struct Value(Arc<i64>);
///
/// impl NodeHandle for Value {
///     fn identity(&self) -> NodeIdentity {
///         NodeIdentity::of_arc(&self.0)
///     }
/// }
///
/// struct KeepAnalyses {
///     id: AnalysisId,
///     kept: Arc<Mutex<Option<DetachedAnalyses>>>,
/// }
///
/// impl CompilerPass<Value> for KeepAnalyses {
///     fn run(&mut self, ir: &Value, cx: &mut PassContext<'_>) -> Result<Value, BoxError> {
///         let doubled = cx.with_detached_analyses(|analyses| {
///             *self.kept.lock().unwrap() = Some(analyses.clone());
///             analyses.analysis_by_id(ir, &self.id, |ir| *ir.0 * 2)
///         })?;
///         Ok(Value(Arc::new(*doubled)))
///     }
///
///     fn did_change(&mut self, input: &Value, output: &Value) -> Result<bool, BoxError> {
///         Ok(input.0 != output.0)
///     }
/// }
///
/// let kept = Arc::new(Mutex::new(None));
/// let mut pass = KeepAnalyses {
///     id: AnalysisId::of_identifier(&Identifier::new("double")),
///     kept: Arc::clone(&kept),
/// };
/// let ir = Value(Arc::new(4));
///
/// assert_eq!(*pass.execute(&ir)?.output().0, 8);
/// let analyses = kept.lock().unwrap().take().expect("the callback ran");
/// assert!(analyses.is_expired());
/// assert!(analyses.analysis_by_id(&ir, &pass.id, |ir| *ir.0).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct DetachedAnalyses {
    slot: Arc<Mutex<Slot>>,
}

impl DetachedAnalyses {
    /// Return the result cached for `ir` under `id`, computing it with
    /// `compute` and caching it on a miss, as
    /// [`PassContext::analysis_by_id`](super::PassContext::analysis_by_id)
    /// does.
    ///
    /// `compute` runs without a lock held, so it may request other results
    /// through this handle, from any thread. When such a request caches a
    /// result for `ir` under `id` while `compute` runs, that result is the
    /// one returned, and the one `compute` returns is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`DetachedAnalysesExpired`], without calling `compute`, when
    /// the callback the handle was lent to has returned, and also when it
    /// returns while `compute` runs.
    pub fn analysis_by_id<T, V>(
        &self,
        ir: &T,
        id: &AnalysisId,
        compute: impl FnOnce(&T) -> V,
    ) -> Result<Arc<V>, DetachedAnalysesExpired>
    where
        T: NodeHandle,
        V: Send + Sync + 'static,
    {
        match &*self.lock() {
            Slot::Cached(cache) => {
                if let Some(result) = cache.cached(ir, id) {
                    return Ok(result);
                }
            }
            Slot::Uncached => {}
            Slot::Expired => return Err(DetachedAnalysesExpired),
        }
        let result = Arc::new(compute(ir));
        match &mut *self.lock() {
            Slot::Cached(cache) => Ok(cache.insert(ir, id, result)),
            Slot::Uncached => Ok(result),
            Slot::Expired => Err(DetachedAnalysesExpired),
        }
    }

    /// Return whether the callback the handle was lent to has returned.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        matches!(*self.lock(), Slot::Expired)
    }

    /// Lock the slot. A panic while it was locked left the cache with every
    /// result either cached or not, so a poisoned lock is used as it is.
    fn lock(&self) -> MutexGuard<'_, Slot> {
        self.slot.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Error for a request through a [`DetachedAnalyses`] handle whose callback
/// has returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DetachedAnalysesExpired;

impl fmt::Display for DetachedAnalysesExpired {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the detached analyses expired when their callback returned")
    }
}

impl Error for DetachedAnalysesExpired {}

/// The cache of a pass context moved into a [`DetachedAnalyses`] handle,
/// which moves back when this is dropped, also while unwinding.
#[derive(Debug)]
pub(super) struct Detachment<'c> {
    cache: Option<&'c mut AnalysisCache>,
    handle: DetachedAnalyses,
}

impl<'c> Detachment<'c> {
    /// Move the contents of `cache`, if there is one, into a new handle.
    pub(super) fn new(mut cache: Option<&'c mut AnalysisCache>) -> Self {
        let slot = cache
            .as_deref_mut()
            .map_or(Slot::Uncached, |cache| Slot::Cached(mem::take(cache)));
        Self {
            cache,
            handle: DetachedAnalyses {
                slot: Arc::new(Mutex::new(slot)),
            },
        }
    }

    /// Return the handle the cache moved into.
    pub(super) const fn handle(&self) -> &DetachedAnalyses {
        &self.handle
    }
}

impl Drop for Detachment<'_> {
    fn drop(&mut self) {
        let slot = mem::replace(&mut *self.handle.lock(), Slot::Expired);
        if let (Some(cache), Slot::Cached(detached)) = (self.cache.as_deref_mut(), slot) {
            *cache = detached;
        }
    }
}
