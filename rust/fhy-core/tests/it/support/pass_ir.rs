//! A toy IR, analyses over it, and configurable passes for the pass
//! infrastructure tests.
//!
//! Every node counts the analysis runs made over it in counters it shares
//! with the nodes derived from it, so a test observes how often an analysis
//! ran across a whole pipeline without any process-global state.

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fhy_core::pass::{Analysis, CompilerPass, PassContext, PassFailure};
use fhy_core::tree::{NodeHandle, NodeIdentity};

/// Analysis run counts shared by a node and every node derived from it.
#[derive(Debug, Default)]
struct AnalysisRunCounters {
    double: AtomicUsize,
    parity: AtomicUsize,
}

/// One immutable node of the toy IR.
#[derive(Debug)]
struct BoxNode {
    value: i64,
    counters: Arc<AnalysisRunCounters>,
}

/// A handle to a toy IR node holding one integer.
#[derive(Debug, Clone)]
pub(crate) struct BoxIr(Arc<BoxNode>);

impl BoxIr {
    /// Build a node holding `value` with fresh analysis run counters.
    #[must_use]
    pub(crate) fn new(value: i64) -> Self {
        Self(Arc::new(BoxNode {
            value,
            counters: Arc::default(),
        }))
    }

    /// Build a new node holding `value` that shares this node's counters.
    #[must_use]
    pub(crate) fn derive(&self, value: i64) -> Self {
        Self(Arc::new(BoxNode {
            value,
            counters: Arc::clone(&self.0.counters),
        }))
    }

    /// Return the node's value.
    #[must_use]
    pub(crate) fn value(&self) -> i64 {
        self.0.value
    }

    /// Return how many times [`DoubleAnalysis`] ran over this node's family.
    #[must_use]
    pub(crate) fn double_runs(&self) -> usize {
        self.0.counters.double.load(Ordering::SeqCst)
    }

    /// Return how many times [`ParityAnalysis`] ran over this node's family.
    #[must_use]
    pub(crate) fn parity_runs(&self) -> usize {
        self.0.counters.parity.load(Ordering::SeqCst)
    }

    /// Return how many handles to this node are alive.
    #[must_use]
    pub(crate) fn handle_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }

    /// Return whether both handles point to the same node.
    #[must_use]
    pub(crate) fn is_same_node(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl NodeHandle for BoxIr {
    fn identity(&self) -> NodeIdentity {
        NodeIdentity::of_arc(&self.0)
    }
}

/// Doubles a node's value, counting its runs.
#[derive(Debug, Default)]
pub(crate) struct DoubleAnalysis;

impl Analysis for DoubleAnalysis {
    type Ir = BoxIr;
    type Output = i64;

    fn run(&self, ir: &BoxIr) -> i64 {
        ir.0.counters.double.fetch_add(1, Ordering::SeqCst);
        ir.value() * 2
    }
}

/// Computes a node's value modulo two, counting its runs.
#[derive(Debug, Default)]
pub(crate) struct ParityAnalysis;

impl Analysis for ParityAnalysis {
    type Ir = BoxIr;
    type Output = i64;

    fn run(&self, ir: &BoxIr) -> i64 {
        ir.0.counters.parity.fetch_add(1, Ordering::SeqCst);
        ir.value().rem_euclid(2)
    }
}

/// The hook of a [`ClosurePass`].
type RunHook<'a> =
    Box<dyn FnMut(&BoxIr, &mut PassContext<'_>) -> Result<BoxIr, PassFailure> + Send + 'a>;

/// A pass over the toy IR named explicitly, whose run is a closure and which
/// reports a change exactly when the output holds a different value.
pub(crate) struct ClosurePass<'a> {
    name: String,
    run: RunHook<'a>,
}

impl<'a> ClosurePass<'a> {
    /// Build the pass `name` whose run is `run`.
    pub(crate) fn new(
        name: &str,
        run: impl FnMut(&BoxIr, &mut PassContext<'_>) -> Result<BoxIr, PassFailure> + Send + 'a,
    ) -> Self {
        Self {
            name: name.to_owned(),
            run: Box::new(run),
        }
    }
}

impl fmt::Debug for ClosurePass<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClosurePass")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl CompilerPass<BoxIr> for ClosurePass<'_> {
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
    }

    fn run(&mut self, ir: &BoxIr, cx: &mut PassContext<'_>) -> Result<BoxIr, PassFailure> {
        (self.run)(ir, cx)
    }

    fn did_change(&mut self, input: &BoxIr, output: &BoxIr) -> Result<bool, PassFailure> {
        Ok(input.value() != output.value())
    }
}
