//! Reusable analyses and the identity-keyed analysis cache a pass manager
//! keeps for one run.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use super::preserved::{AnalysisId, PreservedAnalyses};
use crate::tree::{BuildIdentityHasher, NodeHandle, NodeIdentity};

/// A reusable computation over IR of type [`Ir`](Self::Ir) whose result a
/// pass manager can cache.
///
/// Passes obtain results through
/// [`PassContext::analysis`](super::PassContext::analysis), which builds the
/// analysis with [`Default`] and caches its result per node while the pass
/// runs under a [`PassManager`](super::PassManager). A type is an analysis
/// of one IR type; a computation over several IR types is one analysis type
/// per IR, such as a generic type. An analysis cannot fail; one whose
/// computation can fail makes the failure part of its
/// [`Output`](Self::Output).
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
///
/// use fhy_core::pass::{Analysis, CompilerPass, ExecutePass, PassContext, PassFailure};
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
/// #[derive(Default)]
/// struct IsEven;
///
/// impl Analysis for IsEven {
///     type Ir = Value;
///     type Output = bool;
///
///     fn run(&self, ir: &Value) -> bool {
///         *ir.0 % 2 == 0
///     }
/// }
///
/// struct HalveEven;
///
/// impl CompilerPass<Value> for HalveEven {
///     fn run(&mut self, ir: &Value, cx: &mut PassContext<'_>) -> Result<Value, PassFailure> {
///         Ok(if *cx.analysis::<IsEven>(ir) { Value(Arc::new(*ir.0 / 2)) } else { ir.clone() })
///     }
///
///     fn did_change(&mut self, input: &Value, output: &Value) -> Result<bool, PassFailure> {
///         Ok(input.0 != output.0)
///     }
/// }
///
/// assert_eq!(*HalveEven.execute(&Value(Arc::new(8)))?.output().0, 4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait Analysis: 'static {
    /// The IR the analysis runs over.
    type Ir;

    /// The result the analysis computes.
    type Output: Send + Sync + 'static;

    /// Compute the analysis result for `ir`.
    fn run(&self, ir: &Self::Ir) -> Self::Output;
}

/// The key of one cached node: its identity and the handle type that
/// reported it, so handles of two types that share a node never share
/// results.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct CacheKey {
    identity: NodeIdentity,
    handle_type: TypeId,
}

/// A cached analysis result, type-erased.
type CachedResult = Arc<dyn Any + Send + Sync>;

/// Return the key `ir` is cached under.
fn find_cache_key<T: NodeHandle>(ir: &T) -> CacheKey {
    CacheKey {
        identity: ir.identity(),
        handle_type: TypeId::of::<T>(),
    }
}

/// The results cached for one node, by analysis.
type Results = HashMap<AnalysisId, CachedResult, BuildIdentityHasher>;

/// The cached results for one node, together with a handle that keeps the
/// node, and so its identity, alive while the bucket exists.
#[derive(Debug)]
struct Bucket {
    /// Never read: holding it is what keeps the node alive.
    _handle: Box<dyn Any + Send + Sync>,
    results: Results,
}

impl Bucket {
    /// Create the empty bucket for `ir`.
    fn new<T: NodeHandle>(ir: &T) -> Self {
        Self {
            _handle: Box::new(ir.clone()),
            results: Results::default(),
        }
    }
}

/// Analysis results cached per node identity for one pass-manager run.
///
/// Each cached node's bucket holds a clone of the node's handle, so no other
/// node can take over its identity while the cache holds results for it.
/// Nothing is removed while the run lasts: a node identity is an address,
/// so releasing a node mid-run would let a new node reuse the address and
/// inherit the results cached for the old one. Dropping the cache at the
/// end of the run releases every handle.
#[derive(Debug, Default)]
pub(super) struct AnalysisCache {
    buckets: HashMap<CacheKey, Bucket, BuildIdentityHasher>,
}

impl AnalysisCache {
    /// Create an empty cache.
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Return the result of the analysis `A` for `ir`, computing and caching
    /// it on a miss.
    pub(super) fn get<A>(&mut self, ir: &A::Ir) -> Arc<A::Output>
    where
        A: Analysis + Default,
        A::Ir: NodeHandle,
    {
        self.get_or_insert_with(ir, AnalysisId::of::<A>(), || A::default().run(ir))
    }

    /// Return the result cached for `ir` under `id`, computing it with
    /// `compute` and caching it on a miss.
    fn get_or_insert_with<T, V>(
        &mut self,
        ir: &T,
        id: AnalysisId,
        compute: impl FnOnce() -> V,
    ) -> Arc<V>
    where
        T: NodeHandle,
        V: Send + Sync + 'static,
    {
        let bucket = self
            .buckets
            .entry(find_cache_key(ir))
            .or_insert_with(|| Bucket::new(ir));
        // The key's handle type and `id` fix the result type, so a cached
        // result always downcasts; one that did not would be recomputed and
        // replaced rather than served.
        if let Some(cached) = bucket.results.get(&id) {
            if let Ok(result) = Arc::clone(cached).downcast::<V>() {
                return result;
            }
        }
        let result = Arc::new(compute());
        let erased: CachedResult = Arc::clone(&result) as CachedResult;
        bucket.results.insert(id, erased);
        result
    }

    /// Carry the results `preserved` preserves from `from` over to its
    /// replacement `to`, merging them into what `to` has.
    ///
    /// Nothing is removed. When both are the same node nothing changes: a
    /// node cannot change, so its own results stay valid whatever a pass
    /// reports. Otherwise `to` gains each preserved result of `from` for an
    /// analysis it has no result of its own for, and `from` keeps its
    /// results.
    pub(super) fn transfer<T: NodeHandle>(
        &mut self,
        from: &T,
        to: &T,
        preserved: &PreservedAnalyses,
    ) {
        let from_key = find_cache_key(from);
        let to_key = find_cache_key(to);
        if from_key == to_key {
            return;
        }
        let Some(from_bucket) = self.buckets.get(&from_key) else {
            return;
        };
        let carried: Vec<(AnalysisId, CachedResult)> = from_bucket
            .results
            .iter()
            .filter(|(id, _)| preserved.is_id_preserved(**id))
            .map(|(id, result)| (*id, Arc::clone(result)))
            .collect();
        if carried.is_empty() {
            return;
        }
        let to_bucket = self
            .buckets
            .entry(to_key)
            .or_insert_with(|| Bucket::new(to));
        for (id, result) in carried {
            to_bucket.results.entry(id).or_insert(result);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A node that counts the analysis runs made over it.
    #[derive(Debug, Default)]
    struct TestNode {
        value: i64,
        double_runs: AtomicUsize,
        parity_runs: AtomicUsize,
        label_runs: AtomicUsize,
    }

    /// A handle to a [`TestNode`].
    #[derive(Debug, Clone)]
    struct TestIr(Arc<TestNode>);

    impl TestIr {
        fn new(value: i64) -> Self {
            Self(Arc::new(TestNode {
                value,
                ..TestNode::default()
            }))
        }

        fn double_runs(&self) -> usize {
            self.0.double_runs.load(Ordering::SeqCst)
        }

        fn parity_runs(&self) -> usize {
            self.0.parity_runs.load(Ordering::SeqCst)
        }

        fn label_runs(&self) -> usize {
            self.0.label_runs.load(Ordering::SeqCst)
        }

        fn handle_count(&self) -> usize {
            Arc::strong_count(&self.0)
        }
    }

    impl NodeHandle for TestIr {
        fn identity(&self) -> NodeIdentity {
            NodeIdentity::of_arc(&self.0)
        }
    }

    /// A second handle type over the same nodes as [`TestIr`].
    #[derive(Debug, Clone)]
    struct AliasIr(Arc<TestNode>);

    impl NodeHandle for AliasIr {
        fn identity(&self) -> NodeIdentity {
            NodeIdentity::of_arc(&self.0)
        }
    }

    #[derive(Default)]
    struct DoubleAnalysis;

    impl Analysis for DoubleAnalysis {
        type Ir = TestIr;
        type Output = i64;

        fn run(&self, ir: &TestIr) -> i64 {
            ir.0.double_runs.fetch_add(1, Ordering::SeqCst);
            ir.0.value * 2
        }
    }

    #[derive(Default)]
    struct ParityAnalysis;

    impl Analysis for ParityAnalysis {
        type Ir = TestIr;
        type Output = i64;

        fn run(&self, ir: &TestIr) -> i64 {
            ir.0.parity_runs.fetch_add(1, Ordering::SeqCst);
            ir.0.value.rem_euclid(2)
        }
    }

    /// An analysis over either handle type, with a different output for
    /// each.
    struct LabelAnalysis<T>(PhantomData<T>);

    impl<T> Default for LabelAnalysis<T> {
        fn default() -> Self {
            Self(PhantomData)
        }
    }

    impl Analysis for LabelAnalysis<TestIr> {
        type Ir = TestIr;
        type Output = i64;

        fn run(&self, ir: &TestIr) -> i64 {
            ir.0.label_runs.fetch_add(1, Ordering::SeqCst);
            ir.0.value
        }
    }

    impl Analysis for LabelAnalysis<AliasIr> {
        type Ir = AliasIr;
        type Output = String;

        fn run(&self, ir: &AliasIr) -> String {
            ir.0.label_runs.fetch_add(1, Ordering::SeqCst);
            format!("alias {}", ir.0.value)
        }
    }

    /// Seed `cache` with both counting analyses for `ir`.
    fn seed_both(cache: &mut AnalysisCache, ir: &TestIr) {
        cache.get::<DoubleAnalysis>(ir);
        cache.get::<ParityAnalysis>(ir);
    }

    /// Test a result is computed on the first request and served from the
    /// cache afterwards.
    #[test]
    fn analysis_cache_get_computes_once_per_node() {
        let mut cache = AnalysisCache::new();
        let ir = TestIr::new(3);

        let first = cache.get::<DoubleAnalysis>(&ir);
        let second = cache.get::<DoubleAnalysis>(&ir.clone());

        assert_eq!((*first, *second), (6, 6));
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(ir.double_runs(), 1);
    }

    /// Test results are kept per analysis and per node.
    #[test]
    fn analysis_cache_get_keeps_results_per_analysis_and_node() {
        let mut cache = AnalysisCache::new();
        let first = TestIr::new(3);
        let second = TestIr::new(4);

        seed_both(&mut cache, &first);
        seed_both(&mut cache, &second);
        seed_both(&mut cache, &first);

        assert_eq!(*cache.get::<ParityAnalysis>(&second), 0);
        assert_eq!((first.double_runs(), first.parity_runs()), (1, 1));
        assert_eq!((second.double_runs(), second.parity_runs()), (1, 1));
    }

    /// Test two handle types over one node never share results, even for
    /// analyses of one generic type.
    #[test]
    fn analysis_cache_separates_handle_types_over_one_node() {
        let mut cache = AnalysisCache::new();
        let ir = TestIr::new(5);
        let alias = AliasIr(Arc::clone(&ir.0));

        let number = cache.get::<LabelAnalysis<TestIr>>(&ir);
        let text = cache.get::<LabelAnalysis<AliasIr>>(&alias);
        let number_again = cache.get::<LabelAnalysis<TestIr>>(&ir);

        assert_eq!(*number, 5);
        assert_eq!(*text, "alias 5");
        assert_eq!(*number_again, 5);
        assert_eq!(ir.label_runs(), 2);
    }

    /// Test `get_or_insert_with` computes on a miss only.
    #[test]
    fn analysis_cache_get_or_insert_with_computes_on_a_miss_only() {
        struct Report;
        impl Analysis for Report {
            type Ir = TestIr;
            type Output = String;

            fn run(&self, _ir: &TestIr) -> String {
                "report".to_owned()
            }
        }
        let mut cache = AnalysisCache::new();
        let ir = TestIr::new(1);
        let mut computations = 0;

        let first = cache.get_or_insert_with(&ir, AnalysisId::of::<Report>(), || {
            computations += 1;
            "report".to_owned()
        });
        let second = cache.get_or_insert_with(&ir, AnalysisId::of::<Report>(), || {
            computations += 1;
            "other".to_owned()
        });

        assert_eq!((first.as_str(), second.as_str()), ("report", "report"));
        assert_eq!(computations, 1);
    }

    /// Test the cache holds a handle to each cached node until it is dropped.
    #[test]
    fn analysis_cache_holds_a_handle_to_each_cached_node() {
        let mut cache = AnalysisCache::new();
        let ir = TestIr::new(1);

        cache.get::<DoubleAnalysis>(&ir);
        let while_cached = ir.handle_count();
        drop(cache);

        assert_eq!(while_cached, 2);
        assert_eq!(ir.handle_count(), 1);
    }

    /// Test transferring with the all-preserving set copies every result to
    /// the replacement and leaves the replaced node's results in place.
    #[test]
    fn analysis_cache_transfer_copies_preserved_results() {
        let mut cache = AnalysisCache::new();
        let from = TestIr::new(3);
        let to = TestIr::new(4);
        seed_both(&mut cache, &from);

        cache.transfer(&from, &to, &PreservedAnalyses::all());
        let transferred = *cache.get::<DoubleAnalysis>(&to);
        seed_both(&mut cache, &from);

        assert_eq!(transferred, 6);
        assert_eq!(to.double_runs(), 0);
        assert_eq!((from.double_runs(), from.parity_runs()), (1, 1));
        assert_eq!(from.handle_count(), 2);
        assert_eq!(to.handle_count(), 2);
    }

    /// Test transferring with the empty set drops every result.
    #[test]
    fn analysis_cache_transfer_drops_unpreserved_results() {
        let mut cache = AnalysisCache::new();
        let from = TestIr::new(3);
        let to = TestIr::new(4);
        seed_both(&mut cache, &from);

        cache.transfer(&from, &to, &PreservedAnalyses::none());
        let to_handles = to.handle_count();
        let recomputed = *cache.get::<DoubleAnalysis>(&to);

        assert_eq!(to_handles, 1);
        assert_eq!(recomputed, 8);
        assert_eq!(to.double_runs(), 1);
    }

    /// Test transferring copies exactly the preserved results.
    #[test]
    fn analysis_cache_transfer_copies_only_preserved_results() {
        let mut cache = AnalysisCache::new();
        let from = TestIr::new(3);
        let to = TestIr::new(4);
        seed_both(&mut cache, &from);

        cache.transfer(
            &from,
            &to,
            &PreservedAnalyses::none().preserve::<ParityAnalysis>(),
        );

        assert_eq!(*cache.get::<ParityAnalysis>(&to), 1);
        assert_eq!(*cache.get::<DoubleAnalysis>(&to), 8);
        assert_eq!((to.parity_runs(), to.double_runs()), (0, 1));
    }

    /// Test transferring merges into the replacement's results: it keeps the
    /// results it has and gains those it lacks.
    #[test]
    fn analysis_cache_transfer_merges_into_the_replacements_results() {
        let mut cache = AnalysisCache::new();
        let from = TestIr::new(3);
        let to = TestIr::new(4);
        cache.get::<DoubleAnalysis>(&from);
        cache.get::<ParityAnalysis>(&to);

        cache.transfer(&from, &to, &PreservedAnalyses::all());
        cache.get::<ParityAnalysis>(&to);
        let double = *cache.get::<DoubleAnalysis>(&to);

        assert_eq!(double, 6);
        assert_eq!((to.parity_runs(), to.double_runs()), (1, 0));
    }

    /// Test a result the replacement computed itself wins over the preserved
    /// result of the node it replaces.
    #[test]
    fn analysis_cache_transfer_keeps_the_replacements_own_result() {
        let mut cache = AnalysisCache::new();
        let from = TestIr::new(3);
        let to = TestIr::new(4);
        cache.get::<DoubleAnalysis>(&from);
        cache.get::<DoubleAnalysis>(&to);

        cache.transfer(&from, &to, &PreservedAnalyses::all());
        let double = *cache.get::<DoubleAnalysis>(&to);

        assert_eq!(double, 8);
        assert_eq!(to.double_runs(), 1);
    }

    /// Test transferring from a node without results keeps the replacement's
    /// results.
    #[test]
    fn analysis_cache_transfer_from_an_uncached_node_keeps_the_replacements_results() {
        let mut cache = AnalysisCache::new();
        let from = TestIr::new(3);
        let to = TestIr::new(4);
        cache.get::<DoubleAnalysis>(&to);

        cache.transfer(&from, &to, &PreservedAnalyses::all());
        cache.get::<DoubleAnalysis>(&to);

        assert_eq!(to.handle_count(), 2);
        assert_eq!(from.handle_count(), 1);
        assert_eq!(to.double_runs(), 1);
    }

    /// Test transferring a node to itself keeps every result, whatever is
    /// preserved.
    #[test]
    fn analysis_cache_transfer_to_the_same_node_keeps_every_result() {
        let mut cache = AnalysisCache::new();
        let ir = TestIr::new(3);
        seed_both(&mut cache, &ir);

        cache.transfer(&ir, &ir.clone(), &PreservedAnalyses::none());
        seed_both(&mut cache, &ir);

        assert_eq!((ir.double_runs(), ir.parity_runs()), (1, 1));
    }
}
