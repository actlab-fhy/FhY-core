//! Node identity, reusable analyses, and the identity-keyed analysis cache a
//! pass manager keeps for one run.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use super::preserved::{AnalysisId, PreservedAnalyses};

/// The opaque identity of a live IR node.
///
/// Two identities are equal only when they come from handles to the same
/// node while that node is alive. Once every handle to a node is dropped, a
/// later node may receive the same identity; a holder that needs an identity
/// to stay unique keeps a handle alive alongside it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeIdentity(usize);

impl NodeIdentity {
    /// Return the identity of the allocation `node` points to.
    ///
    /// Clones of one [`Arc`] share an identity; two separately allocated
    /// `Arc`s have different identities while both are alive.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use fhy_core::pass::NodeIdentity;
    ///
    /// let node = Arc::new(5);
    /// let alias = Arc::clone(&node);
    /// let other = Arc::new(5);
    ///
    /// assert_eq!(NodeIdentity::of_arc(&node), NodeIdentity::of_arc(&alias));
    /// assert_ne!(NodeIdentity::of_arc(&node), NodeIdentity::of_arc(&other));
    /// ```
    #[must_use]
    pub fn of_arc<T: ?Sized>(node: &Arc<T>) -> Self {
        Self(Arc::as_ptr(node).cast::<()>().addr())
    }
}

/// A cheap-to-clone handle to an immutable IR node with a stable identity.
///
/// Pipeline IR and every IR an analysis is cached for is a node handle.
/// Cloning a handle must not copy the node: a clone reports the same
/// [`identity`](Self::identity), and the node cannot change while any
/// handle to it is alive, so a result computed for one handle holds for
/// every clone.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
///
/// use fhy_core::pass::{NodeHandle, NodeIdentity};
///
/// #[derive(Clone)]
/// struct Module(Arc<Vec<String>>);
///
/// impl NodeHandle for Module {
///     fn identity(&self) -> NodeIdentity {
///         NodeIdentity::of_arc(&self.0)
///     }
/// }
///
/// let module = Module(Arc::new(vec!["main".to_owned()]));
/// assert_eq!(module.identity(), module.clone().identity());
/// ```
pub trait NodeHandle: Clone + Send + Sync + 'static {
    /// Return the identity of the node this handle points to.
    #[must_use]
    fn identity(&self) -> NodeIdentity;
}

/// A reusable computation over IR whose result a pass manager can cache.
///
/// Passes obtain results through
/// [`PassContext::analysis`](super::PassContext::analysis), which builds the
/// analysis with [`Default`] and caches its result per node while the pass
/// runs under a [`PassManager`](super::PassManager). An analysis cannot
/// fail; one whose computation can fail makes the failure part of its
/// [`Output`](Self::Output).
pub trait Analysis<T>: 'static {
    /// The result the analysis computes.
    type Output: Send + Sync + 'static;

    /// Compute the analysis result for `ir`.
    fn run(&self, ir: &T) -> Self::Output;
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

/// The cached results for one node, together with a handle that keeps the
/// node, and so its identity, alive while the bucket exists.
#[derive(Debug)]
struct Bucket {
    /// Never read: holding it is what keeps the node alive.
    _handle: Box<dyn Any + Send + Sync>,
    results: HashMap<AnalysisId, CachedResult>,
}

impl Bucket {
    /// Create the bucket for `ir` holding `results`.
    fn new<T: NodeHandle>(ir: &T, results: HashMap<AnalysisId, CachedResult>) -> Self {
        Self {
            _handle: Box::new(ir.clone()),
            results,
        }
    }
}

/// Analysis results cached per node identity for one pass-manager run.
///
/// Each cached node's bucket holds a clone of the node's handle, so no other
/// node can take over its identity while the cache holds results for it.
/// Dropping the cache releases every handle.
#[derive(Debug, Default)]
pub(super) struct AnalysisCache {
    buckets: HashMap<CacheKey, Bucket>,
}

impl AnalysisCache {
    /// Create an empty cache.
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Return the result of the analysis `A` for `ir`, computing and caching
    /// it on a miss.
    pub(super) fn get<A, T>(&mut self, ir: &T) -> Arc<A::Output>
    where
        A: Analysis<T> + Default,
        T: NodeHandle,
    {
        self.get_or_insert_with(ir, AnalysisId::of::<A>(), || A::default().run(ir))
    }

    /// Return the result cached for `ir` under `id`, computing it with
    /// `compute` and caching it on a miss.
    pub(super) fn get_or_insert_with<T, V>(
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
            .or_insert_with(|| Bucket::new(ir, HashMap::new()));
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

    /// Drop the results cached for `ir` that `preserved` does not preserve.
    pub(super) fn invalidate<T: NodeHandle>(&mut self, ir: &T, preserved: &PreservedAnalyses) {
        if preserved.preserves_all() {
            return;
        }
        let key = find_cache_key(ir);
        let Some(bucket) = self.buckets.get_mut(&key) else {
            return;
        };
        bucket
            .results
            .retain(|id, _| preserved.is_id_preserved(*id));
        if bucket.results.is_empty() {
            self.buckets.remove(&key);
        }
    }

    /// Carry the results `preserved` preserves from `from` over to its
    /// replacement `to`.
    ///
    /// When both are the same node this is [`invalidate`](Self::invalidate).
    /// Otherwise the bucket of `from` and any bucket of `to` are dropped, and
    /// `to` receives the preserved results of `from`.
    pub(super) fn transfer<T: NodeHandle>(
        &mut self,
        from: &T,
        to: &T,
        preserved: &PreservedAnalyses,
    ) {
        let from_key = find_cache_key(from);
        let to_key = find_cache_key(to);
        if from_key == to_key {
            self.invalidate(from, preserved);
            return;
        }
        let from_bucket = self.buckets.remove(&from_key);
        self.buckets.remove(&to_key);
        let Some(Bucket { mut results, .. }) = from_bucket else {
            return;
        };
        results.retain(|id, _| preserved.is_id_preserved(*id));
        if !results.is_empty() {
            self.buckets.insert(to_key, Bucket::new(to, results));
        }
    }
}

#[cfg(test)]
mod tests {
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

    impl Analysis<TestIr> for DoubleAnalysis {
        type Output = i64;

        fn run(&self, ir: &TestIr) -> i64 {
            ir.0.double_runs.fetch_add(1, Ordering::SeqCst);
            ir.0.value * 2
        }
    }

    #[derive(Default)]
    struct ParityAnalysis;

    impl Analysis<TestIr> for ParityAnalysis {
        type Output = i64;

        fn run(&self, ir: &TestIr) -> i64 {
            ir.0.parity_runs.fetch_add(1, Ordering::SeqCst);
            ir.0.value.rem_euclid(2)
        }
    }

    /// An analysis over both handle types, with a different output for each.
    #[derive(Default)]
    struct LabelAnalysis;

    impl Analysis<TestIr> for LabelAnalysis {
        type Output = i64;

        fn run(&self, ir: &TestIr) -> i64 {
            ir.0.label_runs.fetch_add(1, Ordering::SeqCst);
            ir.0.value
        }
    }

    impl Analysis<AliasIr> for LabelAnalysis {
        type Output = String;

        fn run(&self, ir: &AliasIr) -> String {
            ir.0.label_runs.fetch_add(1, Ordering::SeqCst);
            format!("alias {}", ir.0.value)
        }
    }

    /// Seed `cache` with both counting analyses for `ir`.
    fn seed_both(cache: &mut AnalysisCache, ir: &TestIr) {
        cache.get::<DoubleAnalysis, _>(ir);
        cache.get::<ParityAnalysis, _>(ir);
    }

    /// Test a result is computed on the first request and served from the
    /// cache afterwards.
    #[test]
    fn analysis_cache_get_computes_once_per_node() {
        let mut cache = AnalysisCache::new();
        let ir = TestIr::new(3);

        let first = cache.get::<DoubleAnalysis, _>(&ir);
        let second = cache.get::<DoubleAnalysis, _>(&ir.clone());

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

        assert_eq!(*cache.get::<ParityAnalysis, _>(&second), 0);
        assert_eq!((first.double_runs(), first.parity_runs()), (1, 1));
        assert_eq!((second.double_runs(), second.parity_runs()), (1, 1));
    }

    /// Test two handle types over one node never share results, even for the
    /// same analysis.
    #[test]
    fn analysis_cache_separates_handle_types_over_one_node() {
        let mut cache = AnalysisCache::new();
        let ir = TestIr::new(5);
        let alias = AliasIr(Arc::clone(&ir.0));

        let number = cache.get::<LabelAnalysis, _>(&ir);
        let text = cache.get::<LabelAnalysis, _>(&alias);
        let number_again = cache.get::<LabelAnalysis, _>(&ir);

        assert_eq!(*number, 5);
        assert_eq!(*text, "alias 5");
        assert_eq!(*number_again, 5);
        assert_eq!(ir.label_runs(), 2);
    }

    /// Test `get_or_insert_with` computes on a miss only.
    #[test]
    fn analysis_cache_get_or_insert_with_computes_on_a_miss_only() {
        struct Marker;
        let mut cache = AnalysisCache::new();
        let ir = TestIr::new(1);
        let mut computations = 0;

        let first = cache.get_or_insert_with(&ir, AnalysisId::of::<Marker>(), || {
            computations += 1;
            "report".to_owned()
        });
        let second = cache.get_or_insert_with(&ir, AnalysisId::of::<Marker>(), || {
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

        cache.get::<DoubleAnalysis, _>(&ir);
        let while_cached = ir.handle_count();
        drop(cache);

        assert_eq!(while_cached, 2);
        assert_eq!(ir.handle_count(), 1);
    }

    /// Test invalidating with the empty set drops every result.
    #[test]
    fn analysis_cache_invalidate_with_none_drops_everything() {
        let mut cache = AnalysisCache::new();
        let ir = TestIr::new(3);
        seed_both(&mut cache, &ir);

        cache.invalidate(&ir, &PreservedAnalyses::none());
        let handles_after_invalidate = ir.handle_count();
        seed_both(&mut cache, &ir);

        assert_eq!(handles_after_invalidate, 1);
        assert_eq!((ir.double_runs(), ir.parity_runs()), (2, 2));
    }

    /// Test invalidating with the all-preserving set keeps everything.
    #[test]
    fn analysis_cache_invalidate_with_all_keeps_everything() {
        let mut cache = AnalysisCache::new();
        let ir = TestIr::new(3);
        seed_both(&mut cache, &ir);

        cache.invalidate(&ir, &PreservedAnalyses::all());
        seed_both(&mut cache, &ir);

        assert_eq!((ir.double_runs(), ir.parity_runs()), (1, 1));
    }

    /// Test invalidating keeps exactly the preserved results.
    #[test]
    fn analysis_cache_invalidate_keeps_only_preserved_results() {
        let mut cache = AnalysisCache::new();
        let ir = TestIr::new(3);
        seed_both(&mut cache, &ir);

        cache.invalidate(&ir, &PreservedAnalyses::none().preserve::<DoubleAnalysis>());
        seed_both(&mut cache, &ir);

        assert_eq!((ir.double_runs(), ir.parity_runs()), (1, 2));
    }

    /// Test invalidating a node without results caches nothing for it.
    #[test]
    fn analysis_cache_invalidate_of_an_uncached_node_holds_no_handle() {
        let mut cache = AnalysisCache::new();
        let ir = TestIr::new(3);

        cache.invalidate(&ir, &PreservedAnalyses::all());

        assert_eq!(ir.handle_count(), 1);
    }

    /// Test transferring with the all-preserving set moves every result to
    /// the replacement and drops the replaced node's bucket.
    #[test]
    fn analysis_cache_transfer_moves_preserved_results() {
        let mut cache = AnalysisCache::new();
        let from = TestIr::new(3);
        let to = TestIr::new(4);
        seed_both(&mut cache, &from);

        cache.transfer(&from, &to, &PreservedAnalyses::all());
        let from_handles = from.handle_count();
        let transferred = *cache.get::<DoubleAnalysis, _>(&to);

        assert_eq!(transferred, 6);
        assert_eq!(to.double_runs(), 0);
        assert_eq!(from_handles, 1);
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
        let recomputed = *cache.get::<DoubleAnalysis, _>(&to);

        assert_eq!(to_handles, 1);
        assert_eq!(recomputed, 8);
        assert_eq!(to.double_runs(), 1);
    }

    /// Test transferring moves exactly the preserved results.
    #[test]
    fn analysis_cache_transfer_moves_only_preserved_results() {
        let mut cache = AnalysisCache::new();
        let from = TestIr::new(3);
        let to = TestIr::new(4);
        seed_both(&mut cache, &from);

        cache.transfer(
            &from,
            &to,
            &PreservedAnalyses::none().preserve::<ParityAnalysis>(),
        );

        assert_eq!(*cache.get::<ParityAnalysis, _>(&to), 1);
        assert_eq!(*cache.get::<DoubleAnalysis, _>(&to), 8);
        assert_eq!((to.parity_runs(), to.double_runs()), (0, 1));
    }

    /// Test transferring replaces whatever the replacement had cached.
    #[test]
    fn analysis_cache_transfer_replaces_the_replacements_results() {
        let mut cache = AnalysisCache::new();
        let from = TestIr::new(3);
        let to = TestIr::new(4);
        cache.get::<DoubleAnalysis, _>(&from);
        cache.get::<ParityAnalysis, _>(&to);

        cache.transfer(&from, &to, &PreservedAnalyses::all());
        cache.get::<ParityAnalysis, _>(&to);
        let double = *cache.get::<DoubleAnalysis, _>(&to);

        assert_eq!(double, 6);
        assert_eq!(to.parity_runs(), 2);
    }

    /// Test transferring from a node without results drops the replacement's
    /// results.
    #[test]
    fn analysis_cache_transfer_from_an_uncached_node_drops_the_replacements_results() {
        let mut cache = AnalysisCache::new();
        let from = TestIr::new(3);
        let to = TestIr::new(4);
        cache.get::<DoubleAnalysis, _>(&to);

        cache.transfer(&from, &to, &PreservedAnalyses::all());
        let to_handles = to.handle_count();
        cache.get::<DoubleAnalysis, _>(&to);

        assert_eq!(to_handles, 1);
        assert_eq!(to.double_runs(), 2);
    }

    /// Test transferring a node to itself invalidates what is not preserved.
    #[test]
    fn analysis_cache_transfer_to_the_same_node_invalidates() {
        let mut cache = AnalysisCache::new();
        let ir = TestIr::new(3);
        seed_both(&mut cache, &ir);

        cache.transfer(
            &ir,
            &ir.clone(),
            &PreservedAnalyses::none().preserve::<DoubleAnalysis>(),
        );
        seed_both(&mut cache, &ir);

        assert_eq!((ir.double_runs(), ir.parity_runs()), (1, 2));
    }
}
