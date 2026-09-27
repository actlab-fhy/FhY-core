//! Tests for analyses that no Rust type names, such as ones a language
//! binding defines at run time: analysis ids built from identifiers, the
//! pipeline cache by id, and the detached cache handle a hook lends out.
//!
//! Each computation counts its runs in a counter the test owns, so nothing
//! here reads process-global state.

use crate::support::pass_ir;

use std::any::type_name;
use std::collections::HashSet;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use fhy_core::foreign::BoxError;
use fhy_core::identifier::Identifier;
use fhy_core::pass::{
    AnalysisId, CompilerPass, DetachedAnalyses, DetachedAnalysesExpired, ExecutePass, PassContext,
    PassManager, PreservedAnalyses,
};
use pass_ir::{BoxIr, ClosurePass, DoubleAnalysis};

// =============================================================================
// Helpers
// =============================================================================

/// Return the value of `ir` plus one, counting the call in `runs`.
fn count_successor(ir: &BoxIr, runs: &AtomicUsize) -> i64 {
    runs.fetch_add(1, Ordering::SeqCst);
    ir.value() + 1
}

/// Return how many calls `runs` counted.
fn read(runs: &AtomicUsize) -> usize {
    runs.load(Ordering::SeqCst)
}

/// Build a pipeline named `name` of `passes`.
fn build_pipeline<'p>(
    name: &str,
    passes: impl IntoIterator<Item = ClosurePass<'p>>,
) -> PassManager<'p, BoxIr> {
    let mut manager = PassManager::new(Identifier::new(name));
    for pass in passes {
        manager.add_pass(pass);
    }
    manager
}

/// Build the pass `name` that reads the successor analysis `id` of its
/// input, counting computations in `runs`, and returns its input.
fn build_successor_reading_pass<'a>(
    name: &str,
    id: &'a AnalysisId,
    runs: &'a AtomicUsize,
) -> ClosurePass<'a> {
    ClosurePass::new(name, move |ir, cx| {
        cx.analysis_by_id(ir, id, |ir| count_successor(ir, runs));
        Ok(ir.clone())
    })
}

// =============================================================================
// Analysis ids from identifiers
// =============================================================================

/// Test an id built from an identifier equals exactly the ids built from
/// that identifier, restored copies included.
#[test]
fn analysis_id_of_identifier_is_equal_exactly_for_one_identifier() {
    let name = Identifier::new("tests.dynamic.liveness");
    let restored = Identifier::try_restore(name.id(), name.name_hint()).expect("the id is valid");
    let namesake = Identifier::new("tests.dynamic.liveness");

    let ids: HashSet<_> = [
        AnalysisId::of_identifier(&name),
        AnalysisId::of_identifier(&restored),
        AnalysisId::of_identifier(&namesake),
    ]
    .into_iter()
    .collect();

    assert_eq!(
        AnalysisId::of_identifier(&name),
        AnalysisId::of_identifier(&name.clone())
    );
    assert_eq!(
        AnalysisId::of_identifier(&name),
        AnalysisId::of_identifier(&restored)
    );
    assert_ne!(
        AnalysisId::of_identifier(&name),
        AnalysisId::of_identifier(&namesake)
    );
    assert_eq!(ids.len(), 2);
}

/// Test an id built from an identifier never equals the id of a type, even
/// one whose name is the identifier's name hint.
#[test]
fn analysis_id_of_identifier_never_equals_a_type_id() {
    let namesake = Identifier::new(type_name::<DoubleAnalysis>());

    assert_ne!(
        AnalysisId::of_identifier(&namesake),
        AnalysisId::of::<DoubleAnalysis>()
    );
}

#[test]
fn analysis_id_of_identifier_displays_its_name_hint() {
    let name = Identifier::new("tests.dynamic.dominance");

    assert_eq!(
        AnalysisId::of_identifier(&name).to_string(),
        "tests.dynamic.dominance"
    );
}

/// Test an id returns the identifier it was built from, and a type's id
/// none.
#[test]
fn analysis_id_identifier_returns_the_identifier_it_was_built_from() {
    let name = Identifier::new("tests.dynamic.aliasing");

    assert_eq!(AnalysisId::of_identifier(&name).identifier(), Some(&name));
    assert_eq!(AnalysisId::of::<DoubleAnalysis>().identifier(), None);
}

/// Test ids of types order before ids of identifiers, and ids of
/// identifiers order by the identifier, agreeing with equality for
/// restored copies whose name hints differ.
#[test]
fn analysis_ids_order_types_first_then_identifiers_by_identifier() {
    let first = Identifier::new("tests.dynamic.z_first");
    let second = Identifier::new("tests.dynamic.a_second");
    let renamed =
        Identifier::try_restore(first.id(), "tests.dynamic.renamed").expect("the id is valid");

    assert!(AnalysisId::of::<DoubleAnalysis>() < AnalysisId::of_identifier(&first));
    assert!(AnalysisId::of_identifier(&first) < AnalysisId::of_identifier(&second));
    assert_eq!(
        AnalysisId::of_identifier(&first).cmp(&AnalysisId::of_identifier(&renamed)),
        std::cmp::Ordering::Equal
    );
}

/// Test a preservation set keeps ids of identifiers as it keeps ids of
/// types, and lists them after the types.
#[test]
fn preserved_analyses_preserve_ids_of_identifiers() {
    let kept = Identifier::new("tests.dynamic.kept");
    let dropped = Identifier::new("tests.dynamic.dropped");

    let preserved = PreservedAnalyses::none()
        .preserve_id(AnalysisId::of_identifier(&kept))
        .preserve::<DoubleAnalysis>();

    assert!(preserved.is_id_preserved(&AnalysisId::of_identifier(&kept)));
    assert!(!preserved.is_id_preserved(&AnalysisId::of_identifier(&dropped)));
    assert!(PreservedAnalyses::all().is_id_preserved(&AnalysisId::of_identifier(&dropped)));
    assert_eq!(
        preserved.preserved_ids().cloned().collect::<Vec<_>>(),
        [
            AnalysisId::of::<DoubleAnalysis>(),
            AnalysisId::of_identifier(&kept)
        ]
    );
}

// =============================================================================
// The cache by id
// =============================================================================

/// Test a result requested twice by id in one managed pass is computed
/// once and shared.
#[test]
fn analysis_by_id_is_cached_within_a_managed_pass() {
    let id = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.successor"));
    let runs = AtomicUsize::new(0);
    let observed = Mutex::new(Vec::new());
    let mut manager = build_pipeline(
        "pipeline",
        [ClosurePass::new("tests.dynamic.read_twice", |ir, cx| {
            let first = cx.analysis_by_id(ir, &id, |ir| count_successor(ir, &runs));
            let second =
                cx.analysis_by_id(ir, &id, |_| -> i64 { unreachable!("the result is cached") });
            observed
                .lock()
                .expect("no test thread panicked")
                .push((*first, Arc::ptr_eq(&first, &second)));
            Ok(ir.clone())
        })],
    );

    manager.run(&BoxIr::new(4)).expect("the run succeeds");
    drop(manager);

    assert_eq!(
        observed.into_inner().expect("no test thread panicked"),
        [(5, true)]
    );
    assert_eq!(read(&runs), 1);
}

/// Test a pass run on its own computes a result by id on every request.
#[test]
fn analysis_by_id_computes_afresh_outside_a_pipeline() {
    let id = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.standalone"));
    let runs = AtomicUsize::new(0);
    let mut pass = ClosurePass::new("tests.dynamic.standalone_reader", |ir, cx| {
        cx.analysis_by_id(ir, &id, |ir| count_successor(ir, &runs));
        cx.analysis_by_id(ir, &id, |ir| count_successor(ir, &runs));
        Ok(ir.clone())
    });

    pass.execute(&BoxIr::new(1)).expect("the run succeeds");

    assert_eq!(read(&runs), 2);
}

/// Test results are kept per id and per node.
#[test]
fn analysis_by_id_keeps_results_per_id_and_node() {
    let successor = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.per_id_a"));
    let other = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.per_id_b"));
    let side = BoxIr::new(10);
    let runs = AtomicUsize::new(0);
    let observed = Mutex::new(Vec::new());
    let mut manager = build_pipeline(
        "pipeline",
        [ClosurePass::new("tests.dynamic.read_each", |ir, cx| {
            let push = |value: i64| {
                observed
                    .lock()
                    .expect("no test thread panicked")
                    .push(value);
            };
            push(*cx.analysis_by_id(ir, &successor, |ir| count_successor(ir, &runs)));
            push(*cx.analysis_by_id(ir, &other, |ir| -count_successor(ir, &runs)));
            push(*cx.analysis_by_id(&side, &successor, |ir| count_successor(ir, &runs)));
            push(*cx.analysis_by_id(ir, &successor, |ir| count_successor(ir, &runs)));
            Ok(ir.clone())
        })],
    );

    manager.run(&BoxIr::new(1)).expect("the run succeeds");
    drop(manager);

    assert_eq!(
        observed.into_inner().expect("no test thread panicked"),
        [2, -2, 11, 2]
    );
    assert_eq!(read(&runs), 3);
}

/// Test the id of an analysis type reaches that type's cached result.
#[test]
fn analysis_by_id_of_a_type_id_serves_that_types_result() {
    let observed = Mutex::new(Vec::new());
    let mut manager = build_pipeline(
        "pipeline",
        [ClosurePass::new(
            "tests.dynamic.typed_then_by_id",
            |ir, cx| {
                cx.analysis::<DoubleAnalysis>(ir);
                let by_id =
                    cx.analysis_by_id(ir, &AnalysisId::of::<DoubleAnalysis>(), |_| -> i64 {
                        unreachable!("the typed result is cached")
                    });
                observed
                    .lock()
                    .expect("no test thread panicked")
                    .push(*by_id);
                Ok(ir.clone())
            },
        )],
    );
    let input = BoxIr::new(3);

    manager.run(&input).expect("the run succeeds");
    drop(manager);

    assert_eq!(observed.into_inner().expect("no test thread panicked"), [6]);
    assert_eq!(input.double_runs(), 1);
}

/// Test a request by id for a result of another type than the one cached
/// under the id recomputes it and replaces the cached result.
#[test]
fn analysis_by_id_recomputes_a_result_cached_with_another_type() {
    let id = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.retyped"));
    let observed = Mutex::new(Vec::new());
    let mut manager = build_pipeline(
        "pipeline",
        [ClosurePass::new("tests.dynamic.retype", |ir, cx| {
            let number = *cx.analysis_by_id(ir, &id, BoxIr::value);
            let text = cx.analysis_by_id(ir, &id, |ir| format!("value {}", ir.value()));
            let text_again = cx.analysis_by_id(ir, &id, |_| String::from("recomputed"));
            observed.lock().expect("no test thread panicked").push((
                number,
                text.to_string(),
                text_again.to_string(),
            ));
            Ok(ir.clone())
        })],
    );

    manager.run(&BoxIr::new(7)).expect("the run succeeds");
    drop(manager);

    assert_eq!(
        observed.into_inner().expect("no test thread panicked"),
        [(7, "value 7".to_owned(), "value 7".to_owned())]
    );
}

/// Test a result by id follows an unchanged pass to the new node it
/// returns.
#[test]
fn analysis_by_id_follows_an_unchanged_pass_to_its_new_node() {
    let id = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.follow"));
    let runs = AtomicUsize::new(0);
    let mut manager = build_pipeline(
        "pipeline",
        [
            ClosurePass::new("tests.dynamic.compute_then_copy", |ir, cx| {
                cx.analysis_by_id(ir, &id, |ir| count_successor(ir, &runs));
                Ok(ir.derive(ir.value()))
            }),
            build_successor_reading_pass("tests.dynamic.read_copy", &id, &runs),
        ],
    );

    manager.run(&BoxIr::new(5)).expect("the run succeeds");
    drop(manager);

    assert_eq!(read(&runs), 1);
}

/// Adds one to the value and preserves exactly the analysis `kept`.
struct AddOnePreserving<'a> {
    kept: &'a AnalysisId,
}

impl CompilerPass<BoxIr> for AddOnePreserving<'_> {
    fn run(&mut self, ir: &BoxIr, _cx: &mut PassContext<'_>) -> Result<BoxIr, BoxError> {
        Ok(ir.derive(ir.value() + 1))
    }

    fn did_change(&mut self, input: &BoxIr, output: &BoxIr) -> Result<bool, BoxError> {
        Ok(input.value() != output.value())
    }

    fn preserved_analyses(
        &mut self,
        _input: &BoxIr,
        _output: &BoxIr,
        _changed: bool,
    ) -> Result<PreservedAnalyses, BoxError> {
        Ok(PreservedAnalyses::none().preserve_id(self.kept.clone()))
    }
}

/// Test a changing pass carries over exactly the ids of identifiers it
/// preserves.
#[test]
fn pass_manager_carries_preserved_ids_of_identifiers_to_a_changed_output() {
    let kept = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.kept_result"));
    let dropped = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.dropped_result"));
    let kept_runs = AtomicUsize::new(0);
    let dropped_runs = AtomicUsize::new(0);
    let observed = Mutex::new(Vec::new());
    let read_both = |ir: &BoxIr, cx: &mut PassContext<'_>| {
        let kept_value = *cx.analysis_by_id(ir, &kept, |ir| count_successor(ir, &kept_runs));
        let dropped_value =
            *cx.analysis_by_id(ir, &dropped, |ir| count_successor(ir, &dropped_runs));
        observed
            .lock()
            .expect("no test thread panicked")
            .push((kept_value, dropped_value));
        Ok(ir.clone())
    };
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(ClosurePass::new("tests.dynamic.seed_both", read_both));
    manager.add_pass(AddOnePreserving { kept: &kept });
    manager.add_pass(ClosurePass::new("tests.dynamic.read_both", read_both));

    manager.run(&BoxIr::new(1)).expect("the run succeeds");
    drop(manager);

    assert_eq!(
        observed.into_inner().expect("no test thread panicked"),
        [(2, 2), (2, 3)]
    );
    assert_eq!((read(&kept_runs), read(&dropped_runs)), (1, 2));
}

/// Adds one, computes `id` of its output, and preserves `id`.
struct ComputeOnOutput<'a> {
    id: &'a AnalysisId,
    runs: &'a AtomicUsize,
}

impl CompilerPass<BoxIr> for ComputeOnOutput<'_> {
    fn run(&mut self, ir: &BoxIr, cx: &mut PassContext<'_>) -> Result<BoxIr, BoxError> {
        let output = ir.derive(ir.value() + 1);
        cx.analysis_by_id(&output, self.id, |ir| count_successor(ir, self.runs));
        Ok(output)
    }

    fn did_change(&mut self, input: &BoxIr, output: &BoxIr) -> Result<bool, BoxError> {
        Ok(input.value() != output.value())
    }

    fn preserved_analyses(
        &mut self,
        _input: &BoxIr,
        _output: &BoxIr,
        _changed: bool,
    ) -> Result<PreservedAnalyses, BoxError> {
        Ok(PreservedAnalyses::none().preserve_id(self.id.clone()))
    }
}

/// Test an output that already has a result by id of its own keeps it
/// rather than inheriting its input's preserved result.
#[test]
fn pass_manager_prefers_an_outputs_own_result_by_id_to_its_inputs() {
    let id = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.own_result"));
    let runs = AtomicUsize::new(0);
    let observed = Mutex::new(Vec::new());
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(build_successor_reading_pass(
        "tests.dynamic.seed_input",
        &id,
        &runs,
    ));
    manager.add_pass(ComputeOnOutput {
        id: &id,
        runs: &runs,
    });
    manager.add_pass(ClosurePass::new("tests.dynamic.reread", |ir, cx| {
        let value = *cx.analysis_by_id(ir, &id, |_| -> i64 {
            unreachable!("the output has its own")
        });
        observed
            .lock()
            .expect("no test thread panicked")
            .push(value);
        Ok(ir.clone())
    }));

    manager.run(&BoxIr::new(3)).expect("the run succeeds");
    drop(manager);

    assert_eq!(observed.into_inner().expect("no test thread panicked"), [5]);
    assert_eq!(read(&runs), 2);
}

// =============================================================================
// The detached cache
// =============================================================================

/// Assert `T` is `Send`, `Sync` and `'static`.
fn assert_send_sync_static<T: Send + Sync + 'static>() {}

#[test]
fn detached_analyses_are_send_sync_and_static() {
    assert_send_sync_static::<DetachedAnalyses>();
    assert_send_sync_static::<DetachedAnalysesExpired>();
}

/// Test a detached handle serves the pipeline's cache, both ways: it finds
/// results cached before it was detached, and results it caches are there
/// after the callback returns.
#[test]
fn detached_analyses_serve_the_pipeline_cache() {
    let id = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.detached"));
    let runs = AtomicUsize::new(0);
    let observed = Mutex::new(Vec::new());
    let mut manager = build_pipeline(
        "pipeline",
        [ClosurePass::new("tests.dynamic.detach", |ir, cx| {
            cx.analysis::<DoubleAnalysis>(ir);
            let (double, successor) = cx.with_detached_analyses(|analyses| {
                let double = analyses
                    .analysis_by_id(ir, &AnalysisId::of::<DoubleAnalysis>(), |_| -> i64 {
                        unreachable!("the typed result is cached")
                    })
                    .expect("the handle is attached");
                let successor = analyses
                    .analysis_by_id(ir, &id, |ir| count_successor(ir, &runs))
                    .expect("the handle is attached");
                (*double, *successor)
            });
            let again =
                *cx.analysis_by_id(ir, &id, |_| -> i64 { unreachable!("the result came back") });
            observed
                .lock()
                .expect("no test thread panicked")
                .push((double, successor, again));
            Ok(ir.clone())
        })],
    );
    let input = BoxIr::new(4);

    manager.run(&input).expect("the run succeeds");
    drop(manager);

    assert_eq!(
        observed.into_inner().expect("no test thread panicked"),
        [(8, 5, 5)]
    );
    assert_eq!((input.double_runs(), read(&runs)), (1, 1));
}

/// Test a handle detached from a pass run on its own computes afresh on
/// every request.
#[test]
fn detached_analyses_compute_afresh_outside_a_pipeline() {
    let id = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.detached_standalone"));
    let runs = AtomicUsize::new(0);
    let mut pass = ClosurePass::new("tests.dynamic.detach_standalone", |ir, cx| {
        cx.with_detached_analyses(|analyses| {
            for _ in 0..2 {
                analyses
                    .analysis_by_id(ir, &id, |ir| count_successor(ir, &runs))
                    .expect("the handle is attached");
            }
        });
        Ok(ir.clone())
    });

    pass.execute(&BoxIr::new(1)).expect("the run succeeds");

    assert_eq!(read(&runs), 2);
}

/// Test a clone of the handle kept past its callback finds nothing: every
/// request fails without computing, whether the run was managed or not.
#[test]
fn detached_analyses_expire_when_their_callback_returns() {
    let id = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.expired"));
    let kept: Mutex<Vec<DetachedAnalyses>> = Mutex::new(Vec::new());
    let keep = |ir: &BoxIr, cx: &mut PassContext<'_>| {
        let attached = cx.with_detached_analyses(|analyses| {
            kept.lock()
                .expect("no test thread panicked")
                .push(analyses.clone());
            !analyses.is_expired()
        });
        assert!(attached, "a handle is attached during its callback");
        Ok(ir.clone())
    };
    let mut manager = build_pipeline("pipeline", [ClosurePass::new("tests.dynamic.keep", keep)]);
    manager.run(&BoxIr::new(1)).expect("the run succeeds");
    drop(manager);
    ClosurePass::new("tests.dynamic.keep_standalone", keep)
        .execute(&BoxIr::new(1))
        .expect("the run succeeds");
    let ir = BoxIr::new(2);

    let outcomes: Vec<_> = kept
        .into_inner()
        .expect("no test thread panicked")
        .iter()
        .map(|analyses| {
            let request = analyses.analysis_by_id(&ir, &id, |_| -> i64 {
                unreachable!("an expired handle computes nothing")
            });
            (
                analyses.is_expired(),
                matches!(request, Err(DetachedAnalysesExpired { .. })),
            )
        })
        .collect();

    assert_eq!(outcomes, [(true, true), (true, true)]);
}

/// Test a handle kept past its callback holds no node: the run releases
/// every cached handle when it ends.
#[test]
fn detached_analyses_kept_past_their_callback_hold_no_node() {
    let id = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.release"));
    let kept = Mutex::new(None);
    let mut manager = build_pipeline(
        "pipeline",
        [ClosurePass::new(
            "tests.dynamic.keep_after_caching",
            |ir, cx| {
                cx.with_detached_analyses(|analyses| {
                    analyses
                        .analysis_by_id(ir, &id, BoxIr::value)
                        .expect("the handle is attached");
                    *kept.lock().expect("no test thread panicked") = Some(analyses.clone());
                });
                Ok(ir.clone())
            },
        )],
    );
    let input = BoxIr::new(1);

    manager.run(&input).expect("the run succeeds");
    drop(manager);

    assert!(kept.lock().expect("no test thread panicked").is_some());
    assert_eq!(input.handle_count(), 1);
}

/// Test the cache comes back to the context when the callback panics.
#[test]
fn detached_analyses_come_back_after_a_panicking_callback() {
    let id = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.panic"));
    let runs = AtomicUsize::new(0);
    let mut manager = build_pipeline(
        "pipeline",
        [ClosurePass::new(
            "tests.dynamic.panicking_callback",
            |ir, cx| {
                let unwound = panic::catch_unwind(AssertUnwindSafe(|| {
                    cx.with_detached_analyses(|analyses| {
                        analyses
                            .analysis_by_id(ir, &id, |ir| count_successor(ir, &runs))
                            .expect("the handle is attached");
                        panic::resume_unwind(Box::new("the callback fails"));
                    })
                }));
                assert!(unwound.is_err(), "the callback panicked");
                cx.analysis_by_id(ir, &id, |_| -> i64 { unreachable!("the result came back") });
                Ok(ir.clone())
            },
        )],
    );

    manager.run(&BoxIr::new(1)).expect("the run succeeds");
    drop(manager);

    assert_eq!(read(&runs), 1);
}

/// Test a detached handle serves requests from another thread while its
/// callback runs.
#[test]
fn detached_analyses_serve_another_thread_during_their_callback() {
    let id = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.threads"));
    let runs = AtomicUsize::new(0);
    let observed = Mutex::new(Vec::new());
    let mut manager = build_pipeline(
        "pipeline",
        [ClosurePass::new(
            "tests.dynamic.across_threads",
            |ir, cx| {
                let from_thread = cx.with_detached_analyses(|analyses| {
                    thread::scope(|scope| {
                        scope
                            .spawn(|| {
                                *analyses
                                    .analysis_by_id(ir, &id, |ir| count_successor(ir, &runs))
                                    .expect("the handle is attached")
                            })
                            .join()
                            .expect("the request thread does not panic")
                    })
                });
                let here = *cx
                    .analysis_by_id(ir, &id, |_| -> i64 { unreachable!("the result came back") });
                observed
                    .lock()
                    .expect("no test thread panicked")
                    .push((from_thread, here));
                Ok(ir.clone())
            },
        )],
    );

    manager.run(&BoxIr::new(8)).expect("the run succeeds");
    drop(manager);

    assert_eq!(
        observed.into_inner().expect("no test thread panicked"),
        [(9, 9)]
    );
    assert_eq!(read(&runs), 1);
}

/// Test a computation may request other results through the same handle,
/// and a result the computation's own request cached is the one both
/// requests return.
#[test]
fn detached_analyses_serve_requests_made_while_computing() {
    let outer = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.outer"));
    let inner = AnalysisId::of_identifier(&Identifier::new("tests.dynamic.inner"));
    let observed = Mutex::new(Vec::new());
    let mut manager = build_pipeline(
        "pipeline",
        [ClosurePass::new(
            "tests.dynamic.nested_requests",
            |ir, cx| {
                cx.with_detached_analyses(|analyses| {
                    let mut nested_same = None;
                    let result = analyses
                        .analysis_by_id(ir, &outer, |ir| {
                            let inner_value = *analyses
                                .analysis_by_id(ir, &inner, BoxIr::value)
                                .expect("the handle is attached");
                            nested_same = Some(
                                analyses
                                    .analysis_by_id(ir, &outer, |_| inner_value * 100)
                                    .expect("the handle is attached"),
                            );
                            inner_value * 10
                        })
                        .expect("the handle is attached");
                    let nested_same = nested_same.expect("the computation ran");
                    observed.lock().expect("no test thread panicked").push((
                        *result,
                        *nested_same,
                        Arc::ptr_eq(&result, &nested_same),
                    ));
                });
                Ok(ir.clone())
            },
        )],
    );

    manager.run(&BoxIr::new(3)).expect("the run succeeds");
    drop(manager);

    assert_eq!(
        observed.into_inner().expect("no test thread panicked"),
        [(300, 300, true)]
    );
}

/// Return the error a request through an expired handle fails with.
fn produce_expired_error() -> DetachedAnalysesExpired {
    let kept = Mutex::new(None);
    ClosurePass::new("tests.dynamic.expire", |ir, cx| {
        cx.with_detached_analyses(|analyses| {
            *kept.lock().expect("no test thread panicked") = Some(analyses.clone());
        });
        Ok(ir.clone())
    })
    .execute(&BoxIr::new(1))
    .expect("the run succeeds");
    let analyses = kept
        .into_inner()
        .expect("no test thread panicked")
        .expect("the callback ran");
    analyses
        .analysis_by_id(
            &BoxIr::new(1),
            &AnalysisId::of::<DoubleAnalysis>(),
            BoxIr::value,
        )
        .expect_err("the handle expired")
}

/// Test the expired error renders one line, without a source.
#[test]
fn detached_analyses_expired_displays_one_line_without_a_source() {
    let error = produce_expired_error();

    assert_eq!(
        error.to_string(),
        "the detached analyses expired when their callback returned"
    );
    assert!(std::error::Error::source(&error).is_none());
}
