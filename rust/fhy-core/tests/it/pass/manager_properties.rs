//! Property tests for `fhy_core::pass::PassManager`: run
//! order, fixpoint termination, and the analysis cache's preservation
//! contract.
//!
//! Public API only; nothing here reads process-global state.

use crate::support::pass_ir;

use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::sync::Arc;

use fhy_core::identifier::Identifier;
use fhy_core::pass::{
    CompilerPass, FixpointPassGroup, NodeHandle, NodeIdentity, PassContext, PassFailure,
    PassManager, PipelineRecord, PreservedAnalyses,
};
use pass_ir::{BoxIr, ClosurePass, DoubleAnalysis};
use proptest::prelude::*;

/// A toy IR holding a list of integers.
#[derive(Debug, Clone)]
struct ListIr(Arc<Vec<i64>>);

impl NodeHandle for ListIr {
    fn identity(&self) -> NodeIdentity {
        NodeIdentity::of_arc(&self.0)
    }
}

/// Decrements every positive element by one.
struct DecrementPositive;

impl CompilerPass<ListIr> for DecrementPositive {
    fn run(&mut self, ir: &ListIr, _cx: &mut PassContext<'_>) -> Result<ListIr, PassFailure> {
        let values =
            ir.0.iter()
                .map(|value| if *value > 0 { value - 1 } else { *value })
                .collect();
        Ok(ListIr(Arc::new(values)))
    }

    fn did_change(&mut self, input: &ListIr, output: &ListIr) -> Result<bool, PassFailure> {
        Ok(input.0 != output.0)
    }
}

/// One step of a pipeline over [`BoxIr`] in the cache property.
#[derive(Debug, Clone, Copy)]
enum CacheStep {
    /// Read [`DoubleAnalysis`] of the current node.
    Read,
    /// Add to the value, preserving no analysis when that changes it.
    Add(i64),
    /// Add to the value, declaring [`DoubleAnalysis`] preserved.
    AddPreservingDouble(i64),
    /// Return the current node itself.
    Keep,
}

/// Generate one cache step.
fn generate_cache_step() -> impl Strategy<Value = CacheStep> {
    prop_oneof![
        Just(CacheStep::Read),
        (-2_i64..=2).prop_map(CacheStep::Add),
        (-2_i64..=2).prop_map(CacheStep::AddPreservingDouble),
        Just(CacheStep::Keep),
    ]
}

/// Adds to the value and declares [`DoubleAnalysis`] preserved.
struct AddPreservingDouble(i64);

impl CompilerPass<BoxIr> for AddPreservingDouble {
    fn run(&mut self, ir: &BoxIr, _cx: &mut PassContext<'_>) -> Result<BoxIr, PassFailure> {
        Ok(ir.derive(ir.value() + self.0))
    }

    fn did_change(&mut self, input: &BoxIr, output: &BoxIr) -> Result<bool, PassFailure> {
        Ok(input.value() != output.value())
    }

    fn preserved_analyses(
        &mut self,
        _input: &BoxIr,
        _output: &BoxIr,
        _changed: bool,
    ) -> Result<PreservedAnalyses, PassFailure> {
        Ok(PreservedAnalyses::none().preserve::<DoubleAnalysis>())
    }
}

/// The values [`DoubleAnalysis`] reads return and the number of times it
/// runs, as the preservation contract prescribes for `steps` from `start`:
/// a result is recomputed only after a change that does not preserve it,
/// and is otherwise served from the cache, stale or not.
fn model_cache(start: i64, steps: &[CacheStep]) -> (Vec<i64>, usize) {
    let mut value = start;
    let mut cached = None;
    let mut reads = Vec::new();
    let mut runs = 0;
    for step in steps {
        match *step {
            CacheStep::Read => {
                let result = *cached.get_or_insert_with(|| {
                    runs += 1;
                    value * 2
                });
                reads.push(result);
            }
            CacheStep::Add(delta) => {
                value += delta;
                if delta != 0 {
                    cached = None;
                }
            }
            CacheStep::AddPreservingDouble(delta) => value += delta,
            CacheStep::Keep => {}
        }
    }
    (reads, runs)
}

proptest! {
    /// Test the pipeline runs its passes in the order they were added, as
    /// the passes themselves record and as the records list them.
    #[test]
    fn pass_manager_runs_passes_in_the_order_added(
        indices in prop::collection::vec(0_usize..5, 1..=5),
    ) {
        let recorder = RefCell::new(Vec::new());
        let expected: Vec<String> = indices
            .iter()
            .map(|index| format!("tests.prop.recording_{index}"))
            .collect();
        let mut manager = PassManager::new(Identifier::new("pipeline"));
        for name in &expected {
            let recorder = &recorder;
            let recorded_name = name.clone();
            manager.add_pass(ClosurePass::new(name, move |ir, _| {
                recorder.borrow_mut().push(recorded_name.clone());
                Ok(ir.clone())
            }));
        }

        let result = manager.run(&BoxIr::new(0)).expect("the run succeeds");
        let record_names: Vec<String> = result
            .records()
            .iter()
            .map(|record| match record {
                PipelineRecord::Pass(pass) => pass.pass_name().to_owned(),
                PipelineRecord::FixpointGroup(group) => format!("{group:?}"),
            })
            .collect();
        drop(manager);

        prop_assert_eq!(recorder.into_inner(), expected.clone());
        prop_assert_eq!(record_names, expected);
    }

    /// Test a group of a pass that decrements every positive element reaches
    /// the all-zero list in one changing iteration per unit of the largest
    /// element, plus one iteration that changes nothing.
    #[test]
    fn fixpoint_group_of_a_decrementing_pass_converges_after_the_largest_element(
        values in prop::collection::vec(0_i64..=5, 0..=5),
    ) {
        let largest = usize::try_from(values.iter().copied().max().unwrap_or(0))
            .expect("the values are non-negative");
        let mut group = FixpointPassGroup::new(Identifier::new("decrement-to-fixpoint"))
            .with_max_iterations(NonZeroUsize::new(largest + 2).expect("positive"));
        group.add_pass(DecrementPositive);
        let mut manager = PassManager::new(Identifier::new("pipeline"));
        manager.add_fixpoint_group(group);

        let result = manager
            .run(&ListIr(Arc::new(values.clone())))
            .expect("the group converges within its budget");

        let zeros = vec![0; values.len()];
        prop_assert_eq!(result.output().0.as_slice(), zeros.as_slice());
        let PipelineRecord::FixpointGroup(record) = &result.records()[0] else {
            panic!("expected a group record, got {:?}", result.records());
        };
        prop_assert!(record.is_converged());
        prop_assert_eq!(record.iterations(), largest + 1);
        prop_assert_eq!(record.iterations(), record.iteration_records().len());
    }

    /// Test the cache never serves a result across a change that did not
    /// preserve it, and serves every other repeated read from the cache.
    #[test]
    fn pass_manager_cache_honors_the_preservation_contract(
        start in -5_i64..=5,
        steps in prop::collection::vec(generate_cache_step(), 0..12),
    ) {
        let reads = RefCell::new(Vec::new());
        let mut manager = PassManager::new(Identifier::new("pipeline"));
        for (index, step) in steps.iter().enumerate() {
            let name = format!("tests.prop.step_{index}");
            match *step {
                CacheStep::Read => {
                    let reads = &reads;
                    manager.add_pass(ClosurePass::new(&name, move |ir, cx| {
                        reads.borrow_mut().push(*cx.analysis::<DoubleAnalysis, _>(ir));
                        Ok(ir.clone())
                    }));
                }
                CacheStep::Add(delta) => {
                    manager.add_pass(ClosurePass::new(&name, move |ir, _| {
                        Ok(ir.derive(ir.value() + delta))
                    }));
                }
                CacheStep::AddPreservingDouble(delta) => {
                    manager.add_pass(AddPreservingDouble(delta));
                }
                CacheStep::Keep => {
                    manager.add_pass(ClosurePass::new(&name, |ir, _| Ok(ir.clone())));
                }
            }
        }
        let input = BoxIr::new(start);
        let (expected_reads, expected_runs) = model_cache(start, &steps);

        manager.run(&input).expect("the run succeeds");
        drop(manager);

        prop_assert_eq!(reads.into_inner(), expected_reads);
        prop_assert_eq!(input.double_runs(), expected_runs);
    }
}
