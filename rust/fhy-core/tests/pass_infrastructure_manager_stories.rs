//! Tests for `fhy_core::pass_infrastructure::PassManager` and
//! `FixpointPassGroup`: pipeline order, records, the per-run analysis cache,
//! fixpoint iteration, and opt-in verification.
//!
//! Public API only. Analyses count their runs in counters the toy IR nodes
//! carry, so nothing here reads process-global state.

#[path = "common/pass_ir.rs"]
pub mod pass_ir;

use std::cell::{Cell, RefCell};
use std::num::NonZeroUsize;

use fhy_core::diagnostic::DiagnosticLevel;
use fhy_core::identifier::{HasIdentifier, Identifier};
use fhy_core::pass_infrastructure::{
    CompilerPass, ExecutePass, FixpointGroupRecord, FixpointPassGroup, PassContext, PassFailure,
    PassManager, PassRunRecord, PipelineRecord, PreservedAnalyses, ValidationManager,
};
use pass_ir::{
    BoxIr, ClosurePass, DoubleAnalysis, ParityAnalysis, build_add_pass, build_identity_pass,
};

// =============================================================================
// Helpers
// =============================================================================

/// Return the pass record of a pipeline record, failing the test otherwise.
fn expect_pass_record(record: &PipelineRecord) -> &PassRunRecord {
    match record {
        PipelineRecord::Pass(pass) => pass,
        PipelineRecord::FixpointGroup(group) => panic!("expected a pass record, got {group:?}"),
    }
}

/// Return the group record of a pipeline record, failing the test otherwise.
fn expect_group_record(record: &PipelineRecord) -> &FixpointGroupRecord {
    match record {
        PipelineRecord::FixpointGroup(group) => group,
        PipelineRecord::Pass(pass) => panic!("expected a group record, got {pass:?}"),
    }
}

/// Return the pass names of pipeline records that are pass records.
fn collect_pass_names(records: &[PipelineRecord]) -> Vec<&str> {
    records
        .iter()
        .map(|record| expect_pass_record(record).pass_name())
        .collect()
}

/// Build the pass `name` that decrements a positive value by one.
fn build_decrement_to_zero_pass(name: &str) -> ClosurePass<'static> {
    ClosurePass::new(name, |ir, _| Ok(ir.derive((ir.value() - 1).max(0))))
}

/// Build the pass `name` that flips a value between zero and one.
fn build_flip_pass(name: &str) -> ClosurePass<'static> {
    ClosurePass::new(name, |ir, _| Ok(ir.derive(1 - ir.value())))
}

/// Build the fixpoint group `name` with a budget of `max_iterations`.
fn build_group<'p>(name: &str, max_iterations: usize) -> FixpointPassGroup<'p, BoxIr> {
    FixpointPassGroup::new(Identifier::new(name))
        .with_max_iterations(NonZeroUsize::new(max_iterations).expect("the budget is positive"))
}

/// Reports an error for every negative value.
struct NegativeValueCheck;

impl CompilerPass<BoxIr, ()> for NegativeValueCheck {
    fn run(&mut self, ir: &BoxIr, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        if ir.value() < 0 {
            cx.report_text(
                DiagnosticLevel::Error,
                format!("negative value: {}", ir.value()),
                None,
            );
        }
        Ok(())
    }

    fn did_change(&mut self, _input: &BoxIr, _output: &()) -> Result<bool, PassFailure> {
        Ok(false)
    }
}

/// Counts the nodes it validates and reports nothing.
#[derive(Default)]
struct CountingCheck {
    invocations: usize,
}

impl CompilerPass<BoxIr, ()> for CountingCheck {
    fn run(&mut self, _ir: &BoxIr, _cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        self.invocations += 1;
        Ok(())
    }

    fn did_change(&mut self, _input: &BoxIr, _output: &()) -> Result<bool, PassFailure> {
        Ok(false)
    }
}

/// Build the verifier that rejects negative values.
fn build_negative_value_verifier<'p>() -> ValidationManager<'p, BoxIr> {
    let mut verifier = ValidationManager::new(Identifier::new("verifier"));
    verifier.add(NegativeValueCheck);
    verifier
}

// =============================================================================
// Pipeline order and records
// =============================================================================

/// Test the pipeline runs its passes in the order they were added.
#[test]
fn pass_manager_runs_passes_in_insertion_order() {
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(build_add_pass("tests.pm.add_one", 1));
    manager.add_pass(ClosurePass::new("tests.pm.double", |ir, _| {
        Ok(ir.derive(ir.value() * 2))
    }));

    let result = manager.run(&BoxIr::new(3)).expect("the run succeeds");

    assert_eq!(result.output().value(), 8);
    assert_eq!(
        collect_pass_names(result.records()),
        ["tests.pm.add_one", "tests.pm.double"]
    );
}

/// Test a pipeline without items returns its input node.
#[test]
fn pass_manager_without_items_returns_the_input() {
    let mut manager = PassManager::new(Identifier::new("empty"));
    let input = BoxIr::new(3);

    let result = manager.run(&input).expect("the run succeeds");

    assert!(result.output().is_same_node(&input));
    assert!(result.records().is_empty());
}

/// Test a pass record holds the run's name, change flag, diagnostics, and
/// preserved analyses.
#[test]
fn pass_run_record_holds_the_outcome_of_the_run() {
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(ClosurePass::new("tests.pm.warn_and_add", |ir, cx| {
        cx.report_text(DiagnosticLevel::Warning, "careful", None);
        Ok(ir.derive(ir.value() + 1))
    }));
    manager.add_pass(build_identity_pass("tests.pm.identity"));

    let result = manager.run(&BoxIr::new(0)).expect("the run succeeds");

    let changing = expect_pass_record(&result.records()[0]);
    assert_eq!(changing.pass_name(), "tests.pm.warn_and_add");
    assert!(changing.is_changed());
    assert_eq!(changing.diagnostics().len(), 1);
    assert_eq!(changing.diagnostics()[0].message_text(), "careful");
    assert_eq!(changing.preserved_analyses(), &PreservedAnalyses::none());
    let unchanged = expect_pass_record(&result.records()[1]);
    assert!(!unchanged.is_changed());
    assert!(unchanged.diagnostics().is_empty());
    assert_eq!(unchanged.preserved_analyses(), &PreservedAnalyses::all());
}

/// Changes the value and preserves only [`DoubleAnalysis`].
struct PreserveDoubleOnly;

impl CompilerPass<BoxIr> for PreserveDoubleOnly {
    fn run(&mut self, ir: &BoxIr, _cx: &mut PassContext<'_>) -> Result<BoxIr, PassFailure> {
        Ok(ir.derive(ir.value() + 1))
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

/// Test a pass record holds exactly the analyses the pass preserved.
#[test]
fn pass_run_record_holds_a_specific_preservation_set() {
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(PreserveDoubleOnly);

    let result = manager.run(&BoxIr::new(0)).expect("the run succeeds");

    let record = expect_pass_record(&result.records()[0]);
    assert_eq!(record.pass_name(), "PreserveDoubleOnly");
    assert!(!record.preserved_analyses().preserves_all());
    assert!(record.preserved_analyses().is_preserved::<DoubleAnalysis>());
    assert!(!record.preserved_analyses().is_preserved::<ParityAnalysis>());
}

/// Test the pipeline stops at the first failing pass and returns its error.
#[test]
fn pass_manager_stops_at_the_first_failing_pass() {
    let later_ran = Cell::new(false);
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(build_add_pass("tests.pm.before_failure", 1));
    manager.add_pass(ClosurePass::new("tests.pm.failing", |_, _| {
        Err("broken".into())
    }));
    manager.add_pass(ClosurePass::new("tests.pm.after_failure", |ir, _| {
        later_ran.set(true);
        Ok(ir.clone())
    }));

    let error = manager
        .run(&BoxIr::new(0))
        .expect_err("the second pass fails");
    drop(manager);

    assert_eq!(
        error.to_string(),
        "Pass \"tests.pm.failing\" failed run with broken"
    );
    assert_eq!(error.pass_name(), Some("tests.pm.failing"));
    assert!(!later_ran.get());
}

/// Test the pipeline's name is its identifier.
#[test]
fn pass_manager_name_is_its_identifier() {
    let name = Identifier::new("named-pipeline");

    let manager: PassManager<'_, BoxIr> = PassManager::new(name.clone());

    assert_eq!(manager.name(), &name);
    assert_eq!(manager.identifier(), &name);
}

/// Test a default pipeline is named `pipeline` and holds no items.
#[test]
fn pass_manager_default_is_an_empty_pipeline_named_pipeline() {
    let input = BoxIr::new(7);

    let mut manager: PassManager<'_, BoxIr> = PassManager::default();
    let result = manager.run(&input).expect("an empty pipeline cannot fail");

    assert_eq!(manager.name().name_hint(), "pipeline");
    assert!(result.records().is_empty());
}

// =============================================================================
// Analyses under a manager
// =============================================================================

/// Test an analysis requested twice by one managed pass runs once.
#[test]
fn pass_context_analysis_is_cached_within_a_managed_pass() {
    let mut observed = Vec::new();
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(ClosurePass::new("tests.pm.twice_read", |ir, cx| {
        observed.push(*cx.analysis::<DoubleAnalysis, _>(ir));
        observed.push(*cx.analysis::<DoubleAnalysis, _>(ir));
        Ok(ir.clone())
    }));
    let input = BoxIr::new(5);

    manager.run(&input).expect("the run succeeds");
    drop(manager);

    assert_eq!(observed, [10, 10]);
    assert_eq!(input.double_runs(), 1);
}

/// Test an analysis computed before an unchanged pass is reused after it.
#[test]
fn pass_context_analysis_is_reused_across_an_unchanged_pass() {
    let mut observed = Vec::new();
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(ClosurePass::new("tests.pm.compute", |ir, cx| {
        cx.analysis::<DoubleAnalysis, _>(ir);
        Ok(ir.clone())
    }));
    manager.add_pass(ClosurePass::new("tests.pm.read_again", |ir, cx| {
        observed.push(*cx.analysis::<DoubleAnalysis, _>(ir));
        Ok(ir.clone())
    }));
    let input = BoxIr::new(5);

    manager.run(&input).expect("the run succeeds");
    drop(manager);

    assert_eq!(observed, [10]);
    assert_eq!(input.double_runs(), 1);
}

/// Test an unchanged pass that returns a new node hands the cached results
/// on to that node.
#[test]
fn pass_context_analysis_follows_an_unchanged_pass_to_its_new_node() {
    let mut observed = Vec::new();
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(ClosurePass::new("tests.pm.compute_then_copy", |ir, cx| {
        cx.analysis::<DoubleAnalysis, _>(ir);
        Ok(ir.derive(ir.value()))
    }));
    manager.add_pass(ClosurePass::new("tests.pm.read_copy", |ir, cx| {
        observed.push(*cx.analysis::<DoubleAnalysis, _>(ir));
        Ok(ir.clone())
    }));
    let input = BoxIr::new(5);

    manager.run(&input).expect("the run succeeds");
    drop(manager);

    assert_eq!(observed, [10]);
    assert_eq!(input.double_runs(), 1);
}

/// Test an analysis is recomputed after a pass that changed the IR without
/// preserving it.
#[test]
fn pass_context_analysis_recomputes_after_a_changing_pass() {
    let mut observed = Vec::new();
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(ClosurePass::new("tests.pm.seed", |ir, cx| {
        cx.analysis::<DoubleAnalysis, _>(ir);
        Ok(ir.clone())
    }));
    manager.add_pass(build_add_pass("tests.pm.mutate", 1));
    manager.add_pass(ClosurePass::new("tests.pm.reread", |ir, cx| {
        observed.push(*cx.analysis::<DoubleAnalysis, _>(ir));
        Ok(ir.clone())
    }));
    let input = BoxIr::new(5);

    manager.run(&input).expect("the run succeeds");
    drop(manager);

    assert_eq!(observed, [12]);
    assert_eq!(input.double_runs(), 2);
}

/// Test a changing pass carries over exactly the analyses it preserves.
#[test]
fn pass_manager_carries_only_preserved_analyses_to_a_changed_output() {
    let mut observed = Vec::new();
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(ClosurePass::new("tests.pm.seed_both", |ir, cx| {
        cx.analysis::<DoubleAnalysis, _>(ir);
        cx.analysis::<ParityAnalysis, _>(ir);
        Ok(ir.clone())
    }));
    manager.add_pass(PreserveDoubleOnly);
    manager.add_pass(ClosurePass::new("tests.pm.read_both", |ir, cx| {
        observed.push(*cx.analysis::<DoubleAnalysis, _>(ir));
        observed.push(*cx.analysis::<ParityAnalysis, _>(ir));
        Ok(ir.clone())
    }));
    let input = BoxIr::new(2);

    manager.run(&input).expect("the run succeeds");
    drop(manager);

    assert_eq!(observed, [4, 1]);
    assert_eq!(input.double_runs(), 1);
    assert_eq!(input.parity_runs(), 2);
}

/// Test an analysis of a node other than the pass's input is cached too.
#[test]
fn pass_context_analysis_caches_a_node_other_than_the_input() {
    let side = BoxIr::new(21);
    let observed = RefCell::new(Vec::new());
    let read_side = |ir: &BoxIr, cx: &mut PassContext<'_>| {
        observed
            .borrow_mut()
            .push(*cx.analysis::<DoubleAnalysis, _>(&side));
        Ok(ir.clone())
    };
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(ClosurePass::new("tests.pm.read_side", read_side));
    manager.add_pass(build_add_pass("tests.pm.change_input", 1));
    manager.add_pass(ClosurePass::new("tests.pm.read_side_again", read_side));

    manager.run(&BoxIr::new(0)).expect("the run succeeds");
    drop(manager);

    assert_eq!(observed.into_inner(), [42, 42]);
    assert_eq!(side.double_runs(), 1);
}

/// Test passes inside a fixpoint group see analyses of their current input.
#[test]
fn pass_context_analysis_serves_passes_inside_a_fixpoint_group() {
    let mut observed = Vec::new();
    let mut group = build_group("read-then-decrement", 10);
    group.add_pass(ClosurePass::new("tests.pm.fixpoint_reader", |ir, cx| {
        observed.push(*cx.analysis::<DoubleAnalysis, _>(ir));
        Ok(ir.derive((ir.value() - 1).max(0)))
    }));
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_fixpoint_group(group);

    manager.run(&BoxIr::new(2)).expect("the run succeeds");
    drop(manager);

    assert_eq!(observed, [4, 2, 0]);
}

/// Test every run starts with an empty cache.
#[test]
fn pass_manager_starts_every_run_with_an_empty_cache() {
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(ClosurePass::new("tests.pm.read", |ir, cx| {
        cx.analysis::<DoubleAnalysis, _>(ir);
        Ok(ir.clone())
    }));
    let input = BoxIr::new(1);

    manager.run(&input).expect("the first run succeeds");
    manager.run(&input).expect("the second run succeeds");

    assert_eq!(input.double_runs(), 2);
}

/// Test a pass kept by its caller runs standalone after a managed run and
/// computes analyses afresh there.
#[test]
fn borrowed_pass_runs_standalone_after_a_managed_run() {
    let mut pass = ClosurePass::new("tests.pm.state_check", |ir, cx| {
        cx.analysis::<DoubleAnalysis, _>(ir);
        Ok(ir.clone())
    });
    let managed_input = BoxIr::new(5);
    let standalone_input = BoxIr::new(7);

    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(&mut pass);
    manager
        .run(&managed_input)
        .expect("the managed run succeeds");
    drop(manager);
    pass.execute(&standalone_input).expect("the run succeeds");
    pass.execute(&standalone_input).expect("the run succeeds");

    assert_eq!(managed_input.double_runs(), 1);
    assert_eq!(standalone_input.double_runs(), 2);
}

/// Test a run holds no handle to any node once it returns.
#[test]
fn pass_manager_run_releases_every_cached_handle() {
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(ClosurePass::new("tests.pm.read_input", |ir, cx| {
        cx.analysis::<DoubleAnalysis, _>(ir);
        Ok(ir.clone())
    }));
    manager.add_pass(PreserveDoubleOnly);
    manager.add_pass(ClosurePass::new("tests.pm.read_output", |ir, cx| {
        cx.analysis::<ParityAnalysis, _>(ir);
        Ok(ir.clone())
    }));
    manager.set_verifier(build_negative_value_verifier());
    let input = BoxIr::new(1);

    let result = manager.run(&input).expect("the run succeeds");

    assert_eq!(input.handle_count(), 1);
    assert_eq!(result.output().handle_count(), 1);
}

// =============================================================================
// Fixpoint groups
// =============================================================================

/// Test a fixpoint group iterates until an iteration changes nothing and
/// records every iteration.
#[test]
fn fixpoint_group_converges_and_records_each_iteration() {
    let mut group = build_group("decrement-group", 10);
    group.add_pass(build_decrement_to_zero_pass("tests.pm.decrement"));
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_fixpoint_group(group);

    let result = manager.run(&BoxIr::new(3)).expect("the group converges");

    assert_eq!(result.output().value(), 0);
    assert_eq!(result.records().len(), 1);
    let record = expect_group_record(&result.records()[0]);
    assert_eq!(record.group_name().name_hint(), "decrement-group");
    assert!(record.is_converged());
    assert_eq!(record.iterations(), 4);
    assert_eq!(record.iteration_records().len(), 4);
    let iterations: Vec<_> = record
        .iteration_records()
        .iter()
        .map(|iteration| (iteration.iteration(), iteration.is_changed()))
        .collect();
    assert_eq!(iterations, [(1, true), (2, true), (3, true), (4, false)]);
    for iteration in record.iteration_records() {
        let names: Vec<_> = iteration
            .pass_runs()
            .iter()
            .map(PassRunRecord::pass_name)
            .collect();
        assert_eq!(names, ["tests.pm.decrement"]);
    }
}

/// Test a group runs all its passes, in order, in every iteration.
#[test]
fn fixpoint_group_runs_its_passes_in_order_each_iteration() {
    let mut group = build_group("two-pass-group", 10);
    group.add_pass(build_decrement_to_zero_pass("tests.pm.first_decrement"));
    group.add_pass(build_identity_pass("tests.pm.second_identity"));
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_fixpoint_group(group);

    let result = manager.run(&BoxIr::new(1)).expect("the group converges");

    let record = expect_group_record(&result.records()[0]);
    let runs: Vec<Vec<(&str, bool)>> = record
        .iteration_records()
        .iter()
        .map(|iteration| {
            iteration
                .pass_runs()
                .iter()
                .map(|run| (run.pass_name(), run.is_changed()))
                .collect()
        })
        .collect();
    assert_eq!(
        runs,
        [
            vec![
                ("tests.pm.first_decrement", true),
                ("tests.pm.second_identity", false)
            ],
            vec![
                ("tests.pm.first_decrement", false),
                ("tests.pm.second_identity", false)
            ],
        ]
    );
}

/// Test a group that fails on non-convergence fails the run when its budget
/// runs out.
#[test]
fn fixpoint_group_fails_the_run_when_it_does_not_converge() {
    let mut group = build_group("flip-group", 3);
    group.add_pass(build_flip_pass("tests.pm.flip"));
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_fixpoint_group(group);

    let error = manager
        .run(&BoxIr::new(0))
        .expect_err("the group never converges");

    assert_eq!(
        error.to_string(),
        "Fixpoint group \"flip-group\" did not converge in 3 iterations."
    );
    assert!(error.is_non_convergence());
    assert!(error.is_execution_failure());
    assert!(!error.is_validation_failure());
    assert_eq!(error.pass_name(), None);
    assert_eq!(error.failed_hook(), None);
    assert!(error.diagnostics().is_empty());
}

/// Test a group allowed not to converge hands on the IR of its last
/// iteration and records that it did not converge.
#[test]
fn fixpoint_group_hands_on_the_last_ir_when_allowed_not_to_converge() {
    let mut group = build_group("lenient-flip-group", 3).with_fail_on_non_convergence(false);
    group.add_pass(build_flip_pass("tests.pm.lenient_flip"));
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_fixpoint_group(group);
    manager.add_pass(build_add_pass("tests.pm.after_group", 10));

    let result = manager
        .run(&BoxIr::new(0))
        .expect("the group may not converge");

    assert_eq!(result.output().value(), 11);
    let record = expect_group_record(&result.records()[0]);
    assert!(!record.is_converged());
    assert_eq!(record.iterations(), 3);
    assert_eq!(
        expect_pass_record(&result.records()[1]).pass_name(),
        "tests.pm.after_group"
    );
}

/// Test a group with a budget of one converges only if its first iteration
/// changes nothing.
#[test]
fn fixpoint_group_with_a_budget_of_one_needs_an_unchanged_first_iteration() {
    let mut converging = build_group("one-shot-identity", 1);
    converging.add_pass(build_identity_pass("tests.pm.one_shot_identity"));
    let mut failing = build_group("one-shot-add", 1);
    failing.add_pass(build_add_pass("tests.pm.one_shot_add", 1));
    let mut converging_manager = PassManager::new(Identifier::new("converging"));
    converging_manager.add_fixpoint_group(converging);
    let mut failing_manager = PassManager::new(Identifier::new("failing"));
    failing_manager.add_fixpoint_group(failing);

    let converged = converging_manager.run(&BoxIr::new(0));
    let error = failing_manager
        .run(&BoxIr::new(0))
        .expect_err("one changing iteration exhausts the budget");

    let result = converged.expect("an unchanged iteration converges");
    assert_eq!(expect_group_record(&result.records()[0]).iterations(), 1);
    assert_eq!(
        error.to_string(),
        "Fixpoint group \"one-shot-add\" did not converge in 1 iterations."
    );
}

/// Test a group without passes converges in its first iteration.
#[test]
fn fixpoint_group_without_passes_converges_immediately() {
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_fixpoint_group(build_group("empty-group", 5));

    let result = manager.run(&BoxIr::new(4)).expect("the group converges");

    let record = expect_group_record(&result.records()[0]);
    assert!(record.is_converged());
    assert_eq!(record.iterations(), 1);
    assert!(record.iteration_records()[0].pass_runs().is_empty());
    assert_eq!(result.output().value(), 4);
}

/// Test a pipeline mixes passes and groups, feeding each item the previous
/// item's IR.
#[test]
fn pass_manager_runs_passes_and_groups_in_order() {
    let mut group = build_group("middle-group", 10);
    group.add_pass(build_decrement_to_zero_pass("tests.pm.middle_decrement"));
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(build_add_pass("tests.pm.leading_add", 2));
    manager.add_fixpoint_group(group);
    manager.add_pass(build_add_pass("tests.pm.trailing_add", 5));

    let result = manager.run(&BoxIr::new(1)).expect("the run succeeds");

    assert_eq!(result.output().value(), 5);
    assert_eq!(
        expect_pass_record(&result.records()[0]).pass_name(),
        "tests.pm.leading_add"
    );
    assert_eq!(expect_group_record(&result.records()[1]).iterations(), 4);
    assert_eq!(
        expect_pass_record(&result.records()[2]).pass_name(),
        "tests.pm.trailing_add"
    );
}

/// Test a new group has a budget of ten and fails on non-convergence, and
/// the builders change both.
#[test]
fn fixpoint_pass_group_new_uses_the_default_configuration() {
    let name = Identifier::new("configured");

    let default: FixpointPassGroup<'_, BoxIr> = FixpointPassGroup::new(name.clone());
    let configured: FixpointPassGroup<'_, BoxIr> = FixpointPassGroup::new(name.clone())
        .with_max_iterations(NonZeroUsize::new(2).expect("positive"))
        .with_fail_on_non_convergence(false);

    assert_eq!(default.name(), &name);
    assert_eq!(default.identifier(), &name);
    assert_eq!(default.max_iterations().get(), 10);
    assert!(default.fails_on_non_convergence());
    assert_eq!(configured.max_iterations().get(), 2);
    assert!(!configured.fails_on_non_convergence());
}

// =============================================================================
// Verification
// =============================================================================

/// Test the verifier rejects invalid pipeline input before any pass runs,
/// blaming the first pass.
#[test]
fn pass_manager_verifier_rejects_invalid_input_blaming_the_first_pass() {
    let first_ran = Cell::new(false);
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(ClosurePass::new("tests.pm.first", |ir, _| {
        first_ran.set(true);
        Ok(ir.clone())
    }));
    manager.set_verifier(build_negative_value_verifier());

    let error = manager
        .run(&BoxIr::new(-1))
        .expect_err("the input is invalid");
    drop(manager);

    let message = "Pass \"tests.pm.first\" rejected input IR: verification reported 1 error(s).";
    assert_eq!(error.to_string(), message);
    assert!(error.is_validation_failure());
    assert_eq!(error.pass_name(), Some("tests.pm.first"));
    assert_eq!(error.failed_hook(), None);
    let report = error.verification_report().expect("the report is attached");
    let errors: Vec<_> = report
        .errors()
        .map(fhy_core::diagnostic::Diagnostic::message_text)
        .collect();
    assert_eq!(errors, ["negative value: -1"]);
    assert_eq!(error.diagnostics().len(), 1);
    let diagnostic = &error.diagnostics()[0];
    assert_eq!(diagnostic.level(), DiagnosticLevel::Error);
    assert_eq!(diagnostic.message_text(), message);
    assert_eq!(diagnostic.source(), "tests.pm.first");
    assert_eq!(
        diagnostic.detail(),
        Some("[ERROR] NegativeValueCheck: negative value: -1")
    );
    assert!(!first_ran.get());
}

/// Test input verification blames the first pass of a leading fixpoint
/// group.
#[test]
fn pass_manager_verifier_blames_the_first_pass_of_a_leading_group() {
    let mut group = build_group("leading-group", 3);
    group.add_pass(build_identity_pass("tests.pm.group_first"));
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_fixpoint_group(build_group("empty-leading-group", 3));
    manager.add_fixpoint_group(group);
    manager.add_pass(build_identity_pass("tests.pm.after_group"));
    manager.set_verifier(build_negative_value_verifier());

    let error = manager
        .run(&BoxIr::new(-2))
        .expect_err("the input is invalid");

    assert_eq!(error.pass_name(), Some("tests.pm.group_first"));
}

/// Test a pipeline without passes verifies nothing.
#[test]
fn pass_manager_verifier_skips_a_pipeline_without_passes() {
    let mut check = CountingCheck::default();
    let mut verifier = ValidationManager::new(Identifier::new("verifier"));
    verifier.add(&mut check);
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_fixpoint_group(build_group("empty-group", 2));
    manager.set_verifier(verifier);

    let result = manager.run(&BoxIr::new(-1)).expect("nothing is verified");
    drop(manager);

    assert_eq!(result.output().value(), -1);
    assert_eq!(check.invocations, 0);
}

/// Test verification of unchanged IR happens once per run.
#[test]
fn pass_manager_verifier_validates_unchanged_ir_once() {
    let mut check = CountingCheck::default();
    let mut verifier = ValidationManager::new(Identifier::new("verifier"));
    verifier.add(&mut check);
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(build_identity_pass("tests.pm.identity_a"));
    manager.add_pass(build_identity_pass("tests.pm.identity_b"));
    manager.set_verifier(verifier);

    manager.run(&BoxIr::new(0)).expect("the run succeeds");
    drop(manager);

    assert_eq!(check.invocations, 1);
}

/// Test every changed output is verified.
#[test]
fn pass_manager_verifier_validates_each_changed_output() {
    let mut check = CountingCheck::default();
    let mut verifier = ValidationManager::new(Identifier::new("verifier"));
    verifier.add(&mut check);
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(build_add_pass("tests.pm.increment", 1));
    manager.add_pass(build_identity_pass("tests.pm.keep"));
    manager.add_pass(build_add_pass("tests.pm.increment_again", 1));
    manager.set_verifier(verifier);

    manager.run(&BoxIr::new(0)).expect("the run succeeds");
    drop(manager);

    assert_eq!(check.invocations, 3);
}

/// Test a corrupting pass in the middle of a pipeline is blamed for the
/// invalid output it produced, with its own diagnostics kept, and later
/// passes do not run.
#[test]
fn pass_manager_verifier_blames_the_pass_that_produced_invalid_output() {
    let last_ran = Cell::new(false);
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(build_add_pass("tests.pm.first_clean", 1));
    manager.add_pass(ClosurePass::new("tests.pm.corrupt", |ir, cx| {
        cx.report_text(DiagnosticLevel::Info, "rewriting", None);
        Ok(ir.derive(-100))
    }));
    manager.add_pass(ClosurePass::new("tests.pm.last_clean", |ir, _| {
        last_ran.set(true);
        Ok(ir.derive(ir.value() + 1))
    }));
    manager.set_verifier(build_negative_value_verifier());

    let error = manager
        .run(&BoxIr::new(0))
        .expect_err("the corrupt output is invalid");
    drop(manager);

    let message =
        "Pass \"tests.pm.corrupt\" produced invalid output IR: verification reported 1 error(s).";
    assert_eq!(error.to_string(), message);
    assert!(error.is_validation_failure());
    assert_eq!(error.pass_name(), Some("tests.pm.corrupt"));
    let report = error.verification_report().expect("the report is attached");
    let errors: Vec<_> = report
        .errors()
        .map(fhy_core::diagnostic::Diagnostic::message_text)
        .collect();
    assert_eq!(errors, ["negative value: -100"]);
    let diagnostics: Vec<_> = error
        .diagnostics()
        .iter()
        .map(|d| (d.level(), d.message_text(), d.detail()))
        .collect();
    assert_eq!(
        diagnostics,
        [
            (DiagnosticLevel::Info, "rewriting", None),
            (
                DiagnosticLevel::Error,
                message,
                Some("[ERROR] NegativeValueCheck: negative value: -100")
            ),
        ]
    );
    assert!(!last_ran.get());
}

/// Changes the value to -5 but reports no change.
struct SilentCorruption;

impl CompilerPass<BoxIr> for SilentCorruption {
    fn run(&mut self, ir: &BoxIr, _cx: &mut PassContext<'_>) -> Result<BoxIr, PassFailure> {
        Ok(ir.derive(-5))
    }

    fn did_change(&mut self, _input: &BoxIr, _output: &BoxIr) -> Result<bool, PassFailure> {
        Ok(false)
    }
}

/// Test an output its pass reports as unchanged is not verified.
#[test]
fn pass_manager_verifier_skips_an_output_reported_unchanged() {
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(SilentCorruption);
    manager.set_verifier(build_negative_value_verifier());

    let result = manager.run(&BoxIr::new(1)).expect("nothing changed");

    assert_eq!(result.output().value(), -5);
}

/// Test outputs changed inside a fixpoint group are verified, blaming the
/// group's pass.
#[test]
fn pass_manager_verifier_validates_changed_outputs_inside_a_group() {
    let mut group = build_group("descending-group", 10);
    group.add_pass(build_add_pass("tests.pm.descend", -1));
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_fixpoint_group(group);
    manager.set_verifier(build_negative_value_verifier());

    let error = manager
        .run(&BoxIr::new(2))
        .expect_err("the third iteration goes negative");

    assert_eq!(
        error.to_string(),
        "Pass \"tests.pm.descend\" produced invalid output IR: verification reported 1 error(s)."
    );
}

/// Test a verifier without validators accepts any IR.
#[test]
fn pass_manager_with_an_empty_verifier_accepts_any_ir() {
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(build_add_pass("tests.pm.go_negative", -10));
    manager.set_verifier(ValidationManager::new(Identifier::new("empty-verifier")));

    let result = manager.run(&BoxIr::new(-1)).expect("nothing is rejected");

    assert_eq!(result.output().value(), -11);
}

/// Test a pipeline without a verifier runs over invalid IR.
#[test]
fn pass_manager_without_a_verifier_runs_on_invalid_ir() {
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(build_add_pass("tests.pm.unverified", -10));

    let result = manager.run(&BoxIr::new(-1)).expect("nothing is verified");

    assert_eq!(result.output().value(), -11);
}

/// Test setting a verifier replaces the one set before.
#[test]
fn set_verifier_replaces_the_previous_verifier() {
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(build_identity_pass("tests.pm.replaced_verifier"));
    manager.set_verifier(build_negative_value_verifier());
    manager.set_verifier(ValidationManager::new(Identifier::new("lenient")));

    let result = manager
        .run(&BoxIr::new(-3))
        .expect("the lenient verifier accepts");

    assert_eq!(result.output().value(), -3);
}
