//! Tests for the expression passes: `RewriteRuleApplier`, which applies
//! rewrite rules bottom-up, `ExpressionPrettyFormatter`, which formats an
//! expression as text, and `register_expression_passes`, which registers the
//! rule applier by name. Each runs standalone, and the rule applier also in
//! a pipeline and a fixpoint group.
//!
//! Public API only.

use crate::support::expression as expression_support;
use crate::support::pattern as pattern_support;

use std::any::TypeId;

use expression_support::{
    build_call_or_panic, build_doubling_dag, build_identifier, build_literal,
    build_piecewise_or_panic,
};
use fhy_core::diagnostic::{Diagnostic, DiagnosticLevel};
use fhy_core::expr::pattern::{
    CallbackError, FiredRule, MatchBindings, Pattern, RewriteError, RewriteRule,
    RewriteRuleApplier, apply_rewrite_rules,
};
use fhy_core::expr::{
    BinaryOperation, Expression, ExpressionPrettyFormatter, FormatOptions, IdentifierStyle,
    Notation, PiecewiseError, RebuildError, register_expression_passes,
};
use fhy_core::identifier::Identifier;
use fhy_core::pass::{
    CompilerPass, ExecutePass, FailureClass, FixpointIterationRecord, FixpointPassGroup, PassError,
    PassErrorKind, PassHook, PassManager, PassRegistry, PipelineRecord, PreservedAnalyses,
};
use pattern_support::{
    ProbeError, build_capture, build_literal_pattern, build_x_plus_zero_rule,
    build_x_times_one_rule, expect_probe_error, rewrite_to_capture, rewrite_to_literal,
};
use rstest::rstest;

/// The name the rule applier is registered under.
const RULE_APPLIER_NAME: &str = "fhy_core.symbolic.expression.apply_rewrite_rules";

/// The description the rule applier is registered with.
const RULE_APPLIER_DESCRIPTION: &str =
    "Apply a sequence of rewrite rules bottom-up over an expression tree.";

/// Return `x + 0` for the reference `x`.
fn build_plus_zero(x: &Expression) -> Expression {
    Expression::new_binary(BinaryOperation::Add, x, 0)
}

/// Return the `(rule index, name)` of every firing in `fired`.
fn describe_fired(fired: &[FiredRule]) -> Vec<(usize, Option<&str>)> {
    fired
        .iter()
        .map(|firing| (firing.rule_index(), firing.name()))
        .collect()
}

/// Return the rule `0 + x -> x + 0`, named so, whose output the rule
/// `x + 0 -> x` rewrites only in a later run.
fn build_move_zero_right_rule() -> RewriteRule {
    RewriteRule::new(
        Pattern::binary(
            Some(BinaryOperation::Add),
            build_literal_pattern(0),
            build_capture("x"),
        ),
        |bindings: &MatchBindings| {
            let x = bindings
                .get("x")
                .ok_or_else(|| CallbackError::new(ProbeError("unbound capture")))?;
            Ok(Expression::new_binary(BinaryOperation::Add, x, 0))
        },
    )
    .with_name("0 + x -> x + 0")
}

/// Return the rewrite error a failed rule-applier run carries as its
/// source.
fn expect_rewrite_error(error: &PassError) -> &RewriteError {
    std::error::Error::source(error)
        .and_then(|source| source.downcast_ref::<RewriteError>())
        .unwrap_or_else(|| panic!("expected a rewrite error as the source of {error:?}"))
}

/// Return `{1 if c; 0 otherwise}` over the reference `c`, and the rule
/// rewriting `c` to the literal `1`, named `c -> 1`, which the piecewise
/// refuses as a case condition.
fn build_refused_condition(c: &Identifier) -> (Expression, RewriteRule) {
    let expression = build_piecewise_or_panic([(Expression::from(c.clone()), 1)], 0);
    let rule = RewriteRule::new(Pattern::identifier(Some(c.clone())), rewrite_to_literal(1))
        .with_name("c -> 1");
    (expression, rule)
}

// =============================================================================
// RewriteRuleApplier
// =============================================================================

/// Test the applier keeps the rules it was built with, in order.
#[test]
fn rewrite_rule_applier_rules_are_the_rules_it_was_built_with() {
    let applier = RewriteRuleApplier::new([build_x_plus_zero_rule(), build_x_times_one_rule()]);

    let names: Vec<Option<&str>> = applier.rules().iter().map(RewriteRule::name).collect();

    assert_eq!(names, [Some("x + 0 -> x"), Some("x * 1 -> x")]);
}

/// Test executing the applier gives the output, change flag, and firings
/// `apply_rewrite_rules` gives for the same rules.
#[test]
fn rewrite_rule_applier_execute_matches_apply_rewrite_rules() {
    let (_, a) = build_identifier("a");
    let expression = Expression::new_binary(
        BinaryOperation::Multiply,
        build_plus_zero(&a),
        build_plus_zero(&build_literal(2)),
    ) * 1;
    let rules = [build_x_plus_zero_rule(), build_x_times_one_rule()];
    let mut applier = RewriteRuleApplier::new(rules.clone());

    let outcome = applier.execute(&expression).expect("no rule fails");

    let expected = apply_rewrite_rules(&expression, &rules).expect("no rule fails");
    assert_eq!(outcome.output(), expected.output());
    assert_eq!(outcome.is_changed(), expected.is_changed());
    assert_eq!(
        describe_fired(applier.fired()),
        describe_fired(expected.fired())
    );
}

/// Test a run in which no rule fires returns the input itself, unchanged,
/// preserving every analysis.
#[test]
fn rewrite_rule_applier_execute_without_a_firing_returns_the_input_unchanged() {
    let (_, a) = build_identifier("a");
    let expression = &a + 1;
    let mut applier = RewriteRuleApplier::new([build_x_plus_zero_rule()]);

    let outcome = applier.execute(&expression).expect("no rule fails");

    assert!(Expression::ptr_eq(outcome.output(), &expression));
    assert!(!outcome.is_changed());
    assert_eq!(outcome.preserved_analyses(), &PreservedAnalyses::all());
    assert!(applier.fired().is_empty());
}

/// Test a run in which a rule fires reports a change and preserves no
/// analysis.
#[test]
fn rewrite_rule_applier_execute_reports_a_change_when_a_rule_fires() {
    let (_, a) = build_identifier("a");
    let mut applier = RewriteRuleApplier::new([build_x_plus_zero_rule()]);

    let outcome = applier
        .execute(&build_plus_zero(&a))
        .expect("no rule fails");

    assert!(Expression::ptr_eq(outcome.output(), &a));
    assert!(outcome.is_changed());
    assert_eq!(outcome.preserved_analyses(), &PreservedAnalyses::none());
}

/// Test a rule firing at the root and returning the root itself leaves the
/// input unchanged.
#[test]
fn rewrite_rule_applier_execute_with_an_identity_rewrite_at_the_root_is_unchanged() {
    let expression = build_literal(5);
    let rule = RewriteRule::new(build_capture("x"), rewrite_to_capture("x"));
    let mut applier = RewriteRuleApplier::new([rule]);

    let outcome = applier.execute(&expression).expect("no rule fails");

    assert!(Expression::ptr_eq(outcome.output(), &expression));
    assert!(!outcome.is_changed());
    assert_eq!(describe_fired(applier.fired()), [(0, None)]);
}

/// Test the applier's firings are those of its last run: none before the
/// first, and a later run replaces them.
#[test]
fn rewrite_rule_applier_fired_lists_the_firings_of_the_last_run() {
    let (_, a) = build_identifier("a");
    let mut applier = RewriteRuleApplier::new([build_x_plus_zero_rule()]);
    let before = applier.fired().len();

    applier
        .execute(&build_plus_zero(&build_plus_zero(&a)))
        .expect("no rule fails");
    let after_first = describe_fired(applier.fired())
        .into_iter()
        .map(|(index, name)| (index, name.map(str::to_owned)))
        .collect::<Vec<_>>();
    applier.execute(&(&a + 1)).expect("no rule fails");

    assert_eq!(before, 0);
    assert_eq!(
        after_first,
        [
            (0, Some("x + 0 -> x".to_owned())),
            (0, Some("x + 0 -> x".to_owned()))
        ]
    );
    assert!(applier.fired().is_empty());
}

/// Test each firing of a named rule reports an informational diagnostic
/// naming the rule, from the pass.
#[test]
fn rewrite_rule_applier_reports_each_named_firing() {
    let (_, a) = build_identifier("a");
    let expression = build_plus_zero(&build_plus_zero(&a));
    let mut applier = RewriteRuleApplier::new([build_x_plus_zero_rule()]);

    let outcome = applier.execute(&expression).expect("no rule fails");

    let reported: Vec<(DiagnosticLevel, &str, &str)> = outcome
        .diagnostics()
        .iter()
        .map(|diagnostic| {
            (
                diagnostic.level(),
                diagnostic.message_text(),
                diagnostic.source(),
            )
        })
        .collect();
    let expected = (
        DiagnosticLevel::Info,
        "Applied rewrite rule \"x + 0 -> x\".",
        RULE_APPLIER_NAME,
    );
    assert_eq!(reported, [expected, expected]);
}

/// Test a firing of an unnamed rule reports nothing.
#[test]
fn rewrite_rule_applier_does_not_report_unnamed_firings() {
    let rule = RewriteRule::new(Pattern::wildcard(), rewrite_to_literal(0));
    let mut applier = RewriteRuleApplier::new([rule]);

    let outcome = applier.execute(&build_literal(5)).expect("no rule fails");

    assert!(
        outcome.diagnostics().is_empty(),
        "{:?}",
        outcome.diagnostics()
    );
    assert_eq!(describe_fired(applier.fired()), [(0, None)]);
}

/// Test a subtree occurring twice is rewritten once: one firing, one
/// diagnostic, and both occurrences replaced by the same node.
#[test]
fn rewrite_rule_applier_rewrites_a_shared_subtree_once() {
    let (_, a) = build_identifier("a");
    let shared = build_plus_zero(&a);
    let expression = Expression::new_binary(BinaryOperation::Multiply, &shared, &shared);
    let mut applier = RewriteRuleApplier::new([build_x_plus_zero_rule()]);

    let outcome = applier.execute(&expression).expect("no rule fails");

    assert_eq!(
        outcome.output(),
        &Expression::new_binary(BinaryOperation::Multiply, &a, &a)
    );
    assert_eq!(describe_fired(applier.fired()), [(0, Some("x + 0 -> x"))]);
    assert_eq!(outcome.diagnostics().len(), 1);
}

/// Test a doubling DAG 64 levels deep over `a + 0` is rewritten once per
/// distinct node.
#[test]
fn rewrite_rule_applier_rewrites_a_doubling_dag_once_per_distinct_node() {
    let (_, a) = build_identifier("a");
    let dag = build_doubling_dag(&build_plus_zero(&a), 64);
    let mut applier = RewriteRuleApplier::new([build_x_plus_zero_rule()]);

    let outcome = applier.execute(&dag).expect("no rule fails");

    assert_eq!(outcome.output(), &build_doubling_dag(&a, 64));
    assert_eq!(applier.fired().len(), 1);
}

/// Test the applier's name and description are the ones it is registered
/// under, whether or not it is registered.
#[test]
fn rewrite_rule_applier_name_and_description_are_its_registered_ones() {
    let applier = RewriteRuleApplier::new([]);

    assert_eq!(applier.name(), RULE_APPLIER_NAME);
    assert_eq!(applier.description(), RULE_APPLIER_DESCRIPTION);
}

/// Test the applier counts a change exactly when the output is a different
/// node, even an equal one.
#[test]
fn rewrite_rule_applier_did_change_compares_identity() {
    let (_, a) = build_identifier("a");
    let expression = &a + 1;
    let equal = &a + 1;
    let mut applier = RewriteRuleApplier::new([]);

    let same = applier.did_change(&expression, &expression.clone());
    let distinct = applier.did_change(&expression, &equal);

    assert!(!same.expect("the comparison cannot fail"));
    assert!(distinct.expect("the comparison cannot fail"));
}

/// Test a failing guard or rewrite fails the run with an execution failure
/// of the run hook whose source is the rule's callback error.
#[rstest]
#[case::guard(RewriteRule::new(Pattern::wildcard(), rewrite_to_literal(0)).with_guard(
    |_: &MatchBindings| Err(CallbackError::new(ProbeError("guard failed")))
), "guard failed")]
#[case::rewrite(RewriteRule::new(Pattern::wildcard(), |_: &MatchBindings| {
    Err(CallbackError::new(ProbeError("rewrite failed")))
}), "rewrite failed")]
fn rewrite_rule_applier_execute_fails_with_the_callback_error(
    #[case] failing: RewriteRule,
    #[case] message: &'static str,
) {
    let mut applier =
        RewriteRuleApplier::new([build_x_plus_zero_rule(), failing.with_name("failing")]);

    let error = applier
        .execute(&build_literal(5))
        .expect_err("the second rule fails");

    assert!(
        matches!(
            error.kind(),
            PassErrorKind::Hook {
                hook: PassHook::Run,
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(error.class(), FailureClass::Execution);
    assert_eq!(error.pass_name(), Some(RULE_APPLIER_NAME));
    let RewriteError::Callback {
        rule_index,
        rule_name,
        source,
    } = expect_rewrite_error(&error)
    else {
        panic!("expected a callback failure, got {error:?}");
    };
    assert_eq!(*rule_index, 1);
    assert_eq!(rule_name.as_deref(), Some("failing"));
    assert_eq!(expect_probe_error(source), &ProbeError(message));
}

/// Test a rule whose rewrite a piecewise refuses as a case condition fails
/// the run with the rebuild error, after reporting the firing: the error
/// carries the firing's diagnostic, then the failure.
#[test]
fn rewrite_rule_applier_execute_fails_with_the_rebuild_error() {
    let c = Identifier::new("c");
    let (expression, rule) = build_refused_condition(&c);
    let mut applier = RewriteRuleApplier::new([rule]);

    let error = applier
        .execute(&expression)
        .expect_err("the piecewise refuses the literal condition");

    let RewriteError::Rebuild {
        rule_index,
        rule_name,
        source,
    } = expect_rewrite_error(&error)
    else {
        panic!("expected a rebuild failure, got {error:?}");
    };
    assert_eq!(*rule_index, 0);
    assert_eq!(rule_name.as_deref(), Some("c -> 1"));
    assert_eq!(
        source,
        &RebuildError::Piecewise(PiecewiseError::NonBooleanConditionLiteral { case_index: 0 })
    );
    let levels: Vec<DiagnosticLevel> = error.diagnostics().iter().map(Diagnostic::level).collect();
    assert_eq!(levels, [DiagnosticLevel::Info, DiagnosticLevel::Error]);
    assert_eq!(describe_fired(applier.fired()), [(0, Some("c -> 1"))]);
}

/// Test the applier runs as a pipeline step, and its firings are readable
/// after the pipeline ran.
#[test]
fn rewrite_rule_applier_runs_in_a_pass_manager() {
    let (_, a) = build_identifier("a");
    let expression = build_plus_zero(&a) * 1;
    let mut applier = RewriteRuleApplier::new([build_x_plus_zero_rule(), build_x_times_one_rule()]);
    let mut manager = PassManager::default();
    manager.add_pass(&mut applier);

    let result = manager.run(&expression).expect("no rule fails");
    drop(manager);

    assert!(Expression::ptr_eq(result.output(), &a));
    let [PipelineRecord::Pass(record)] = result.records() else {
        panic!("expected one pass record, got {:?}", result.records());
    };
    assert_eq!(record.pass_name(), RULE_APPLIER_NAME);
    assert!(record.is_changed());
    assert_eq!(record.diagnostics().len(), 2);
    assert_eq!(
        describe_fired(applier.fired()),
        [(0, Some("x + 0 -> x")), (1, Some("x * 1 -> x"))]
    );
}

/// Test the applier converges in a fixpoint group when a rule's rewrite is
/// rewritten only in the next run: `0 + a` becomes `a + 0`, then `a`, and a
/// third run changes nothing.
#[test]
fn rewrite_rule_applier_converges_in_a_fixpoint_pass_group() {
    let (_, a) = build_identifier("a");
    let expression = Expression::new_binary(BinaryOperation::Add, 0, &a);
    let mut group = FixpointPassGroup::new(Identifier::new("simplify"));
    group.add_pass(RewriteRuleApplier::new([
        build_move_zero_right_rule(),
        build_x_plus_zero_rule(),
    ]));
    let mut manager = PassManager::default();
    manager.add_fixpoint_group(group);

    let result = manager.run(&expression).expect("the group converges");

    assert!(Expression::ptr_eq(result.output(), &a));
    let [PipelineRecord::FixpointGroup(record)] = result.records() else {
        panic!("expected one group record, got {:?}", result.records());
    };
    assert!(record.is_converged());
    let changes: Vec<bool> = record
        .iteration_records()
        .iter()
        .map(FixpointIterationRecord::is_changed)
        .collect();
    assert_eq!(changes, [true, true, false]);
}

// =============================================================================
// ExpressionPrettyFormatter
// =============================================================================

/// Return a tree with every node kind over the identifier `x`.
fn build_every_kind(x: &Expression) -> Expression {
    build_piecewise_or_panic(
        [(x.less(3), build_call_or_panic("f", [-x, build_literal(1)]))],
        x.power(2) + 1,
    )
}

/// Test executing the formatter gives the text `Expression::display` gives
/// under the same options.
#[rstest]
#[case::symbolic_name_hint(FormatOptions::default())]
#[case::functional_name_hint(FormatOptions::default().with_notation(Notation::Functional))]
#[case::symbolic_with_id(
    FormatOptions::default().with_identifier_style(IdentifierStyle::NameHintWithId)
)]
#[case::functional_with_id(
    FormatOptions::default()
        .with_notation(Notation::Functional)
        .with_identifier_style(IdentifierStyle::NameHintWithId)
)]
fn expression_pretty_formatter_execute_matches_display(#[case] options: FormatOptions) {
    let (_, x) = build_identifier("x");
    let expression = build_every_kind(&x);
    let mut formatter = ExpressionPrettyFormatter::new(options);

    let outcome = formatter
        .execute(&expression)
        .expect("formatting cannot fail");

    assert_eq!(outcome.output(), &expression.display(options).to_string());
}

/// Test the default formatter writes symbolic notation without ids.
#[test]
fn expression_pretty_formatter_default_formats_symbolically_without_ids() {
    let (_, x) = build_identifier("x");
    let mut formatter = ExpressionPrettyFormatter::default();

    let outcome = formatter
        .execute(&(&x + 2))
        .expect("formatting cannot fail");

    assert_eq!(outcome.output(), "(x + 2)");
    assert_eq!(formatter.options(), FormatOptions::default());
}

/// Test the formatter keeps the options it was built with.
#[test]
fn expression_pretty_formatter_options_are_the_options_it_was_built_with() {
    let options = FormatOptions::default().with_notation(Notation::Functional);

    let formatter = ExpressionPrettyFormatter::new(options);

    assert_eq!(formatter.options(), options);
}

/// Test every formatting run counts as a change, since text is never the
/// input expression.
#[test]
fn expression_pretty_formatter_execute_reports_a_change() {
    let mut formatter = ExpressionPrettyFormatter::default();

    let outcome = formatter
        .execute(&build_literal(1))
        .expect("formatting cannot fail");

    assert!(outcome.is_changed());
    assert_eq!(outcome.preserved_analyses(), &PreservedAnalyses::none());
}

/// Test the formatter is named after its type and described by its name.
#[test]
fn expression_pretty_formatter_name_is_its_type_name() {
    let formatter = ExpressionPrettyFormatter::default();

    assert_eq!(formatter.name(), "ExpressionPrettyFormatter");
    assert_eq!(formatter.description(), "ExpressionPrettyFormatter");
}

// =============================================================================
// Registration
// =============================================================================

/// Test registering the expression passes, once or again, registers the
/// rule applier under its name and description, and the registered factory
/// builds an applier with no rules.
#[test]
fn register_expression_passes_registers_the_rule_applier() {
    let (_, a) = build_identifier("a");
    let expression = build_plus_zero(&a);

    let mut registry = PassRegistry::new();

    let first = register_expression_passes(&mut registry);
    let second = register_expression_passes(&mut registry);

    assert_eq!(first, Ok(()));
    assert_eq!(second, Ok(()));
    assert_eq!(registry.len(), 1);
    let info = registry.info(RULE_APPLIER_NAME).expect("registered");
    assert_eq!(info.description(), RULE_APPLIER_DESCRIPTION);
    assert_eq!(info.pass_type_id(), TypeId::of::<RewriteRuleApplier>());
    assert_eq!(info.input_type_id(), TypeId::of::<Expression>());
    assert_eq!(info.output_type_id(), TypeId::of::<Expression>());
    let mut created = registry
        .create::<Expression, Expression>(RULE_APPLIER_NAME)
        .expect("the rule applier is registered");
    assert_eq!(created.name(), RULE_APPLIER_NAME);
    let outcome = created.execute(&expression).expect("no rules, no failure");
    assert!(Expression::ptr_eq(outcome.output(), &expression));
    assert!(!outcome.is_changed());
}
