//! Tests for `Expression` as pass-infrastructure IR: its node identity, its
//! children and rebuilds seen through `Tree`, and the tree walk, the
//! memoized rewrite, and the walk and rewrite passes over expressions,
//! including trees too deep or too shared for a recursive traversal.
//!
//! Public API only.

#[path = "common/expression.rs"]
pub mod expression_support;
#[path = "common/stack.rs"]
pub mod stack_support;

use expression_support::{build_call_or_panic, build_deep_sum, build_identifier, build_literal};
use fhy_core::identifier::Identifier;
use fhy_core::pass_infrastructure::{
    ExecutePass, NodeHandle, PassContext, RewritePass, RewriteTreeError, Rewriter, TraversalOrder,
    Tree, TreeVisitor, WalkPass,
};
use fhy_core::symbolic::expression::{
    BinaryOperation, Expression, ExpressionBuildError, ExpressionKind, LiteralValue,
    UnaryOperation, build_piecewise,
};
use stack_support::{SMALL_STACK_DEPTH, run_on_small_stack};

// =============================================================================
// Helpers
// =============================================================================

/// The error the test visitors and rewriters never return.
type Never = std::convert::Infallible;

/// Records the kind of every expression it visits.
#[derive(Debug, Default)]
struct KindRecorder {
    kinds: Vec<String>,
}

/// Return a short label of `expression`'s node kind.
fn label_kind(expression: &Expression) -> String {
    match expression.kind() {
        ExpressionKind::Unary(node) => format!("unary {:?}", node.operation()),
        ExpressionKind::Binary(node) => format!("binary {:?}", node.operation()),
        ExpressionKind::Identifier(identifier) => format!("identifier {}", identifier.name_hint()),
        ExpressionKind::Literal(literal) => format!("literal {literal}"),
        ExpressionKind::Piecewise(_) => "piecewise".to_owned(),
        ExpressionKind::Call(node) => format!("call {}", node.function_name()),
    }
}

impl TreeVisitor<Expression> for KindRecorder {
    type Error = Never;

    fn visit(&mut self, node: &Expression, _cx: &mut PassContext<'_>) -> Result<(), Never> {
        self.kinds.push(label_kind(node));
        Ok(())
    }
}

/// Counts the expressions it visits.
#[derive(Debug, Default)]
struct VisitCounter {
    count: usize,
}

impl TreeVisitor<Expression> for VisitCounter {
    type Error = Never;

    fn visit(&mut self, _node: &Expression, _cx: &mut PassContext<'_>) -> Result<(), Never> {
        self.count += 1;
        Ok(())
    }
}

/// Replaces every reference to one identifier by a replacement, counting
/// the nodes it is asked to rewrite.
#[derive(Debug)]
struct IdentifierReplacer {
    target: Identifier,
    replacement: Expression,
    calls: usize,
}

impl IdentifierReplacer {
    /// Create the rewriter replacing references to `target` by
    /// `replacement`.
    fn new(target: &Identifier, replacement: &Expression) -> Self {
        Self {
            target: target.clone(),
            replacement: replacement.clone(),
            calls: 0,
        }
    }
}

impl Rewriter<Expression> for IdentifierReplacer {
    type Error = Never;

    fn rewrite(
        &mut self,
        node: &Expression,
        _cx: &mut PassContext<'_>,
    ) -> Result<Option<Expression>, Never> {
        self.calls += 1;
        Ok(match node.kind() {
            ExpressionKind::Identifier(identifier) if identifier == &self.target => {
                Some(self.replacement.clone())
            }
            _ => None,
        })
    }
}

/// Rewrites the literal `true` to the literal `1`.
#[derive(Debug, Default)]
struct TrueToOne;

impl Rewriter<Expression> for TrueToOne {
    type Error = Never;

    fn rewrite(
        &mut self,
        node: &Expression,
        _cx: &mut PassContext<'_>,
    ) -> Result<Option<Expression>, Never> {
        Ok(match node.kind() {
            ExpressionKind::Literal(literal) if literal == &LiteralValue::from(true) => {
                Some(build_literal(1))
            }
            _ => None,
        })
    }
}

// =============================================================================
// NodeHandle and Tree
// =============================================================================

/// Test clones of an expression share its identity.
#[test]
fn expression_identity_is_shared_by_clones() {
    let (_, x) = build_identifier("x");
    let expression = &x + 1;

    let identity = expression.identity();

    assert_eq!(identity, expression.clone().identity());
}

/// Test two separately built equal expressions have different identities.
#[test]
fn expression_identity_differs_between_equal_nodes() {
    let (_, x) = build_identifier("x");
    let first = &x + 1;
    let second = &x + 1;

    assert_eq!(first, second);
    assert_ne!(first.identity(), second.identity());
}

/// Test the tree children of an expression are its own children, in their
/// order.
#[test]
fn expression_tree_children_are_the_expression_children() {
    let (_, x) = build_identifier("x");
    let expression = build_piecewise([(x.clone(), build_literal(1))], build_literal(2))
        .expect("a valid piecewise");

    let tree_children: Vec<&Expression> = Tree::children(&expression).collect();

    let own_children: Vec<&Expression> = expression.children().collect();
    assert_eq!(tree_children.len(), 3);
    for (tree_child, own_child) in tree_children.iter().zip(&own_children) {
        assert!(Expression::ptr_eq(tree_child, own_child));
    }
}

/// Test a tree rebuild of an expression builds the node around the new
/// children.
#[test]
fn expression_tree_rebuild_uses_the_new_children() {
    let (_, x) = build_identifier("x");
    let (_, y) = build_identifier("y");
    let expression = -&x;

    let rebuilt = Tree::rebuild_with_children(&expression, vec![y.clone()]).expect("one child");

    assert_eq!(rebuilt, Expression::new_unary(UnaryOperation::Negate, &y));
}

/// Test a tree rebuild of an expression refuses a child list of the wrong
/// length.
#[test]
fn expression_tree_rebuild_refuses_a_wrong_child_count() {
    let (_, x) = build_identifier("x");
    let expression = -&x;

    let result = Tree::rebuild_with_children(&expression, Vec::new());

    assert_eq!(
        result.expect_err("the child count is wrong"),
        ExpressionBuildError::ChildCountMismatch {
            expected: 1,
            actual: 0
        }
    );
}

// =============================================================================
// Walks
// =============================================================================

/// Test a walk pass visits an expression's nodes in pre-order.
#[test]
fn walk_pass_visits_an_expression_in_pre_order() {
    let (_, a) = build_identifier("a");
    let (_, b) = build_identifier("b");
    let expression = build_call_or_panic("f", [&a + 1, -&b]);
    let mut pass = WalkPass::new(KindRecorder::default(), TraversalOrder::Pre);

    pass.execute(&expression).expect("no hook fails");

    assert_eq!(
        pass.visitor().kinds,
        [
            "call f",
            "binary Add",
            "identifier a",
            "literal 1",
            "unary Negate",
            "identifier b",
        ]
    );
}

/// Test walking an expression [`SMALL_STACK_DEPTH`] levels deep on a small
/// stack visits every node.
#[test]
fn walk_tree_walks_a_deep_expression_on_a_small_stack() {
    let count = run_on_small_stack(|| {
        let (_, x) = build_identifier("x");
        let expression = build_deep_sum(&x, SMALL_STACK_DEPTH);
        let mut pass = WalkPass::new(VisitCounter::default(), TraversalOrder::Post);

        pass.execute(&expression).expect("no hook fails");

        pass.into_visitor().count
    });

    assert_eq!(count, 2 * SMALL_STACK_DEPTH + 1);
}

// =============================================================================
// Rewrites
// =============================================================================

/// Test a rewrite pass replaces an identifier and keeps the handles of the
/// subtrees it does not touch.
#[test]
fn rewrite_pass_replaces_an_identifier_in_an_expression() {
    let (x_identifier, x) = build_identifier("x");
    let (_, y) = build_identifier("y");
    let (_, z) = build_identifier("z");
    let untouched = &z * 2;
    let expression = Expression::new_binary(BinaryOperation::Add, &x, &untouched);
    let mut pass = RewritePass::new(IdentifierReplacer::new(&x_identifier, &y));

    let outcome = pass.execute(&expression).expect("no rewrite fails");

    assert!(outcome.is_changed());
    assert_eq!(
        outcome.output(),
        &Expression::new_binary(BinaryOperation::Add, &y, &untouched)
    );
    let ExpressionKind::Binary(node) = outcome.output().kind() else {
        panic!("expected a binary node, got {:?}", outcome.output());
    };
    assert!(Expression::ptr_eq(node.right(), &untouched));
}

/// Test a rewrite pass that touches nothing returns the expression itself.
#[test]
fn rewrite_pass_keeps_an_expression_it_does_not_touch() {
    let (x_identifier, _) = build_identifier("x");
    let (_, y) = build_identifier("y");
    let (_, z) = build_identifier("z");
    let expression = &z * 2;
    let mut pass = RewritePass::new(IdentifierReplacer::new(&x_identifier, &y));

    let outcome = pass.execute(&expression).expect("no rewrite fails");

    assert!(!outcome.is_changed());
    assert!(Expression::ptr_eq(outcome.output(), &expression));
}

/// Test a rewrite making a piecewise case condition a non-Boolean literal
/// fails with the refused rebuild of the piecewise.
#[test]
fn rewrite_tree_reports_a_refused_expression_rebuild() {
    let expression = build_piecewise([(build_literal(true), build_literal(5))], build_literal(6))
        .expect("a valid piecewise");
    let mut pass = RewritePass::new(TrueToOne);

    let result = pass.execute(&expression);

    let error = result.expect_err("the rebuild is refused");
    let source = std::error::Error::source(&error).expect("the rewrite error is the source");
    let tree_error = source
        .downcast_ref::<RewriteTreeError<Expression, Never>>()
        .expect("the source is the rewrite error");
    let RewriteTreeError::Rebuild {
        node,
        children,
        source,
    } = tree_error
    else {
        panic!("expected a rebuild failure, got {tree_error:?}");
    };
    assert!(Expression::ptr_eq(node, &expression));
    assert_eq!(children[0], build_literal(1));
    assert_eq!(
        source,
        &ExpressionBuildError::NonBooleanConditionLiteral { case_index: 0 }
    );
}

/// Test rewriting the bottom of an expression [`SMALL_STACK_DEPTH`] levels
/// deep on a small stack rebuilds every level above it.
#[test]
fn rewrite_pass_rebuilds_a_deep_expression_on_a_small_stack() {
    let is_expected = run_on_small_stack(|| {
        let (x_identifier, x) = build_identifier("x");
        let (_, y) = build_identifier("y");
        let expression = build_deep_sum(&x, SMALL_STACK_DEPTH);
        let mut pass = RewritePass::new(IdentifierReplacer::new(&x_identifier, &y));

        let outcome = pass.execute(&expression).expect("no rewrite fails");

        outcome.is_changed() && outcome.output() == &build_deep_sum(&y, SMALL_STACK_DEPTH)
    });

    assert!(is_expected);
}

/// Test the DAG `x_0 = a`, `x_{k+1} = x_k + x_k` with 64 levels, which has
/// 65 distinct nodes and 2^65 - 1 occurrences, is rewritten with one call
/// per distinct node into a DAG sharing its nodes the same way.
#[test]
fn rewrite_pass_rewrites_a_doubling_expression_dag_once_per_distinct_node() {
    let levels = 64;
    let (a_identifier, a) = build_identifier("a");
    let (_, b) = build_identifier("b");
    let mut dag = a.clone();
    for _ in 0..levels {
        dag = &dag + &dag;
    }
    let mut pass = RewritePass::new(IdentifierReplacer::new(&a_identifier, &b));

    let outcome = pass.execute(&dag).expect("no rewrite fails");

    assert_eq!(pass.rewriter().calls, levels + 1);
    let mut node = outcome.output();
    for _ in 0..levels {
        let ExpressionKind::Binary(binary) = node.kind() else {
            panic!("expected a sum, got {node:?}");
        };
        assert!(Expression::ptr_eq(binary.left(), binary.right()));
        node = binary.left();
    }
    assert!(Expression::ptr_eq(node, &b));
}
