//! Properties of the checker, as `test_type_checker_properties.py` states
//! them: synthesis of well-typed integer trees agrees with the promotion of
//! their leaves' types, and checking a synthesized type succeeds. And the
//! checker's memo of shared nodes: a DAG checks as its unshared tree does.

use std::collections::HashMap;

use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{Expression, LiteralValue};
use fhy_core::identifier::Identifier;
use fhy_core::types::checking::TypeChecker;
use fhy_core::types::{CoreDataType, NumericalType, Type, TypeQualifier};
use proptest::prelude::*;

/// A tree of additions and multiplications over identifiers of sized
/// signed integer types and non-negative literals below 100.
#[derive(Debug, Clone)]
enum Tree {
    Leaf(usize),
    Literal(u8),
    Add(Box<Tree>, Box<Tree>),
    Multiply(Box<Tree>, Box<Tree>),
}

fn tree_strategy() -> impl Strategy<Value = Tree> {
    let leaf = prop_oneof![
        (0_usize..3).prop_map(Tree::Leaf),
        (0_u8..100).prop_map(Tree::Literal)
    ];
    leaf.prop_recursive(4, 16, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone())
                .prop_map(|(left, right)| Tree::Add(Box::new(left), Box::new(right))),
            (inner.clone(), inner)
                .prop_map(|(left, right)| Tree::Multiply(Box::new(left), Box::new(right))),
        ]
    })
}

const TYPES: [CoreDataType; 3] = [CoreDataType::Int8, CoreDataType::Int16, CoreDataType::Int32];

fn build(tree: &Tree, identifiers: &[Identifier]) -> Expression {
    match tree {
        Tree::Leaf(index) => Expression::from(identifiers[*index].clone()),
        Tree::Literal(value) => Expression::from(LiteralValue::from(i64::from(*value))),
        Tree::Add(left, right) => build(left, identifiers) + build(right, identifiers),
        Tree::Multiply(left, right) => build(left, identifiers) * build(right, identifiers),
    }
}

/// Return the reference type: the promotion of the identifiers' types the
/// tree reaches, or `None` for a tree of literals only.
fn reference_type(tree: &Tree) -> Option<CoreDataType> {
    match tree {
        Tree::Leaf(index) => Some(TYPES[*index]),
        Tree::Literal(_) => None,
        Tree::Add(left, right) | Tree::Multiply(left, right) => {
            match (reference_type(left), reference_type(right)) {
                (Some(left), Some(right)) => left.promote(right).ok(),
                (Some(side), None) | (None, Some(side)) => Some(side),
                (None, None) => None,
            }
        }
    }
}

proptest! {
    #[test]
    fn synthesis_agrees_with_the_promotion_of_the_leaves(tree in tree_strategy()) {
        let identifiers: Vec<Identifier> = (0..3).map(|index| Identifier::new(&format!("v{index}"))).collect();
        let bindings: HashMap<Identifier, (Type, TypeQualifier)> = identifiers
            .iter()
            .zip(TYPES)
            .map(|(identifier, core)| (identifier.clone(), (Type::Numerical(NumericalType::scalar(core)), TypeQualifier::Param)))
            .collect();
        let registry = FunctionRegistry::new();
        let checker = TypeChecker::new(&bindings, &registry, &registry);
        let expression = build(&tree, &identifiers);

        let (synthesized, qualifier) = checker.synthesize(&expression).expect("a well-typed tree");

        prop_assert_eq!(qualifier, TypeQualifier::Param);
        let expected = reference_type(&tree).unwrap_or(CoreDataType::Uint);
        prop_assert_eq!(&synthesized, &Type::Numerical(NumericalType::scalar(expected)));
        prop_assert!(checker.check(&expression, &synthesized).is_ok());
    }
}

/// One node of a DAG: a leaf, or an operation on two earlier nodes.
#[derive(Debug, Clone)]
enum DagNode {
    Leaf(usize),
    Literal(u8),
    Boolean(bool),
    Add(usize, usize),
    Multiply(usize, usize),
    Less(usize, usize),
}

/// A DAG as its nodes in order, each operation on earlier nodes; the last
/// node is the root.
fn dag_strategy() -> impl Strategy<Value = Vec<DagNode>> {
    (1_usize..=12).prop_flat_map(|length| {
        (0..length)
            .map(|position| {
                let leaf = prop_oneof![
                    4 => (0_usize..3).prop_map(DagNode::Leaf),
                    2 => (0_u8..100).prop_map(DagNode::Literal),
                    1 => any::<bool>().prop_map(DagNode::Boolean),
                ];
                if position == 0 {
                    return leaf.boxed();
                }
                let earlier = 0..position;
                prop_oneof![
                    1 => leaf,
                    3 => (earlier.clone(), earlier.clone()).prop_map(|(l, r)| DagNode::Add(l, r)),
                    3 => (earlier.clone(), earlier.clone())
                        .prop_map(|(l, r)| DagNode::Multiply(l, r)),
                    1 => (earlier.clone(), earlier).prop_map(|(l, r)| DagNode::Less(l, r)),
                ]
                .boxed()
            })
            .collect::<Vec<_>>()
    })
}

fn build_leaf(node: &DagNode, identifiers: &[Identifier]) -> Option<Expression> {
    match node {
        DagNode::Leaf(index) => Some(Expression::from(identifiers[*index].clone())),
        DagNode::Literal(value) => Some(Expression::from(LiteralValue::from(i64::from(*value)))),
        DagNode::Boolean(value) => Some(Expression::from(LiteralValue::Bool(*value))),
        _ => None,
    }
}

fn combine(node: &DagNode, left: &Expression, right: &Expression) -> Expression {
    match node {
        DagNode::Add(..) => left + right,
        DagNode::Multiply(..) => left * right,
        DagNode::Less(..) => left.less(right),
        _ => unreachable!("an operation"),
    }
}

/// Return the DAG's root, each node built once and shared by its users.
fn build_shared(dag: &[DagNode], identifiers: &[Identifier]) -> Expression {
    let mut built: Vec<Expression> = Vec::with_capacity(dag.len());
    for node in dag {
        let expression = match node {
            DagNode::Add(left, right)
            | DagNode::Multiply(left, right)
            | DagNode::Less(left, right) => combine(node, &built[*left], &built[*right]),
            leaf => build_leaf(leaf, identifiers).expect("a leaf"),
        };
        built.push(expression);
    }
    built.pop().expect("a root")
}

/// Return the node at `position` of the DAG as a tree: every occurrence
/// built afresh, so no node is shared.
fn build_unshared(dag: &[DagNode], position: usize, identifiers: &[Identifier]) -> Expression {
    let node = &dag[position];
    match node {
        DagNode::Add(left, right) | DagNode::Multiply(left, right) | DagNode::Less(left, right) => {
            combine(
                node,
                &build_unshared(dag, *left, identifiers),
                &build_unshared(dag, *right, identifiers),
            )
        }
        leaf => build_leaf(leaf, identifiers).expect("a leaf"),
    }
}

proptest! {
    #[test]
    fn checking_a_shared_dag_equals_checking_its_unshared_tree(dag in dag_strategy()) {
        let identifiers: Vec<Identifier> = (0..3).map(|index| Identifier::new(&format!("v{index}"))).collect();
        let bindings: HashMap<Identifier, (Type, TypeQualifier)> = identifiers
            .iter()
            .zip(TYPES)
            .map(|(identifier, core)| (identifier.clone(), (Type::Numerical(NumericalType::scalar(core)), TypeQualifier::Param)))
            .collect();
        let registry = FunctionRegistry::new();
        let checker = TypeChecker::new(&bindings, &registry, &registry);
        let shared = build_shared(&dag, &identifiers);
        let unshared = build_unshared(&dag, dag.len() - 1, &identifiers);
        prop_assert_eq!(&shared, &unshared);

        let text = |result: Result<(Type, TypeQualifier), _>| result.map_err(|error: fhy_core::types::checking::TypeCheckError| error.to_string());
        prop_assert_eq!(text(checker.synthesize(&shared)), text(checker.synthesize(&unshared)));
        for expected in [CoreDataType::Int64, CoreDataType::Int8, CoreDataType::Bool] {
            let expected = Type::Numerical(NumericalType::scalar(expected));
            prop_assert_eq!(
                text(checker.check(&shared, &expected)),
                text(checker.check(&unshared, &expected))
            );
        }
    }
}
