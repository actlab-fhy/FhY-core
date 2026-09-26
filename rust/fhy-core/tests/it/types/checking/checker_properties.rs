//! Properties of the checker, as `test_type_checker_properties.py` states
//! them: synthesis of well-typed integer trees agrees with the promotion of
//! their leaves' types, and checking a synthesized type succeeds.

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
