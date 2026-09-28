//! Properties of the checker, as `test_type_checker_properties.py` states
//! them: synthesis of well-typed integer trees agrees with the promotion of
//! their leaves' types, and checking a synthesized type succeeds. And the
//! checker's memo of shared nodes: a DAG checks as its unshared tree does.
//! Over trees of signed, unsigned, float and Boolean leaves, with
//! comparisons, connectives, unary operators and piecewise nodes: synthesis
//! is symmetric for commutative operations, and a well-typed tree's type
//! has the value kind the evaluator computes for it.

use std::collections::HashMap;

use fhy_core::expression::evaluate::{Evaluator, Scalar};
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{BinaryOperation, Expression, LiteralValue, UnaryOperation};
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
    Add(Box<Self>, Box<Self>),
    Multiply(Box<Self>, Box<Self>),
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

/// The types of the identifiers the broad trees draw from.
const BROAD_TYPES: [CoreDataType; 8] = [
    CoreDataType::Int8,
    CoreDataType::Int16,
    CoreDataType::Int32,
    CoreDataType::Uint8,
    CoreDataType::Uint16,
    CoreDataType::Float32,
    CoreDataType::Float64,
    CoreDataType::Bool,
];

/// A tree over the identifiers of [`BROAD_TYPES`] and literals, well typed
/// or not.
#[derive(Debug, Clone)]
enum Broad {
    Leaf(usize),
    Integer(i8),
    /// A float literal, the half of the value.
    Float(i8),
    Boolean(bool),
    Unary(UnaryOperation, Box<Self>),
    Binary(BinaryOperation, Box<Self>, Box<Self>),
    /// A conjunction when `true`, a disjunction otherwise.
    Logical(bool, Box<Self>, Box<Self>),
    Piecewise(Box<Self>, Box<Self>, Box<Self>),
}

const UNARY: [UnaryOperation; 3] = [
    UnaryOperation::Negate,
    UnaryOperation::Positive,
    UnaryOperation::LogicalNot,
];

const BINARY: [BinaryOperation; 10] = [
    BinaryOperation::Add,
    BinaryOperation::Subtract,
    BinaryOperation::Multiply,
    BinaryOperation::Divide,
    BinaryOperation::FloorDivide,
    BinaryOperation::Equal,
    BinaryOperation::NotEqual,
    BinaryOperation::Less,
    BinaryOperation::LessEqual,
    BinaryOperation::Greater,
];

/// The commutative operations: two binary ones and the two connectives.
const COMMUTATIVE: [BinaryOperation; 4] = [
    BinaryOperation::Add,
    BinaryOperation::Multiply,
    BinaryOperation::Equal,
    BinaryOperation::NotEqual,
];

/// Return trees of any shape, most of them ill-typed.
fn any_broad_strategy() -> impl Strategy<Value = Broad> {
    let leaf = prop_oneof![
        4 => (0..BROAD_TYPES.len()).prop_map(Broad::Leaf),
        1 => (-4_i8..=8).prop_map(Broad::Integer),
        1 => (-8_i8..=8).prop_map(Broad::Float),
        1 => any::<bool>().prop_map(Broad::Boolean),
    ];
    leaf.prop_recursive(4, 24, 3, |inner| {
        prop_oneof![
            1 => (0..UNARY.len(), inner.clone())
                .prop_map(|(index, operand)| Broad::Unary(UNARY[index], Box::new(operand))),
            4 => (0..BINARY.len(), inner.clone(), inner.clone()).prop_map(|(index, left, right)| {
                Broad::Binary(BINARY[index], Box::new(left), Box::new(right))
            }),
            1 => (any::<bool>(), inner.clone(), inner.clone()).prop_map(|(and, left, right)| {
                Broad::Logical(and, Box::new(left), Box::new(right))
            }),
            1 => (inner.clone(), inner.clone(), inner).prop_map(|(condition, value, otherwise)| {
                Broad::Piecewise(Box::new(condition), Box::new(value), Box::new(otherwise))
            }),
        ]
    })
}

/// The value kind a typed tree is drawn for.
#[derive(Debug, Clone, Copy)]
enum Kind {
    Integer,
    Float,
    Boolean,
}

/// Return the leaves of `kind`: its identifiers, and literals the checker
/// takes in any context of the kind.
fn typed_leaf(kind: Kind) -> BoxedStrategy<Broad> {
    match kind {
        Kind::Integer => prop_oneof![
            3 => (0_usize..5).prop_map(Broad::Leaf),
            1 => (0_i8..=100).prop_map(Broad::Integer),
        ]
        .boxed(),
        Kind::Float => prop_oneof![
            3 => (5_usize..7).prop_map(Broad::Leaf),
            1 => (-8_i8..=8).prop_map(Broad::Float),
        ]
        .boxed(),
        Kind::Boolean => prop_oneof![
            3 => Just(Broad::Leaf(7)),
            1 => any::<bool>().prop_map(Broad::Boolean),
        ]
        .boxed(),
    }
}

/// Return trees of `kind` that the checker accepts by construction, `depth`
/// levels deep at most.
fn typed_strategy(kind: Kind, depth: u32) -> BoxedStrategy<Broad> {
    if depth == 0 {
        return typed_leaf(kind);
    }
    let same = move || typed_strategy(kind, depth - 1);
    let boolean = move || typed_strategy(Kind::Boolean, depth - 1);
    let binary = |operations: &'static [BinaryOperation],
                  left: BoxedStrategy<Broad>,
                  right: BoxedStrategy<Broad>| {
        (0..operations.len(), left, right).prop_map(move |(index, left, right)| {
            Broad::Binary(operations[index], Box::new(left), Box::new(right))
        })
    };
    let piecewise = (boolean(), same(), same()).prop_map(|(condition, value, otherwise)| {
        Broad::Piecewise(Box::new(condition), Box::new(value), Box::new(otherwise))
    });
    match kind {
        Kind::Integer => prop_oneof![
            2 => typed_leaf(kind),
            1 => (prop_oneof![Just(UnaryOperation::Negate), Just(UnaryOperation::Positive)], same())
                .prop_map(|(operation, operand)| Broad::Unary(operation, Box::new(operand))),
            3 => binary(
                &[BinaryOperation::Add, BinaryOperation::Subtract, BinaryOperation::Multiply, BinaryOperation::FloorDivide],
                same(),
                same(),
            ),
            1 => piecewise,
        ]
        .boxed(),
        Kind::Float => {
            let integer = move || typed_strategy(Kind::Integer, depth - 1);
            prop_oneof![
                2 => typed_leaf(kind),
                1 => (prop_oneof![Just(UnaryOperation::Negate), Just(UnaryOperation::Positive)], same())
                    .prop_map(|(operation, operand)| Broad::Unary(operation, Box::new(operand))),
                2 => binary(
                    &[BinaryOperation::Add, BinaryOperation::Subtract, BinaryOperation::Multiply, BinaryOperation::Divide],
                    same(),
                    same(),
                ),
                1 => binary(&[BinaryOperation::Divide], integer(), integer()),
                1 => binary(&[BinaryOperation::Divide, BinaryOperation::FloorDivide], same(), integer()),
                1 => piecewise,
            ]
            .boxed()
        }
        Kind::Boolean => {
            let comparisons: &'static [BinaryOperation] = &[
                BinaryOperation::Equal,
                BinaryOperation::NotEqual,
                BinaryOperation::Less,
                BinaryOperation::LessEqual,
                BinaryOperation::Greater,
            ];
            prop_oneof![
                2 => typed_leaf(kind),
                1 => same().prop_map(|operand| Broad::Unary(UnaryOperation::LogicalNot, Box::new(operand))),
                1 => (any::<bool>(), same(), same()).prop_map(|(and, left, right)| {
                    Broad::Logical(and, Box::new(left), Box::new(right))
                }),
                1 => binary(comparisons, typed_strategy(Kind::Integer, depth - 1), typed_strategy(Kind::Integer, depth - 1)),
                1 => binary(comparisons, typed_strategy(Kind::Float, depth - 1), typed_strategy(Kind::Float, depth - 1)),
                1 => binary(&[BinaryOperation::Equal, BinaryOperation::NotEqual], same(), same()),
                1 => piecewise,
            ]
            .boxed()
        }
    }
}

/// Return trees of every kind, mostly well typed, some of any shape.
fn broad_strategy() -> impl Strategy<Value = Broad> {
    prop_oneof![
        2 => typed_strategy(Kind::Integer, 3),
        2 => typed_strategy(Kind::Float, 3),
        2 => typed_strategy(Kind::Boolean, 3),
        1 => any_broad_strategy(),
    ]
}

/// Return pairs of trees of one kind, for the commutative operations of
/// that kind: the operation's index in [`COMMUTATIVE`], or past it for a
/// conjunction or disjunction.
fn commutative_strategy() -> impl Strategy<Value = (usize, Broad, Broad)> {
    let numeric = |kind| {
        (
            0..COMMUTATIVE.len(),
            typed_strategy(kind, 3),
            typed_strategy(kind, 3),
        )
    };
    prop_oneof![
        numeric(Kind::Integer),
        numeric(Kind::Float),
        (
            2..COMMUTATIVE.len() + 2,
            typed_strategy(Kind::Boolean, 3),
            typed_strategy(Kind::Boolean, 3)
        ),
        (
            0..COMMUTATIVE.len() + 2,
            any_broad_strategy(),
            any_broad_strategy()
        ),
    ]
}

fn build_broad(tree: &Broad, identifiers: &[Identifier]) -> Expression {
    match tree {
        Broad::Leaf(index) => Expression::from(identifiers[*index].clone()),
        Broad::Integer(value) => Expression::from(i64::from(*value)),
        Broad::Float(half) => Expression::from(f64::from(*half) / 2.0),
        Broad::Boolean(value) => Expression::from(LiteralValue::Bool(*value)),
        Broad::Unary(operation, operand) => {
            Expression::new_unary(*operation, build_broad(operand, identifiers))
        }
        Broad::Binary(operation, left, right) => Expression::new_binary(
            *operation,
            build_broad(left, identifiers),
            build_broad(right, identifiers),
        ),
        Broad::Logical(and, left, right) => {
            let (left, right) = (
                build_broad(left, identifiers),
                build_broad(right, identifiers),
            );
            if *and {
                left.and(right)
            } else {
                left.or(right)
            }
        }
        Broad::Piecewise(condition, value, otherwise) => {
            let otherwise = build_broad(otherwise, identifiers);
            // The builder refuses a non-Boolean literal condition; the tree
            // is then its otherwise branch.
            Expression::piecewise(
                [(
                    build_broad(condition, identifiers),
                    build_broad(value, identifiers),
                )],
                otherwise.clone(),
            )
            .unwrap_or(otherwise)
        }
    }
}

/// Return the broad identifiers and their bindings as `Param`s.
fn broad_bindings() -> (Vec<Identifier>, HashMap<Identifier, (Type, TypeQualifier)>) {
    let identifiers: Vec<Identifier> = BROAD_TYPES
        .iter()
        .map(|core| Identifier::new(&format!("v_{core}")))
        .collect();
    let bindings = identifiers
        .iter()
        .zip(BROAD_TYPES)
        .map(|(identifier, core)| {
            (
                identifier.clone(),
                (
                    Type::Numerical(NumericalType::scalar(core)),
                    TypeQualifier::Param,
                ),
            )
        })
        .collect();
    (identifiers, bindings)
}

/// Return the value of the identifier of `core`, from the drawn `seed`.
fn broad_value(core: CoreDataType, seed: i8) -> Scalar {
    if core == CoreDataType::Bool {
        Scalar::Bool(seed % 2 == 0)
    } else if core.is_real_float() {
        Scalar::Real(f64::from(seed) / 4.0)
    } else if core.is_unsigned() {
        Scalar::Int(i64::from(seed.unsigned_abs()))
    } else {
        Scalar::Int(i64::from(seed))
    }
}

/// Return the value kind of a scalar of `core`: 0 for a Boolean, 1 for an
/// integer and 2 for a real float.
fn kind_of_type(core: CoreDataType) -> u8 {
    if core == CoreDataType::Bool {
        0
    } else if core.is_integral() {
        1
    } else {
        2
    }
}

/// Return the value kind of `value`, as [`kind_of_type`] numbers them.
const fn kind_of_value(value: Scalar) -> u8 {
    match value {
        Scalar::Bool(_) => 0,
        Scalar::Int(_) => 1,
        Scalar::Real(_) => 2,
    }
}

proptest! {
    #[test]
    fn synthesis_is_symmetric_for_commutative_operations(
        (operation, left, right) in commutative_strategy(),
    ) {
        let (identifiers, bindings) = broad_bindings();
        let registry = FunctionRegistry::new();
        let checker = TypeChecker::new(&bindings, &registry, &registry);
        let (left, right) = (build_broad(&left, &identifiers), build_broad(&right, &identifiers));
        let apply = |first: &Expression, second: &Expression| match COMMUTATIVE.get(operation) {
            Some(binary) => Expression::new_binary(*binary, first.clone(), second.clone()),
            None if operation == COMMUTATIVE.len() => first.and(second),
            None => first.or(second),
        };

        let forward = checker.synthesize(&apply(&left, &right)).ok();
        let backward = checker.synthesize(&apply(&right, &left)).ok();

        prop_assert_eq!(forward, backward);
    }

    #[test]
    fn a_well_typed_tree_s_type_has_the_evaluator_s_value_kind(
        tree in broad_strategy(),
        seeds in proptest::collection::vec(-6_i8..=6, BROAD_TYPES.len()),
    ) {
        let (identifiers, bindings) = broad_bindings();
        let registry = FunctionRegistry::new();
        let checker = TypeChecker::new(&bindings, &registry, &registry);
        let expression = build_broad(&tree, &identifiers);
        let Ok((synthesized, _)) = checker.synthesize(&expression) else {
            return Ok(());
        };
        let environment: HashMap<Identifier, Scalar> = identifiers
            .iter()
            .zip(BROAD_TYPES)
            .zip(&seeds)
            .map(|((identifier, core), seed)| (identifier.clone(), broad_value(core, *seed)))
            .collect();
        let Ok(value) = Evaluator::new(&registry).evaluate(&expression, &environment) else {
            // A division by zero or an overflow: no value to compare.
            return Ok(());
        };

        let Type::Numerical(numerical) = &synthesized else {
            panic!("a numerical type, got {synthesized}");
        };
        let fhy_core::types::DataType::Primitive(core) = numerical.data_type() else {
            panic!("a primitive type, got {synthesized}");
        };
        prop_assert_eq!(kind_of_type(*core), kind_of_value(value), "{} : {}", expression, synthesized);
    }
}
