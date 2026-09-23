//! Property tests for expressions and literal values.
//!
//! Covers the free-identifier law of substitution, structural equality as an
//! equivalence consistent with hashing, rebuilding and substituting as
//! identities, the JSON round trip, renaming free identifiers, and the laws
//! of literal equality, canonical keys, the integer-bucket predicate, and
//! text `Display`.
//!
//! Public API only (`fhy_core::symbolic::expression`).

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::LazyLock;

use fhy_core::identifier::Identifier;
use fhy_core::symbolic::expression::{
    BinaryOperation, Expression, ExpressionKind, LiteralKind, LiteralValue, UnaryOperation,
    build_call, build_piecewise,
};
use num_bigint::BigInt;
use proptest::prelude::*;
use proptest::sample::select;

/// Identifiers the generated trees refer to.
static POOL: LazyLock<[Identifier; 3]> = LazyLock::new(|| {
    [
        Identifier::new("v0"),
        Identifier::new("v1"),
        Identifier::new("v2"),
    ]
});

/// Identifiers no generated tree refers to, the targets of renamings.
static FRESH_POOL: LazyLock<[Identifier; 3]> = LazyLock::new(|| {
    [
        Identifier::new("w0"),
        Identifier::new("w1"),
        Identifier::new("w2"),
    ]
});

/// Every unary operation.
const UNARY_OPERATIONS: [UnaryOperation; 3] = [
    UnaryOperation::Negate,
    UnaryOperation::Positive,
    UnaryOperation::LogicalNot,
];

/// Every binary operation.
const BINARY_OPERATIONS: [BinaryOperation; 15] = [
    BinaryOperation::Add,
    BinaryOperation::Subtract,
    BinaryOperation::Multiply,
    BinaryOperation::Divide,
    BinaryOperation::FloorDivide,
    BinaryOperation::Modulo,
    BinaryOperation::Power,
    BinaryOperation::LogicalAnd,
    BinaryOperation::LogicalOr,
    BinaryOperation::Equal,
    BinaryOperation::NotEqual,
    BinaryOperation::Less,
    BinaryOperation::LessEqual,
    BinaryOperation::Greater,
    BinaryOperation::GreaterEqual,
];

/// Literal spellings drawn from a small value space, so that equal literals
/// in different spellings and buckets are drawn often: `b:` Boolean, `i:`
/// integer, `f:` float, `t:` text.
const LITERAL_SPELLINGS: [&str; 22] = [
    "b:true", "b:false", "i:0", "i:1", "i:5", "t:0", "t:00", "t:5", "t:05", "t:1", "f:0.0",
    "f:-0.0", "f:5.0", "f:1.0", "f:NaN", "f:-NaN", "t:5.0", "t:5.00", "t:05.", "t:1.0", "t:.0",
    "t:0.0",
];

/// Build a literal from a spelling of [`LITERAL_SPELLINGS`].
fn build_spelled_literal(spelling: &str) -> LiteralValue {
    let (kind, value) = spelling.split_at(2);
    match kind {
        "b:" => LiteralValue::from(value == "true"),
        "i:" => LiteralValue::from(value.parse::<i64>().expect("an integer")),
        "f:" => LiteralValue::from(value.parse::<f64>().expect("a float")),
        "t:" => LiteralValue::parse_text(value).expect("a literal text"),
        _ => panic!("unknown literal spelling {spelling:?}"),
    }
}

/// Return the default hash of `value`.
fn hash_of<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Return a copy of `expression` sharing no node with it.
fn copy_deeply(expression: &Expression) -> Expression {
    match expression.kind() {
        ExpressionKind::Identifier(identifier) => Expression::from(identifier.clone()),
        ExpressionKind::Literal(literal) => Expression::from(literal.clone()),
        _ => expression
            .rebuild_with_children(expression.children().map(copy_deeply).collect())
            .expect("a node rebuilds from copies of its own children"),
    }
}

/// Return `expression` unless it is a literal other than a Boolean, which
/// becomes a Boolean literal, so it can stand as a case condition.
fn coerce_to_condition(expression: Expression) -> Expression {
    match expression.kind() {
        ExpressionKind::Literal(literal) if !matches!(literal.kind(), LiteralKind::Bool(_)) => {
            Expression::from(LiteralValue::from(true))
        }
        _ => expression,
    }
}

/// Return a strategy for literals of every kind; floats are finite unless
/// `with_non_finite_floats` is set.
fn build_literal_strategy(with_non_finite_floats: bool) -> BoxedStrategy<LiteralValue> {
    let finite_float = any::<f64>().prop_filter("a finite float", |value| value.is_finite());
    let float = if with_non_finite_floats {
        prop_oneof![
            4 => finite_float,
            1 => select(vec![f64::NAN, -f64::NAN, f64::INFINITY, f64::NEG_INFINITY]),
        ]
        .boxed()
    } else {
        finite_float.boxed()
    };
    prop_oneof![
        any::<bool>().prop_map(LiteralValue::from),
        (-1000_i64..1000).prop_map(LiteralValue::from),
        "-?[1-9][0-9]{18,40}".prop_map(|digits| {
            LiteralValue::from(digits.parse::<BigInt>().expect("generated digits"))
        }),
        float.prop_map(LiteralValue::from),
        "[0-9]{1,6}".prop_map(|text| LiteralValue::parse_text(&text).expect("an integer text")),
        "[0-9]{0,4}\\.[0-9]{1,4}|[0-9]{1,4}\\.[0-9]{0,4}"
            .prop_map(|text| LiteralValue::parse_text(&text).expect("a decimal text")),
    ]
    .boxed()
}

/// Return a strategy for trees over [`POOL`] of every node kind.
fn build_expression_strategy(with_non_finite_floats: bool) -> BoxedStrategy<Expression> {
    let leaf = prop_oneof![
        (0..POOL.len()).prop_map(|index| Expression::from(POOL[index].clone())),
        build_literal_strategy(with_non_finite_floats).prop_map(Expression::from),
    ];
    leaf.prop_recursive(4, 24, 3, |inner| {
        prop_oneof![
            (select(UNARY_OPERATIONS.to_vec()), inner.clone())
                .prop_map(|(operation, operand)| Expression::new_unary(operation, operand)),
            (
                select(BINARY_OPERATIONS.to_vec()),
                inner.clone(),
                inner.clone()
            )
                .prop_map(|(operation, left, right)| Expression::new_binary(
                    operation, left, right
                )),
            (
                prop::collection::vec(
                    (inner.clone().prop_map(coerce_to_condition), inner.clone()),
                    1..3
                ),
                inner.clone(),
            )
                .prop_map(|(cases, otherwise)| {
                    build_piecewise(cases, otherwise).expect("conditions are coerced")
                }),
            (select(vec!["f", "g"]), prop::collection::vec(inner, 0..3)).prop_map(
                |(function_name, arguments)| {
                    build_call(function_name, arguments).expect("a named call")
                }
            ),
        ]
    })
    .boxed()
}

/// Return a strategy for piecewise trees over integer and Boolean literals,
/// with a piecewise at the root.
fn build_piecewise_strategy() -> BoxedStrategy<Expression> {
    /// Build a piecewise from drawn cases, coercing each condition.
    fn build_node((cases, otherwise): (Vec<(Expression, Expression)>, Expression)) -> Expression {
        let cases: Vec<(Expression, Expression)> = cases
            .into_iter()
            .map(|(condition, value)| (coerce_to_condition(condition), value))
            .collect();
        build_piecewise(cases, otherwise).expect("conditions are coerced")
    }

    let leaf = prop_oneof![
        (-1000_i64..1000).prop_map(|value| Expression::from(LiteralValue::from(value))),
        any::<bool>().prop_map(|value| Expression::from(LiteralValue::from(value))),
    ];
    let tree = leaf.prop_recursive(3, 24, 7, |inner| {
        (
            prop::collection::vec((inner.clone(), inner.clone()), 1..4),
            inner,
        )
            .prop_map(build_node)
    });
    (
        prop::collection::vec((tree.clone(), tree.clone()), 1..4),
        tree,
    )
        .prop_map(build_node)
        .boxed()
}

proptest! {
    /// Test substitution removes the mapped identifiers and brings in the
    /// free identifiers of the replacements it uses, and nothing else.
    #[test]
    fn expression_substitute_updates_free_identifiers_per_specification(
        expression in build_expression_strategy(false),
        domain in prop::sample::subsequence(vec![0_usize, 1, 2], 1..=2),
        replacements in prop::collection::vec(
            build_expression_strategy(false).prop_map(coerce_to_condition),
            2,
        ),
    ) {
        let substitution: HashMap<Identifier, Expression> = domain
            .iter()
            .zip(replacements)
            .map(|(index, replacement)| (POOL[*index].clone(), replacement))
            .collect();
        let free_before = expression.free_identifiers();
        let mut expected: HashSet<Identifier> = free_before
            .iter()
            .filter(|identifier| !substitution.contains_key(identifier))
            .cloned()
            .collect();
        for (identifier, replacement) in &substitution {
            if free_before.contains(identifier) {
                expected.extend(replacement.free_identifiers());
            }
        }

        let substituted = expression.substitute(&substitution);

        let substituted = substituted.expect("no replacement is a literal other than a Boolean");
        prop_assert_eq!(substituted.free_identifiers(), expected);
    }

    /// Test a tree equals a copy sharing no node with it, NaN literals
    /// included, and the two hash equally.
    #[test]
    fn expression_equality_is_reflexive_and_agrees_with_hash(
        expression in build_expression_strategy(true),
    ) {
        let copy = copy_deeply(&expression);

        prop_assert_eq!(&copy, &expression);
        prop_assert_eq!(hash_of(&copy), hash_of(&expression));
    }

    /// Test equality answers alike in both directions.
    #[test]
    fn expression_equality_is_symmetric(
        left in build_expression_strategy(true),
        right in build_expression_strategy(true),
    ) {
        let left_copy = copy_deeply(&left);

        prop_assert_eq!(left == right, right == left);
        prop_assert_eq!(left == left_copy, left_copy == left);
    }

    /// Test rebuilding a node from its own children yields an equal tree.
    #[test]
    fn expression_rebuild_with_own_children_is_an_identity(
        expression in build_expression_strategy(true),
    ) {
        let children: Vec<Expression> = expression.children().cloned().collect();

        let rebuilt = expression.rebuild_with_children(children);

        prop_assert_eq!(rebuilt.expect("a node's own children rebuild it"), expression);
    }

    /// Test substituting with an empty map yields an equal tree.
    #[test]
    fn expression_substitute_with_empty_map_is_an_identity(
        expression in build_expression_strategy(true),
    ) {
        let substituted = expression.substitute(&HashMap::new());

        prop_assert_eq!(substituted.expect("nothing to refuse"), expression);
    }

    /// Test a JSON round trip yields an equal tree.
    #[test]
    fn expression_json_round_trip_is_an_identity(expression in build_expression_strategy(false)) {
        let wire = serde_json::to_value(&expression).expect("finite trees serialize");

        let restored: Expression = serde_json::from_value(wire).expect("the wire form decodes");

        prop_assert_eq!(restored, expression);
    }

    /// Test a random piecewise tree survives a JSON round trip.
    #[test]
    fn expression_piecewise_tree_round_trips_through_json(expression in build_piecewise_strategy()) {
        let text = serde_json::to_string(&expression).expect("finite trees serialize");

        let restored: Expression = serde_json::from_str(&text).expect("the text decodes");

        prop_assert_eq!(restored, expression);
    }

    /// Test renaming every pool identifier to a fresh one is equivalence
    /// under that renaming, and, when the tree has a free identifier, is
    /// neither equality nor equivalence under no renaming.
    #[test]
    fn expression_renaming_free_identifiers_holds_only_under_the_declared_renaming(
        expression in build_expression_strategy(false),
    ) {
        let renaming: HashMap<Identifier, Identifier> =
            POOL.iter().cloned().zip(FRESH_POOL.iter().cloned()).collect();
        let substitution: HashMap<Identifier, Expression> = renaming
            .iter()
            .map(|(from, to)| (from.clone(), Expression::from(to.clone())))
            .collect();
        let renamed = expression.substitute(&substitution).expect("identifiers replace identifiers");

        let under_renaming = expression.is_alpha_equivalent_under(&renamed, &renaming);
        let under_no_renaming = expression.is_alpha_equivalent_under(&renamed, &HashMap::new());

        prop_assert!(under_renaming);
        let has_free_identifiers = !expression.free_identifiers().is_empty();
        prop_assert_eq!(expression == renamed, !has_free_identifiers);
        prop_assert_eq!(under_no_renaming, !has_free_identifiers);
    }

    /// Test an integer and its digit text, zero-padded or not, share a key.
    #[test]
    fn literal_value_canonical_key_agrees_for_integer_and_digit_text(
        value in 0_i64..=1000,
        padding in 0_usize..=5,
    ) {
        let integer_key = LiteralValue::from(value).canonical_key();
        let text = value.to_string();
        let padded = format!("{}{text}", "0".repeat(padding));

        let text_key = LiteralValue::parse_text(&text).expect("digits").canonical_key();
        let padded_key = LiteralValue::parse_text(&padded).expect("digits").canonical_key();

        prop_assert_eq!(&integer_key, &text_key);
        prop_assert_eq!(&integer_key, &padded_key);
    }

    /// Test appending zeros after the decimal point keeps the key.
    #[test]
    fn literal_value_canonical_key_ignores_trailing_decimal_zeros(
        base in "[0-9]{1,6}\\.[0-9]{0,6}|\\.[0-9]{1,6}",
        extra_zeros in 0_usize..=5,
    ) {
        let padded = format!("{base}{}", "0".repeat(extra_zeros));

        let base_key = LiteralValue::parse_text(&base).expect("a decimal text").canonical_key();
        let padded_key = LiteralValue::parse_text(&padded).expect("a decimal text").canonical_key();

        prop_assert_eq!(base_key, padded_key);
    }

    /// Test two literals are equal exactly when their keys are, equal
    /// literals hash equally, and the integer-bucket predicate agrees on
    /// them.
    #[test]
    fn literal_value_equality_agrees_with_key_hash_and_bucket(
        left in select(LITERAL_SPELLINGS.to_vec()),
        right in select(LITERAL_SPELLINGS.to_vec()),
    ) {
        let left = build_spelled_literal(left);
        let right = build_spelled_literal(right);

        let equal = left == right;

        prop_assert_eq!(equal, left.canonical_key() == right.canonical_key());
        if equal {
            prop_assert_eq!(hash_of(&left), hash_of(&right));
            prop_assert_eq!(left.is_integer_valued(), right.is_integer_valued());
        }
    }

    /// Test a literal text displays as given and an integer displays as its
    /// digits.
    #[test]
    fn literal_value_display_writes_texts_verbatim_and_integers_as_digits(
        text in "[0-9]{1,8}(\\.[0-9]{0,8})?|\\.[0-9]{1,8}",
        integer in any::<i64>(),
    ) {
        let text_literal = LiteralValue::parse_text(&text).expect("a literal text");

        prop_assert_eq!(text_literal.to_string(), text);
        prop_assert_eq!(LiteralValue::from(integer).to_string(), integer.to_string());
    }
}
