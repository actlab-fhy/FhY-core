//! Property tests for printing expressions.
//!
//! Covers a JSON round trip printing identically, every free identifier's
//! name hint (and id, when shown) appearing in the text, ids changing only
//! the identifier references, one bracket pair per inner node, and a literal
//! printing as its `Display` text.
//!
//! Public API only (`fhy_core::symbolic::expression`).

use std::sync::LazyLock;

use fhy_core::identifier::Identifier;
use fhy_core::symbolic::expression::{
    BinaryOperation, Expression, ExpressionKind, FormatOptions, IdentifierStyle, LiteralKind,
    LiteralValue, Notation, UnaryOperation, build_call, build_piecewise, format_expression,
};
use num_bigint::BigInt;
use proptest::prelude::*;
use proptest::sample::select;

/// Identifiers the generated trees refer to, with distinct name hints.
static POOL: LazyLock<[Identifier; 3]> = LazyLock::new(|| {
    [
        Identifier::new("v0"),
        Identifier::new("v1"),
        Identifier::new("v2"),
    ]
});

/// Both notations.
const NOTATIONS: [Notation; 2] = [Notation::Symbolic, Notation::Functional];

/// Every option combination.
const ALL_OPTIONS: [(Notation, IdentifierStyle); 4] = [
    (Notation::Symbolic, IdentifierStyle::NameHint),
    (Notation::Symbolic, IdentifierStyle::NameHintWithId),
    (Notation::Functional, IdentifierStyle::NameHint),
    (Notation::Functional, IdentifierStyle::NameHintWithId),
];

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

/// The number of inner nodes of a tree, by kind.
#[derive(Debug, Default, PartialEq, Eq)]
struct InnerNodeCounts {
    unary: usize,
    binary: usize,
    piecewise: usize,
    call: usize,
}

/// Return how many inner nodes of each kind `expression` holds, counting a
/// shared subtree at every occurrence.
fn count_inner_nodes(expression: &Expression) -> InnerNodeCounts {
    let mut counts = InnerNodeCounts::default();
    let mut pending = vec![expression];
    while let Some(node) = pending.pop() {
        match node.kind() {
            ExpressionKind::Unary(_) => counts.unary += 1,
            ExpressionKind::Binary(_) => counts.binary += 1,
            ExpressionKind::Piecewise(_) => counts.piecewise += 1,
            ExpressionKind::Call(_) => counts.call += 1,
            ExpressionKind::Identifier(_) | ExpressionKind::Literal(_) => {}
        }
        pending.extend(node.children());
    }
    counts
}

/// Return how many times `character` occurs in `text`.
fn count_occurrences(text: &str, character: char) -> usize {
    text.chars()
        .filter(|&candidate| candidate == character)
        .count()
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

/// Return a strategy for trees over [`POOL`] of every node kind; float
/// literals are finite unless `with_non_finite_floats` is set.
fn build_expression_strategy(with_non_finite_floats: bool) -> BoxedStrategy<Expression> {
    let leaf = prop_oneof![
        (0..POOL.len()).prop_map(|index| Expression::from(POOL[index].clone())),
        build_literal_strategy(with_non_finite_floats).prop_map(Expression::from),
    ];
    leaf.prop_recursive(5, 32, 4, |inner| {
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
                    1..4
                ),
                inner.clone(),
            )
                .prop_map(|(cases, otherwise)| {
                    build_piecewise(cases, otherwise).expect("conditions are coerced")
                }),
            (select(vec!["f", "g"]), prop::collection::vec(inner, 0..4)).prop_map(
                |(function_name, arguments)| {
                    build_call(function_name, arguments).expect("a named call")
                }
            ),
        ]
    })
    .boxed()
}

proptest! {
    /// Test a tree read back from its JSON form prints exactly as the
    /// original under every option combination.
    #[test]
    fn format_expression_of_a_json_round_trip_is_unchanged(
        expression in build_expression_strategy(false),
    ) {
        let json = serde_json::to_string(&expression).expect("a finite tree serializes");
        let restored: Expression = serde_json::from_str(&json).expect("its own JSON decodes");

        for (notation, identifiers) in ALL_OPTIONS {
            let options = FormatOptions::new(notation, identifiers);
            prop_assert_eq!(
                format_expression(&restored, options),
                format_expression(&expression, options),
                "under {:?}", options
            );
        }
    }

    /// Test every free identifier's name hint appears in the text written
    /// with name hints, in both notations.
    #[test]
    fn format_expression_writes_every_free_identifier_name_hint(
        expression in build_expression_strategy(true),
    ) {
        for notation in NOTATIONS {
            let text = format_expression(
                &expression,
                FormatOptions::new(notation, IdentifierStyle::NameHint),
            );

            for identifier in expression.free_identifiers() {
                prop_assert!(
                    text.contains(identifier.name_hint()),
                    "{} missing from {:?}", identifier.name_hint(), text
                );
            }
        }
    }

    /// Test every free identifier appears as `name::id` in the text written
    /// with ids, in both notations.
    #[test]
    fn format_expression_writes_every_free_identifier_with_its_id(
        expression in build_expression_strategy(true),
    ) {
        for notation in NOTATIONS {
            let text = format_expression(
                &expression,
                FormatOptions::new(notation, IdentifierStyle::NameHintWithId),
            );

            for identifier in expression.free_identifiers() {
                let written = format!("{}::{}", identifier.name_hint(), identifier.id());
                prop_assert!(text.contains(&written), "{} missing from {:?}", written, text);
            }
        }
    }

    /// Test showing ids changes only the identifier references: removing
    /// each `::id` from the text written with ids gives the text written
    /// with name hints.
    #[test]
    fn format_expression_ids_change_only_identifier_references(
        expression in build_expression_strategy(true),
    ) {
        for notation in NOTATIONS {
            let with_ids = format_expression(
                &expression,
                FormatOptions::new(notation, IdentifierStyle::NameHintWithId),
            );
            let name_hints = format_expression(
                &expression,
                FormatOptions::new(notation, IdentifierStyle::NameHint),
            );

            let stripped = POOL.iter().fold(with_ids, |text, identifier| {
                text.replace(
                    &format!("{}::{}", identifier.name_hint(), identifier.id()),
                    identifier.name_hint(),
                )
            });

            prop_assert_eq!(stripped, name_hints, "under {:?}", notation);
        }
    }

    /// Test functional notation writes one pair of parentheses per inner
    /// node and none for a leaf.
    #[test]
    fn format_expression_writes_one_parenthesis_pair_per_inner_node_functionally(
        expression in build_expression_strategy(true),
    ) {
        let counts = count_inner_nodes(&expression);
        let inner_nodes = counts.unary + counts.binary + counts.piecewise + counts.call;

        let text = format_expression(
            &expression,
            FormatOptions::new(Notation::Functional, IdentifierStyle::NameHint),
        );

        prop_assert_eq!(count_occurrences(&text, '('), inner_nodes);
        prop_assert_eq!(count_occurrences(&text, ')'), inner_nodes);
        prop_assert_eq!(count_occurrences(&text, '{'), 0);
    }

    /// Test symbolic notation writes one pair of parentheses per unary,
    /// binary, and call node, and one pair of braces per piecewise node.
    #[test]
    fn format_expression_writes_one_bracket_pair_per_inner_node_symbolically(
        expression in build_expression_strategy(true),
    ) {
        let counts = count_inner_nodes(&expression);
        let parenthesized = counts.unary + counts.binary + counts.call;

        let text = format_expression(
            &expression,
            FormatOptions::new(Notation::Symbolic, IdentifierStyle::NameHint),
        );

        prop_assert_eq!(count_occurrences(&text, '('), parenthesized);
        prop_assert_eq!(count_occurrences(&text, ')'), parenthesized);
        prop_assert_eq!(count_occurrences(&text, '{'), counts.piecewise);
        prop_assert_eq!(count_occurrences(&text, '}'), counts.piecewise);
    }

    /// Test a literal prints as its `Display` text under every option
    /// combination.
    #[test]
    fn format_expression_writes_a_literal_as_its_display_text(
        value in build_literal_strategy(true),
    ) {
        let expected = value.to_string();
        let literal = Expression::from(value);

        for (notation, identifiers) in ALL_OPTIONS {
            let options = FormatOptions::new(notation, identifiers);
            prop_assert_eq!(format_expression(&literal, options), expected.as_str());
        }
    }
}
