//! Property tests for printing expressions.
//!
//! Covers a JSON round trip printing identically, every free identifier's
//! name hint (and id, when shown) appearing in the text, ids changing only
//! the identifier references, one bracket pair per inner node, and a literal
//! printing as its `Display` text.
//!
//! Public API only (`fhy_core::expr`).

use crate::support::expression as expression_support;

use expression_support::{
    IDENTIFIER_POOL as POOL, build_expression_strategy, build_literal_strategy,
};
use fhy_core::expr::{Expression, ExpressionKind, FormatOptions, IdentifierStyle, Notation};
use proptest::prelude::*;

/// Both notations.
const NOTATIONS: [Notation; 2] = [Notation::Symbolic, Notation::Functional];

/// Every option combination.
const ALL_OPTIONS: [(Notation, IdentifierStyle); 4] = [
    (Notation::Symbolic, IdentifierStyle::NameHint),
    (Notation::Symbolic, IdentifierStyle::NameHintWithId),
    (Notation::Functional, IdentifierStyle::NameHint),
    (Notation::Functional, IdentifierStyle::NameHintWithId),
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
            let options = FormatOptions::default()
.with_notation(notation)
.with_identifier_style(identifiers);
            prop_assert_eq!(
                restored.display(options).to_string(),
                expression.display(options).to_string(),
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
            let text = expression.display(FormatOptions::default().with_notation(notation)).to_string();

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
            let text = expression.display(FormatOptions::default()
.with_notation(notation)
.with_identifier_style(IdentifierStyle::NameHintWithId)).to_string();

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
            let with_ids = expression.display(FormatOptions::default()
.with_notation(notation)
.with_identifier_style(IdentifierStyle::NameHintWithId)).to_string();
            let name_hints = expression.display(FormatOptions::default().with_notation(notation)).to_string();

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

        let text = expression.display(FormatOptions::default().with_notation(Notation::Functional)).to_string();

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

        let text = expression.display(FormatOptions::default().with_notation(Notation::Symbolic)).to_string();

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
            let options = FormatOptions::default()
.with_notation(notation)
.with_identifier_style(identifiers);
            prop_assert_eq!(literal.display(options).to_string(), expected.as_str());
        }
    }
}
