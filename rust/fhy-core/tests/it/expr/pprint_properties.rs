//! Property tests for printing expressions: a JSON round trip printing
//! identically, every free identifier's name hint (and id, when shown)
//! appearing in the text, ids changing only the identifier references, one
//! bracket pair per inner node, and a literal printing as its `Display` text.

use crate::support::expression as expression_support;

use expression_support::{
    IDENTIFIER_POOL as POOL, build_expression_strategy, build_literal_strategy,
};
use fhy_core::expr::{Expression, ExpressionKind, FormatOptions, IdentifierStyle, Notation};
use proptest::prelude::*;

const NOTATIONS: [Notation; 2] = [Notation::Symbolic, Notation::Functional];

const ALL_OPTIONS: [(Notation, IdentifierStyle); 4] = [
    (Notation::Symbolic, IdentifierStyle::NameHint),
    (Notation::Symbolic, IdentifierStyle::NameHintWithId),
    (Notation::Functional, IdentifierStyle::NameHint),
    (Notation::Functional, IdentifierStyle::NameHintWithId),
];

#[derive(Debug, Default, PartialEq, Eq)]
struct InnerNodeCounts {
    unary: usize,
    binary: usize,
    logical: usize,
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
            ExpressionKind::Logical(_) => counts.logical += 1,
            ExpressionKind::Piecewise(_) => counts.piecewise += 1,
            ExpressionKind::Call(_) => counts.call += 1,
            ExpressionKind::Identifier(_) | ExpressionKind::Literal(_) => {}
        }
        pending.extend(node.children());
    }
    counts
}

fn build_options((notation, identifiers): (Notation, IdentifierStyle)) -> FormatOptions {
    FormatOptions::default()
        .with_notation(notation)
        .with_identifier_style(identifiers)
}

fn count_occurrences(text: &str, character: char) -> usize {
    text.chars()
        .filter(|&candidate| candidate == character)
        .count()
}

proptest! {
    /// Test a tree read back from its JSON form prints as the original under
    /// every option combination.
    #[test]
    fn format_expression_of_a_json_round_trip_is_unchanged(
        expression in build_expression_strategy(true),
    ) {
        let json = serde_json::to_string(&expression).expect("every tree serializes");
        let restored: Expression = serde_json::from_str(&json).expect("its own JSON decodes");

        for options in ALL_OPTIONS.map(build_options) {
            prop_assert_eq!(
                restored.display(options).to_string(),
                expression.display(options).to_string(),
                "under {:?}", options
            );
        }
    }

    #[test]
    fn format_expression_writes_every_free_identifier_name_hint(
        expression in build_expression_strategy(true),
    ) {
        for notation in NOTATIONS {
            let options = FormatOptions::default().with_notation(notation);
            let text = expression.display(options).to_string();

            for identifier in expression.free_identifiers() {
                prop_assert!(
                    text.contains(identifier.name_hint()),
                    "{} missing from {:?}", identifier.name_hint(), text
                );
            }
        }
    }

    #[test]
    fn format_expression_writes_every_free_identifier_with_its_id(
        expression in build_expression_strategy(true),
    ) {
        for notation in NOTATIONS {
            let options = FormatOptions::default()
                .with_notation(notation)
                .with_identifier_style(IdentifierStyle::NameHintWithId);
            let text = expression.display(options).to_string();

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
            let name_hint_options = FormatOptions::default().with_notation(notation);
            let id_options =
                name_hint_options.with_identifier_style(IdentifierStyle::NameHintWithId);
            let with_ids = expression.display(id_options).to_string();
            let name_hints = expression.display(name_hint_options).to_string();

            let stripped = POOL.iter().fold(with_ids, |text, identifier| {
                text.replace(
                    &format!("{}::{}", identifier.name_hint(), identifier.id()),
                    identifier.name_hint(),
                )
            });

            prop_assert_eq!(stripped, name_hints, "under {:?}", notation);
        }
    }

    /// Test functional notation writes one pair of parentheses per inner node
    /// and no braces.
    #[test]
    fn format_expression_writes_one_parenthesis_pair_per_inner_node_functionally(
        expression in build_expression_strategy(true),
    ) {
        let counts = count_inner_nodes(&expression);
        let inner_nodes =
            counts.unary + counts.binary + counts.logical + counts.piecewise + counts.call;
        let options = FormatOptions::default().with_notation(Notation::Functional);

        let text = expression.display(options).to_string();

        prop_assert_eq!(count_occurrences(&text, '('), inner_nodes);
        prop_assert_eq!(count_occurrences(&text, ')'), inner_nodes);
        prop_assert_eq!(count_occurrences(&text, '{'), 0);
    }

    /// Test symbolic notation writes one pair of parentheses per unary,
    /// binary, logical, and call node, and one pair of braces per piecewise
    /// node.
    #[test]
    fn format_expression_writes_one_bracket_pair_per_inner_node_symbolically(
        expression in build_expression_strategy(true),
    ) {
        let counts = count_inner_nodes(&expression);
        let parenthesized = counts.unary + counts.binary + counts.logical + counts.call;
        let options = FormatOptions::default().with_notation(Notation::Symbolic);

        let text = expression.display(options).to_string();

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

        for options in ALL_OPTIONS.map(build_options) {
            prop_assert_eq!(literal.display(options).to_string(), expected.as_str());
        }
    }
}
