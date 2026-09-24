//! Property tests for expressions and literal values.
//!
//! Covers the free-identifier law of substitution, structural equality as an
//! equivalence consistent with hashing, rebuilding and substituting as
//! identities, the JSON round trip, renaming free identifiers, the laws
//! of literal equality, canonical keys, the integer-bucket predicate, and
//! text `Display`, and, over DAGs sharing their subtrees at random, that
//! substitution answers as it does for an unshared copy.
//!
//! Public API only (`fhy_core::symbolic::expression`).

#[path = "common/expression.rs"]
pub mod expression_support;
#[path = "common/hashing.rs"]
pub mod hashing_support;

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use expression_support::{
    IDENTIFIER_POOL as POOL, build_expression_dag_strategy, build_expression_strategy,
    build_literal_strategy, coerce_to_condition, copy_deeply,
};
use fhy_core::identifier::Identifier;
use fhy_core::symbolic::expression::{
    AlphaRenaming, Expression, ExpressionBuildError, ExpressionKind, LiteralKind, LiteralValue,
    build_piecewise,
};
use hashing_support::hash_of;
use proptest::prelude::*;
use proptest::sample::select;

/// Identifiers no generated tree refers to, the targets of renamings.
static FRESH_POOL: LazyLock<[Identifier; 3]> = LazyLock::new(|| {
    [
        Identifier::new("w0"),
        Identifier::new("w1"),
        Identifier::new("w2"),
    ]
});

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

/// Return whether substituting `substitution` into `expression` puts a
/// literal other than a Boolean in a piecewise case condition: whether some
/// case condition is a reference to an identifier mapped to such a literal.
fn does_substitution_break_a_condition(
    expression: &Expression,
    substitution: &HashMap<Identifier, Expression>,
) -> bool {
    let breaks_here = match expression.kind() {
        ExpressionKind::Piecewise(piecewise) => piecewise.cases().iter().any(|(condition, _)| {
            let ExpressionKind::Identifier(identifier) = condition.kind() else {
                return false;
            };
            substitution.get(identifier).is_some_and(|replacement| {
                matches!(
                    replacement.kind(),
                    ExpressionKind::Literal(literal) if !matches!(literal.kind(), LiteralKind::Bool(_))
                )
            })
        }),
        _ => false,
    };
    breaks_here
        || expression
            .children()
            .any(|child| does_substitution_break_a_condition(child, substitution))
}

/// The renamings of [`POOL`] onto itself other than the identity, as
/// permutations of pool indices: the three swaps and the two rotations.
const POOL_PERMUTATIONS: [[usize; 3]; 5] = [[1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [2, 0, 1]];

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
        (-1000_i64..=1000).prop_map(|value| Expression::from(LiteralValue::from(value))),
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
    /// Test substitution is refused exactly when it puts a literal other
    /// than a Boolean in a piecewise case condition, and otherwise removes
    /// the mapped identifiers and brings in the free identifiers of the
    /// replacements it uses, and nothing else.
    #[test]
    fn expression_substitute_updates_free_identifiers_per_specification(
        expression in build_expression_strategy(false),
        domain in prop::sample::subsequence(vec![0_usize, 1, 2], 1..=2),
        replacements in prop::collection::vec(
            prop_oneof![
                build_expression_strategy(false),
                build_literal_strategy(false).prop_map(Expression::from),
            ],
            2,
        ),
    ) {
        let substitution: HashMap<Identifier, Expression> = domain
            .iter()
            .zip(replacements)
            .map(|(index, replacement)| (POOL[*index].clone(), replacement))
            .collect();
        let is_refused = does_substitution_break_a_condition(&expression, &substitution);
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

        match substituted {
            Ok(substituted) => {
                prop_assert!(!is_refused, "a number reached a case condition unrefused");
                prop_assert_eq!(substituted.free_identifiers(), expected);
            }
            Err(error) => {
                prop_assert!(is_refused, "refused with {:?}", error);
                prop_assert!(
                    matches!(error, ExpressionBuildError::NonBooleanConditionLiteral { .. }),
                    "refused with {:?}",
                    error
                );
            }
        }
    }

    /// Test a tree equals, and is equivalent under the empty renaming to,
    /// itself and a copy sharing no node with it, NaN literals included,
    /// and the two hash equally.
    #[test]
    fn expression_equality_is_reflexive_and_agrees_with_hash(
        expression in build_expression_strategy(true),
    ) {
        let copy = copy_deeply(&expression);

        prop_assert_eq!(&copy, &expression);
        prop_assert_eq!(hash_of(&copy), hash_of(&expression));
        prop_assert!(expression.is_alpha_equivalent_under(&expression, &AlphaRenaming::default()));
        prop_assert!(expression.is_alpha_equivalent_under(&copy, &AlphaRenaming::default()));
    }

    /// Test equality and equivalence under the empty renaming answer alike
    /// in both directions.
    #[test]
    fn expression_equality_is_symmetric(
        left in build_expression_strategy(true),
        right in build_expression_strategy(true),
    ) {
        let left_copy = copy_deeply(&left);
        let empty = AlphaRenaming::default();

        prop_assert_eq!(left == right, right == left);
        prop_assert_eq!(left == left_copy, left_copy == left);
        prop_assert_eq!(
            left.is_alpha_equivalent_under(&right, &empty),
            right.is_alpha_equivalent_under(&left, &empty)
        );
        prop_assert_eq!(
            left.is_alpha_equivalent_under(&left_copy, &empty),
            left_copy.is_alpha_equivalent_under(&left, &empty)
        );
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

    /// Test re-encoding a decoded tree reproduces the same JSON text.
    #[test]
    fn expression_json_text_is_stable_across_a_round_trip(
        expression in build_expression_strategy(false),
    ) {
        let text = serde_json::to_string(&expression).expect("finite trees serialize");
        let restored: Expression = serde_json::from_str(&text).expect("the text decodes");

        let re_encoded = serde_json::to_string(&restored).expect("finite trees serialize");

        prop_assert_eq!(re_encoded, text);
    }

    /// Test a random piecewise tree survives a JSON round trip.
    #[test]
    fn expression_piecewise_tree_round_trips_through_json(expression in build_piecewise_strategy()) {
        let text = serde_json::to_string(&expression).expect("finite trees serialize");

        let restored: Expression = serde_json::from_str(&text).expect("the text decodes");

        prop_assert_eq!(restored, expression);
    }

    /// Test renaming every pool identifier to a fresh one is equivalence
    /// under that renaming, and of the renamed tree to the original under
    /// the inverse renaming, and, when the tree has a free identifier, is
    /// neither equality nor equivalence under no renaming.
    #[test]
    fn expression_renaming_free_identifiers_holds_only_under_the_declared_renaming(
        expression in build_expression_strategy(false),
    ) {
        let pairs: HashMap<Identifier, Identifier> =
            POOL.iter().cloned().zip(FRESH_POOL.iter().cloned()).collect();
        let inverse_pairs: HashMap<Identifier, Identifier> =
            pairs.iter().map(|(from, to)| (to.clone(), from.clone())).collect();
        let substitution: HashMap<Identifier, Expression> = pairs
            .iter()
            .map(|(from, to)| (from.clone(), Expression::from(to.clone())))
            .collect();
        let renaming = AlphaRenaming::try_new(pairs).expect("the pools are distinct");
        let inverse = AlphaRenaming::try_new(inverse_pairs).expect("the pools are distinct");
        let renamed = expression.substitute(&substitution).expect("identifiers replace identifiers");

        let under_renaming = expression.is_alpha_equivalent_under(&renamed, &renaming);
        let under_inverse = renamed.is_alpha_equivalent_under(&expression, &inverse);
        let under_no_renaming =
            expression.is_alpha_equivalent_under(&renamed, &AlphaRenaming::default());

        prop_assert!(under_renaming);
        prop_assert!(under_inverse);
        let has_free_identifiers = !expression.free_identifiers().is_empty();
        prop_assert_eq!(expression == renamed, !has_free_identifiers);
        prop_assert_eq!(under_no_renaming, !has_free_identifiers);
    }

    /// Test renaming the pool identifiers among themselves, onto
    /// identifiers the tree also refers to, is equivalence under that
    /// renaming and of the renamed tree to the original under the inverse,
    /// and is equality exactly when the renaming fixes every free identifier
    /// of the tree.
    #[test]
    fn expression_renaming_within_the_tree_identifiers_holds_under_the_declared_renaming(
        expression in build_expression_strategy(false),
        permutation in select(POOL_PERMUTATIONS.to_vec()),
    ) {
        let pairs: HashMap<Identifier, Identifier> = permutation
            .iter()
            .enumerate()
            .filter(|(from, to)| from != *to)
            .map(|(from, to)| (POOL[from].clone(), POOL[*to].clone()))
            .collect();
        let inverse_pairs: HashMap<Identifier, Identifier> =
            pairs.iter().map(|(from, to)| (to.clone(), from.clone())).collect();
        let substitution: HashMap<Identifier, Expression> = pairs
            .iter()
            .map(|(from, to)| (from.clone(), Expression::from(to.clone())))
            .collect();
        let is_fixed = expression
            .free_identifiers()
            .iter()
            .all(|identifier| !pairs.contains_key(identifier));
        let renaming = AlphaRenaming::try_new(pairs).expect("a permutation is injective");
        let inverse = AlphaRenaming::try_new(inverse_pairs).expect("a permutation is injective");
        let renamed = expression.substitute(&substitution).expect("identifiers replace identifiers");

        let under_renaming = expression.is_alpha_equivalent_under(&renamed, &renaming);
        let under_inverse = renamed.is_alpha_equivalent_under(&expression, &inverse);

        prop_assert!(under_renaming);
        prop_assert!(under_inverse);
        prop_assert_eq!(expression == renamed, is_fixed);
    }

    /// Test an integer and its digit text, zero-padded or not, share a key.
    #[test]
    fn literal_value_canonical_key_agrees_for_integer_and_digit_text(
        value in prop_oneof![0_i64..=1000, 0_i64..=i64::MAX],
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

    /// Test two literals of any kind and size are equal exactly when their
    /// keys are, and equal literals hash equally.
    #[test]
    fn literal_value_equality_agrees_with_key_and_hash_over_every_literal(
        left in build_literal_strategy(true),
        right in build_literal_strategy(true),
    ) {
        let equal = left == right;

        prop_assert_eq!(equal, left.canonical_key() == right.canonical_key());
        prop_assert_eq!(&left, &left.clone());
        if equal {
            prop_assert_eq!(hash_of(&left), hash_of(&right));
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

proptest! {
    /// Test substituting into a DAG gives what substituting into its
    /// unshared copy gives, the same tree or the same refusal.
    #[test]
    fn expression_substitute_into_a_dag_answers_as_for_its_unshared_copy(
        dag in build_expression_dag_strategy(),
        domain in prop::sample::subsequence(vec![0_usize, 1, 2], 1..=2),
        replacements in prop::collection::vec(
            prop_oneof![
                build_expression_dag_strategy(),
                build_literal_strategy(false).prop_map(Expression::from),
            ],
            2,
        ),
    ) {
        let substitution: HashMap<Identifier, Expression> = domain
            .iter()
            .zip(replacements)
            .map(|(index, replacement)| (POOL[*index].clone(), replacement))
            .collect();

        let substituted = dag.substitute(&substitution);

        prop_assert_eq!(substituted, copy_deeply(&dag).substitute(&substitution));
    }

    /// Test substituting for identifiers a DAG does not refer to returns the
    /// DAG itself.
    #[test]
    fn expression_substitute_replacing_nothing_returns_the_dag_itself(
        dag in build_expression_dag_strategy(),
        replacement in build_expression_dag_strategy(),
    ) {
        let absent: HashMap<Identifier, Expression> = POOL
            .iter()
            .filter(|identifier| !dag.free_identifiers().contains(*identifier))
            .map(|identifier| (identifier.clone(), replacement.clone()))
            .collect();

        let substituted = dag.substitute(&absent).expect("nothing is replaced");

        prop_assert!(Expression::ptr_eq(&substituted, &dag));
    }
}
