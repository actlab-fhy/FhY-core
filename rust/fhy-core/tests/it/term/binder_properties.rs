//! Property tests for `Binder`'s provided methods over the test lambda
//! calculus: alpha equivalence against a de Bruijn model, its laws, and
//! capture-avoiding substitution.
//!
//! A binder list that repeats an identifier pairs with nothing (N-S10-2
//! (b)), so reflexivity, transitivity and "structurally equal implies
//! alpha-equivalent" hold for terms whose binder lists repeat no
//! identifier. The properties of those laws state that precondition: their
//! generator, `build_distinct_term_strategy`, builds only such terms.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use fhy_core::identifier::Identifier;
use fhy_core::term::{AlphaEquivalence, Binder, FreeIdentifiers, Term};
use proptest::prelude::*;

use crate::support::lambda::{Lam, Lambda, app};

/// The identifiers the generated terms bind and reference, few enough that
/// binders often shadow each other and capture is often possible.
static POOL: LazyLock<[Identifier; 3]> = LazyLock::new(|| {
    [
        Identifier::new("p0"),
        Identifier::new("p1"),
        Identifier::new("p2"),
    ]
});

/// A term in de Bruijn form: a bound reference is the level of its binder,
/// innermost first, and its position in that binder's list.
#[derive(Debug, PartialEq, Eq)]
enum DeBruijn {
    Free(u64),
    Bound { level: usize, position: usize },
    App(Box<DeBruijn>, Box<DeBruijn>),
    Lam { arity: usize, body: Vec<DeBruijn> },
}

/// Return the de Bruijn form of `term` under the binder lists `scopes`,
/// outermost first, or `None` if a binder list in `term` repeats an
/// identifier, since such a list pairs with none.
fn to_de_bruijn(term: &Lambda, scopes: &mut Vec<Vec<Identifier>>) -> Option<DeBruijn> {
    match term {
        Lambda::Var(identifier) => Some(
            scopes
                .iter()
                .rev()
                .enumerate()
                .find_map(|(level, scope)| {
                    scope
                        .iter()
                        .position(|bound| bound == identifier)
                        .map(|position| DeBruijn::Bound { level, position })
                })
                .unwrap_or(DeBruijn::Free(identifier.id())),
        ),
        Lambda::App(function, argument) => Some(DeBruijn::App(
            Box::new(to_de_bruijn(function, scopes)?),
            Box::new(to_de_bruijn(argument, scopes)?),
        )),
        Lambda::Lam(lam) => {
            let parameters = lam.bound_identifiers();
            if parameters.iter().collect::<HashSet<_>>().len() != parameters.len() {
                return None;
            }
            scopes.push(parameters.to_vec());
            let body: Option<Vec<_>> = lam
                .scoped_children()
                .iter()
                .map(|child| to_de_bruijn(child, scopes))
                .collect();
            scopes.pop();
            Some(DeBruijn::Lam {
                arity: parameters.len(),
                body: body?,
            })
        }
    }
}

/// Return whether a binder list in `term` repeats an identifier.
fn has_repeated_binder(term: &Lambda) -> bool {
    to_de_bruijn(term, &mut Vec::new()).is_none()
}

/// Return `term` with every binder's parameters renamed to fresh
/// identifiers, which is alpha-equivalent to it when no binder list repeats
/// an identifier.
fn refresh(term: &Lambda) -> Lambda {
    match term {
        Lambda::Var(_) => term.clone(),
        Lambda::App(function, argument) => app(refresh(function), refresh(argument)),
        Lambda::Lam(lam) => {
            let mut renamed = lam.clone();
            for parameter in lam.bound_identifiers() {
                renamed = renamed
                    .rename_bound_identifier(parameter, Identifier::new(parameter.name_hint()))
                    .expect("infallible");
            }
            let body = renamed.scoped_children().iter().map(refresh).collect();
            Lambda::Lam(
                renamed
                    .rebuild_with_scoped_children(body)
                    .expect("infallible"),
            )
        }
    }
}

fn build_term_strategy(distinct_binders: bool) -> impl Strategy<Value = Lambda> {
    let leaf = (0..POOL.len()).prop_map(|index| Lambda::Var(POOL[index].clone()));
    leaf.prop_recursive(4, 24, 3, move |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(function, argument)| app(function, argument)),
            (
                prop::collection::vec(0..POOL.len(), 0..3),
                prop::collection::vec(inner, 1..3)
            )
                .prop_map(move |(indices, body)| {
                    let mut parameters: Vec<Identifier> = Vec::new();
                    for index in indices {
                        let parameter = &POOL[index];
                        if !distinct_binders || !parameters.contains(parameter) {
                            parameters.push(parameter.clone());
                        }
                    }
                    Lambda::Lam(Lam::new(parameters, body))
                }),
        ]
    })
}

/// Terms whose binder lists may repeat an identifier.
fn build_any_term_strategy() -> impl Strategy<Value = Lambda> {
    build_term_strategy(false)
}

/// Terms whose binder lists repeat no identifier: the precondition of the
/// laws.
fn build_distinct_term_strategy() -> impl Strategy<Value = Lambda> {
    build_term_strategy(true)
}

/// Return `term` with `replacements` substituted by the algorithm before
/// R2-035: every replacement's free identifiers are capturable, whether its
/// key occurs or not, and each bound identifier of the original list is
/// renamed when capturable.
fn substitute_as_before(term: &Lambda, replacements: &HashMap<Identifier, Lambda>) -> Lambda {
    match term {
        Lambda::Var(identifier) => replacements
            .get(identifier)
            .cloned()
            .unwrap_or_else(|| term.clone()),
        Lambda::App(function, argument) => app(
            substitute_as_before(function, replacements),
            substitute_as_before(argument, replacements),
        ),
        Lambda::Lam(lam) => {
            let bound: HashSet<&Identifier> = lam.bound_identifiers().iter().collect();
            let active: HashMap<Identifier, Lambda> = replacements
                .iter()
                .filter(|(identifier, _)| !bound.contains(identifier))
                .map(|(identifier, term)| (identifier.clone(), term.clone()))
                .collect();
            if active.is_empty() {
                return term.clone();
            }
            let capturable: HashSet<Identifier> = active
                .values()
                .flat_map(FreeIdentifiers::free_identifiers)
                .collect();
            let mut safe = lam.clone();
            for bound_identifier in lam.bound_identifiers() {
                if capturable.contains(bound_identifier) {
                    safe = safe
                        .rename_bound_identifier(
                            bound_identifier,
                            Identifier::new(bound_identifier.name_hint()),
                        )
                        .expect("infallible");
                }
            }
            let body = safe
                .scoped_children()
                .iter()
                .map(|child| substitute_as_before(child, &active))
                .collect();
            Lambda::Lam(safe.rebuild_with_scoped_children(body).expect("infallible"))
        }
    }
}

proptest! {
    /// Test a substitution's result is alpha-equivalent to the one the
    /// algorithm before R2-035 gave, which renamed more binders.
    #[test]
    fn substitution_agrees_with_the_algorithm_that_renamed_more(
        term in build_distinct_term_strategy(),
        replacement in build_distinct_term_strategy(),
        key in 0..POOL.len(),
    ) {
        let replacements = HashMap::from([(POOL[key].clone(), replacement)]);

        let result = term.substitute(&replacements).expect("infallible");
        let before = substitute_as_before(&term, &replacements);

        prop_assert!(result.is_alpha_equivalent(&before), "{result:?} against {before:?}");
    }

    /// Test alpha equivalence holds exactly when both terms have one de
    /// Bruijn form, repeated binders included.
    #[test]
    fn alpha_equivalence_agrees_with_the_de_bruijn_model(
        left in build_any_term_strategy(),
        right in build_any_term_strategy(),
    ) {
        let left_form = to_de_bruijn(&left, &mut Vec::new());
        let right_form = to_de_bruijn(&right, &mut Vec::new());
        let expected = left_form.is_some() && left_form == right_form;

        prop_assert_eq!(left.is_alpha_equivalent(&right), expected);
        prop_assert_eq!(left.is_alpha_equivalent(&refresh(&right)), expected);
    }

    /// Test alpha equivalence is symmetric, repeated binders included.
    #[test]
    fn alpha_equivalence_is_symmetric(
        left in build_any_term_strategy(),
        right in build_any_term_strategy(),
    ) {
        let right = if left == right { refresh(&left) } else { right };

        prop_assert_eq!(left.is_alpha_equivalent(&right), right.is_alpha_equivalent(&left));
    }

    /// Test a term whose binder list repeats an identifier is
    /// alpha-equivalent to no term, itself included. The term is a random
    /// term applied to, or applying, a lambda that binds one identifier
    /// twice, or that lambda alone.
    #[test]
    fn a_term_with_a_repeated_binder_is_alpha_equivalent_to_no_term(
        base in build_any_term_strategy(),
        body in build_any_term_strategy(),
        repeated in 0..POOL.len(),
        placement in 0..3_usize,
        other in build_any_term_strategy(),
    ) {
        let parameter = POOL[repeated].clone();
        let repeating = Lambda::Lam(Lam::new(vec![parameter.clone(), parameter], vec![body]));
        let term = match placement {
            0 => repeating,
            1 => app(repeating, base),
            _ => app(base, repeating),
        };
        prop_assert!(has_repeated_binder(&term));

        prop_assert!(!term.is_alpha_equivalent(&term));
        prop_assert!(!term.is_alpha_equivalent(&term.clone()));
        prop_assert!(!term.is_alpha_equivalent(&other));
        prop_assert!(!other.is_alpha_equivalent(&term));
    }

    /// Test alpha equivalence is reflexive, and implied by structural
    /// equality, on terms whose binder lists repeat no identifier.
    #[test]
    fn alpha_equivalence_is_reflexive_on_terms_without_repeated_binders(
        term in build_distinct_term_strategy(),
    ) {
        prop_assert!(!has_repeated_binder(&term), "the precondition of the law");

        prop_assert!(term.is_alpha_equivalent(&term));
        prop_assert!(term.is_alpha_equivalent(&term.clone()));
    }

    /// Test alpha equivalence is transitive over renamed copies, on terms
    /// whose binder lists repeat no identifier.
    #[test]
    fn alpha_equivalence_is_transitive_on_terms_without_repeated_binders(
        term in build_distinct_term_strategy(),
    ) {
        prop_assert!(!has_repeated_binder(&term), "the precondition of the law");
        let renamed = refresh(&term);
        let renamed_again = refresh(&renamed);

        prop_assert!(term.is_alpha_equivalent(&renamed));
        prop_assert!(renamed.is_alpha_equivalent(&renamed_again));
        prop_assert!(term.is_alpha_equivalent(&renamed_again));
    }

    /// Test substituting into two alpha-equivalent terms gives
    /// alpha-equivalent results, on terms whose binder lists repeat no
    /// identifier.
    #[test]
    fn substitution_respects_alpha_equivalence(
        term in build_distinct_term_strategy(),
        replacement in build_distinct_term_strategy(),
        key in 0..POOL.len(),
    ) {
        prop_assert!(!has_repeated_binder(&term), "the precondition of the law");
        let replacements = HashMap::from([(POOL[key].clone(), replacement)]);

        let left = term.substitute(&replacements).expect("infallible");
        let right = refresh(&term).substitute(&replacements).expect("infallible");

        prop_assert!(left.is_alpha_equivalent(&right));
    }

    /// Test a substitution captures nothing: the result's free identifiers
    /// are the term's minus the replaced key, plus the replacement's when
    /// the key occurs free.
    #[test]
    fn substitution_never_captures(
        term in build_any_term_strategy(),
        replacement in build_any_term_strategy(),
        key in 0..POOL.len(),
    ) {
        let key = POOL[key].clone();
        let mut expected = term.free_identifiers();
        if expected.remove(&key) {
            expected.extend(replacement.free_identifiers());
        }
        let replacements = HashMap::from([(key, replacement)]);

        let result = term.substitute(&replacements).expect("infallible");

        prop_assert_eq!(result.free_identifiers(), expected);
    }
}
