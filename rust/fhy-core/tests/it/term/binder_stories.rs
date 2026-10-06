//! Tests for `Binder`'s provided methods, over the test lambda calculus and
//! over a binder whose children are expressions: alpha equivalence up to
//! renaming the bound identifiers, free identifiers, capture-avoiding
//! substitution, and the refusal of a binder list that repeats an
//! identifier. Also `Expression` through the term traits.
//!
//! The lambda cases mirror `tests/test_binder.py` and the binder
//! cases of `tests/test_alpha_equivalence.py`.

use std::collections::{HashMap, HashSet};

use fhy_core::expression::{Expression, PiecewiseError};
use fhy_core::identifier::Identifier;
use fhy_core::term::{AlphaEquivalence, AlphaRenaming, Binder, FreeIdentifiers, Term};

use crate::support::lambda::{
    Alpha, Free, Lam, Lambda, app, block, build_identifiers, expect_lam, lam, var,
};

// =============================================================================
// Alpha equivalence
// =============================================================================

#[test]
fn identity_lambdas_over_different_parameters_are_alpha_equivalent() {
    let [x, y] = build_identifiers(["x", "y"]);

    assert!(lam([&x], var(&x)).alpha_equivalent(&lam([&y], var(&y))));
}

#[test]
fn lambdas_over_distinct_free_bodies_are_not_alpha_equivalent() {
    let [x, y, a, b] = build_identifiers(["x", "y", "a", "b"]);

    assert!(!lam([&x], var(&a)).alpha_equivalent(&lam([&y], var(&b))));
}

#[test]
fn lambdas_sharing_a_free_identifier_are_alpha_equivalent() {
    let [x, y, shared] = build_identifiers(["x", "y", "shared"]);

    assert!(lam([&x], var(&shared)).alpha_equivalent(&lam([&y], var(&shared))));
}

#[test]
fn lambdas_binding_different_numbers_of_identifiers_are_not_alpha_equivalent() {
    let [x, y, z] = build_identifiers(["x", "y", "z"]);

    assert!(!lam([&x], var(&x)).alpha_equivalent(&lam([&y, &z], var(&y))));
}

#[test]
fn blocks_with_different_numbers_of_children_are_not_alpha_equivalent() {
    let [x, y] = build_identifiers(["x", "y"]);

    let one = block([&x], vec![var(&x)]);
    let two = block([&y], vec![var(&y), var(&y)]);

    assert!(!one.alpha_equivalent(&two));
}

#[test]
fn block_alpha_equivalence_compares_every_child_under_the_frame() {
    let [x, y, free] = build_identifiers(["x", "y", "free"]);

    let left = block([&x], vec![var(&x), var(&x)]);
    let right = block([&y], vec![var(&y), var(&y)]);
    let mismatch = block([&y], vec![var(&y), var(&free)]);

    assert!(left.alpha_equivalent(&right));
    assert!(!left.alpha_equivalent(&mismatch));
}

#[test]
fn lambdas_nested_over_one_name_match_the_inner_binder() {
    let [x, a, b] = build_identifiers(["x", "a", "b"]);

    let left = lam([&x], lam([&x], var(&x)));

    assert!(left.alpha_equivalent(&lam([&a], lam([&b], var(&b)))));
    assert!(!left.alpha_equivalent(&lam([&a], lam([&b], var(&a)))));
}

#[test]
fn a_lambda_refuses_to_capture_a_free_identifier() {
    let [x, y] = build_identifiers(["x", "y"]);

    // `\x. y` binds nothing its body uses; `\y. y` is the identity.
    assert!(!lam([&x], var(&y)).alpha_equivalent(&lam([&y], var(&y))));
    assert!(!lam([&y], var(&y)).alpha_equivalent(&lam([&x], var(&y))));
}

#[test]
fn lambdas_swapping_their_parameters_and_arguments_are_alpha_equivalent() {
    let [x, y, a, b] = build_identifiers(["x", "y", "a", "b"]);

    let left = lam([&x, &y], app(var(&x), var(&y)));
    let right = lam([&b, &a], app(var(&b), var(&a)));

    assert!(left.alpha_equivalent(&right));
}

#[test]
fn a_lambda_compares_its_free_identifiers_under_the_free_renaming() {
    let [x, y, free_a, free_b] = build_identifiers(["x", "y", "a", "b"]);
    let renaming = AlphaRenaming::new(HashMap::from([(free_a.clone(), free_b.clone())]))
        .expect("one pair is injective");

    let left = lam([&x], app(var(&x), var(&free_a)));
    let right = lam([&y], app(var(&y), var(&free_b)));

    assert!(left.alpha_equivalent_under(&right, &renaming));
    assert!(!left.alpha_equivalent(&right));
}

#[test]
fn a_lambda_is_not_alpha_equivalent_to_a_variable() {
    let [x] = build_identifiers(["x"]);

    assert!(!lam([&x], var(&x)).alpha_equivalent(&var(&x)));
}

// =============================================================================
// Repeated bound identifiers
// =============================================================================

#[test]
fn a_lambda_repeating_a_parameter_matches_no_lambda_on_either_side() {
    let [x, y, a, b] = build_identifiers(["x", "y", "a", "b"]);

    let repeats = lam([&x, &x], var(&x));
    let distinct = lam([&a, &b], var(&b));

    assert!(!repeats.alpha_equivalent(&distinct));
    assert!(!distinct.alpha_equivalent(&repeats));
    assert!(!lam([&x, &y], var(&x)).alpha_equivalent(&lam([&a, &a], var(&a))));
}

#[test]
fn a_lambda_repeating_a_parameter_is_not_alpha_equivalent_to_itself() {
    let [x] = build_identifiers(["x"]);
    let repeats = lam([&x, &x], var(&x));

    assert!(!repeats.alpha_equivalent(&repeats));
    assert!(!repeats.alpha_equivalent(&repeats.clone()));
}

#[test]
fn a_repeated_parameter_nested_inside_a_term_makes_the_whole_term_match_nothing() {
    let [x, f] = build_identifiers(["x", "f"]);
    let term = app(var(&f), lam([&x, &x], var(&x)));

    assert!(!term.alpha_equivalent(&term.clone()));
}

// =============================================================================
// Free identifiers
// =============================================================================

#[test]
fn free_identifiers_leave_out_a_bound_parameter() {
    let [x] = build_identifiers(["x"]);

    assert_eq!(lam([&x], var(&x)).free(), HashSet::new());
}

#[test]
fn free_identifiers_hold_an_unbound_body_reference() {
    let [x, y] = build_identifiers(["x", "y"]);

    assert_eq!(lam([&x], var(&y)).free(), HashSet::from([y]));
}

#[test]
fn free_identifiers_are_the_union_over_the_children_minus_the_bound_set() {
    let [x, y, z] = build_identifiers(["x", "y", "z"]);

    let nested = lam([&x], app(var(&x), app(var(&y), var(&z))));
    let multi = block([&x], vec![var(&x), var(&y), var(&z)]);

    assert_eq!(nested.free(), HashSet::from([y.clone(), z.clone()]));
    assert_eq!(multi.free(), HashSet::from([y, z]));
}

// =============================================================================
// Substitution
// =============================================================================

#[test]
fn substitute_leaves_a_key_the_lambda_binds_shadowed() {
    let [x, y] = build_identifiers(["x", "y"]);
    let identity = lam([&x], var(&x));

    let result = identity
        .substitute(&HashMap::from([(x, var(&y))]))
        .expect("infallible");

    assert!(expect_lam(&result).ptr_eq(expect_lam(&identity)));
}

#[test]
fn substitute_with_no_applying_key_returns_the_same_handle() {
    let [x, unrelated] = build_identifiers(["x", "unrelated"]);
    let identity = expect_lam(&lam([&x], var(&x))).clone();

    let empty: HashMap<Identifier, Lambda> = HashMap::new();
    let result = identity
        .substitute_avoiding_capture(&empty)
        .expect("infallible");
    let bound_only = identity
        .substitute_avoiding_capture(&HashMap::from([(x, var(&unrelated))]))
        .expect("infallible");

    assert!(result.ptr_eq(&identity));
    assert!(bound_only.ptr_eq(&identity));
}

#[test]
fn substitute_rewrites_a_free_identifier_of_the_body_in_place() {
    let [x, y, z] = build_identifiers(["x", "y", "z"]);
    let term = lam([&x], app(var(&x), var(&y)));

    let result = term
        .substitute(&HashMap::from([(y, var(&z))]))
        .expect("infallible");

    assert_eq!(result, lam([&x], app(var(&x), var(&z))));
}

#[test]
fn substitute_renames_a_binder_that_would_capture_a_replacement() {
    let [x, y] = build_identifiers(["x", "y"]);
    let term = lam([&x], var(&y));

    let result = term
        .substitute(&HashMap::from([(y, var(&x))]))
        .expect("infallible");

    let renamed = expect_lam(&result);
    let fresh = &renamed.parameters()[0];
    assert_ne!(fresh, &x);
    assert_eq!(fresh.name_hint(), "x");
    assert_eq!(renamed.body(), &[var(&x)]);
    assert_eq!(result.free(), HashSet::from([x.clone()]));
    assert!(!result.alpha_equivalent(&lam([&x], var(&x))));
}

#[test]
fn substitute_renames_only_the_parameters_a_replacement_would_capture() {
    let [x, y, z, w] = build_identifiers(["x", "y", "z", "w"]);
    let term = lam([&x, &z], app(var(&x), app(var(&z), var(&w))));

    let result = term
        .substitute(&HashMap::from([(w, var(&x))]))
        .expect("infallible");

    let renamed = expect_lam(&result);
    assert_ne!(renamed.parameters()[0], x);
    assert_eq!(renamed.parameters()[1], z);
    assert!(result.alpha_equivalent(&lam([&y, &z], app(var(&y), app(var(&z), var(&x))))));
}

/// A lambda whose `rename_bound_identifier` records each identifier it is
/// asked to rename, and whether the binder bound it.
#[derive(Debug, Clone)]
struct Recording {
    inner: Lam,
    renamed: std::sync::Arc<std::sync::Mutex<Vec<(Identifier, bool)>>>,
}

impl Binder for Recording {
    type Child = Lambda;
    type RebuildError = std::convert::Infallible;

    fn bound_identifiers(&self) -> &[Identifier] {
        self.inner.bound_identifiers()
    }

    fn scoped_children(&self) -> &[Lambda] {
        self.inner.scoped_children()
    }

    fn rename_bound_identifier(
        &self,
        old: &Identifier,
        new: Identifier,
    ) -> Result<Self, std::convert::Infallible> {
        let is_bound = self.inner.bound_identifiers().contains(old);
        self.renamed
            .lock()
            .expect("unpoisoned")
            .push((old.clone(), is_bound));
        Ok(Self {
            inner: self.inner.rename_bound_identifier(old, new)?,
            renamed: std::sync::Arc::clone(&self.renamed),
        })
    }

    fn rebuild_with_scoped_children(
        &self,
        children: Vec<Lambda>,
    ) -> Result<Self, std::convert::Infallible> {
        Ok(Self {
            inner: self.inner.rebuild_with_scoped_children(children)?,
            renamed: std::sync::Arc::clone(&self.renamed),
        })
    }
}

#[test]
fn a_substitution_that_replaces_nothing_inside_returns_the_same_handle() {
    let [x, y, z] = build_identifiers(["x", "y", "z"]);
    // `\y. x`: `z` does not occur, although its replacement mentions `y`.
    let binder = Recording {
        inner: Lam::new(vec![y.clone()], vec![var(&x)]),
        renamed: std::sync::Arc::default(),
    };

    let substituted = binder
        .substitute_avoiding_capture(&HashMap::from([(z.clone(), var(&y))]))
        .expect("infallible");

    assert!(substituted.inner.ptr_eq(&binder.inner));
    assert!(
        binder.renamed.lock().expect("unpoisoned").is_empty(),
        "nothing is renamed, so no fresh id is drawn"
    );
    let term = lam([&y], var(&x));
    let result = term
        .substitute(&HashMap::from([(z, var(&y))]))
        .expect("infallible");
    assert!(expect_lam(&result).ptr_eq(expect_lam(&term)));
}

#[test]
fn a_repeated_binder_is_renamed_once() {
    let [x, z] = build_identifiers(["x", "z"]);
    // `\x x. z` with `z := x`: `x` would be captured.
    let binder = Recording {
        inner: Lam::new(vec![x.clone(), x.clone()], vec![var(&z)]),
        renamed: std::sync::Arc::default(),
    };

    let substituted = binder
        .substitute_avoiding_capture(&HashMap::from([(z, var(&x))]))
        .expect("infallible");

    assert_eq!(
        *binder.renamed.lock().expect("unpoisoned"),
        vec![(x.clone(), true)]
    );
    let parameters = substituted.bound_identifiers();
    assert_eq!(parameters.len(), 2);
    assert_eq!(parameters[0], parameters[1]);
    assert_ne!(parameters[0], x);
    assert_eq!(substituted.scoped_children(), &[var(&x)]);
}

// =============================================================================
// A binder over expressions
// =============================================================================

/// A function-like binder: parameters over expression bodies, whose
/// rebuilding reports a piecewise refusal.
#[derive(Debug, Clone)]
struct Function {
    parameters: Vec<Identifier>,
    bodies: Vec<Expression>,
}

impl Binder for Function {
    type Child = Expression;
    type RebuildError = PiecewiseError;

    fn bound_identifiers(&self) -> &[Identifier] {
        &self.parameters
    }

    fn scoped_children(&self) -> &[Expression] {
        &self.bodies
    }

    fn rename_bound_identifier(
        &self,
        old: &Identifier,
        new: Identifier,
    ) -> Result<Self, PiecewiseError> {
        let renamed = HashMap::from([(old.clone(), Expression::from(new.clone()))]);
        Ok(Self {
            parameters: self
                .parameters
                .iter()
                .map(|parameter| {
                    if parameter == old {
                        new.clone()
                    } else {
                        parameter.clone()
                    }
                })
                .collect(),
            bodies: self
                .bodies
                .iter()
                .map(|body| body.substitute(&renamed))
                .collect::<Result<_, _>>()?,
        })
    }

    fn rebuild_with_scoped_children(
        &self,
        children: Vec<Expression>,
    ) -> Result<Self, PiecewiseError> {
        Ok(Self {
            parameters: self.parameters.clone(),
            bodies: children,
        })
    }
}

#[test]
fn a_binder_over_expressions_compares_its_bodies_under_the_parameter_frame() {
    let [x, y, z] = build_identifiers(["x", "y", "z"]);
    let left = Function {
        parameters: vec![x.clone()],
        bodies: vec![Expression::from(x) + Expression::from(z.clone())],
    };
    let right = Function {
        parameters: vec![y.clone()],
        bodies: vec![Expression::from(y) + Expression::from(z.clone())],
    };

    assert_eq!(
        left.is_binder_alpha_equivalent_under(&right, &AlphaRenaming::default()),
        Ok(true)
    );
    assert_eq!(left.binder_free_identifiers(), Ok(HashSet::from([z])));
}

#[test]
fn a_binder_over_expressions_substitutes_without_capture() {
    let [x, y] = build_identifiers(["x", "y"]);
    let function = Function {
        parameters: vec![x.clone()],
        bodies: vec![Expression::from(x.clone()) + Expression::from(y.clone())],
    };

    let result = function
        .substitute_avoiding_capture(&HashMap::from([(y.clone(), Expression::from(x.clone()))]))
        .expect("a sum never breaks a piecewise");

    assert_ne!(result.parameters[0], x);
    assert_eq!(
        result.binder_free_identifiers(),
        Ok(HashSet::from([x.clone()]))
    );
    let expected = Function {
        parameters: vec![y.clone()],
        bodies: vec![Expression::from(y) + Expression::from(x)],
    };
    assert_eq!(
        result.is_binder_alpha_equivalent_under(&expected, &AlphaRenaming::default()),
        Ok(true)
    );
}

#[test]
fn a_binder_over_expressions_reports_a_substitution_that_breaks_a_piecewise() {
    let [x, c] = build_identifiers(["x", "c"]);
    let function = Function {
        parameters: vec![x.clone()],
        bodies: vec![
            Expression::piecewise([(Expression::from(c.clone()), Expression::from(x))], 0)
                .expect("an identifier condition is valid"),
        ],
    };

    let error = function
        .substitute_avoiding_capture(&HashMap::from([(c, Expression::from(1))]))
        .expect_err("a numeric literal cannot be a condition");

    assert!(
        matches!(error, PiecewiseError::NonBooleanConditionLiteral { .. }),
        "{error:?}"
    );
}

// =============================================================================
// Expression through the traits
// =============================================================================

#[test]
fn expression_through_the_traits_agrees_with_its_methods() {
    let [x, y, z] = build_identifiers(["x", "y", "z"]);
    let left = Expression::from(x.clone()) + Expression::from(z.clone());
    let right = Expression::from(y.clone()) + Expression::from(z.clone());
    let mut renaming = AlphaRenaming::default();
    renaming
        .enter_binders(std::slice::from_ref(&x), std::slice::from_ref(&y))
        .expect("one pair");
    let replacements = HashMap::from([(z, Expression::from(2))]);

    let Ok(under_renaming) = AlphaEquivalence::is_alpha_equivalent_under(&left, &right, &renaming);
    let Ok(without_renaming) = AlphaEquivalence::is_alpha_equivalent(&left, &right);
    let Ok(with_itself) = AlphaEquivalence::is_alpha_equivalent(&left, &left.clone());
    let Ok(free) = FreeIdentifiers::free_identifiers(&left);
    assert!(under_renaming);
    assert!(!without_renaming);
    assert!(with_itself);
    assert_eq!(free, left.free_identifiers());
    assert_eq!(
        Term::substitute(&left, &replacements).expect("no piecewise"),
        left.substitute(&replacements).expect("no piecewise")
    );
}

#[test]
fn lam_helper_exposes_its_parameters_and_body() {
    let [x] = build_identifiers(["x"]);
    let term = Lam::new(vec![x.clone()], vec![var(&x)]);

    assert_eq!(term.parameters(), std::slice::from_ref(&x));
    assert_eq!(term.body(), &[var(&x)]);
}
