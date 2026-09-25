//! Tests for `AlphaRenaming` with binder frames: entering and leaving
//! frames, how an identifier resolves through the frames and the free
//! renaming, the capture rules of `is_corresponding`, value equality, and
//! expressions compared under frames the way a binder term compares its
//! body.
//!
//! Most cases are ported from the Python suites of `AlphaRenaming`
//! (`tests/test_alpha_equivalence.py`, `tests/test_binder.py` and
//! `tests/test_derived_equivalence.py`); the traceability table is in
//! `docs/design/python-switch.md`, "S4.2 implementation notes". A Rust
//! expression has no binder, so where Python compares two binder terms,
//! these tests enter one frame per binder level and compare the bodies.

use crate::support::expression as expression_support;

use std::collections::HashMap;

use expression_support::{build_identifier, build_literal};
use fhy_core::expression::{AlphaRenaming, Expression, RenamingPart};
use fhy_core::identifier::Identifier;
use rstest::rstest;

/// Return the frame or free renaming pairing each identifier with its image.
fn build_map<const N: usize>(
    pairs: [(&Identifier, &Identifier); N],
) -> HashMap<Identifier, Identifier> {
    pairs
        .into_iter()
        .map(|(from, to)| (from.clone(), to.clone()))
        .collect()
}

/// Return the renaming with the free renaming `free_pairs` and one frame per
/// entry of `frames`, outermost first.
fn build_renaming(
    free_pairs: HashMap<Identifier, Identifier>,
    frames: Vec<HashMap<Identifier, Identifier>>,
) -> AlphaRenaming {
    let mut renaming = AlphaRenaming::try_new(free_pairs).expect("the free renaming is injective");
    for frame in frames {
        renaming
            .enter_binder(frame)
            .expect("every frame is injective");
    }
    renaming
}

/// Return the renaming with no free renaming and one frame per entry of
/// `frames`, outermost first.
fn build_framed_renaming(frames: Vec<HashMap<Identifier, Identifier>>) -> AlphaRenaming {
    build_renaming(HashMap::new(), frames)
}

fn build_identifiers<const N: usize>(names: [&str; N]) -> [Identifier; N] {
    names.map(Identifier::new)
}

// =============================================================================
// Construction
// =============================================================================

#[test]
fn alpha_renaming_default_resolves_every_identifier_to_itself() {
    let [x] = build_identifiers(["x"]);

    let renaming = AlphaRenaming::default();

    assert_eq!(renaming.resolve(&x), &x);
    assert!(renaming.is_empty());
    assert_eq!(renaming.binder_depth(), 0);
}

#[test]
fn alpha_renaming_try_new_resolves_a_mapped_identifier_to_its_image() {
    let [x, y] = build_identifiers(["x", "y"]);

    let renaming = AlphaRenaming::try_new(build_map([(&x, &y)])).expect("one pair is injective");

    assert_eq!(renaming.resolve(&x), &y);
}

#[test]
fn alpha_renaming_try_new_of_an_empty_map_is_the_default() {
    let [x] = build_identifiers(["x"]);

    let renaming = AlphaRenaming::try_new(HashMap::new()).expect("an empty map is injective");

    assert_eq!(renaming, AlphaRenaming::default());
    assert_eq!(renaming.resolve(&x), &x);
}

#[test]
fn alpha_renaming_try_new_names_the_free_renaming_in_its_refusal() {
    let [a, b, target] = build_identifiers(["a", "b", "t"]);

    let error = AlphaRenaming::try_new(build_map([(&a, &target), (&b, &target)]))
        .expect_err("two identifiers share the image t");

    assert_eq!(error.image(), &target);
    assert_eq!(error.part(), RenamingPart::FreeRenaming);
}

// =============================================================================
// Entering and leaving binder frames
// =============================================================================

#[test]
fn alpha_renaming_enter_binder_resolves_a_bound_identifier_to_its_image() {
    let [x, y] = build_identifiers(["x", "y"]);
    let mut renaming = AlphaRenaming::default();

    renaming
        .enter_binder(build_map([(&x, &y)]))
        .expect("one pair is injective");

    assert_eq!(renaming.resolve(&x), &y);
    assert_eq!(renaming.binder_depth(), 1);
    assert!(!renaming.is_empty());
}

#[test]
fn alpha_renaming_enter_binder_of_an_empty_frame_keeps_resolution_at_identity() {
    let [x] = build_identifiers(["x"]);

    let renaming = build_framed_renaming(vec![HashMap::new()]);

    assert_eq!(renaming.resolve(&x), &x);
    assert!(renaming.is_corresponding(&x, &x));
    assert!(renaming.is_empty());
    assert_eq!(renaming.binder_depth(), 1);
}

#[test]
fn alpha_renaming_inner_frame_shadows_an_outer_frame() {
    let [x, a, b] = build_identifiers(["x", "a", "b"]);

    let renaming = build_framed_renaming(vec![build_map([(&x, &a)]), build_map([(&x, &b)])]);

    assert_eq!(renaming.resolve(&x), &b);
}

#[test]
fn alpha_renaming_frames_may_share_an_image() {
    let [x, y, shared] = build_identifiers(["x", "y", "a"]);

    let renaming =
        build_framed_renaming(vec![build_map([(&x, &shared)]), build_map([(&y, &shared)])]);

    assert_eq!(renaming.resolve(&x), &shared);
    assert_eq!(renaming.resolve(&y), &shared);
}

/// Test a frame sending two identifiers to one image is refused, names the
/// frame, and leaves the renaming as it was.
#[test]
fn alpha_renaming_enter_binder_refuses_a_non_injective_frame() {
    let [a, b, shared] = build_identifiers(["a", "b", "t"]);
    let mut renaming = AlphaRenaming::default();

    let error = renaming
        .enter_binder(build_map([(&a, &shared), (&b, &shared)]))
        .expect_err("two identifiers share the image t");

    assert_eq!(error.image(), &shared);
    assert_eq!(error.part(), RenamingPart::BinderFrame);
    assert_eq!(
        error.to_string(),
        format!(
            "a binder frame must be injective, but more than one identifier maps to t::{}",
            shared.id()
        )
    );
    assert_eq!(renaming, AlphaRenaming::default());
}

#[test]
fn alpha_renaming_enter_binder_keeps_the_free_renaming_of_unbound_identifiers() {
    let [x, y, free_a, free_b] = build_identifiers(["x", "y", "a", "b"]);

    let renaming = build_renaming(build_map([(&free_a, &free_b)]), vec![build_map([(&x, &y)])]);

    assert_eq!(renaming.resolve(&free_a), &free_b);
}

#[test]
fn alpha_renaming_leave_binder_restores_the_renaming_before_the_frame() {
    let [x, y, free_a, free_b] = build_identifiers(["x", "y", "a", "b"]);
    let outer = build_renaming(build_map([(&free_a, &free_b)]), vec![build_map([(&x, &y)])]);
    let mut renaming = outer.clone();
    renaming
        .enter_binder(build_map([(&x, &free_a)]))
        .expect("one pair is injective");

    let left = renaming.leave_binder();

    assert!(left);
    assert_eq!(renaming, outer);
    assert_eq!(renaming.resolve(&x), &y);
}

#[test]
fn alpha_renaming_leave_binder_without_a_frame_changes_nothing() {
    let [free_a, free_b] = build_identifiers(["a", "b"]);
    let outer = build_renaming(build_map([(&free_a, &free_b)]), Vec::new());
    let mut renaming = outer.clone();

    let left = renaming.leave_binder();

    assert!(!left);
    assert_eq!(renaming, outer);
}

// =============================================================================
// Resolution order
// =============================================================================

#[test]
fn alpha_renaming_resolve_falls_back_to_the_free_renaming_outside_every_frame() {
    let [free_a, free_b, x, y] = build_identifiers(["a", "b", "x", "y"]);

    let renaming = build_renaming(build_map([(&free_a, &free_b)]), vec![build_map([(&x, &y)])]);

    assert_eq!(renaming.resolve(&free_a), &free_b);
}

#[test]
fn alpha_renaming_resolve_falls_back_to_identity_for_an_unmapped_identifier() {
    let [free_a, free_b, x, y, unrelated] = build_identifiers(["a", "b", "x", "y", "z"]);

    let renaming = build_renaming(build_map([(&free_a, &free_b)]), vec![build_map([(&x, &y)])]);

    assert_eq!(renaming.resolve(&unrelated), &unrelated);
}

#[test]
fn alpha_renaming_frame_takes_precedence_over_the_free_renaming() {
    let [key, free_value, bound_value] = build_identifiers(["k", "free", "bound"]);

    let renaming = build_renaming(
        build_map([(&key, &free_value)]),
        vec![build_map([(&key, &bound_value)])],
    );

    assert_eq!(renaming.resolve(&key), &bound_value);
}

// =============================================================================
// Correspondence
// =============================================================================

#[rstest]
#[case::bound_to_its_image("x", "y", true)]
#[case::bound_to_another("x", "z", false)]
#[case::unbound_to_itself("z", "z", true)]
#[case::unbound_to_another("z", "w", false)]
fn alpha_renaming_is_corresponding_follows_one_frame(
    #[case] left: &str,
    #[case] right: &str,
    #[case] expected: bool,
) {
    let identifiers: HashMap<&str, Identifier> = ["x", "y", "z", "w"]
        .into_iter()
        .map(|name| (name, Identifier::new(name)))
        .collect();
    let renaming = build_framed_renaming(vec![build_map([(&identifiers["x"], &identifiers["y"])])]);

    let corresponding = renaming.is_corresponding(&identifiers[left], &identifiers[right]);

    assert_eq!(corresponding, expected);
}

/// Test correspondence agrees with resolution through a frame, the free
/// renaming and identity, for every pair where neither side is captured.
#[test]
fn alpha_renaming_is_corresponding_agrees_with_resolve() {
    let [x, y, free_a, free_b, unrelated] = build_identifiers(["x", "y", "a", "b", "z"]);
    let renaming = build_renaming(build_map([(&free_a, &free_b)]), vec![build_map([(&x, &y)])]);

    for left in [&x, &free_a, &unrelated] {
        for right in [&x, &y, &free_a, &free_b, &unrelated] {
            assert_eq!(
                renaming.is_corresponding(left, right),
                renaming.resolve(left) == right,
                "{left:?} against {right:?}"
            );
        }
    }
}

/// Test an identifier bound only on the other side is captured: nothing on
/// this side corresponds to it, although it resolves to itself.
#[test]
fn alpha_renaming_is_corresponding_refuses_an_identifier_bound_only_on_the_other_side() {
    let [x, y] = build_identifiers(["x", "y"]);
    let renaming = build_framed_renaming(vec![build_map([(&x, &y)])]);

    assert_eq!(renaming.resolve(&y), &y);
    assert!(!renaming.is_corresponding(&y, &y));
}

/// Test an image of the free renaming corresponds only to the identifier
/// mapped to it, inside a frame too.
#[test]
fn alpha_renaming_is_corresponding_refuses_an_unmapped_identifier_matching_a_free_image() {
    let [free_a, free_b, x, y] = build_identifiers(["a", "b", "x", "y"]);
    let renaming = build_renaming(build_map([(&free_a, &free_b)]), vec![build_map([(&x, &y)])]);

    assert!(!renaming.is_corresponding(&free_b, &free_b));
    assert!(renaming.is_corresponding(&free_a, &free_b));
}

/// Test a frame that binds the other side's identifier shadows an outer
/// frame binding this side's: `\x. \y. x` is not `\a. \a. a`, whose body
/// refers to the inner `a`. The Python renaming answers `true` here, and
/// `false` for the same pair asked the other way round; see the S4.2
/// implementation notes.
#[test]
fn alpha_renaming_is_corresponding_lets_an_inner_image_shadow_an_outer_binding() {
    let [x, y, a] = build_identifiers(["x", "y", "a"]);
    let forward = build_framed_renaming(vec![build_map([(&x, &a)]), build_map([(&y, &a)])]);
    let backward = build_framed_renaming(vec![build_map([(&a, &x)]), build_map([(&a, &y)])]);

    assert!(!forward.is_corresponding(&x, &a));
    assert!(!backward.is_corresponding(&a, &x));
    assert!(forward.is_corresponding(&y, &a));
    assert!(backward.is_corresponding(&a, &y));
}

// =============================================================================
// Value equality
// =============================================================================

#[test]
fn alpha_renaming_equal_frames_make_equal_renamings() {
    let [x, y] = build_identifiers(["x", "y"]);

    let left = build_framed_renaming(vec![build_map([(&x, &y)])]);
    let right = build_framed_renaming(vec![build_map([(&x, &y)])]);

    assert_eq!(left, right);
}

#[test]
fn alpha_renaming_frame_order_distinguishes_renamings() {
    let [x, y, a, b] = build_identifiers(["x", "y", "a", "b"]);

    let left = build_framed_renaming(vec![build_map([(&x, &y)]), build_map([(&a, &b)])]);
    let right = build_framed_renaming(vec![build_map([(&a, &b)]), build_map([(&x, &y)])]);

    assert_ne!(left, right);
}

#[test]
fn alpha_renaming_free_renaming_distinguishes_renamings() {
    let [free_a, free_b, free_c] = build_identifiers(["a", "b", "c"]);

    let left = build_renaming(build_map([(&free_a, &free_b)]), Vec::new());
    let right = build_renaming(build_map([(&free_a, &free_c)]), Vec::new());

    assert_ne!(left, right);
}

#[test]
fn alpha_renaming_empty_frame_distinguishes_renamings() {
    let with_frame = build_framed_renaming(vec![HashMap::new()]);

    assert_ne!(with_frame, AlphaRenaming::default());
    assert_eq!(AlphaRenaming::default(), AlphaRenaming::default());
}

// =============================================================================
// Expressions under binder frames
// =============================================================================

/// A term of nested one-parameter binders over an expression body, the shape
/// of the Python tests' binder terms; `parameters[0]` is the outermost.
#[derive(Debug, Clone)]
struct Binders {
    parameters: Vec<Identifier>,
    body: Expression,
}

impl Binders {
    fn new<const N: usize>(parameters: [&Identifier; N], body: Expression) -> Self {
        Self {
            parameters: parameters.into_iter().cloned().collect(),
            body,
        }
    }

    /// Return whether `other` is this term up to renaming its binders, and
    /// its free identifiers by `renaming`: enter one frame per binder level,
    /// pairing the parameters, and compare the bodies under them.
    fn is_alpha_equivalent_under(&self, other: &Self, renaming: &AlphaRenaming) -> bool {
        if self.parameters.len() != other.parameters.len() {
            return false;
        }
        let mut renaming = renaming.clone();
        for (parameter, other_parameter) in self.parameters.iter().zip(&other.parameters) {
            let frame = build_map([(parameter, other_parameter)]);
            if renaming.enter_binder(frame).is_err() {
                return false;
            }
        }
        self.body.is_alpha_equivalent_under(&other.body, &renaming)
    }

    fn is_alpha_equivalent(&self, other: &Self) -> bool {
        self.is_alpha_equivalent_under(other, &AlphaRenaming::default())
    }
}

#[test]
fn binders_renaming_their_parameter_are_alpha_equivalent() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");

    let left = Binders::new([&x], x_reference);
    let right = Binders::new([&y], y_reference);

    assert!(left.is_alpha_equivalent(&right));
}

#[test]
fn binders_with_a_free_body_on_one_side_are_not_alpha_equivalent() {
    let (x, x_reference) = build_identifier("x");
    let (y, _) = build_identifier("y");
    let (_, free_reference) = build_identifier("z");

    let left = Binders::new([&x], x_reference);
    let right = Binders::new([&y], free_reference);

    assert!(!left.is_alpha_equivalent(&right));
}

#[test]
fn binders_sharing_a_free_body_are_alpha_equivalent() {
    let (x, _) = build_identifier("x");
    let (y, _) = build_identifier("y");
    let (_, shared_reference) = build_identifier("shared");

    let left = Binders::new([&x], shared_reference.clone());
    let right = Binders::new([&y], shared_reference);

    assert!(left.is_alpha_equivalent(&right));
}

#[test]
fn binders_over_distinct_free_bodies_are_not_alpha_equivalent() {
    let (x, _) = build_identifier("x");
    let (y, _) = build_identifier("y");
    let (_, a_reference) = build_identifier("a");
    let (_, b_reference) = build_identifier("b");

    let left = Binders::new([&x], a_reference);
    let right = Binders::new([&y], b_reference);

    assert!(!left.is_alpha_equivalent(&right));
}

/// Test `\x. \x. x` is `\a. \b. b`: the inner binder shadows the outer.
#[test]
fn binders_nested_over_one_name_match_the_inner_binder() {
    let (x, x_reference) = build_identifier("x");
    let (a, _) = build_identifier("a");
    let (b, b_reference) = build_identifier("b");

    let left = Binders::new([&x, &x], x_reference);
    let right = Binders::new([&a, &b], b_reference);

    assert!(left.is_alpha_equivalent(&right));
}

/// Test `\x. \x. x` is not `\a. \b. a`, whose body refers to the outer
/// binder.
#[test]
fn binders_nested_over_one_name_do_not_match_the_outer_binder() {
    let (x, x_reference) = build_identifier("x");
    let (a, a_reference) = build_identifier("a");
    let (b, _) = build_identifier("b");

    let left = Binders::new([&x, &x], x_reference);
    let right = Binders::new([&a, &b], a_reference);

    assert!(!left.is_alpha_equivalent(&right));
}

/// Test `\x. x + y` is not `\y. y + y`: the free `y` would be captured.
#[test]
fn binders_refuse_to_capture_a_free_identifier() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");

    let left = Binders::new([&x], &x_reference + &y_reference);
    let right = Binders::new([&y], &y_reference + &y_reference);

    assert!(!left.is_alpha_equivalent(&right));
}

/// Test `\x. \y. x - y` is `\y. \x. y - x`.
#[test]
fn binders_swapped_with_their_operands_are_alpha_equivalent() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");

    let left = Binders::new([&x, &y], &x_reference - &y_reference);
    let right = Binders::new([&y, &x], &y_reference - &x_reference);

    assert!(left.is_alpha_equivalent(&right));
}

#[test]
fn binders_compare_free_identifiers_of_their_body_under_the_free_renaming() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let (free_a, a_reference) = build_identifier("a");
    let (free_b, b_reference) = build_identifier("b");
    let renaming = build_renaming(build_map([(&free_a, &free_b)]), Vec::new());

    let left = Binders::new([&x], &x_reference + &a_reference);
    let right = Binders::new([&y], &y_reference + &b_reference);

    assert!(left.is_alpha_equivalent_under(&right, &renaming));
}

#[test]
fn binders_over_the_same_parameter_are_alpha_equivalent() {
    let (x, x_reference) = build_identifier("x");

    let left = Binders::new([&x], x_reference.clone());
    let right = Binders::new([&x], x_reference);

    assert!(left.is_alpha_equivalent(&right));
}

/// Test an expression referring to one identifier twice is not equivalent
/// to itself under a frame binding that identifier to another, although the
/// two trees are one handle.
#[test]
fn expression_is_not_alpha_equivalent_to_itself_under_a_frame_renaming_its_identifier() {
    let (x, x_reference) = build_identifier("x");
    let (y, _) = build_identifier("y");
    let tree = &x_reference * 2 + &x_reference;
    let renaming = build_framed_renaming(vec![build_map([(&x, &y)])]);

    assert!(!tree.is_alpha_equivalent_under(&tree, &renaming));
}

/// Test a two-parameter binder whose frame is not injective is refused, so
/// `\(x, y). x` is not `\(z, z). z`.
#[test]
fn binder_frame_pairing_two_parameters_with_one_is_refused() {
    let [x, y, z] = build_identifiers(["x", "y", "z"]);
    let mut renaming = AlphaRenaming::default();

    let refused = renaming.enter_binder(build_map([(&x, &z), (&y, &z)]));

    assert_eq!(
        refused.expect_err("x and y share the image z").part(),
        RenamingPart::BinderFrame
    );
    assert_eq!(renaming.binder_depth(), 0);
}

#[test]
fn binders_of_different_depths_are_not_alpha_equivalent() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let (z, _) = build_identifier("z");

    let one_level = Binders::new([&x], x_reference);
    let two_levels = Binders::new([&y, &z], y_reference);

    assert!(!one_level.is_alpha_equivalent(&two_levels));
}

// =============================================================================
// Equivalence laws over a fixed set of binder terms
// =============================================================================

/// Return terms covering leaves, references, operations, binders, nested
/// binders, and binders shadowing each other on one side.
fn build_example_terms() -> Vec<Binders> {
    let (p, p_reference) = build_identifier("p");
    let (q, q_reference) = build_identifier("q");
    let (_, r_reference) = build_identifier("r");
    let (a, a_reference) = build_identifier("a");
    let one = build_literal(1);
    let two = build_literal(2);
    vec![
        Binders::new([], one.clone()),
        Binders::new([], two.clone()),
        Binders::new([], p_reference.clone()),
        Binders::new([], q_reference.clone()),
        Binders::new([], &one + &p_reference),
        Binders::new([], &q_reference + &two),
        Binders::new([&p], p_reference.clone()),
        Binders::new([&q], q_reference.clone()),
        Binders::new([&p], &p_reference + &r_reference),
        Binders::new([&p, &q], &p_reference + &q_reference),
        Binders::new([&p, &q], p_reference.clone()),
        Binders::new([&p, &q], q_reference.clone()),
        Binders::new([&a, &a], a_reference),
        Binders::new([&q, &p], q_reference),
    ]
}

#[test]
fn binders_alpha_equivalence_is_reflexive() {
    for term in build_example_terms() {
        assert!(term.is_alpha_equivalent(&term), "{term:?}");
    }
}

#[test]
fn binders_alpha_equivalence_is_symmetric() {
    let terms = build_example_terms();

    for left in &terms {
        for right in &terms {
            assert_eq!(
                left.is_alpha_equivalent(right),
                right.is_alpha_equivalent(left),
                "{left:?} against {right:?}"
            );
        }
    }
}

#[test]
fn binders_alpha_equivalence_is_transitive() {
    let terms = build_example_terms();
    let mut non_trivial_triples = 0;

    for left in &terms {
        for middle in &terms {
            for right in &terms {
                if left.is_alpha_equivalent(middle) && middle.is_alpha_equivalent(right) {
                    if !std::ptr::eq(left, right) {
                        non_trivial_triples += 1;
                    }
                    assert!(
                        left.is_alpha_equivalent(right),
                        "{left:?}, {middle:?}, {right:?}"
                    );
                }
            }
        }
    }

    assert!(non_trivial_triples > 0);
}
