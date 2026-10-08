//! Tests for the equivalence of whole spaces and configurations: one frame
//! pairs every name of two spaces in canonical order, identifier members
//! and the members of the conditions' and forbidden clauses' set
//! constraints are references resolved through it, and structural
//! equivalence compares names by `==`. They include the audit's
//! counterexamples A1 to A7, each in both directions.

#![expect(
    clippy::many_single_char_names,
    reason = "the stories name decisions as the design's examples do: c, a, b, x, y"
)]

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use fhy_core::constraint::{
    Bindings, Constraint, ConstraintContext, ConstraintError, CustomConstraint, Outcome, Value,
};
use fhy_core::diagnostic::Note;
use fhy_core::expression::Expression;
use fhy_core::foreign::{BoxError, ForeignPart, Part};
use fhy_core::identifier::Identifier;
use fhy_core::param::Param;
use fhy_core::search_space::{Configuration, EquivalenceError, Space};
use fhy_core::term::{AlphaEquivalence, AlphaRenaming};
use rstest::rstest;

use crate::support::constraint::{TestCustom, int};
use crate::support::param::{in_set, less_than};
use crate::support::search_space::{
    Realization, TileKnob, bare_alternative, categorical, categorical_of, categorical_where,
    choice_of, chooses, chosen, compare_alpha_both_ways, condition, configure, forbidden,
    int_param, int_values, plain_alternative, plain_variable,
};

/// The names of the standard space.
#[derive(Clone)]
struct Labels {
    space: Identifier,
    v: Identifier,
    c: Identifier,
    a: Identifier,
    b: Identifier,
    x: Identifier,
    y: Identifier,
    /// The variable of `y`'s param.
    p: Identifier,
}

impl Labels {
    /// Return fresh names.
    fn fresh() -> Self {
        let [space, v, c, a, b, x, y, p] =
            ["space", "v", "c", "a", "b", "x", "y", "p"].map(Identifier::new);
        Self {
            space,
            v,
            c,
            a,
            b,
            x,
            y,
            p,
        }
    }

    /// Return fresh names minted in the reverse order, so every id order
    /// among them is the opposite of [`fresh`](Self::fresh)'s.
    fn fresh_reversed() -> Self {
        let [p, y, x, b, a, c, v, space] =
            ["p", "y", "x", "b", "a", "c", "v", "space"].map(Identifier::new);
        Self {
            space,
            v,
            c,
            a,
            b,
            x,
            y,
            p,
        }
    }
}

/// The params of the standard space whose values are not identifiers.
#[derive(Clone)]
struct Params {
    v: Param,
    x: Param,
}

impl Params {
    /// Return the params: `v` over `{1, 2}`, `x` over `{1, 2, 3}`.
    fn new() -> Self {
        Self {
            v: int_param(&[1, 2]),
            x: int_param(&[1, 2, 3]),
        }
    }
}

/// The parts of the standard space a test may change.
#[derive(Clone)]
struct Shape {
    /// The alternatives `v`'s condition requires `c` to choose.
    guard: Vec<&'static str>,
    /// The values of `x` the forbidden clause pairs with `v = 2`.
    forbidden_x: Vec<i64>,
    /// The space's notes.
    notes: Vec<&'static str>,
}

impl Shape {
    /// Return the standard shape.
    fn standard() -> Self {
        Self {
            guard: vec!["a"],
            forbidden_x: vec![3],
            notes: vec!["standard"],
        }
    }
}

/// Return the standard space over `labels`: the variable `v` over
/// `params.v`, active while `c` chooses one of `shape.guard`; the choice
/// `c` among `a`, holding `x` over `params.x`, and `b`, holding `y` over
/// the categories `{a, b}` (references to the alternatives' names); and the
/// clause forbidding `x` in `shape.forbidden_x` with `v = 2`.
fn build_standard(labels: &Labels, params: &Params, shape: &Shape) -> Space {
    let alternative = |name: &str| match name {
        "a" => &labels.a,
        "b" => &labels.b,
        _ => unreachable!("no alternative {name}"),
    };
    let guard: Vec<&Identifier> = shape.guard.iter().map(|name| alternative(name)).collect();
    let y_param = categorical_of(&labels.p, vec![chosen(&labels.a), chosen(&labels.b)]);
    let choice = choice_of(
        &labels.c,
        vec![
            plain_alternative(
                &labels.a,
                vec![plain_variable(&labels.x, params.x.clone())],
                vec![],
            ),
            plain_alternative(&labels.b, vec![plain_variable(&labels.y, y_param)], vec![]),
        ],
    );
    Space::new(
        labels.space.clone(),
        vec![plain_variable(&labels.v, params.v.clone())],
        vec![choice],
        vec![condition(&labels.v, [chooses(&labels.c, &guard)])],
        vec![forbidden([
            in_set(&labels.x, int_values(&shape.forbidden_x)),
            in_set(&labels.v, [int(2)]),
        ])],
    )
    .expect("the standard space is valid")
    .with_notes(
        shape
            .notes
            .iter()
            .map(|&note| Note::with_other_kind(note))
            .collect(),
    )
}

/// Return whether the spaces `left` and `right` are structurally
/// equivalent, in each direction.
fn compare_structural_both_ways(left: &Space, right: &Space) -> [bool; 2] {
    [
        left.is_structurally_equivalent(right)
            .expect("the comparison succeeds"),
        right
            .is_structurally_equivalent(left)
            .expect("the comparison succeeds"),
    ]
}

// ---------------------------------------------------------------------------
// Reflexivity and structural equivalence
// ---------------------------------------------------------------------------

#[rstest]
#[case::a_plain_variable("variable")]
#[case::a_plain_alternative("plain_alternative")]
#[case::an_implementor_alternative("implementor_alternative")]
#[case::a_choice("choice")]
#[case::a_space("space")]
#[case::a_configuration("configuration")]
fn structural_equivalence_is_reflexive(#[case] which: &str) {
    let labels = Labels::fresh();
    let space = build_standard(&labels, &Params::new(), &Shape::standard());
    let choice = space.choices()[0].clone();

    let (structural, alpha) = match which {
        "variable" => {
            let variable = space.variables()[0].clone();
            (
                variable.is_structurally_equivalent(&variable),
                variable.is_alpha_equivalent(&variable),
            )
        }
        "plain_alternative" => {
            let alternative = choice.alternatives()[0].clone();
            (
                alternative.is_structurally_equivalent(&alternative),
                alternative.is_alpha_equivalent(&alternative),
            )
        }
        "implementor_alternative" => {
            let axis = Identifier::new("axis");
            let alternative = Realization::new(
                &Identifier::new("realized"),
                vec![TileKnob::part(
                    &Identifier::new("tile"),
                    int_param(&[4, 8]),
                    &[&axis],
                )],
                &[&axis],
                &[&axis],
                1,
            )
            .into_part();
            (
                alternative.is_structurally_equivalent(&alternative),
                alternative.is_alpha_equivalent(&alternative),
            )
        }
        "choice" => (
            choice.is_structurally_equivalent(&choice),
            choice.is_alpha_equivalent(&choice),
        ),
        "space" => (
            space.is_structurally_equivalent(&space),
            space.is_alpha_equivalent(&space),
        ),
        "configuration" => {
            let configuration = configure(
                &space,
                [
                    (labels.c.clone(), chosen(&labels.b)),
                    (labels.y.clone(), chosen(&labels.a)),
                ],
            );
            (
                configuration.is_structurally_equivalent(&configuration),
                configuration.is_alpha_equivalent(&configuration),
            )
        }
        _ => unreachable!("unknown case {which}"),
    };

    assert!(structural.expect("the comparison succeeds"));
    assert!(alpha.expect("the comparison succeeds"));
}

#[test]
fn spaces_built_apart_from_one_description_are_structurally_equivalent() {
    let (labels, params) = (Labels::fresh(), Params::new());

    let left = build_standard(&labels, &params, &Shape::standard());
    let right = build_standard(&labels, &params, &Shape::standard());

    assert_eq!(compare_structural_both_ways(&left, &right), [true, true]);
    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
}

#[rstest]
#[case::a_variable_domain("domain")]
#[case::the_condition("condition")]
#[case::the_forbidden_clause("forbidden")]
#[case::the_notes("notes")]
#[case::a_name("name")]
fn structural_equivalence_discriminates_a_perturbed_field(#[case] field: &str) {
    let (labels, params) = (Labels::fresh(), Params::new());
    let (mut perturbed_labels, mut perturbed_params, mut shape) =
        (labels.clone(), params.clone(), Shape::standard());
    match field {
        "domain" => perturbed_params.x = int_param(&[1, 2, 4]),
        "condition" => shape.guard = vec!["b"],
        "forbidden" => shape.forbidden_x = vec![2],
        "notes" => shape.notes = vec!["perturbed"],
        "name" => perturbed_labels.x = Identifier::new("x"),
        _ => unreachable!("unknown field {field}"),
    }

    let left = build_standard(&labels, &params, &Shape::standard());
    let right = build_standard(&perturbed_labels, &perturbed_params, &shape);

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[rstest]
#[case::a_variable_domain("domain")]
#[case::the_condition("condition")]
#[case::the_forbidden_clause("forbidden")]
#[case::the_notes("notes")]
fn alpha_equivalence_discriminates_a_perturbed_field_of_a_relabeled_space(#[case] field: &str) {
    let mut shape = Shape::standard();
    let mut params = Params::new();
    match field {
        "domain" => params.x = int_param(&[1, 2, 4]),
        "condition" => shape.guard = vec!["b"],
        "forbidden" => shape.forbidden_x = vec![2],
        "notes" => shape.notes = vec!["perturbed"],
        _ => unreachable!("unknown field {field}"),
    }

    let left = build_standard(&Labels::fresh(), &Params::new(), &Shape::standard());
    let right = build_standard(&Labels::fresh(), &params, &shape);

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
}

// ---------------------------------------------------------------------------
// Relabeling
// ---------------------------------------------------------------------------

#[rstest]
#[case::in_the_same_id_order(false)]
#[case::in_the_opposite_id_order(true)]
fn relabeled_spaces_are_alpha_equivalent_but_not_structurally(#[case] reversed: bool) {
    let other = if reversed {
        Labels::fresh_reversed()
    } else {
        Labels::fresh()
    };

    let left = build_standard(&Labels::fresh(), &Params::new(), &Shape::standard());
    let right = build_standard(&other, &Params::new(), &Shape::standard());

    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[test]
fn relabeled_conditions_with_several_members_correspond_in_any_canonical_order() {
    let build = |labels: &[Identifier; 4]| {
        let [space, p, q, w] = labels;
        Space::new(
            space.clone(),
            vec![
                plain_variable(p, int_param(&[1, 2])),
                plain_variable(q, int_param(&[1, 2])),
                plain_variable(w, int_param(&[1])),
            ],
            vec![],
            vec![condition(w, [in_set(p, [int(1)]), in_set(q, [int(2)])])],
            vec![forbidden([in_set(p, [int(2)]), in_set(q, [int(1)])])],
        )
        .expect("the space is valid")
    };
    let forward = ["space", "p", "q", "w"].map(Identifier::new);
    let [w, q, p, space] = ["w", "q", "p", "space"].map(Identifier::new);
    let backward = [space, p, q, w];

    let left = build(&forward);
    let right = build(&backward);

    assert_eq!(
        compare_alpha_both_ways(&left, &right),
        [true, true],
        "the systems sort by ids, which the relabeling reverses"
    );
}

#[test]
fn relabeled_equation_conditions_are_alpha_equivalent() {
    let build = || {
        let [space, p, q, w] = ["space", "p", "q", "w"].map(Identifier::new);
        Space::new(
            space,
            vec![
                plain_variable(&p, int_param(&[1, 2])),
                plain_variable(&q, int_param(&[1, 2])),
                plain_variable(&w, int_param(&[1])),
            ],
            vec![],
            vec![condition(&w, [less_than(&p, &q)])],
            vec![],
        )
        .expect("the space is valid")
    };

    let (left, right) = (build(), build());

    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
}

#[test]
fn equation_conditions_with_swapped_operands_are_not_alpha_equivalent() {
    let build = |swapped: bool| {
        let [space, p, q, w] = ["space", "p", "q", "w"].map(Identifier::new);
        let guard = if swapped {
            less_than(&q, &p)
        } else {
            less_than(&p, &q)
        };
        Space::new(
            space,
            vec![
                plain_variable(&p, int_param(&[1, 2])),
                plain_variable(&q, int_param(&[1, 2])),
                plain_variable(&w, int_param(&[1])),
            ],
            vec![],
            vec![condition(&w, [guard])],
            vec![],
        )
        .expect("the space is valid")
    };

    let (left, right) = (build(false), build(true));

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
}

#[test]
fn conditions_on_corresponding_targets_only_correspond() {
    let build = |target_first: bool| {
        let [space, p, q] = ["space", "p", "q"].map(Identifier::new);
        let (target, gate) = if target_first { (&p, &q) } else { (&q, &p) };
        Space::new(
            space.clone(),
            vec![
                plain_variable(&p, int_param(&[1, 2])),
                plain_variable(&q, int_param(&[1, 2])),
            ],
            vec![],
            vec![condition(target, [in_set(gate, [int(1)])])],
            vec![],
        )
        .expect("the space is valid")
    };

    let (left, right) = (build(true), build(false));

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
}

#[test]
fn forbidden_clauses_correspond_in_order() {
    let build = |reversed: bool| {
        let [space, p, q] = ["space", "p", "q"].map(Identifier::new);
        let mut clauses = vec![
            forbidden([in_set(&p, [int(1)])]),
            forbidden([in_set(&q, [int(2)])]),
        ];
        if reversed {
            clauses.reverse();
        }
        Space::new(
            space,
            vec![
                plain_variable(&p, int_param(&[1, 2])),
                plain_variable(&q, int_param(&[1, 2])),
            ],
            vec![],
            vec![],
            clauses,
        )
        .expect("the space is valid")
    };

    let (left, same, reversed) = (build(false), build(false), build(true));

    assert_eq!(compare_alpha_both_ways(&left, &same), [true, true]);
    assert_eq!(compare_alpha_both_ways(&left, &reversed), [false, false]);
}

#[test]
fn spaces_of_different_shapes_are_not_equivalent_without_an_error() {
    let labels = Labels::fresh();
    let standard = build_standard(&labels, &Params::new(), &Shape::standard());
    let extra = Identifier::new("extra");
    let larger = Space::new(
        Identifier::new("larger"),
        vec![
            plain_variable(&labels.v, int_param(&[1, 2])),
            plain_variable(&extra, int_param(&[1])),
        ],
        standard.choices().to_vec(),
        vec![],
        vec![],
    )
    .expect("the space is valid");

    assert_eq!(compare_alpha_both_ways(&standard, &larger), [false, false]);
    assert_eq!(
        compare_structural_both_ways(&standard, &larger),
        [false, false]
    );
}

#[test]
fn free_identifier_members_correspond_through_the_free_renaming() {
    let build = |free: &Identifier| {
        let v = Identifier::new("v");
        Space::new(
            Identifier::new("space"),
            vec![plain_variable(&v, categorical(vec![chosen(free)]))],
            vec![],
            vec![],
            vec![],
        )
        .expect("the space is valid")
    };
    let (z, z_prime) = (Identifier::new("z"), Identifier::new("z"));
    let (left, right) = (build(&z), build(&z_prime));
    let renaming = AlphaRenaming::new(HashMap::from([(z.clone(), z_prime.clone())]))
        .expect("one pair is injective");

    let unrenamed = left
        .is_alpha_equivalent(&right)
        .expect("the comparison succeeds");
    let renamed = left
        .is_alpha_equivalent_under(&right, &renaming)
        .expect("the comparison succeeds");

    assert!(!unrenamed, "a free identifier corresponds only to itself");
    assert!(renamed);
}

// ---------------------------------------------------------------------------
// The audit's counterexamples
// ---------------------------------------------------------------------------

#[test]
fn audit_a1_relabeled_alternatives_are_equivalent_in_both_directions() {
    let build = || {
        let [space, d, a, b] = ["space", "d", "A", "B"].map(Identifier::new);
        Space::new(
            space,
            vec![],
            vec![choice_of(
                &d,
                vec![bare_alternative(&a), bare_alternative(&b)],
            )],
            vec![],
            vec![],
        )
        .expect("the space is valid")
    };

    let (left, right) = (build(), build());

    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
    assert_eq!(
        compare_alpha_both_ways(&left.choices()[0], &right.choices()[0]),
        [true, true],
        "the choice compared on its own agrees with the space"
    );
}

#[test]
fn audit_a5_constraints_of_a_categorical_variable_count() {
    let build = |narrowed: bool| {
        let [space, k] = ["space", "k"].map(Identifier::new);
        let param = categorical_where(int_values(&[1, 2]), |p| {
            if narrowed {
                vec![in_set(p, [int(1)])]
            } else {
                vec![]
            }
        });
        Space::new(
            space,
            vec![plain_variable(&k, param)],
            vec![],
            vec![],
            vec![],
        )
        .expect("the space is valid")
    };

    let (plain, narrowed) = (build(false), build(true));

    assert_eq!(compare_alpha_both_ways(&plain, &narrowed), [false, false]);
}

#[rstest]
#[case::an_integer_and_a_boolean(Value::Int(1.into()), Value::Bool(true))]
#[case::an_integer_and_a_string(Value::Int(1.into()), Value::Str("1".into()))]
fn audit_a6_members_of_different_types_never_correspond(
    #[case] left_value: Value,
    #[case] right_value: Value,
) {
    let build = |value: &Value| {
        let [space, k] = ["space", "k"].map(Identifier::new);
        let space = Space::new(
            space,
            vec![plain_variable(&k, categorical(vec![value.clone()]))],
            vec![],
            vec![],
            vec![],
        )
        .expect("the space is valid");
        let configuration = configure(&space, [(k, value.clone())]);
        (space, configuration)
    };

    let (left_space, left_configuration) = build(&left_value);
    let (right_space, right_configuration) = build(&right_value);

    assert_eq!(
        compare_alpha_both_ways(&left_space, &right_space),
        [false, false]
    );
    assert_eq!(
        compare_alpha_both_ways(&left_configuration, &right_configuration),
        [false, false]
    );
}

#[test]
fn audit_a7_a_free_member_never_matches_a_bound_name() {
    let z = Identifier::new("Z");
    let left = Space::new(
        Identifier::new("left"),
        vec![],
        vec![choice_of(
            &Identifier::new("d"),
            vec![plain_alternative(
                &Identifier::new("A"),
                vec![plain_variable(
                    &Identifier::new("kz"),
                    categorical(vec![chosen(&z)]),
                )],
                vec![],
            )],
        )],
        vec![],
        vec![],
    )
    .expect("the space is valid");
    let right = Space::new(
        Identifier::new("right"),
        vec![],
        vec![choice_of(
            &Identifier::new("d"),
            vec![plain_alternative(
                &z,
                vec![plain_variable(
                    &Identifier::new("kz"),
                    categorical(vec![chosen(&z)]),
                )],
                vec![],
            )],
        )],
        vec![],
        vec![],
    )
    .expect("the space is valid");

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
}

#[test]
fn members_naming_alternatives_correspond_to_the_relabeled_alternatives() {
    let left = build_standard(&Labels::fresh(), &Params::new(), &Shape::standard());
    let other = Labels::fresh();
    let swapped = {
        let y_param = categorical(vec![chosen(&other.b), chosen(&other.c)]);
        let choice = choice_of(
            &other.c,
            vec![
                plain_alternative(
                    &other.a,
                    vec![plain_variable(&other.x, int_param(&[1, 2, 3]))],
                    vec![],
                ),
                plain_alternative(&other.b, vec![plain_variable(&other.y, y_param)], vec![]),
            ],
        );
        Space::new(
            other.space.clone(),
            vec![plain_variable(&other.v, int_param(&[1, 2]))],
            vec![choice],
            vec![condition(&other.v, [chooses(&other.c, &[&other.a])])],
            vec![forbidden([
                in_set(&other.x, [int(3)]),
                in_set(&other.v, [int(2)]),
            ])],
        )
        .expect("the space is valid")
        .with_notes(vec![Note::with_other_kind("standard")])
    };

    assert_eq!(
        compare_alpha_both_ways(&left, &swapped),
        [false, false],
        "y's categories name {{a, b}} on the left and {{b, c}} on the right"
    );
}

// ---------------------------------------------------------------------------
// Configurations
// ---------------------------------------------------------------------------

#[test]
fn configuration_entries_correspond_under_the_space_frame() {
    let (left_labels, right_labels) = (Labels::fresh(), Labels::fresh_reversed());
    let left_space = build_standard(&left_labels, &Params::new(), &Shape::standard());
    let right_space = build_standard(&right_labels, &Params::new(), &Shape::standard());

    let left = configure(
        &left_space,
        [
            (left_labels.c.clone(), chosen(&left_labels.b)),
            (left_labels.y.clone(), chosen(&left_labels.a)),
        ],
    );
    let right = configure(
        &right_space,
        [
            (right_labels.c.clone(), chosen(&right_labels.b)),
            (right_labels.y.clone(), chosen(&right_labels.a)),
        ],
    );
    let mismatched = configure(
        &right_space,
        [
            (right_labels.c.clone(), chosen(&right_labels.b)),
            (right_labels.y.clone(), chosen(&right_labels.b)),
        ],
    );

    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
    assert_eq!(compare_alpha_both_ways(&left, &mismatched), [false, false]);
    assert_eq!(left.key(), right.key());
}

#[test]
fn configurations_over_unrelated_spaces_are_not_equivalent() {
    let labels = Labels::fresh();
    let standard = build_standard(&labels, &Params::new(), &Shape::standard());
    let [other_space, q] = ["other", "q"].map(Identifier::new);
    let other = Space::new(
        other_space,
        vec![plain_variable(&q, int_param(&[1, 2]))],
        vec![],
        vec![],
        vec![],
    )
    .expect("the space is valid");

    let left = configure(
        &standard,
        [
            (labels.c.clone(), chosen(&labels.a)),
            (labels.v.clone(), int(1)),
        ],
    );
    let right = configure(&other, [(q.clone(), int(1))]);

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
    assert!(
        !left
            .is_structurally_equivalent(&right)
            .expect("plain parts")
    );
}

#[rstest]
#[case::another_value("value")]
#[case::another_alternative("alternative")]
#[case::an_unassigned_decision("unassigned")]
fn configurations_of_one_space_with_different_entries_are_not_equivalent(#[case] which: &str) {
    let labels = Labels::fresh();
    let space = build_standard(&labels, &Params::new(), &Shape::standard());
    let base = configure(
        &space,
        [
            (labels.c.clone(), chosen(&labels.a)),
            (labels.x.clone(), int(1)),
        ],
    );

    let other: Configuration = match which {
        "value" => configure(
            &space,
            [
                (labels.c.clone(), chosen(&labels.a)),
                (labels.x.clone(), int(2)),
            ],
        ),
        "alternative" => configure(&space, [(labels.c.clone(), chosen(&labels.b))]),
        "unassigned" => configure(&space, [(labels.c.clone(), chosen(&labels.a))]),
        _ => unreachable!("unknown case {which}"),
    };

    assert_eq!(compare_alpha_both_ways(&base, &other), [false, false]);
    assert!(
        !base
            .is_structurally_equivalent(&other)
            .expect("plain parts")
    );
}

#[test]
fn configurations_of_relabeled_spaces_are_not_structurally_equivalent() {
    let (left_labels, right_labels) = (Labels::fresh(), Labels::fresh());
    let left = configure(
        &build_standard(&left_labels, &Params::new(), &Shape::standard()),
        [(left_labels.c.clone(), chosen(&left_labels.a))],
    );
    let right = configure(
        &build_standard(&right_labels, &Params::new(), &Shape::standard()),
        [(right_labels.c.clone(), chosen(&right_labels.a))],
    );

    assert!(
        !left
            .is_structurally_equivalent(&right)
            .expect("plain parts")
    );
    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
}

// ---------------------------------------------------------------------------
// Custom constraints and failures
// ---------------------------------------------------------------------------

#[test]
fn custom_conditions_correspond_through_their_own_alpha_equivalence() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let build = |label: &str| {
        let [space, p, w] = ["space", "p", "w"].map(Identifier::new);
        Space::new(
            space,
            vec![
                plain_variable(&p, int_param(&[1, 2])),
                plain_variable(&w, int_param(&[1])),
            ],
            vec![],
            vec![condition(
                &w,
                [TestCustom::build(
                    label,
                    Expression::from(&p),
                    Outcome::Satisfied,
                    &log,
                )],
            )],
            vec![],
        )
        .expect("the space is valid")
    };

    let (left, same, other) = (build("one"), build("one"), build("two"));

    assert_eq!(compare_alpha_both_ways(&left, &same), [true, true]);
    assert_eq!(compare_alpha_both_ways(&left, &other), [false, false]);
}

/// A custom constraint naming one identifier whose alpha equivalence
/// fails.
#[derive(Debug)]
struct FailingAlpha(Identifier);

impl ForeignPart for FailingAlpha {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("FailingAlpha")
    }
}

impl CustomConstraint for FailingAlpha {
    fn free_identifiers(&self) -> Result<HashSet<Identifier>, BoxError> {
        Ok(HashSet::from([self.0.clone()]))
    }

    fn evaluate(
        &self,
        _bindings: &Bindings,
        _context: &ConstraintContext<'_>,
    ) -> Result<Outcome, BoxError> {
        Ok(Outcome::Satisfied)
    }

    fn to_expression(&self) -> Result<Expression, BoxError> {
        Ok(Expression::from(&self.0))
    }

    fn ordering_key(&self) -> Result<Cow<'_, str>, BoxError> {
        Ok(Cow::Borrowed("custom|failing_alpha"))
    }

    fn is_alpha_equivalent_under(
        &self,
        _other: &dyn CustomConstraint,
        _renaming: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        Err(BoxError::from("the alpha comparison failed"))
    }
}

#[test]
fn failing_custom_condition_fails_the_comparison() {
    let build = || {
        let [space, p, w] = ["space", "p", "w"].map(Identifier::new);
        Space::new(
            space,
            vec![
                plain_variable(&p, int_param(&[1, 2])),
                plain_variable(&w, int_param(&[1])),
            ],
            vec![],
            vec![condition(
                &w,
                [Constraint::Custom(Part::new(FailingAlpha(p.clone())))],
            )],
            vec![],
        )
        .expect("the space is valid")
    };

    let result = build().is_alpha_equivalent(&build());

    let Err(EquivalenceError::Constraint(ConstraintError::Custom(source))) = &result else {
        panic!("expected a custom constraint's failure, got {result:?}");
    };
    assert_eq!(source.to_string(), "the alpha comparison failed");
}

#[test]
fn realization_with_sub_choices_relabeled_is_alpha_equivalent() {
    let build = || {
        let [space, c, r, axis, s, b, k] =
            ["space", "c", "r", "axis", "s", "b", "k"].map(Identifier::new);
        let sub = choice_of(
            &s,
            vec![plain_alternative(
                &b,
                vec![TileKnob::part(&k, int_param(&[4, 8]), &[&axis])],
                vec![],
            )],
        );
        let realization = Realization::new(&r, vec![], &[&axis], &[&axis], 0)
            .with_choices(vec![sub])
            .into_part();
        Space::new(
            space,
            vec![],
            vec![choice_of(&c, vec![realization])],
            vec![],
            vec![],
        )
        .expect("the space is valid")
    };

    let (left, right) = (build(), build());

    assert_eq!(
        compare_alpha_both_ways(&left, &right),
        [true, true],
        "the knob two levels down names the realization's axis"
    );
}
