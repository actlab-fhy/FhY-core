//! Tests for downstream implementations of `Variable` and `Alternative`:
//! the test implementors `TileKnob` and `Realization`, shaped like
//! MOGA-VM's `ArrayTileKnob` and `RealizationOption`, in comparisons,
//! choices, spaces and configurations; hooks that fail; and, behind the
//! `testing` feature, the conformance checks and one implementor breaking
//! each checkable clause of the contract.
//!
//! The core compares names, params, variables, sub-choices, the number of
//! bound identifiers, kinds and notes itself, and calls an implementor's
//! hooks only for two parts of one kind, after those. A hook's renaming
//! already pairs every name of the space, bound identifiers included, so
//! an identifier in an implementor's own data corresponds through
//! `is_corresponding` only.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fhy_core::constraint::Value;
use fhy_core::foreign::{BoxError, ForeignPart, Part};
use fhy_core::identifier::Identifier;
use fhy_core::param::Param;
use fhy_core::search_space::{
    Alternative, Choice, EquivalenceError, PlainAlternative, PlainVariable, Space, SpaceError,
    Variable,
};
use fhy_core::term::{AlphaEquivalence, AlphaRenaming};
use rstest::rstest;

use crate::support::constraint::int;
use crate::support::hashing::hash_of;
use crate::support::search_space::{
    Realization, TileKnob, bare_alternative, choice_of, chosen, compare_alpha_both_ways, configure,
    int_param, plain_variable, space_of,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the verdicts of `compare` on `left` with `right` and on `right`
/// with `left`.
fn compare_both_ways<T>(
    left: &T,
    right: &T,
    compare: impl Fn(&T, &T) -> Result<bool, EquivalenceError>,
) -> [bool; 2] {
    [
        compare(left, right).expect("compares"),
        compare(right, left).expect("compares"),
    ]
}

/// Return the structural comparison of two variables, both ways.
fn compare_structural_variables(
    left: &Part<dyn Variable>,
    right: &Part<dyn Variable>,
) -> [bool; 2] {
    compare_both_ways(
        left,
        right,
        <Part<dyn Variable>>::is_structurally_equivalent,
    )
}

/// Return the structural comparison of two alternatives, both ways.
fn compare_structural_alternatives(
    left: &Part<dyn Alternative>,
    right: &Part<dyn Alternative>,
) -> [bool; 2] {
    compare_both_ways(
        left,
        right,
        <Part<dyn Alternative>>::is_structurally_equivalent,
    )
}

/// Where an alternative is compared: on its own, inside the choice that
/// holds it, or inside the space that holds that choice.
#[derive(Debug, Clone, Copy)]
enum Level {
    Alone,
    InChoice,
    InSpace,
}

/// Return the alpha comparison, both ways, of `left` with `right`, each
/// the only alternative of a choice named afresh (and of a space named
/// afresh) when `level` says so.
fn compare_alpha_at(
    level: Level,
    left: &Part<dyn Alternative>,
    right: &Part<dyn Alternative>,
) -> [bool; 2] {
    let choice = |alternative: &Part<dyn Alternative>| {
        choice_of(&Identifier::new("c"), vec![alternative.clone()])
    };
    let space = |alternative: &Part<dyn Alternative>| {
        space_of(&Identifier::new("s"), Vec::new(), vec![choice(alternative)])
    };
    match level {
        Level::Alone => compare_alpha_both_ways(left, right),
        Level::InChoice => compare_both_ways(&choice(left), &choice(right), |a, b| {
            a.is_alpha_equivalent(b)
        }),
        Level::InSpace => compare_both_ways(
            &space(left),
            &space(right),
            AlphaEquivalence::is_alpha_equivalent,
        ),
    }
}

/// Return the realization `name` binding `axes`, with walk order `order`
/// and tag 0, holding nothing else.
fn build_walk_realization(
    name: &Identifier,
    axes: &[&Identifier],
    order: &[&Identifier],
) -> Part<dyn Alternative> {
    Realization::new(name, Vec::new(), axes, order, 0).into_part()
}

/// Return the realization `name` binding `axes`, with walk order `order`
/// and tag `tag`.
fn build_tagged_realization(
    name: &Identifier,
    axes: &[&Identifier],
    order: &[&Identifier],
    tag: i64,
) -> Part<dyn Alternative> {
    Realization::new(name, Vec::new(), axes, order, tag).into_part()
}

/// Return the text of the extension error `result` holds.
///
/// # Panics
///
/// Panics if `result` is anything else.
fn render_extension_text(result: Result<bool, EquivalenceError>) -> String {
    match result {
        Err(EquivalenceError::Extension(source)) => source.to_string(),
        other => panic!("expected an Extension error, got {other:?}"),
    }
}

/// Return the name a `DuplicateName` refusal reports.
///
/// # Panics
///
/// Panics if `result` is anything else.
fn find_duplicate_name<T>(result: Result<T, SpaceError>) -> Identifier {
    match result {
        Err(SpaceError::DuplicateName { name }) => name,
        Err(other) => panic!("expected DuplicateName, got {other:?}"),
        Ok(_) => panic!("expected DuplicateName, got a built value"),
    }
}

/// How many times each hook of the parts that share it was called.
#[derive(Debug, Clone, Default)]
struct HookCalls {
    structural: Arc<AtomicUsize>,
    alpha: Arc<AtomicUsize>,
}

impl HookCalls {
    fn structural(&self) -> usize {
        self.structural.load(Ordering::SeqCst)
    }

    fn alpha(&self) -> usize {
        self.alpha.load(Ordering::SeqCst)
    }
}

/// A variable of its own kind whose hooks count their calls and accept.
#[derive(Debug)]
struct RecordingVariable {
    name: Identifier,
    param: Param,
    calls: HookCalls,
}

impl RecordingVariable {
    fn part(name: &Identifier, param: &Param, calls: &HookCalls) -> Part<dyn Variable> {
        Part::new(Self {
            name: name.clone(),
            param: param.clone(),
            calls: calls.clone(),
        })
    }
}

impl ForeignPart for RecordingVariable {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("RecordingVariable")
    }
}

impl Variable for RecordingVariable {
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed("test.recording_variable")
    }

    fn name(&self) -> &Identifier {
        &self.name
    }

    fn param(&self) -> &Param {
        &self.param
    }

    fn is_extension_structurally_equivalent(&self, _: &dyn Variable) -> Result<bool, BoxError> {
        self.calls.structural.fetch_add(1, Ordering::SeqCst);
        Ok(true)
    }

    fn is_extension_alpha_equivalent_under(
        &self,
        _: &dyn Variable,
        _: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        self.calls.alpha.fetch_add(1, Ordering::SeqCst);
        Ok(true)
    }
}

/// An alternative of its own kind whose hooks count their calls and
/// accept.
#[derive(Debug)]
struct RecordingAlternative {
    name: Identifier,
    calls: HookCalls,
}

impl RecordingAlternative {
    fn part(name: &Identifier, calls: &HookCalls) -> Part<dyn Alternative> {
        Part::new(Self {
            name: name.clone(),
            calls: calls.clone(),
        })
    }
}

impl ForeignPart for RecordingAlternative {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("RecordingAlternative")
    }
}

impl Alternative for RecordingAlternative {
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed("test.recording_alternative")
    }

    fn name(&self) -> &Identifier {
        &self.name
    }

    fn variables(&self) -> &[Part<dyn Variable>] {
        &[]
    }

    fn is_extension_structurally_equivalent(&self, _: &dyn Alternative) -> Result<bool, BoxError> {
        self.calls.structural.fetch_add(1, Ordering::SeqCst);
        Ok(true)
    }

    fn is_extension_alpha_equivalent_under(
        &self,
        _: &dyn Alternative,
        _: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        self.calls.alpha.fetch_add(1, Ordering::SeqCst);
        Ok(true)
    }
}

/// The text of the error the failing structural hooks answer.
const STRUCTURAL_FAILURE: &str = "the structural hook failed";
/// The text of the error the failing alpha hooks answer.
const ALPHA_FAILURE: &str = "the alpha hook failed";
/// The text of the error the failing bound identifiers answer.
const BOUND_FAILURE: &str = "the bound identifiers failed";

/// A variable whose hooks both fail.
#[derive(Debug)]
struct FailingVariable {
    name: Identifier,
    param: Param,
}

impl ForeignPart for FailingVariable {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("FailingVariable")
    }
}

impl Variable for FailingVariable {
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed("test.failing_variable")
    }

    fn name(&self) -> &Identifier {
        &self.name
    }

    fn param(&self) -> &Param {
        &self.param
    }

    fn is_extension_structurally_equivalent(&self, _: &dyn Variable) -> Result<bool, BoxError> {
        Err(BoxError::from(STRUCTURAL_FAILURE))
    }

    fn is_extension_alpha_equivalent_under(
        &self,
        _: &dyn Variable,
        _: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        Err(BoxError::from(ALPHA_FAILURE))
    }
}

/// An alternative whose bound identifiers fail, or, when they do not,
/// whose hooks fail.
#[derive(Debug)]
struct FailingAlternative {
    name: Identifier,
    fails_bound_identifiers: bool,
}

impl ForeignPart for FailingAlternative {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("FailingAlternative")
    }
}

impl Alternative for FailingAlternative {
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed("test.failing_alternative")
    }

    fn name(&self) -> &Identifier {
        &self.name
    }

    fn variables(&self) -> &[Part<dyn Variable>] {
        &[]
    }

    fn bound_identifiers(&self) -> Result<Vec<Identifier>, BoxError> {
        if self.fails_bound_identifiers {
            Err(BoxError::from(BOUND_FAILURE))
        } else {
            Ok(Vec::new())
        }
    }

    fn is_extension_structurally_equivalent(&self, _: &dyn Alternative) -> Result<bool, BoxError> {
        Err(BoxError::from(STRUCTURAL_FAILURE))
    }

    fn is_extension_alpha_equivalent_under(
        &self,
        _: &dyn Alternative,
        _: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        Err(BoxError::from(ALPHA_FAILURE))
    }
}

/// Return the failing alternative `name`, whose bound identifiers fail
/// when `fails_bound_identifiers`.
fn build_failing_alternative(
    name: &Identifier,
    fails_bound_identifiers: bool,
) -> Part<dyn Alternative> {
    Part::new(FailingAlternative {
        name: name.clone(),
        fails_bound_identifiers,
    })
}

// ---------------------------------------------------------------------------
// Kinds and the implementor's own data
// ---------------------------------------------------------------------------

#[test]
fn variables_of_different_kinds_are_not_structurally_equivalent() {
    let name = Identifier::new("x");
    let param = int_param(&[1, 2]);
    let knob = TileKnob::part(&name, param.clone(), &[]);
    let plain = plain_variable(&name, param);

    assert_eq!(compare_structural_variables(&knob, &plain), [false, false]);
    assert_eq!(compare_alpha_both_ways(&knob, &plain), [false, false]);
}

#[test]
fn alternatives_with_different_implementor_data_are_not_structurally_equivalent() {
    let name = Identifier::new("r");
    let axis = Identifier::new("i");
    let one = build_tagged_realization(&name, &[&axis], &[&axis], 1);
    let two = build_tagged_realization(&name, &[&axis], &[&axis], 2);
    let one_again = build_tagged_realization(&name, &[&axis], &[&axis], 1);

    assert_eq!(compare_structural_alternatives(&one, &two), [false, false]);
    assert_eq!(
        compare_structural_alternatives(&one, &one_again),
        [true, true]
    );
}

#[test]
fn alternatives_with_different_implementor_data_are_not_alpha_equivalent() {
    let name = Identifier::new("r");
    let axis = Identifier::new("i");
    let one = build_tagged_realization(&name, &[&axis], &[&axis], 1);
    let two = build_tagged_realization(&name, &[&axis], &[&axis], 2);

    assert_eq!(compare_alpha_both_ways(&one, &two), [false, false]);
}

#[test]
fn alternatives_of_different_kinds_are_not_equivalent() {
    let name = Identifier::new("r");
    let realization = build_walk_realization(&name, &[], &[]);
    let plain = bare_alternative(&name);

    assert_eq!(
        compare_structural_alternatives(&realization, &plain),
        [false, false]
    );
    assert_eq!(
        compare_alpha_both_ways(&realization, &plain),
        [false, false]
    );
}

#[test]
fn tile_knobs_over_different_index_symbols_are_not_structurally_equivalent() {
    let name = Identifier::new("t");
    let param = int_param(&[1, 2]);
    let i = Identifier::new("i");
    let j = Identifier::new("j");
    let over_i = TileKnob::part(&name, param.clone(), &[&i]);
    let over_j = TileKnob::part(&name, param.clone(), &[&j]);
    let over_i_again = TileKnob::part(&name, param, &[&i]);

    assert_eq!(
        compare_structural_variables(&over_i, &over_j),
        [false, false]
    );
    assert_eq!(
        compare_structural_variables(&over_i, &over_i_again),
        [true, true]
    );
}

// ---------------------------------------------------------------------------
// The core compares the base fields, and hooks only after them
// ---------------------------------------------------------------------------

#[test]
fn hooks_are_not_called_for_variables_of_different_kinds() {
    let name = Identifier::new("x");
    let param = int_param(&[1, 2]);
    let calls = HookCalls::default();
    let recording = RecordingVariable::part(&name, &param, &calls);
    let plain = plain_variable(&name, param);

    assert_eq!(
        compare_structural_variables(&recording, &plain),
        [false, false]
    );
    assert_eq!(compare_alpha_both_ways(&recording, &plain), [false, false]);
    assert_eq!((calls.structural(), calls.alpha()), (0, 0));
}

#[test]
fn hooks_are_not_called_for_alternatives_of_different_kinds() {
    let name = Identifier::new("a");
    let calls = HookCalls::default();
    let recording = RecordingAlternative::part(&name, &calls);
    let plain = bare_alternative(&name);

    assert_eq!(
        compare_structural_alternatives(&recording, &plain),
        [false, false]
    );
    assert_eq!(compare_alpha_both_ways(&recording, &plain), [false, false]);
    assert_eq!((calls.structural(), calls.alpha()), (0, 0));
}

#[test]
fn a_structural_comparison_of_two_variables_of_one_kind_calls_only_the_structural_hook_once() {
    let name = Identifier::new("x");
    let param = int_param(&[1, 2]);
    let calls = HookCalls::default();
    let left = RecordingVariable::part(&name, &param, &calls);
    let right = RecordingVariable::part(&name, &param, &calls);

    assert!(left.is_structurally_equivalent(&right).expect("compares"));
    assert_eq!((calls.structural(), calls.alpha()), (1, 0));
}

#[test]
fn an_alpha_comparison_of_two_variables_of_one_kind_calls_only_the_alpha_hook_once() {
    let name = Identifier::new("x");
    let param = int_param(&[1, 2]);
    let calls = HookCalls::default();
    let left = RecordingVariable::part(&name, &param, &calls);
    let right = RecordingVariable::part(&Identifier::new("y"), &param, &calls);

    assert!(left.is_alpha_equivalent(&right).expect("compares"));
    assert_eq!((calls.structural(), calls.alpha()), (0, 1));
}

#[test]
fn a_structural_comparison_of_two_alternatives_of_one_kind_calls_only_the_structural_hook_once() {
    let name = Identifier::new("a");
    let calls = HookCalls::default();
    let left = RecordingAlternative::part(&name, &calls);
    let right = RecordingAlternative::part(&name, &calls);

    assert!(left.is_structurally_equivalent(&right).expect("compares"));
    assert_eq!((calls.structural(), calls.alpha()), (1, 0));
}

#[test]
fn an_alpha_comparison_of_two_alternatives_of_one_kind_calls_only_the_alpha_hook_once() {
    let calls = HookCalls::default();
    let left = RecordingAlternative::part(&Identifier::new("a"), &calls);
    let right = RecordingAlternative::part(&Identifier::new("b"), &calls);

    assert!(left.is_alpha_equivalent(&right).expect("compares"));
    assert_eq!((calls.structural(), calls.alpha()), (0, 1));
}

#[test]
fn tile_knobs_with_equal_index_symbols_but_different_params_are_not_equivalent() {
    let name = Identifier::new("t");
    let i = Identifier::new("i");
    let small = TileKnob::part(&name, int_param(&[1, 2]), &[&i]);
    let large = TileKnob::part(&name, int_param(&[1, 3]), &[&i]);

    assert_eq!(compare_structural_variables(&small, &large), [false, false]);
    assert_eq!(compare_alpha_both_ways(&small, &large), [false, false]);
}

#[test]
fn tile_knobs_with_equal_index_symbols_but_different_names_differ_structurally_only() {
    let param = int_param(&[1, 2]);
    let z = Identifier::new("z");
    let first = TileKnob::part(&Identifier::new("t"), param.clone(), &[&z]);
    let second = TileKnob::part(&Identifier::new("u"), param, &[&z]);

    assert_eq!(
        compare_structural_variables(&first, &second),
        [false, false]
    );
    assert_eq!(compare_alpha_both_ways(&first, &second), [true, true]);
}

#[test]
fn realizations_with_equal_data_but_different_names_differ_structurally_only() {
    let axis = Identifier::new("i");
    let first = build_walk_realization(&Identifier::new("r"), &[&axis], &[&axis]);
    let second = build_walk_realization(&Identifier::new("q"), &[&axis], &[&axis]);

    assert_eq!(
        compare_structural_alternatives(&first, &second),
        [false, false]
    );
    assert_eq!(compare_alpha_both_ways(&first, &second), [true, true]);
}

// ---------------------------------------------------------------------------
// Failing hooks
// ---------------------------------------------------------------------------

#[test]
fn a_failing_structural_hook_of_a_variable_fails_the_comparison() {
    let name = Identifier::new("x");
    let param = int_param(&[1]);
    let left = Part::<dyn Variable>::new(FailingVariable {
        name: name.clone(),
        param: param.clone(),
    });
    let right = Part::<dyn Variable>::new(FailingVariable { name, param });

    assert_eq!(
        render_extension_text(left.is_structurally_equivalent(&right)),
        STRUCTURAL_FAILURE
    );
}

#[test]
fn a_failing_alpha_hook_of_a_variable_fails_the_comparison() {
    let param = int_param(&[1]);
    let left = Part::<dyn Variable>::new(FailingVariable {
        name: Identifier::new("x"),
        param: param.clone(),
    });
    let right = Part::<dyn Variable>::new(FailingVariable {
        name: Identifier::new("y"),
        param,
    });

    assert_eq!(
        render_extension_text(left.is_alpha_equivalent(&right)),
        ALPHA_FAILURE
    );
}

#[test]
fn a_failing_structural_hook_of_an_alternative_fails_the_comparison() {
    let name = Identifier::new("a");
    let left = build_failing_alternative(&name, false);
    let right = build_failing_alternative(&name, false);

    assert_eq!(
        render_extension_text(left.is_structurally_equivalent(&right)),
        STRUCTURAL_FAILURE
    );
}

#[test]
fn a_failing_alpha_hook_of_an_alternative_fails_the_comparison() {
    let left = build_failing_alternative(&Identifier::new("a"), false);
    let right = build_failing_alternative(&Identifier::new("b"), false);

    assert_eq!(
        render_extension_text(left.is_alpha_equivalent(&right)),
        ALPHA_FAILURE
    );
}

#[test]
fn a_failing_hook_fails_the_comparison_of_the_choices_that_hold_it() {
    let name = Identifier::new("a");
    let left = choice_of(
        &Identifier::new("c"),
        vec![build_failing_alternative(&name, false)],
    );
    let right = choice_of(
        &Identifier::new("d"),
        vec![build_failing_alternative(&name, false)],
    );
    let same = choice_of(left.name(), vec![build_failing_alternative(&name, false)]);

    assert_eq!(
        render_extension_text(left.is_alpha_equivalent(&right)),
        ALPHA_FAILURE
    );
    assert_eq!(
        render_extension_text(left.is_structurally_equivalent(&same)),
        STRUCTURAL_FAILURE
    );
}

#[test]
fn a_failing_hook_of_a_variable_fails_the_comparison_of_the_spaces_that_hold_it() {
    let name = Identifier::new("x");
    let param = int_param(&[1]);
    let space_name = Identifier::new("s");
    let build = || {
        space_of(
            &space_name,
            vec![Part::new(FailingVariable {
                name: name.clone(),
                param: param.clone(),
            })],
            Vec::new(),
        )
    };
    let left = build();
    let right = build();

    assert_eq!(
        render_extension_text(left.is_alpha_equivalent(&right)),
        ALPHA_FAILURE
    );
    assert_eq!(
        render_extension_text(left.is_structurally_equivalent(&right)),
        STRUCTURAL_FAILURE
    );
}

#[test]
fn a_failing_hook_of_an_alternative_fails_the_comparison_of_the_spaces_that_hold_it() {
    let name = Identifier::new("a");
    let build = |space: &str| {
        space_of(
            &Identifier::new(space),
            Vec::new(),
            vec![choice_of(
                &Identifier::new("c"),
                vec![build_failing_alternative(&name, false)],
            )],
        )
    };
    let left = build("s");
    let right = build("t");

    assert_eq!(
        render_extension_text(left.is_alpha_equivalent(&right)),
        ALPHA_FAILURE
    );
}

#[test]
fn failing_bound_identifiers_refuse_the_choice_that_holds_the_alternative() {
    let name = Identifier::new("a");
    let result = Choice::new(
        Identifier::new("c"),
        vec![build_failing_alternative(&name, true)],
    );

    let Err(SpaceError::Hook {
        alternative,
        source,
    }) = result
    else {
        panic!("expected a Hook refusal");
    };
    assert_eq!(alternative, name);
    assert_eq!(source.to_string(), BOUND_FAILURE);
}

#[test]
fn failing_bound_identifiers_fail_the_comparison_of_the_alternative_on_its_own() {
    let name = Identifier::new("a");
    let left = build_failing_alternative(&name, true);
    let right = build_failing_alternative(&name, true);

    assert_eq!(
        render_extension_text(left.is_structurally_equivalent(&right)),
        BOUND_FAILURE
    );
    assert_eq!(
        render_extension_text(left.is_alpha_equivalent(&right)),
        BOUND_FAILURE
    );
}

// ---------------------------------------------------------------------------
// C-1: compared on its own, in a choice and in a space, the answers agree
// ---------------------------------------------------------------------------

#[rstest]
#[case::alone(Level::Alone)]
#[case::in_a_choice(Level::InChoice)]
#[case::in_a_space(Level::InSpace)]
fn a_realization_matches_its_relabeled_copy(#[case] level: Level) {
    let (i, j) = (Identifier::new("i"), Identifier::new("j"));
    let (i2, j2) = (Identifier::new("i2"), Identifier::new("j2"));
    let left = build_walk_realization(&Identifier::new("r"), &[&i, &j], &[&j, &i]);
    let right = build_walk_realization(&Identifier::new("r2"), &[&i2, &j2], &[&j2, &i2]);

    assert_eq!(compare_alpha_at(level, &left, &right), [true, true]);
}

#[rstest]
#[case::alone(Level::Alone)]
#[case::in_a_choice(Level::InChoice)]
#[case::in_a_space(Level::InSpace)]
fn a_realization_does_not_match_a_copy_whose_order_is_not_the_relabeled_one(#[case] level: Level) {
    let (i, j) = (Identifier::new("i"), Identifier::new("j"));
    let (i2, j2) = (Identifier::new("i2"), Identifier::new("j2"));
    let left = build_walk_realization(&Identifier::new("r"), &[&i, &j], &[&j, &i]);
    let right = build_walk_realization(&Identifier::new("r2"), &[&i2, &j2], &[&i2, &j2]);

    assert_eq!(compare_alpha_at(level, &left, &right), [false, false]);
}

// ---------------------------------------------------------------------------
// C-2: bound identifiers are distinct, and as many for equivalent parts
// ---------------------------------------------------------------------------

#[test]
fn choice_new_refuses_a_realization_that_repeats_an_axis() {
    let axis = Identifier::new("i");
    let result = Choice::new(
        Identifier::new("c"),
        vec![build_walk_realization(
            &Identifier::new("r"),
            &[&axis, &axis],
            &[&axis],
        )],
    );

    assert_eq!(find_duplicate_name(result), axis);
}

#[rstest]
#[case::alone(Level::Alone)]
#[case::in_a_choice(Level::InChoice)]
#[case::in_a_space(Level::InSpace)]
fn realizations_binding_different_numbers_of_axes_are_not_equivalent(#[case] level: Level) {
    let (i, j) = (Identifier::new("i"), Identifier::new("j"));
    let k = Identifier::new("k");
    let two = build_walk_realization(&Identifier::new("r"), &[&i, &j], &[&i]);
    let one = build_walk_realization(&Identifier::new("r2"), &[&k], &[&k]);

    assert_eq!(compare_alpha_at(level, &two, &one), [false, false]);
}

#[test]
fn realizations_binding_different_numbers_of_axes_are_not_structurally_equivalent() {
    let name = Identifier::new("r");
    let (i, j) = (Identifier::new("i"), Identifier::new("j"));
    let two = build_walk_realization(&name, &[&i, &j], &[&i]);
    let one = build_walk_realization(&name, &[&i], &[&i]);

    assert_eq!(compare_structural_alternatives(&two, &one), [false, false]);
}

// ---------------------------------------------------------------------------
// C-4 (F-SS-003): a free identifier never matches a bound one
// ---------------------------------------------------------------------------

#[rstest]
#[case::alone(Level::Alone)]
#[case::in_a_choice(Level::InChoice)]
#[case::in_a_space(Level::InSpace)]
fn a_free_identifier_in_the_order_does_not_match_an_axis_of_the_same_name(#[case] level: Level) {
    let (i, z) = (Identifier::new("i"), Identifier::new("z"));
    let free = build_walk_realization(&Identifier::new("r"), &[&i], &[&z]);
    let bound = build_walk_realization(&Identifier::new("r2"), &[&z], &[&z]);

    assert_eq!(compare_alpha_at(level, &free, &bound), [false, false]);
}

#[rstest]
#[case::alone(Level::Alone)]
#[case::in_a_choice(Level::InChoice)]
#[case::in_a_space(Level::InSpace)]
fn the_same_free_identifier_in_both_orders_matches(#[case] level: Level) {
    let (i, i2, z) = (
        Identifier::new("i"),
        Identifier::new("i2"),
        Identifier::new("z"),
    );
    let left = build_walk_realization(&Identifier::new("r"), &[&i], &[&z]);
    let right = build_walk_realization(&Identifier::new("r2"), &[&i2], &[&z]);

    assert_eq!(compare_alpha_at(level, &left, &right), [true, true]);
}

// ---------------------------------------------------------------------------
// The hooks see the space's frame
// ---------------------------------------------------------------------------

/// Return the space `space` whose one choice holds a realization binding
/// `axis`, walking `[axis]`, and holding a tile knob over `symbol`.
fn build_framed_space(space: &str, axis: &Identifier, symbol: &Identifier) -> Space {
    let knob = TileKnob::part(&Identifier::new("t"), int_param(&[1, 2]), &[symbol]);
    let realization =
        Realization::new(&Identifier::new("r"), vec![knob], &[axis], &[axis], 0).into_part();
    space_of(
        &Identifier::new(space),
        Vec::new(),
        vec![choice_of(&Identifier::new("c"), vec![realization])],
    )
}

#[test]
fn a_knob_over_an_axis_matches_a_relabeled_copy_over_the_relabeled_axis() {
    let (i, i2) = (Identifier::new("i"), Identifier::new("i2"));
    let left = build_framed_space("s", &i, &i);
    let right = build_framed_space("s2", &i2, &i2);

    assert_eq!(
        compare_both_ways(&left, &right, AlphaEquivalence::is_alpha_equivalent),
        [true, true]
    );
}

#[test]
fn a_knob_over_an_axis_does_not_match_a_copy_over_a_free_identifier() {
    let (i, i2, k) = (
        Identifier::new("i"),
        Identifier::new("i2"),
        Identifier::new("k"),
    );
    let left = build_framed_space("s", &i, &i);
    let right = build_framed_space("s2", &i2, &k);

    assert_eq!(
        compare_both_ways(&left, &right, AlphaEquivalence::is_alpha_equivalent),
        [false, false]
    );
}

#[test]
fn a_knob_over_a_free_identifier_matches_a_copy_over_the_same_one() {
    let (i, i2, z) = (
        Identifier::new("i"),
        Identifier::new("i2"),
        Identifier::new("z"),
    );
    let left = build_framed_space("s", &i, &z);
    let right = build_framed_space("s2", &i2, &z);

    assert_eq!(
        compare_both_ways(&left, &right, AlphaEquivalence::is_alpha_equivalent),
        [true, true]
    );
}

// ---------------------------------------------------------------------------
// Bound identifiers are names of the space
// ---------------------------------------------------------------------------

#[test]
fn choice_new_refuses_an_axis_equal_to_the_name_of_a_variable_of_its_realization() {
    let axis = Identifier::new("i");
    let knob = TileKnob::part(&axis, int_param(&[1]), &[]);
    let realization = Realization::new(&Identifier::new("r"), vec![knob], &[&axis], &[], 0);

    let result = Choice::new(Identifier::new("c"), vec![realization.into_part()]);

    assert_eq!(find_duplicate_name(result), axis);
}

#[test]
fn choice_new_refuses_an_axis_equal_to_the_name_of_another_alternative() {
    let other = Identifier::new("other");
    let result = Choice::new(
        Identifier::new("c"),
        vec![
            bare_alternative(&other),
            build_walk_realization(&Identifier::new("r"), &[&other], &[]),
        ],
    );

    assert_eq!(find_duplicate_name(result), other);
}

#[test]
fn choice_new_refuses_an_axis_equal_to_the_name_of_its_own_alternative() {
    let name = Identifier::new("r");
    let result = Choice::new(
        Identifier::new("c"),
        vec![build_walk_realization(&name, &[&name], &[])],
    );

    assert_eq!(find_duplicate_name(result), name);
}

#[test]
fn choice_new_refuses_an_axis_equal_to_the_name_of_the_choice() {
    let name = Identifier::new("c");
    let result = Choice::new(
        name.clone(),
        vec![build_walk_realization(&Identifier::new("r"), &[&name], &[])],
    );

    assert_eq!(find_duplicate_name(result), name);
}

#[test]
fn space_new_refuses_an_axis_equal_to_the_name_of_a_top_level_variable() {
    let axis = Identifier::new("i");
    let choice = choice_of(
        &Identifier::new("c"),
        vec![build_walk_realization(&Identifier::new("r"), &[&axis], &[])],
    );
    let result = Space::new(
        Identifier::new("s"),
        vec![TileKnob::part(&axis, int_param(&[1]), &[])],
        vec![choice],
        Vec::new(),
        Vec::new(),
    );

    assert_eq!(find_duplicate_name(result), axis);
}

#[test]
fn space_new_refuses_an_axis_equal_to_the_name_of_the_space() {
    let name = Identifier::new("s");
    let choice = choice_of(
        &Identifier::new("c"),
        vec![build_walk_realization(&Identifier::new("r"), &[&name], &[])],
    );
    let result = Space::new(
        name.clone(),
        Vec::new(),
        vec![choice],
        Vec::new(),
        Vec::new(),
    );

    assert_eq!(find_duplicate_name(result), name);
}

// ---------------------------------------------------------------------------
// `==` and `Hash` go through eq_part and hash_part
// ---------------------------------------------------------------------------

#[test]
fn spaces_holding_equal_tile_knobs_are_equal_and_hash_alike() {
    let (space, name, i) = (
        Identifier::new("s"),
        Identifier::new("t"),
        Identifier::new("i"),
    );
    let param = int_param(&[1, 2]);
    let build = |symbol: &Identifier| {
        space_of(
            &space,
            vec![TileKnob::part(&name, param.clone(), &[symbol])],
            Vec::new(),
        )
    };
    let left = build(&i);
    let right = build(&i);
    let other = build(&Identifier::new("j"));

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
    assert_ne!(left, other);
    assert_ne!(right, other);
}

#[test]
fn choices_holding_equal_realizations_are_equal_and_hash_alike() {
    let (choice, name, axis) = (
        Identifier::new("c"),
        Identifier::new("r"),
        Identifier::new("i"),
    );
    let build = |tag: i64| {
        choice_of(
            &choice,
            vec![build_tagged_realization(&name, &[&axis], &[&axis], tag)],
        )
    };
    let left = build(1);
    let right = build(1);
    let other = build(2);

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
    assert_ne!(left, other);
}

#[test]
fn equal_tile_knobs_are_equal_parts_with_one_hash() {
    let (name, i) = (Identifier::new("t"), Identifier::new("i"));
    let param = int_param(&[1, 2]);
    let left = TileKnob::part(&name, param.clone(), &[&i]);
    let right = TileKnob::part(&name, param.clone(), &[&i]);
    let other = TileKnob::part(&name, param, &[&Identifier::new("j")]);
    // Building a space of them reads the same parts back through the core.
    let held = space_of(&Identifier::new("s"), vec![left.clone()], Vec::new());

    assert_eq!(held.variables(), std::slice::from_ref(&right));
    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
    assert_ne!(left, other);
}

#[test]
fn a_tile_knob_is_never_equal_to_a_plain_variable() {
    let name = Identifier::new("x");
    let param = int_param(&[1, 2]);
    let knob = TileKnob::part(&name, param.clone(), &[]);
    let plain: Part<dyn Variable> = Part::new(PlainVariable::new(name, param));

    assert_ne!(knob, plain);
    assert_ne!(plain, knob);
}

#[test]
fn a_realization_is_never_equal_to_a_plain_alternative() {
    let name = Identifier::new("a");
    let realization = build_walk_realization(&name, &[], &[]);
    let plain: Part<dyn Alternative> = Part::new(
        PlainAlternative::new(name, Vec::new(), Vec::new()).expect("the names are distinct"),
    );

    assert_ne!(realization, plain);
    assert_ne!(plain, realization);
}

// ---------------------------------------------------------------------------
// Configurations over a space of implementors
// ---------------------------------------------------------------------------

/// The names of the space `implementor_space` builds.
struct SpaceNames {
    space: Identifier,
    choice: Identifier,
    realization: Identifier,
    plain: Identifier,
    axis: Identifier,
    knob: Identifier,
}

impl SpaceNames {
    fn fresh(suffix: &str) -> Self {
        let named = |stem: &str| Identifier::new(&format!("{stem}{suffix}"));
        Self {
            space: named("s"),
            choice: named("c"),
            realization: named("r"),
            plain: named("p"),
            axis: named("i"),
            knob: named("t"),
        }
    }
}

/// Return the space of a choice between a realization binding one axis
/// and a plain alternative, and a tile knob over that axis.
fn build_implementor_space(names: &SpaceNames) -> Space {
    let realization = build_walk_realization(&names.realization, &[&names.axis], &[&names.axis]);
    let choice = choice_of(
        &names.choice,
        vec![realization, bare_alternative(&names.plain)],
    );
    let knob = TileKnob::part(&names.knob, int_param(&[1, 2, 3]), &[&names.axis]);
    space_of(&names.space, vec![knob], vec![choice])
}

/// Return the entries choosing `alternative` and giving the knob `size`.
fn build_entries(
    names: &SpaceNames,
    alternative: &Identifier,
    size: i64,
) -> [(Identifier, Value); 2] {
    [
        (names.choice.clone(), chosen(alternative)),
        (names.knob.clone(), int(size)),
    ]
}

#[test]
fn corresponding_configurations_of_relabeled_implementor_spaces_have_equal_keys() {
    let (left, right) = (SpaceNames::fresh("_l"), SpaceNames::fresh("_r"));
    let left_space = build_implementor_space(&left);
    let right_space = build_implementor_space(&right);

    let left_key = configure(&left_space, build_entries(&left, &left.realization, 2)).key();
    let right_key = configure(&right_space, build_entries(&right, &right.realization, 2)).key();

    assert_eq!(left_key, right_key);
}

#[test]
fn different_configurations_of_an_implementor_space_have_different_keys() {
    let names = SpaceNames::fresh("");
    let space = build_implementor_space(&names);

    let realized = configure(&space, build_entries(&names, &names.realization, 2)).key();
    let plain = configure(&space, build_entries(&names, &names.plain, 2)).key();
    let resized = configure(&space, build_entries(&names, &names.realization, 3)).key();

    assert_ne!(realized, plain);
    assert_ne!(realized, resized);
}

// ---------------------------------------------------------------------------
// Conformance (behind the `testing` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "testing")]
mod conformance {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use fhy_core::diagnostic::Note;
    use fhy_core::foreign::{BoxError, Foreign, ForeignError, ForeignPart, Part, Resolve};
    use fhy_core::identifier::Identifier;
    use fhy_core::param::Param;
    use fhy_core::search_space::testing::{
        ContractClause, check_alternative_conformance, check_variable_conformance,
    };
    use fhy_core::search_space::{Alternative, PlainAlternative, PlainVariable, Variable};
    use fhy_core::term::AlphaRenaming;
    use rstest::rstest;
    use serde::{Deserialize, Serialize};
    use std::borrow::Cow;

    use crate::support::search_space::{
        ImplementorResolver, REALIZATION, Realization, TILE_KNOB, TileKnob, int_param,
    };

    const FLAWED_VARIABLE: &str = "test.flawed_variable";
    const FLAWED_ALTERNATIVE: &str = "test.flawed_alternative";
    const UNSERIALIZABLE: &str = "test.unserializable";

    /// The one way a flawed implementor breaks the contract.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Flaw {
        /// Breaks nothing.
        Sound,
        /// `name` answers differently on a second call.
        UnstableName,
        /// `notes` answers differently on a second call.
        UnstableNotes,
        /// The kind is empty.
        EmptyKind,
        /// The kind is the plain implementation's.
        PlainKind,
        /// The kind depends on the value, so one type has several.
        KindPerTag,
        /// The kind is another test implementor's.
        BorrowedKind,
        /// The bound identifiers repeat.
        RepeatedBound,
        /// The structural hook answers `false` for a value and itself.
        NonReflexive,
        /// Both hooks answer `self.tag <= other.tag`.
        Asymmetric,
        /// The structural hook accepts what the alpha hook refuses.
        StructuralWithoutAlpha,
        /// The structural hook answers `Err`.
        FailingHooks,
        /// `to_foreign` writes a type id that is not the kind.
        WrongWireType,
    }

    /// Return the structural hook's verdict on tags `left` and `right`.
    fn compute_structural_verdict(flaw: Flaw, left: i64, right: i64) -> Result<bool, BoxError> {
        match flaw {
            Flaw::NonReflexive => Ok(false),
            Flaw::StructuralWithoutAlpha => Ok(true),
            Flaw::Asymmetric => Ok(left <= right),
            Flaw::FailingHooks => Err(BoxError::from("the structural hook failed")),
            _ => Ok(left == right),
        }
    }

    /// Return the alpha hook's verdict on tags `left` and `right`.
    fn compute_alpha_verdict(flaw: Flaw, left: i64, right: i64) -> bool {
        match flaw {
            Flaw::Asymmetric => left <= right,
            _ => left == right,
        }
    }

    /// Return the kind a flawed value of `own` answers.
    fn compute_kind(flaw: Flaw, own: &str, plain: &str, borrowed: &str, tag: i64) -> String {
        match flaw {
            Flaw::EmptyKind => String::new(),
            Flaw::PlainKind => plain.to_owned(),
            Flaw::BorrowedKind => borrowed.to_owned(),
            Flaw::KindPerTag => format!("{own}.{tag}"),
            _ => own.to_owned(),
        }
    }

    /// The payload of a flawed value's foreign part.
    #[derive(Serialize, Deserialize)]
    struct FlawedPayload {
        name: Identifier,
        param: Option<Param>,
        bound: Vec<Identifier>,
        tag: i64,
    }

    /// Return the foreign part of `payload`, under `type_id`.
    fn build_foreign(type_id: &str, payload: &FlawedPayload) -> Result<Foreign, ForeignError> {
        let data = serde_json::to_string(payload).map_err(|error| ForeignError::Failed {
            type_id: type_id.to_owned(),
            source: Box::new(error),
        })?;
        Ok(Foreign::new(type_id, data))
    }

    /// Return the payload `foreign` holds.
    fn read_payload(foreign: &Foreign) -> Result<FlawedPayload, ForeignError> {
        serde_json::from_str(foreign.data()).map_err(|error| ForeignError::Failed {
            type_id: foreign.type_id().to_owned(),
            source: Box::new(error),
        })
    }

    /// A variable with the one flaw `flaw`, its hooks comparing `tag`.
    #[derive(Debug)]
    struct FlawedVariable {
        names: [Identifier; 2],
        param: Param,
        notes: Vec<Note>,
        flaw: Flaw,
        tag: i64,
        calls: AtomicUsize,
    }

    impl FlawedVariable {
        fn build(flaw: Flaw, name: Identifier, param: Param, tag: i64) -> Self {
            Self {
                names: [name, Identifier::new("elsewhere")],
                param,
                notes: vec![Note::with_other_kind("a note")],
                flaw,
                tag,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl ForeignPart for FlawedVariable {
        fn type_name(&self) -> Cow<'_, str> {
            Cow::Borrowed("FlawedVariable")
        }

        fn to_foreign(&self) -> Result<Foreign, ForeignError> {
            let type_id = if self.flaw == Flaw::WrongWireType {
                "test.flawed_elsewhere".to_owned()
            } else {
                self.kind().into_owned()
            };
            build_foreign(
                &type_id,
                &FlawedPayload {
                    name: self.names[0].clone(),
                    param: Some(self.param.clone()),
                    bound: Vec::new(),
                    tag: self.tag,
                },
            )
        }
    }

    impl Variable for FlawedVariable {
        fn kind(&self) -> Cow<'_, str> {
            Cow::Owned(compute_kind(
                self.flaw,
                FLAWED_VARIABLE,
                PlainVariable::KIND,
                TILE_KNOB,
                self.tag,
            ))
        }

        fn name(&self) -> &Identifier {
            if self.flaw == Flaw::UnstableName {
                &self.names[self.calls.fetch_add(1, Ordering::SeqCst) % 2]
            } else {
                &self.names[0]
            }
        }

        fn param(&self) -> &Param {
            &self.param
        }

        fn notes(&self) -> &[Note] {
            if self.flaw == Flaw::UnstableNotes {
                &self.notes[..self.calls.fetch_add(1, Ordering::SeqCst) % 2]
            } else {
                &[]
            }
        }

        fn is_extension_structurally_equivalent(
            &self,
            other: &dyn Variable,
        ) -> Result<bool, BoxError> {
            let Some(other) = other.as_any().downcast_ref::<Self>() else {
                return Ok(false);
            };
            compute_structural_verdict(self.flaw, self.tag, other.tag)
        }

        fn is_extension_alpha_equivalent_under(
            &self,
            other: &dyn Variable,
            _: &AlphaRenaming,
        ) -> Result<bool, BoxError> {
            let Some(other) = other.as_any().downcast_ref::<Self>() else {
                return Ok(false);
            };
            Ok(compute_alpha_verdict(self.flaw, self.tag, other.tag))
        }
    }

    /// An alternative with the one flaw `flaw`, binding `bound`, its hooks
    /// comparing `tag`.
    #[derive(Debug)]
    struct FlawedAlternative {
        names: [Identifier; 2],
        bound: Vec<Identifier>,
        notes: Vec<Note>,
        flaw: Flaw,
        tag: i64,
        calls: AtomicUsize,
    }

    impl FlawedAlternative {
        fn build(flaw: Flaw, name: Identifier, bound: Vec<Identifier>, tag: i64) -> Self {
            Self {
                names: [name, Identifier::new("elsewhere")],
                bound,
                notes: vec![Note::with_other_kind("a note")],
                flaw,
                tag,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl ForeignPart for FlawedAlternative {
        fn type_name(&self) -> Cow<'_, str> {
            Cow::Borrowed("FlawedAlternative")
        }

        fn to_foreign(&self) -> Result<Foreign, ForeignError> {
            let type_id = if self.flaw == Flaw::WrongWireType {
                "test.flawed_elsewhere".to_owned()
            } else {
                self.kind().into_owned()
            };
            build_foreign(
                &type_id,
                &FlawedPayload {
                    name: self.names[0].clone(),
                    param: None,
                    bound: self.bound.clone(),
                    tag: self.tag,
                },
            )
        }
    }

    impl Alternative for FlawedAlternative {
        fn kind(&self) -> Cow<'_, str> {
            Cow::Owned(compute_kind(
                self.flaw,
                FLAWED_ALTERNATIVE,
                PlainAlternative::KIND,
                REALIZATION,
                self.tag,
            ))
        }

        fn name(&self) -> &Identifier {
            if self.flaw == Flaw::UnstableName {
                &self.names[self.calls.fetch_add(1, Ordering::SeqCst) % 2]
            } else {
                &self.names[0]
            }
        }

        fn variables(&self) -> &[Part<dyn Variable>] {
            &[]
        }

        fn notes(&self) -> &[Note] {
            if self.flaw == Flaw::UnstableNotes {
                &self.notes[..self.calls.fetch_add(1, Ordering::SeqCst) % 2]
            } else {
                &[]
            }
        }

        fn bound_identifiers(&self) -> Result<Vec<Identifier>, BoxError> {
            Ok(self.bound.clone())
        }

        fn is_extension_structurally_equivalent(
            &self,
            other: &dyn Alternative,
        ) -> Result<bool, BoxError> {
            let Some(other) = other.as_any().downcast_ref::<Self>() else {
                return Ok(false);
            };
            compute_structural_verdict(self.flaw, self.tag, other.tag)
        }

        fn is_extension_alpha_equivalent_under(
            &self,
            other: &dyn Alternative,
            _: &AlphaRenaming,
        ) -> Result<bool, BoxError> {
            let Some(other) = other.as_any().downcast_ref::<Self>() else {
                return Ok(false);
            };
            Ok(compute_alpha_verdict(self.flaw, self.tag, other.tag))
        }
    }

    /// A variable with no `to_foreign` of its own.
    #[derive(Debug)]
    struct UnserializableVariable {
        name: Identifier,
        param: Param,
    }

    impl ForeignPart for UnserializableVariable {
        fn type_name(&self) -> Cow<'_, str> {
            Cow::Borrowed("UnserializableVariable")
        }
    }

    impl Variable for UnserializableVariable {
        fn kind(&self) -> Cow<'_, str> {
            Cow::Borrowed(UNSERIALIZABLE)
        }

        fn name(&self) -> &Identifier {
            &self.name
        }

        fn param(&self) -> &Param {
            &self.param
        }
    }

    /// An alternative with no `to_foreign` of its own.
    #[derive(Debug)]
    struct UnserializableAlternative {
        name: Identifier,
    }

    impl ForeignPart for UnserializableAlternative {
        fn type_name(&self) -> Cow<'_, str> {
            Cow::Borrowed("UnserializableAlternative")
        }
    }

    impl Alternative for UnserializableAlternative {
        fn kind(&self) -> Cow<'_, str> {
            Cow::Borrowed(UNSERIALIZABLE)
        }

        fn name(&self) -> &Identifier {
            &self.name
        }

        fn variables(&self) -> &[Part<dyn Variable>] {
            &[]
        }
    }

    /// The resolver of the flawed implementors: it rebuilds a value with
    /// `flaw`, its tag moved by `tag_shift`, so a nonzero shift gives back
    /// a value that is not equivalent to the sample.
    #[derive(Debug, Clone, Copy)]
    struct FlawedResolver {
        flaw: Flaw,
        tag_shift: i64,
    }

    impl FlawedResolver {
        fn faithful(flaw: Flaw) -> Self {
            Self { flaw, tag_shift: 0 }
        }
    }

    impl Resolve<Part<dyn Variable>> for FlawedResolver {
        fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn Variable>, ForeignError> {
            let payload = read_payload(foreign)?;
            let param = payload.param.ok_or_else(|| ForeignError::Failed {
                type_id: foreign.type_id().to_owned(),
                source: BoxError::from("a variable payload holds a param"),
            })?;
            Ok(Part::new(FlawedVariable::build(
                self.flaw,
                payload.name,
                param,
                payload.tag + self.tag_shift,
            )))
        }
    }

    impl Resolve<Part<dyn Alternative>> for FlawedResolver {
        fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn Alternative>, ForeignError> {
            let payload = read_payload(foreign)?;
            Ok(Part::new(FlawedAlternative::build(
                self.flaw,
                payload.name,
                payload.bound,
                payload.tag + self.tag_shift,
            )))
        }
    }

    /// Return three flawed variables of tags 1, 2 and 1, under fresh names.
    fn build_flawed_variables(flaw: Flaw) -> Vec<Part<dyn Variable>> {
        [1, 2, 1]
            .into_iter()
            .map(|tag| {
                Part::new(FlawedVariable::build(
                    flaw,
                    Identifier::new("v"),
                    int_param(&[1, 2]),
                    tag,
                ))
            })
            .collect()
    }

    /// Return three flawed alternatives of tags 1, 2 and 1, each binding
    /// two fresh identifiers, or one twice for [`Flaw::RepeatedBound`].
    fn build_flawed_alternatives(flaw: Flaw) -> Vec<Part<dyn Alternative>> {
        [1, 2, 1]
            .into_iter()
            .map(|tag| {
                let (first, second) = (Identifier::new("a"), Identifier::new("b"));
                let bound = if flaw == Flaw::RepeatedBound {
                    vec![first.clone(), first]
                } else {
                    vec![first, second]
                };
                Part::new(FlawedAlternative::build(
                    flaw,
                    Identifier::new("alt"),
                    bound,
                    tag,
                ))
            })
            .collect()
    }

    /// Return the violation of `samples` as variables.
    ///
    /// # Panics
    ///
    /// Panics if they conform.
    fn find_variable_violation<R: Resolve<Part<dyn Variable>>>(
        samples: &[Part<dyn Variable>],
        resolver: &R,
    ) -> fhy_core::search_space::testing::ConformanceViolation {
        match check_variable_conformance(samples, resolver) {
            Err(violation) => violation,
            Ok(()) => panic!("expected a violation"),
        }
    }

    /// Return the violation of `samples` as alternatives.
    ///
    /// # Panics
    ///
    /// Panics if they conform.
    fn find_alternative_violation<R: Resolve<Part<dyn Alternative>>>(
        samples: &[Part<dyn Alternative>],
        resolver: &R,
    ) -> fhy_core::search_space::testing::ConformanceViolation {
        match check_alternative_conformance(samples, resolver) {
            Err(violation) => violation,
            Ok(()) => panic!("expected a violation"),
        }
    }

    // -- the test implementors conform ------------------------------------

    #[test]
    fn tile_knobs_conform() {
        let param = int_param(&[1, 2, 3]);
        let (i, j) = (Identifier::new("i"), Identifier::new("j"));
        let samples = [
            TileKnob::part(&Identifier::new("a"), param.clone(), &[&i]),
            TileKnob::part(&Identifier::new("b"), param.clone(), &[&i]),
            TileKnob::part(&Identifier::new("c"), param.clone(), &[&j]),
            TileKnob::part(&Identifier::new("d"), param, &[&i, &j]),
        ];

        let result = check_variable_conformance(&samples, &ImplementorResolver);

        assert_eq!(result.map_err(|violation| violation.to_string()), Ok(()));
    }

    #[test]
    fn realizations_conform() {
        let build = |tag: i64, swapped: bool| {
            let (i, j) = (Identifier::new("i"), Identifier::new("j"));
            let order = if swapped { [&j, &i] } else { [&i, &j] };
            Realization::new(&Identifier::new("r"), Vec::new(), &[&i, &j], &order, tag).into_part()
        };
        let samples = [
            build(1, true),
            build(1, true),
            build(1, false),
            build(2, true),
        ];

        let result = check_alternative_conformance(&samples, &ImplementorResolver);

        assert_eq!(result.map_err(|violation| violation.to_string()), Ok(()));
    }

    #[test]
    fn a_flawed_variable_without_its_flaw_conforms() {
        let samples = build_flawed_variables(Flaw::Sound);

        let result = check_variable_conformance(&samples, &FlawedResolver::faithful(Flaw::Sound));

        assert_eq!(result.map_err(|violation| violation.to_string()), Ok(()));
    }

    #[test]
    fn a_flawed_alternative_without_its_flaw_conforms() {
        let samples = build_flawed_alternatives(Flaw::Sound);

        let result =
            check_alternative_conformance(&samples, &FlawedResolver::faithful(Flaw::Sound));

        assert_eq!(result.map_err(|violation| violation.to_string()), Ok(()));
    }

    // -- one breaker per variable clause -----------------------------------

    #[rstest]
    #[case::name_changes(Flaw::UnstableName, ContractClause::StableGetters, FLAWED_VARIABLE)]
    #[case::notes_change(Flaw::UnstableNotes, ContractClause::StableGetters, FLAWED_VARIABLE)]
    #[case::empty_kind(Flaw::EmptyKind, ContractClause::UniqueKind, "")]
    #[case::plain_kind(Flaw::PlainKind, ContractClause::UniqueKind, PlainVariable::KIND)]
    #[case::not_reflexive(Flaw::NonReflexive, ContractClause::EquivalenceHooks, FLAWED_VARIABLE)]
    #[case::not_symmetric(Flaw::Asymmetric, ContractClause::EquivalenceHooks, FLAWED_VARIABLE)]
    #[case::structural_without_alpha(
        Flaw::StructuralWithoutAlpha,
        ContractClause::EquivalenceHooks,
        FLAWED_VARIABLE
    )]
    #[case::failing_hook(Flaw::FailingHooks, ContractClause::EquivalenceHooks, FLAWED_VARIABLE)]
    #[case::wrong_wire_type(Flaw::WrongWireType, ContractClause::WireForm, FLAWED_VARIABLE)]
    fn a_variable_breaking_one_clause_is_reported_for_that_clause(
        #[case] flaw: Flaw,
        #[case] clause: ContractClause,
        #[case] kind: &str,
    ) {
        let samples = build_flawed_variables(flaw);

        let violation = find_variable_violation(&samples, &FlawedResolver::faithful(flaw));

        assert_eq!(violation.clause(), clause);
        assert_eq!(violation.kind(), kind);
    }

    #[test]
    fn a_variable_answering_a_kind_per_sample_breaks_the_unique_kind_clause() {
        let samples = build_flawed_variables(Flaw::KindPerTag);

        let violation =
            find_variable_violation(&samples, &FlawedResolver::faithful(Flaw::KindPerTag));

        assert_eq!(violation.clause(), ContractClause::UniqueKind);
        assert!(
            violation.kind().starts_with(FLAWED_VARIABLE),
            "the kind {:?} names the flawed type",
            violation.kind()
        );
    }

    #[test]
    fn two_variable_types_sharing_a_kind_break_the_unique_kind_clause() {
        let mut samples = vec![
            TileKnob::part(&Identifier::new("a"), int_param(&[1]), &[]),
            TileKnob::part(&Identifier::new("b"), int_param(&[1]), &[]),
        ];
        samples.extend(build_flawed_variables(Flaw::BorrowedKind));

        let violation = find_variable_violation(&samples, &ImplementorResolver);

        assert_eq!(violation.clause(), ContractClause::UniqueKind);
        assert_eq!(violation.kind(), TILE_KNOB);
    }

    #[test]
    fn a_variable_without_a_wire_form_breaks_the_wire_form_clause() {
        let samples: Vec<Part<dyn Variable>> = ["a", "b"]
            .into_iter()
            .map(|name| {
                Part::new(UnserializableVariable {
                    name: Identifier::new(name),
                    param: int_param(&[1]),
                })
            })
            .collect();

        let violation = find_variable_violation(&samples, &FlawedResolver::faithful(Flaw::Sound));

        assert_eq!(violation.clause(), ContractClause::WireForm);
        assert_eq!(violation.kind(), UNSERIALIZABLE);
    }

    #[test]
    fn a_resolver_rebuilding_a_variable_that_is_not_equivalent_breaks_the_wire_form_clause() {
        let samples = build_flawed_variables(Flaw::Sound);
        let lossy = FlawedResolver {
            flaw: Flaw::Sound,
            tag_shift: 1,
        };

        let violation = find_variable_violation(&samples, &lossy);

        assert_eq!(violation.clause(), ContractClause::WireForm);
        assert_eq!(violation.kind(), FLAWED_VARIABLE);
    }

    // -- one breaker per alternative clause --------------------------------

    #[rstest]
    #[case::name_changes(Flaw::UnstableName, ContractClause::StableGetters, FLAWED_ALTERNATIVE)]
    #[case::notes_change(Flaw::UnstableNotes, ContractClause::StableGetters, FLAWED_ALTERNATIVE)]
    #[case::empty_kind(Flaw::EmptyKind, ContractClause::UniqueKind, "")]
    #[case::plain_kind(Flaw::PlainKind, ContractClause::UniqueKind, PlainAlternative::KIND)]
    #[case::repeated_bound_identifiers(
        Flaw::RepeatedBound,
        ContractClause::DistinctBoundIdentifiers,
        FLAWED_ALTERNATIVE
    )]
    #[case::not_reflexive(
        Flaw::NonReflexive,
        ContractClause::EquivalenceHooks,
        FLAWED_ALTERNATIVE
    )]
    #[case::not_symmetric(Flaw::Asymmetric, ContractClause::EquivalenceHooks, FLAWED_ALTERNATIVE)]
    #[case::structural_without_alpha(
        Flaw::StructuralWithoutAlpha,
        ContractClause::EquivalenceHooks,
        FLAWED_ALTERNATIVE
    )]
    #[case::failing_hook(
        Flaw::FailingHooks,
        ContractClause::EquivalenceHooks,
        FLAWED_ALTERNATIVE
    )]
    #[case::wrong_wire_type(Flaw::WrongWireType, ContractClause::WireForm, FLAWED_ALTERNATIVE)]
    fn an_alternative_breaking_one_clause_is_reported_for_that_clause(
        #[case] flaw: Flaw,
        #[case] clause: ContractClause,
        #[case] kind: &str,
    ) {
        let samples = build_flawed_alternatives(flaw);

        let violation = find_alternative_violation(&samples, &FlawedResolver::faithful(flaw));

        assert_eq!(violation.clause(), clause);
        assert_eq!(violation.kind(), kind);
    }

    #[test]
    fn two_alternative_types_sharing_a_kind_break_the_unique_kind_clause() {
        let (i, j) = (Identifier::new("i"), Identifier::new("j"));
        let mut samples: Vec<Part<dyn Alternative>> = vec![
            Realization::new(&Identifier::new("r"), Vec::new(), &[&i], &[&i], 1).into_part(),
            Realization::new(&Identifier::new("q"), Vec::new(), &[&j], &[&j], 1).into_part(),
        ];
        samples.extend(build_flawed_alternatives(Flaw::BorrowedKind));

        let violation = find_alternative_violation(&samples, &ImplementorResolver);

        assert_eq!(violation.clause(), ContractClause::UniqueKind);
        assert_eq!(violation.kind(), REALIZATION);
    }

    #[test]
    fn an_alternative_without_a_wire_form_breaks_the_wire_form_clause() {
        let samples: Vec<Part<dyn Alternative>> = ["a", "b"]
            .into_iter()
            .map(|name| {
                Part::new(UnserializableAlternative {
                    name: Identifier::new(name),
                })
            })
            .collect();

        let violation =
            find_alternative_violation(&samples, &FlawedResolver::faithful(Flaw::Sound));

        assert_eq!(violation.clause(), ContractClause::WireForm);
        assert_eq!(violation.kind(), UNSERIALIZABLE);
    }

    #[test]
    fn a_resolver_rebuilding_an_alternative_that_is_not_equivalent_breaks_the_wire_form_clause() {
        let samples = build_flawed_alternatives(Flaw::Sound);
        let lossy = FlawedResolver {
            flaw: Flaw::Sound,
            tag_shift: 1,
        };

        let violation = find_alternative_violation(&samples, &lossy);

        assert_eq!(violation.clause(), ContractClause::WireForm);
        assert_eq!(violation.kind(), FLAWED_ALTERNATIVE);
    }

    // -- the text of a violation --------------------------------------------

    #[rstest]
    #[case::stable_getters(Flaw::UnstableName, 1)]
    #[case::unique_kind(Flaw::PlainKind, 2)]
    #[case::equivalence_hooks(Flaw::NonReflexive, 4)]
    #[case::wire_form(Flaw::WrongWireType, 6)]
    fn a_violation_names_its_kind_and_the_clause_number_it_breaks(
        #[case] flaw: Flaw,
        #[case] number: u8,
    ) {
        let samples = build_flawed_variables(flaw);

        let violation = find_variable_violation(&samples, &FlawedResolver::faithful(flaw));

        let expected = format!(
            "the implementation of kind `{}` breaks clause {number}: ",
            violation.kind()
        );
        assert!(
            violation.to_string().starts_with(&expected),
            "the text {:?} starts with {expected:?}",
            violation.to_string()
        );
    }

    #[test]
    fn a_violation_of_the_distinct_bound_identifiers_names_clause_3() {
        let samples = build_flawed_alternatives(Flaw::RepeatedBound);

        let violation =
            find_alternative_violation(&samples, &FlawedResolver::faithful(Flaw::RepeatedBound));

        let expected =
            format!("the implementation of kind `{FLAWED_ALTERNATIVE}` breaks clause 3: ");
        assert!(
            violation.to_string().starts_with(&expected),
            "the text {:?} starts with {expected:?}",
            violation.to_string()
        );
    }
}
