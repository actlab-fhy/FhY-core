//! Tests for editing a `Space` and completing a configuration:
//! `Space::with_decisions`, `Space::without_decisions`,
//! `SpaceError::NotTopLevelDecision` and `Space::complete`.

use fhy_core::constraint::Value;
use fhy_core::diagnostic::Note;
use fhy_core::foreign::Part;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{
    Choice, Condition, Configuration, Forbidden, RandomOracle, Space, SpaceError, TraceError,
    TraceStep, Variable,
};
use rstest::rstest;

use crate::support::constraint::int;
use crate::support::param::in_set;
use crate::support::search::{
    OracleRefusal, RefusingOracle, ScriptedOracle, build_tiling_space, index, with_context,
};
use crate::support::search_space::{
    bare_alternative, choice_of, chooses, chosen, condition, configure, forbidden, int_param,
    int_variable, plain_alternative, plain_variable, space_of,
};
use crate::support::serde::restored;

// -- helpers ----------------------------------------------------------------

/// The names of the table space: top-level variables `v1`, `v2`; the choice
/// `c1` with the alternative `a1` holding `x1`, and the bare alternative
/// `a2`; the choice `c2` with the alternative `b1` holding `y1`. Its
/// canonical order is `v1, v2, c1, x1, c2, y1`.
struct Table {
    name: Identifier,
    v1: Identifier,
    v2: Identifier,
    c1: Identifier,
    a1: Identifier,
    a2: Identifier,
    x1: Identifier,
    c2: Identifier,
    b1: Identifier,
    y1: Identifier,
    ghost: Identifier,
}

impl Table {
    /// Return fresh names.
    fn new() -> Self {
        let [name, v1, v2, c1, a1, a2, x1, c2, b1, y1, ghost] = [
            "table", "v1", "v2", "c1", "a1", "a2", "x1", "c2", "b1", "y1", "ghost",
        ]
        .map(Identifier::new);
        Self {
            name,
            v1,
            v2,
            c1,
            a1,
            a2,
            x1,
            c2,
            b1,
            y1,
            ghost,
        }
    }

    /// Return the name labelled `label`.
    ///
    /// # Panics
    ///
    /// Panics on a label the table does not have.
    fn named(&self, label: &str) -> &Identifier {
        match label {
            "v1" => &self.v1,
            "v2" => &self.v2,
            "c1" => &self.c1,
            "x1" => &self.x1,
            "c2" => &self.c2,
            "y1" => &self.y1,
            "ghost" => &self.ghost,
            _ => panic!("the table has no name {label:?}"),
        }
    }

    /// Return the name labelled by each of `labels`.
    fn named_all(&self, labels: &[&str]) -> Vec<Identifier> {
        labels
            .iter()
            .map(|label| self.named(label).clone())
            .collect()
    }

    /// Return the choice `c1`, its alternative `a1` holding `x1` and
    /// `extra`.
    fn build_c1_holding(&self, extra: Vec<Part<dyn Variable>>) -> Choice {
        let mut variables = vec![int_variable(&self.x1, &[1, 2])];
        variables.extend(extra);
        choice_of(
            &self.c1,
            vec![
                plain_alternative(&self.a1, variables, Vec::new()),
                bare_alternative(&self.a2),
            ],
        )
    }

    /// Return the choice `c2` with the alternative `b1` holding `y1`.
    fn build_c2(&self) -> Choice {
        choice_of(
            &self.c2,
            vec![plain_alternative(
                &self.b1,
                vec![int_variable(&self.y1, &[1, 2])],
                Vec::new(),
            )],
        )
    }

    /// Return the condition that `y1` is active while `x1` is 1.
    fn build_y1_condition(&self) -> Condition {
        condition(&self.y1, [in_set(&self.x1, [int(1)])])
    }

    /// Return the table space with `conditions` and `forbidden_clauses`.
    ///
    /// # Panics
    ///
    /// Panics if the space is refused.
    fn build_space(&self, conditions: Vec<Condition>, forbidden_clauses: Vec<Forbidden>) -> Space {
        Space::new(
            self.name.clone(),
            vec![
                int_variable(&self.v1, &[1, 2]),
                int_variable(&self.v2, &[1, 2]),
            ],
            vec![self.build_c1_holding(Vec::new()), self.build_c2()],
            conditions,
            forbidden_clauses,
        )
        .expect("the table space is valid")
    }

    /// Return the table space with a condition on `y1`, a clause naming
    /// `v1` and `c2`, and two notes.
    fn build_base(&self) -> Space {
        self.build_space(
            vec![self.build_y1_condition()],
            vec![forbidden([
                in_set(&self.v1, [int(2)]),
                chooses(&self.c2, &[&self.b1]),
            ])],
        )
        .with_notes(vec![
            Note::with_other_kind("first"),
            Note::with_other_kind("second"),
        ])
    }

    /// Return the entries giving every decision of the table space a
    /// value.
    fn build_full_entries(&self) -> Vec<(Identifier, Value)> {
        vec![
            (self.v1.clone(), int(1)),
            (self.v2.clone(), int(1)),
            (self.c1.clone(), chosen(&self.a1)),
            (self.x1.clone(), int(1)),
            (self.c2.clone(), chosen(&self.b1)),
            (self.y1.clone(), int(1)),
        ]
    }
}

/// Return the names of `space`'s top-level variables.
fn list_variable_names(space: &Space) -> Vec<Identifier> {
    space
        .variables()
        .iter()
        .map(|variable| variable.get().name().clone())
        .collect()
}

/// Return the names of `space`'s top-level choices.
fn list_choice_names(space: &Space) -> Vec<Identifier> {
    space.choices().iter().map(|c| c.name().clone()).collect()
}

/// Return the names of `space`'s decisions, in canonical order.
fn list_decision_names(space: &Space) -> Vec<Identifier> {
    space
        .decisions()
        .map(|decision| decision.name().clone())
        .collect()
}

/// Return each decision `entries` assigns with the canonical position the
/// trace of the configuration of `space` they build gives its step, in the
/// trace's order.
///
/// # Panics
///
/// Panics if the configuration or its trace cannot be built.
fn record_positions(space: &Space, entries: Vec<(Identifier, Value)>) -> Vec<(Identifier, usize)> {
    let configuration = configure(space, entries);
    let trace = with_context(|context| configuration.trace(context)).expect("the trace is built");
    trace
        .steps()
        .iter()
        .map(|step| {
            (
                step.subject().clone(),
                step.decision().expect("a static step has a position"),
            )
        })
        .collect()
}

// -- with_decisions: where the decisions go ----------------------------------

/// Test `with_decisions` puts a variable in the slot of the top-level
/// variable of its name, with the other variables and the order of the
/// decisions unchanged.
#[rstest]
#[case::first(0)]
#[case::last(1)]
fn with_decisions_replaces_a_top_level_variable_in_its_slot(#[case] slot: usize) {
    let table = Table::new();
    let space = table.build_base();
    let replaced = [&table.v1, &table.v2][slot];
    let replacement = plain_variable(replaced, int_param(&[1, 2, 3]));

    let edited = space
        .with_decisions(vec![replacement.clone()], Vec::new())
        .expect("the edit is valid");

    assert_eq!(edited.variables()[slot], replacement);
    assert_eq!(
        list_variable_names(&edited),
        vec![table.v1.clone(), table.v2.clone()]
    );
    assert_eq!(edited.decision_order(), space.decision_order());
}

/// Test `with_decisions` puts a choice in the slot of the top-level choice
/// of its name, with the other choices and the order of the decisions
/// unchanged.
#[rstest]
#[case::first(0)]
#[case::last(1)]
fn with_decisions_replaces_a_top_level_choice_in_its_slot(#[case] slot: usize) {
    let table = Table::new();
    let space = table.build_base();
    let replacement = [
        table.build_c1_holding(Vec::new()),
        choice_of(
            &table.c2,
            vec![plain_alternative(
                &table.b1,
                vec![int_variable(&table.y1, &[1, 2, 3])],
                Vec::new(),
            )],
        ),
    ][slot]
        .clone();

    let edited = space
        .with_decisions(Vec::new(), vec![replacement.clone()])
        .expect("the edit is valid");

    assert_eq!(edited.choices()[slot], replacement);
    assert_eq!(
        list_choice_names(&edited),
        vec![table.c1.clone(), table.c2.clone()]
    );
    assert_eq!(edited.decision_order(), space.decision_order());
}

/// Test `with_decisions` appends a variable of a new name after the last
/// top-level variable.
#[test]
fn with_decisions_appends_a_new_variable_after_the_last_variable() {
    let table = Table::new();
    let space = table.build_base();
    let v3 = Identifier::new("v3");

    let edited = space
        .with_decisions(vec![int_variable(&v3, &[1, 2])], Vec::new())
        .expect("the edit is valid");

    assert_eq!(
        list_variable_names(&edited),
        vec![table.v1.clone(), table.v2.clone(), v3]
    );
    assert_eq!(
        list_choice_names(&edited),
        vec![table.c1.clone(), table.c2.clone()]
    );
}

/// Test `with_decisions` appends choices of new names after the last
/// top-level choice, in the order given.
#[test]
fn with_decisions_appends_new_choices_after_the_last_choice_in_the_order_given() {
    let table = Table::new();
    let space = table.build_base();
    let [c3, d1, c4, d2] = ["c3", "d1", "c4", "d2"].map(Identifier::new);
    let third = choice_of(&c3, vec![bare_alternative(&d1)]);
    let fourth = choice_of(&c4, vec![bare_alternative(&d2)]);

    let edited = space
        .with_decisions(Vec::new(), vec![third, fourth])
        .expect("the edit is valid");

    assert_eq!(
        list_choice_names(&edited),
        vec![table.c1.clone(), table.c2.clone(), c3, c4]
    );
    assert_eq!(
        list_variable_names(&edited),
        vec![table.v1.clone(), table.v2.clone()]
    );
}

/// Test `with_decisions` does a replacement and an append of one call
/// together.
#[test]
fn with_decisions_replaces_and_appends_in_one_call() {
    let table = Table::new();
    let space = table.build_base();
    let replacement = plain_variable(&table.v1, int_param(&[1, 2, 3]));
    let [c3, d1] = ["c3", "d1"].map(Identifier::new);

    let edited = space
        .with_decisions(
            vec![replacement.clone()],
            vec![choice_of(&c3, vec![bare_alternative(&d1)])],
        )
        .expect("the edit is valid");

    assert_eq!(edited.variables()[0], replacement);
    assert_eq!(
        list_variable_names(&edited),
        vec![table.v1.clone(), table.v2.clone()]
    );
    assert_eq!(
        list_choice_names(&edited),
        vec![table.c1.clone(), table.c2.clone(), c3]
    );
}

/// Test `with_decisions` keeps the space's name, conditions, forbidden
/// clauses and notes.
#[test]
fn with_decisions_keeps_the_name_conditions_clauses_and_notes() {
    let table = Table::new();
    let space = table.build_base();
    let replacement = plain_variable(&table.v2, int_param(&[1, 2, 3]));

    let edited = space
        .with_decisions(vec![replacement], Vec::new())
        .expect("the edit is valid");

    assert_eq!(edited.name(), space.name());
    assert_eq!(edited.conditions(), space.conditions());
    assert_eq!(edited.forbidden(), space.forbidden());
    assert_eq!(edited.notes(), space.notes());
}

// -- with_decisions: errors -------------------------------------------------

/// Test `with_decisions` refuses a variable named like a top-level choice
/// or like a decision nested in one, as `Space::new` does.
#[rstest]
#[case::top_level_choice("c1")]
#[case::nested_variable("x1")]
fn with_decisions_refuses_a_variable_named_like_another_decision(#[case] label: &str) {
    let table = Table::new();
    let space = table.build_base();
    let clash = table.named(label);

    let result = space.with_decisions(vec![int_variable(clash, &[1, 2])], Vec::new());

    let Err(SpaceError::DuplicateName { name }) = result else {
        panic!("expected DuplicateName, got {result:?}");
    };
    assert_eq!(&name, clash);
}

/// Test `with_decisions` refuses a replacement choice that no longer holds
/// a decision a condition names, with `UnknownReference`.
#[test]
fn with_decisions_refuses_a_condition_naming_a_decision_the_replacement_lacks() {
    let table = Table::new();
    let space = table.build_base();
    let x2 = Identifier::new("x2");
    let replacement = choice_of(
        &table.c1,
        vec![
            plain_alternative(&table.a1, vec![int_variable(&x2, &[1, 2])], Vec::new()),
            bare_alternative(&table.a2),
        ],
    );

    let result = space.with_decisions(Vec::new(), vec![replacement]);

    let Err(SpaceError::UnknownReference { name }) = result else {
        panic!("expected UnknownReference, got {result:?}");
    };
    assert_eq!(name, table.x1);
}

// -- with_decisions: trace positions ----------------------------------------

/// Test appending choices keeps every existing position: the trace over
/// the old decisions is a prefix of the trace over the edited space.
#[test]
fn with_decisions_keeps_every_trace_position_when_choices_are_appended() {
    let table = Table::new();
    let space = table.build_base();
    let [c3, d1, z1] = ["c3", "d1", "z1"].map(Identifier::new);
    let appended = choice_of(
        &c3,
        vec![plain_alternative(
            &d1,
            vec![int_variable(&z1, &[1, 2])],
            Vec::new(),
        )],
    );
    let edited = space
        .with_decisions(Vec::new(), vec![appended])
        .expect("the edit is valid");
    let mut edited_entries = table.build_full_entries();
    edited_entries.push((c3.clone(), chosen(&d1)));
    edited_entries.push((z1.clone(), int(1)));

    let before = record_positions(&space, table.build_full_entries());
    let after = record_positions(&edited, edited_entries);

    let expected_before: Vec<(Identifier, usize)> = ["v1", "v2", "c1", "x1", "c2", "y1"]
        .iter()
        .enumerate()
        .map(|(position, label)| (table.named(label).clone(), position))
        .collect();
    assert_eq!(before, expected_before);
    assert_eq!(&after[..before.len()], &before[..]);
    assert_eq!(&after[before.len()..], &[(c3, 6), (z1, 7)]);
}

/// Test appending a variable moves every choice's subtree by one and keeps
/// the positions of the variables before it.
#[test]
fn with_decisions_moves_the_choice_subtrees_by_one_when_a_variable_is_appended() {
    let table = Table::new();
    let space = table.build_base();
    let v3 = Identifier::new("v3");
    let edited = space
        .with_decisions(vec![int_variable(&v3, &[1, 2])], Vec::new())
        .expect("the edit is valid");
    let mut entries = table.build_full_entries();
    entries.push((v3.clone(), int(1)));

    let after = record_positions(&edited, entries);

    let expected = vec![
        (table.v1.clone(), 0),
        (table.v2.clone(), 1),
        (v3, 2),
        (table.c1.clone(), 3),
        (table.x1.clone(), 4),
        (table.c2.clone(), 5),
        (table.y1.clone(), 6),
    ];
    assert_eq!(after, expected);
}

/// Test replacing a choice by one whose subtree holds another number of
/// decisions keeps the positions before it and moves the positions after
/// it.
#[test]
fn with_decisions_moves_the_positions_after_a_replaced_choice_of_another_size() {
    let table = Table::new();
    let space = table.build_base();
    let x1b = Identifier::new("x1b");
    let replacement = table.build_c1_holding(vec![int_variable(&x1b, &[1, 2])]);
    let edited = space
        .with_decisions(Vec::new(), vec![replacement])
        .expect("the edit is valid");
    let mut entries = table.build_full_entries();
    entries.insert(4, (x1b.clone(), int(2)));

    let after = record_positions(&edited, entries);

    let expected = vec![
        (table.v1.clone(), 0),
        (table.v2.clone(), 1),
        (table.c1.clone(), 2),
        (table.x1.clone(), 3),
        (x1b, 4),
        (table.c2.clone(), 5),
        (table.y1.clone(), 6),
    ];
    assert_eq!(after, expected);
}

// -- without_decisions ------------------------------------------------------

/// Test `without_decisions` removes a top-level variable.
#[test]
fn without_decisions_removes_a_top_level_variable() {
    let table = Table::new();
    let space = table.build_space(Vec::new(), Vec::new());

    let edited = space
        .without_decisions([table.v2.clone()])
        .expect("the edit is valid");

    assert_eq!(list_variable_names(&edited), vec![table.v1.clone()]);
    assert_eq!(
        list_decision_names(&edited),
        table.named_all(&["v1", "c1", "x1", "c2", "y1"])
    );
}

/// Test `without_decisions` removes a top-level choice with every decision
/// under it.
#[test]
fn without_decisions_removes_a_choice_with_its_subtree() {
    let table = Table::new();
    let space = table.build_space(Vec::new(), Vec::new());

    let edited = space
        .without_decisions([table.c1.clone()])
        .expect("the edit is valid");

    assert_eq!(list_choice_names(&edited), vec![table.c2.clone()]);
    assert_eq!(
        list_decision_names(&edited),
        table.named_all(&["v1", "v2", "c2", "y1"])
    );
}

/// Test `without_decisions` drops the conditions that target a removed
/// decision or a decision under it, and keeps the others.
#[rstest]
#[case::the_removed_choice("c2")]
#[case::a_decision_under_it("y1")]
fn without_decisions_drops_the_conditions_targeting_the_removed_subtree(#[case] target: &str) {
    let table = Table::new();
    let kept = condition(&table.x1, [in_set(&table.v1, [int(1)])]);
    let dropped = condition(table.named(target), [in_set(&table.v2, [int(1)])]);
    let space = table.build_space(vec![kept.clone(), dropped], Vec::new());

    let edited = space
        .without_decisions([table.c2.clone()])
        .expect("the edit is valid");

    assert_eq!(edited.conditions(), &[kept]);
}

/// Test `without_decisions` refuses a condition that names a removed
/// decision, or one under a removed choice, and does not prune it.
#[rstest]
#[case::a_removed_variable("v2", "v2")]
#[case::a_decision_under_a_removed_choice("c1", "x1")]
fn without_decisions_refuses_a_condition_naming_a_removed_decision(
    #[case] removed: &str,
    #[case] named: &str,
) {
    let table = Table::new();
    let space = table.build_space(
        vec![condition(&table.y1, [in_set(table.named(named), [int(1)])])],
        Vec::new(),
    );

    let result = space.without_decisions([table.named(removed).clone()]);

    let Err(SpaceError::UnknownReference { name }) = result else {
        panic!("expected UnknownReference, got {result:?}");
    };
    assert_eq!(&name, table.named(named));
}

/// Test `without_decisions` refuses a forbidden clause that names a removed
/// decision, or one under a removed choice, and does not prune it.
#[rstest]
#[case::a_removed_variable("v2", "v2")]
#[case::a_decision_under_a_removed_choice("c1", "x1")]
fn without_decisions_refuses_a_forbidden_clause_naming_a_removed_decision(
    #[case] removed: &str,
    #[case] named: &str,
) {
    let table = Table::new();
    let space = table.build_space(
        Vec::new(),
        vec![forbidden([in_set(table.named(named), [int(1)])])],
    );

    let result = space.without_decisions([table.named(removed).clone()]);

    let Err(SpaceError::UnknownReference { name }) = result else {
        panic!("expected UnknownReference, got {result:?}");
    };
    assert_eq!(&name, table.named(named));
}

/// Test `without_decisions` removes a name given twice once.
#[test]
fn without_decisions_removes_a_name_given_twice_once() {
    let table = Table::new();
    let space = table.build_space(Vec::new(), Vec::new());

    let twice = space
        .without_decisions([table.v2.clone(), table.v2.clone()])
        .expect("the edit is valid");

    let once = space
        .without_decisions([table.v2.clone()])
        .expect("the edit is valid");
    assert_eq!(twice, once);
}

/// Test `without_decisions` of no names gives a space equal to the
/// original.
#[test]
fn without_decisions_of_no_names_gives_an_equal_space() {
    let table = Table::new();
    let space = table.build_base();

    let edited = space
        .without_decisions(Vec::new())
        .expect("the edit is valid");

    assert_eq!(edited, space);
}

/// Test `without_decisions` refuses a name that is no top-level decision,
/// nested or unknown, and names the first such name in the order given.
#[rstest]
#[case::nested(&["x1"], "x1")]
#[case::unknown(&["ghost"], "ghost")]
#[case::unknown_after_a_top_level_name(&["v2", "ghost", "x1"], "ghost")]
#[case::nested_before_an_unknown_name(&["x1", "ghost"], "x1")]
fn without_decisions_refuses_the_first_name_that_is_not_top_level(
    #[case] labels: &[&str],
    #[case] offending: &str,
) {
    let table = Table::new();
    let space = table.build_space(Vec::new(), Vec::new());

    let result = space.without_decisions(table.named_all(labels));

    let Err(SpaceError::NotTopLevelDecision { name }) = result else {
        panic!("expected NotTopLevelDecision, got {result:?}");
    };
    assert_eq!(&name, table.named(offending));
}

/// Test `without_decisions` reports a name that is no top-level decision
/// before any error of rebuilding the space.
#[test]
fn without_decisions_reports_a_bad_name_before_a_rebuild_error() {
    let table = Table::new();
    let space = table.build_space(Vec::new(), vec![forbidden([in_set(&table.v2, [int(1)])])]);

    let result = space.without_decisions([table.v2.clone(), table.ghost.clone()]);

    let Err(SpaceError::NotTopLevelDecision { name }) = result else {
        panic!("expected NotTopLevelDecision, got {result:?}");
    };
    assert_eq!(name, table.ghost);
}

/// Test `SpaceError::NotTopLevelDecision` writes the name and says it is
/// not a top-level decision of the space.
#[test]
fn not_top_level_decision_error_writes_the_name_and_the_reason() {
    let error = SpaceError::NotTopLevelDecision {
        name: restored(63_500, "x"),
    };

    let text = error.to_string();

    assert_eq!(text, "x::63500 is not a top-level decision of the space");
}

// -- complete ---------------------------------------------------------------

/// Test `complete` of the empty configuration gives what `sample` gives with
/// an identically seeded random oracle: the configuration and the trace.
#[test]
fn complete_of_the_empty_configuration_equals_sample() {
    let tiling = build_tiling_space();
    let empty = configure(&tiling.space, []);
    let mut completing = RandomOracle::new(7);
    let mut sampling = RandomOracle::new(7);

    let completed = with_context(|context| tiling.space.complete(&empty, &mut completing, context))
        .expect("the space completes");

    let sampled = with_context(|context| tiling.space.sample(&mut sampling, context))
        .expect("the space samples");
    assert_eq!(completed.configuration(), sampled.configuration());
    assert_eq!(completed.trace(), sampled.trace());
}

/// Test `complete` of a configuration that assigns the top variable keeps
/// it, asks the oracle only the choice and the variable under it, and
/// returns a complete configuration whose trace has a step for every
/// assigned decision and replays into it.
#[test]
fn complete_keeps_the_entries_and_asks_only_the_unassigned_active_decisions() {
    let tiling = build_tiling_space();
    let partial = configure(&tiling.space, [(tiling.t.clone(), int(1))]);
    let mut oracle = ScriptedOracle::new([index(0), index(2)]);

    let completed = with_context(|context| tiling.space.complete(&partial, &mut oracle, context))
        .expect("the space completes");

    let configuration = completed.configuration().expect("over a space");
    let asked: Vec<&Identifier> = oracle.seen.iter().map(|step| &step.subject).collect();
    assert_eq!(asked, vec![&tiling.c, &tiling.x]);
    assert_eq!(configuration.value(&tiling.t), Some(&int(1)));
    assert_eq!(configuration.value(&tiling.c), Some(&chosen(&tiling.a)));
    assert_eq!(configuration.value(&tiling.x), Some(&int(3)));
    assert!(configuration.is_complete());
    let subjects: Vec<&Identifier> = completed
        .trace()
        .steps()
        .iter()
        .map(TraceStep::subject)
        .collect();
    assert_eq!(subjects, vec![&tiling.t, &tiling.c, &tiling.x]);
    let replayed = with_context(|context| tiling.space.replay(completed.trace(), context))
        .expect("the trace replays");
    assert_eq!(&replayed, configuration);
}

/// Test `complete` of a configuration that assigns the choice too asks the
/// oracle only the variable left.
#[test]
fn complete_asks_only_the_decisions_a_later_entry_leaves_open() {
    let tiling = build_tiling_space();
    let partial = configure(
        &tiling.space,
        [
            (tiling.t.clone(), int(1)),
            (tiling.c.clone(), chosen(&tiling.a)),
        ],
    );
    let mut oracle = ScriptedOracle::new([index(1)]);

    let completed = with_context(|context| tiling.space.complete(&partial, &mut oracle, context))
        .expect("the space completes");

    let configuration = completed.configuration().expect("over a space");
    let asked: Vec<&Identifier> = oracle.seen.iter().map(|step| &step.subject).collect();
    assert_eq!(asked, vec![&tiling.x]);
    assert_eq!(configuration.value(&tiling.x), Some(&int(2)));
    assert_eq!(completed.trace().len(), 3);
}

/// Test `complete` of a complete configuration never asks the oracle and
/// returns the configuration as it is, with its trace.
#[test]
fn complete_returns_a_complete_configuration_as_it_is() {
    let tiling = build_tiling_space();
    let complete = configure(
        &tiling.space,
        [
            (tiling.t.clone(), int(1)),
            (tiling.c.clone(), chosen(&tiling.b)),
        ],
    );
    let mut oracle = RefusingOracle;

    let completed = with_context(|context| tiling.space.complete(&complete, &mut oracle, context))
        .expect("nothing is asked, so nothing is refused");

    assert_eq!(completed.configuration(), Some(&complete));
    let expected_trace = with_context(|context| complete.trace(context)).expect("the trace");
    assert_eq!(completed.trace(), &expected_trace);
}

/// Test `complete` honours a fixed value when a forbidden clause spans it
/// and an unassigned decision asked earlier: with `a == 1 and b == 1`
/// forbidden and `b` fixed at 1, every run, whatever its oracle's seed,
/// answers `a = 0`, and the trace replays into the completion.
#[test]
fn complete_keeps_an_earlier_decision_clear_of_a_forbidden_clause_with_a_fixed_one() {
    let [name, a, b] = ["pair", "a", "b"].map(Identifier::new);
    let space = Space::new(
        name,
        vec![int_variable(&a, &[0, 1]), int_variable(&b, &[0, 1])],
        Vec::new(),
        Vec::new(),
        vec![forbidden([in_set(&a, [int(1)]), in_set(&b, [int(1)])])],
    )
    .expect("the space is valid");
    let partial = configure(&space, [(b.clone(), int(1))]);

    for seed in 0..64 {
        let mut oracle = RandomOracle::new(seed);

        let completed = with_context(|context| space.complete(&partial, &mut oracle, context))
            .unwrap_or_else(|error| panic!("seed {seed} does not complete: {error}"));

        let configuration = completed.configuration().expect("over a space");
        assert_eq!(configuration.value(&a), Some(&int(0)), "seed {seed}");
        assert_eq!(configuration.value(&b), Some(&int(1)), "seed {seed}");
        let replayed = with_context(|context| space.replay(completed.trace(), context))
            .expect("the trace replays");
        assert_eq!(&replayed, configuration, "seed {seed}");
    }
}

/// Test `complete` of a configuration of another space is refused with
/// `OtherSpace` and does not ask the oracle.
#[test]
fn complete_refuses_a_configuration_of_another_space() {
    let tiling = build_tiling_space();
    let elsewhere = Identifier::new("elsewhere");
    let other = space_of(
        &Identifier::new("other"),
        vec![int_variable(&elsewhere, &[1, 2])],
        Vec::new(),
    );
    let foreign = configure(&other, []);
    let mut oracle = ScriptedOracle::new([index(0)]);

    let result = with_context(|context| tiling.space.complete(&foreign, &mut oracle, context));

    let Err(TraceError::OtherSpace) = result else {
        panic!("expected OtherSpace, got {result:?}");
    };
    assert!(oracle.seen.is_empty(), "{:?}", oracle.seen);
}

/// Test the oracle's error stops `complete` on a partial configuration, as
/// it stops `sample`: an oracle failure carrying the oracle's error.
#[test]
fn complete_stops_with_the_oracles_error() {
    let tiling = build_tiling_space();
    let partial = configure(&tiling.space, [(tiling.t.clone(), int(1))]);
    let mut completing = RefusingOracle;
    let mut sampling = RefusingOracle;

    let completed =
        with_context(|context| tiling.space.complete(&partial, &mut completing, context));

    let sampled = with_context(|context| tiling.space.sample(&mut sampling, context));
    let Err(TraceError::Oracle { source, .. }) = completed else {
        panic!("expected an oracle failure, got {completed:?}");
    };
    let Err(TraceError::Oracle {
        source: expected, ..
    }) = sampled
    else {
        panic!("expected an oracle failure, got {sampled:?}");
    };
    assert_eq!(source.downcast_ref::<OracleRefusal>(), Some(&OracleRefusal));
    assert_eq!(
        expected.downcast_ref::<OracleRefusal>(),
        Some(&OracleRefusal)
    );
}

/// Test carrying values across an edit: the old entries built over the
/// edited space and completed give a complete configuration holding the old
/// values.
#[test]
fn complete_carries_the_old_values_across_an_edit() {
    let table = Table::new();
    let space = table.build_base();
    let old = configure(&space, table.build_full_entries());
    let v3 = Identifier::new("v3");
    let edited = space
        .with_decisions(vec![int_variable(&v3, &[1, 2, 3])], Vec::new())
        .expect("the edit is valid");
    let carried = with_context(|context| {
        Configuration::new(
            &edited,
            old.entries()
                .map(|(name, value)| (name.clone(), value.clone())),
            context,
        )
    })
    .expect("the old values fit the edited space");
    let mut oracle = RandomOracle::new(3);

    let completed = with_context(|context| edited.complete(&carried, &mut oracle, context))
        .expect("the edited space completes");

    let configuration = completed.configuration().expect("over a space");
    assert!(configuration.is_complete());
    for (name, value) in old.entries() {
        assert_eq!(configuration.value(name), Some(value), "{name:?}");
    }
    let drawn = configuration.value(&v3);
    assert!(
        [int(1), int(2), int(3)]
            .iter()
            .any(|value| drawn == Some(value)),
        "{drawn:?}"
    );
}
