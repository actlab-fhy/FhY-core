//! Tests for the serde form of variables, alternatives, choices, spaces and
//! configurations (`fhy_core::search_space::wire`).

use fhy_core::constraint::{ConstraintError, Value};
use fhy_core::diagnostic::Note;
use fhy_core::foreign::{BuildError, Foreign, ForeignError, NoForeign, Part};
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    AssignmentError, CategoricalDomain, IntegerDomain, Param, ParamContext, ParamDomain,
    ParamEvent, ParamObserver, Sign, ZeroInclusion,
};
use fhy_core::search_space::wire::{AlternativeData, ConfigurationData, SpaceData, VariableData};
use fhy_core::search_space::{
    Choice, Condition, Configuration, ConfigurationError, ConfigurationErrors, ConfigurationKey,
    PlainAlternative, PlainVariable, Space, SpaceError, Variable,
};
use fhy_core::solver::Solver;
use rstest::rstest;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::json;

use crate::support::constraint::int;
use crate::support::param::{at_least, at_most, in_set, ints};
use crate::support::search_space::{
    ImplementorResolver, REALIZATION, Realization, TILE_KNOB, TileKnob, bare_alternative,
    categorical, choice_of, chooses, chosen, condition, configure, forbidden, ground_solver,
    int_param, int_variable, natural_param, plain_alternative, plain_variable, space_of,
};
use crate::support::serde::{check_serde_round_trip, restored};

// -- helpers ----------------------------------------------------------------

/// Return the JSON text of the identifier `name` at `id`.
fn build_id_text(id: u64, name: &str) -> String {
    format!(r#"{{"id":{id},"name_hint":"{name}"}}"#)
}

/// Return the categorical param over the integers 1 and 2 whose variable
/// is `variable`.
fn build_categorical_param(variable: Identifier) -> Param {
    let solver = Solver::new();
    Param::new(
        ParamDomain::from(CategoricalDomain::new(ints([1, 2])).expect("the categories are valid")),
        variable,
        Vec::new(),
        &ParamContext::new(&solver),
    )
    .expect("the param is valid")
}

/// Return the JSON text of [`categorical_param`] over the variable `id`.
fn build_param_text(id: u64, name: &str) -> String {
    format!(
        r#"{{"domain":{{"categorical":{{"categories":[{{"int":"1"}},{{"int":"2"}}]}}}},"variable":{},"constraint_system":{{"constraints":[]}}}}"#,
        build_id_text(id, name)
    )
}

/// Return the plain variable `name` at `id` over [`categorical_param`] of
/// the variable `param` at `param_id`.
fn build_plain_variable(id: u64, name: &str, param_id: u64, param: &str) -> PlainVariable {
    PlainVariable::new(
        restored(id, name),
        build_categorical_param(restored(param_id, param)),
    )
}

/// Return the JSON text of [`plain`].
fn build_plain_text(id: u64, name: &str, param_id: u64, param: &str) -> String {
    format!(
        r#"{{"identifier":{},"param":{},"notes":[]}}"#,
        build_id_text(id, name),
        build_param_text(param_id, param)
    )
}

/// Return [`plain`] as a part.
fn build_variable_part(id: u64, name: &str, param_id: u64, param: &str) -> Part<dyn Variable> {
    Part::new(build_plain_variable(id, name, param_id, param))
}

/// Return the JSON text of [`plain`] as a variable in a container.
fn build_tagged_variable_text(id: u64, name: &str, param_id: u64, param: &str) -> String {
    format!(
        r#"{{"plain":{}}}"#,
        build_plain_text(id, name, param_id, param)
    )
}

/// Assert that `value` writes `text` and that `text` reads back as `value`.
fn assert_pinned<T>(value: &T, text: &str)
where
    T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
{
    assert_eq!(serde_json::to_string(value).expect("encodes"), text);
    let decoded: T = serde_json::from_str(text).expect("decodes");
    assert_eq!(&decoded, value);
}

/// Return the note every noted part of [`Rich`] carries.
fn build_notes() -> Vec<Note> {
    vec![Note::with_other_kind("a note")]
}

/// A space with nested choices, conditions, a forbidden clause and notes,
/// with the names its configurations use.
struct Rich {
    space: Space,
    x: Identifier,
    y: Identifier,
    z: Identifier,
    choice: Identifier,
    a1: Identifier,
    av: Identifier,
    sub: Identifier,
    s1: Identifier,
    sv: Identifier,
}

/// Return the [`Rich`] space.
fn build_rich() -> Rich {
    let [x, y, z, choice, a1, a2, av, sub, s1, s2, sv] = [
        "x", "y", "z", "ch", "a1", "a2", "av", "sub", "s1", "s2", "sv",
    ]
    .map(Identifier::new);
    let noted_variable =
        Part::new(PlainVariable::new(av.clone(), int_param(&[1, 2])).with_notes(build_notes()));
    let sub_choice = choice_of(
        &sub,
        vec![
            plain_alternative(&s1, vec![int_variable(&sv, &[1, 2])], Vec::new()),
            bare_alternative(&s2),
        ],
    );
    let first = Part::new(
        PlainAlternative::new(a1.clone(), vec![noted_variable], vec![sub_choice])
            .expect("the names are distinct")
            .with_notes(build_notes()),
    );
    let top_choice =
        choice_of(&choice, vec![first, bare_alternative(&a2)]).with_notes(build_notes());
    let space = Space::new(
        Identifier::new("rich"),
        vec![
            plain_variable(&x, natural_param()),
            int_variable(&y, &[1, 2]),
            int_variable(&z, &[1, 2]),
        ],
        vec![top_choice],
        vec![
            condition(&y, [at_least(&x, 1)]),
            condition(&z, [chooses(&choice, &[&a1])]),
        ],
        vec![forbidden([chooses(&choice, &[&a2]), in_set(&y, ints([2]))])],
    )
    .expect("the space is valid")
    .with_notes(build_notes());
    Rich {
        space,
        x,
        y,
        z,
        choice,
        a1,
        av,
        sub,
        s1,
        sv,
    }
}

impl Rich {
    /// Return the configuration assigning every active decision.
    fn complete(&self) -> Configuration {
        configure(
            &self.space,
            [
                (self.sv.clone(), int(2)),
                (self.sub.clone(), chosen(&self.s1)),
                (self.av.clone(), int(1)),
                (self.choice.clone(), chosen(&self.a1)),
                (self.z.clone(), int(2)),
                (self.y.clone(), int(1)),
                (self.x.clone(), int(3)),
            ],
        )
    }

    /// Return the configuration assigning only `x`.
    fn partial(&self) -> Configuration {
        configure(&self.space, [(self.x.clone(), int(3))])
    }
}

/// The space of the pinned configuration, with the names its texts use: a
/// top-level variable `t`, and a choice `c` of one alternative `a` holding
/// a variable `v`.
struct Pinned {
    space: Space,
    t: Identifier,
    c: Identifier,
    a: Identifier,
    v: Identifier,
}

/// Return the [`Pinned`] space.
fn build_pinned() -> Pinned {
    let t = restored(63_341, "t");
    let c = restored(63_343, "c");
    let a = restored(63_344, "a");
    let v = restored(63_345, "v");
    let alternative = plain_alternative(
        &a,
        vec![build_variable_part(63_345, "v", 63_346, "p")],
        Vec::new(),
    );
    let space = space_of(
        &restored(63_340, "space"),
        vec![build_variable_part(63_341, "t", 63_342, "p")],
        vec![choice_of(&c, vec![alternative])],
    );
    Pinned { space, t, c, a, v }
}

/// Return the JSON text of [`pinned`]'s space.
fn build_pinned_space_text() -> String {
    format!(
        r#"{{"identifier":{},"variables":[{}],"choices":[{{"identifier":{},"alternatives":[{{"plain":{{"identifier":{},"variables":[{}],"choices":[],"notes":[]}}}}],"notes":[]}}],"conditions":[],"forbidden":[],"notes":[]}}"#,
        build_id_text(63_340, "space"),
        build_tagged_variable_text(63_341, "t", 63_342, "p"),
        build_id_text(63_343, "c"),
        build_id_text(63_344, "a"),
        build_tagged_variable_text(63_345, "v", 63_346, "p"),
    )
}

/// Return the names `x`, `y` and `z`.
fn build_numeric_names() -> [Identifier; 3] {
    ["x", "y", "z"].map(Identifier::new)
}

/// Return the space of three variables `names` over the non-negative
/// integers, with `conditions`.
fn build_numeric_space(names: &[Identifier; 3], conditions: Vec<Condition>) -> Space {
    Space::new(
        Identifier::new("numeric"),
        names
            .iter()
            .map(|name| plain_variable(name, natural_param()))
            .collect(),
        Vec::new(),
        conditions,
        Vec::new(),
    )
    .expect("the space is valid")
}

/// Return `value` with the field `extra` added to the object at the JSON
/// pointer `pointer`.
fn add_extra_field(mut value: serde_json::Value, pointer: &str) -> serde_json::Value {
    value
        .pointer_mut(pointer)
        .expect("the pointer names a value")
        .as_object_mut()
        .expect("the pointer names an object")
        .insert("extra".to_owned(), json!(1));
    value
}

/// Return the two-variable space with a condition and a forbidden clause
/// the pinned text of the space shows.
fn build_conditioned_space() -> Space {
    let x = restored(63_331, "x");
    let y = restored(63_333, "y");
    Space::new(
        restored(63_330, "space"),
        vec![
            build_variable_part(63_331, "x", 63_332, "p"),
            build_variable_part(63_333, "y", 63_334, "p"),
        ],
        Vec::new(),
        vec![condition(&y, [in_set(&x, ints([1]))])],
        vec![forbidden([in_set(&y, ints([1]))])],
    )
    .expect("the space is valid")
}

// -- pinned texts -----------------------------------------------------------

#[test]
fn plain_variable_serializes_as_its_fields() {
    assert_pinned(
        &build_plain_variable(63_300, "v", 63_301, "p"),
        &build_plain_text(63_300, "v", 63_301, "p"),
    );
}

#[test]
fn plain_variable_serializes_its_notes() {
    let variable = build_plain_variable(63_300, "v", 63_301, "p").with_notes(build_notes());
    let note = r#"{"message":"a note","kind":{"name":{"id":3,"name_hint":"other"},"description":"Uncategorized note."}}"#;

    assert_pinned(
        &variable,
        &build_plain_text(63_300, "v", 63_301, "p")
            .replace(r#""notes":[]"#, &format!(r#""notes":[{note}]"#)),
    );
}

#[test]
fn plain_alternative_serializes_as_its_fields() {
    let alternative = PlainAlternative::new(
        restored(63_310, "alt"),
        vec![build_variable_part(63_311, "av", 63_312, "p")],
        Vec::new(),
    )
    .expect("the names are distinct");
    let text = format!(
        r#"{{"identifier":{},"variables":[{}],"choices":[],"notes":[]}}"#,
        build_id_text(63_310, "alt"),
        build_tagged_variable_text(63_311, "av", 63_312, "p"),
    );

    assert_pinned(&alternative, &text);
}

#[test]
fn choice_serializes_as_its_fields() {
    let alternative = PlainAlternative::new(
        restored(63_321, "a1"),
        vec![build_variable_part(63_322, "av", 63_323, "p")],
        Vec::new(),
    )
    .expect("the names are distinct");
    let choice = Choice::new(restored(63_320, "c"), vec![Part::new(alternative)])
        .expect("the choice is valid");
    let text = format!(
        r#"{{"identifier":{},"alternatives":[{{"plain":{{"identifier":{},"variables":[{}],"choices":[],"notes":[]}}}}],"notes":[]}}"#,
        build_id_text(63_320, "c"),
        build_id_text(63_321, "a1"),
        build_tagged_variable_text(63_322, "av", 63_323, "p"),
    );

    assert_pinned(&choice, &text);
}

#[test]
fn space_serializes_with_its_condition_and_its_forbidden_clause() {
    let text = format!(
        r#"{{"identifier":{},"variables":[{},{}],"choices":[],"conditions":[{{"target":{},"when":{{"constraints":[{{"in_set":{{"variable":{},"values":[{{"int":"1"}}]}}}}]}}}}],"forbidden":[{{"when":{{"constraints":[{{"in_set":{{"variable":{},"values":[{{"int":"1"}}]}}}}]}}}}],"notes":[]}}"#,
        build_id_text(63_330, "space"),
        build_tagged_variable_text(63_331, "x", 63_332, "p"),
        build_tagged_variable_text(63_333, "y", 63_334, "p"),
        build_id_text(63_333, "y"),
        build_id_text(63_331, "x"),
        build_id_text(63_333, "y"),
    );

    assert_pinned(&build_conditioned_space(), &text);
}

#[test]
fn configuration_serializes_its_space_and_its_entries_in_canonical_order() {
    let pinned = build_pinned();
    let configuration = configure(
        &pinned.space,
        [
            (pinned.v.clone(), int(2)),
            (pinned.c.clone(), chosen(&pinned.a)),
            (pinned.t.clone(), int(1)),
        ],
    );
    let text = format!(
        r#"{{"space":{},"entries":[{{"name":{},"value":{{"int":"1"}}}},{{"name":{},"value":{{"identifier":{}}}}},{{"name":{},"value":{{"int":"2"}}}}]}}"#,
        build_pinned_space_text(),
        build_id_text(63_341, "t"),
        build_id_text(63_343, "c"),
        build_id_text(63_344, "a"),
        build_id_text(63_345, "v"),
    );

    assert_pinned(&configuration, &text);
}

#[test]
fn partial_configuration_serializes_only_its_entries() {
    let pinned = build_pinned();
    let configuration = configure(&pinned.space, [(pinned.t.clone(), int(2))]);
    let text = format!(
        r#"{{"space":{},"entries":[{{"name":{},"value":{{"int":"2"}}}}]}}"#,
        build_pinned_space_text(),
        build_id_text(63_341, "t"),
    );

    assert_pinned(&configuration, &text);
}

#[test]
fn a_payload_listing_entries_in_another_order_decodes_to_the_same_configuration() {
    let pinned = build_pinned();
    let configuration = configure(
        &pinned.space,
        [
            (pinned.t.clone(), int(1)),
            (pinned.c.clone(), chosen(&pinned.a)),
            (pinned.v.clone(), int(2)),
        ],
    );
    let mut value = serde_json::to_value(&configuration).expect("encodes");
    value["entries"]
        .as_array_mut()
        .expect("the entries are a list")
        .reverse();

    let decoded: Configuration = serde_json::from_value(value).expect("decodes");

    assert_eq!(decoded, configuration);
}

// -- merged conditions and canonical order ----------------------------------

#[test]
fn two_conditions_on_one_target_serialize_as_one_condition_holding_both() {
    let names = build_numeric_names();
    let [x, y, _] = &names;
    let space = build_numeric_space(
        &names,
        vec![
            condition(y, [at_least(x, 1)]),
            condition(y, [at_most(x, 5)]),
        ],
    );

    let value = serde_json::to_value(&space).expect("encodes");

    let conditions = value["conditions"].as_array().expect("a list");
    assert_eq!(conditions.len(), 1, "{value}");
    assert_eq!(
        conditions[0]["target"],
        serde_json::to_value(y).expect("encodes")
    );
    let when = conditions[0]["when"]["constraints"]
        .as_array()
        .expect("a list");
    assert_eq!(when.len(), 2, "{value}");
    for constraint in [at_least(x, 1), at_most(x, 5)] {
        let expected = serde_json::to_value(&constraint).expect("encodes");
        assert!(when.contains(&expected), "{expected} is not in {when:?}");
    }
}

#[test]
fn conditions_serialize_in_canonical_order_of_their_targets() {
    let names = build_numeric_names();
    let [x, y, z] = &names;
    let space = build_numeric_space(
        &names,
        vec![
            condition(z, [at_least(x, 1)]),
            condition(y, [at_least(x, 1)]),
        ],
    );

    let value = serde_json::to_value(&space).expect("encodes");

    let targets: Vec<&serde_json::Value> = value["conditions"]
        .as_array()
        .expect("a list")
        .iter()
        .map(|condition| &condition["target"])
        .collect();
    assert_eq!(
        targets,
        [
            &serde_json::to_value(y).expect("encodes"),
            &serde_json::to_value(z).expect("encodes")
        ]
    );
}

// -- round trips ------------------------------------------------------------

#[test]
fn plain_variable_round_trips() {
    let variable =
        PlainVariable::new(Identifier::new("v"), int_param(&[1, 2, 3])).with_notes(build_notes());

    check_serde_round_trip(&variable).expect("round trips");
}

#[test]
fn plain_alternative_round_trips() {
    let sub = choice_of(
        &Identifier::new("sub"),
        vec![bare_alternative(&Identifier::new("s"))],
    );
    let alternative = PlainAlternative::new(
        Identifier::new("alt"),
        vec![int_variable(&Identifier::new("v"), &[1, 2])],
        vec![sub],
    )
    .expect("the names are distinct")
    .with_notes(build_notes());

    check_serde_round_trip(&alternative).expect("round trips");
}

#[test]
fn choice_round_trips() {
    let choice = choice_of(
        &Identifier::new("c"),
        vec![
            plain_alternative(
                &Identifier::new("a"),
                vec![int_variable(&Identifier::new("v"), &[1, 2])],
                Vec::new(),
            ),
            bare_alternative(&Identifier::new("b")),
        ],
    )
    .with_notes(build_notes());

    check_serde_round_trip(&choice).expect("round trips");
}

#[test]
fn space_with_nested_choices_conditions_forbidden_clauses_and_notes_round_trips() {
    check_serde_round_trip(&build_rich().space).expect("round trips");
}

#[test]
fn partial_configuration_round_trips() {
    check_serde_round_trip(&build_rich().partial()).expect("round trips");
}

#[test]
fn complete_configuration_round_trips() {
    let configuration = build_rich().complete();

    assert!(configuration.is_complete());
    check_serde_round_trip(&configuration).expect("round trips");
}

// -- foreign parts ----------------------------------------------------------

/// Return a space holding a tile knob and a realization, and a
/// configuration of it.
fn build_foreign_space() -> (Space, Configuration) {
    let [knob, index, choice, realization, axis, inner] =
        ["knob", "index", "c", "r", "axis", "inner"].map(Identifier::new);
    let space = space_of(
        &Identifier::new("foreign"),
        vec![TileKnob::part(&knob, int_param(&[1, 2]), &[&index])],
        vec![choice_of(
            &choice,
            vec![
                Realization::new(
                    &realization,
                    vec![int_variable(&inner, &[1, 2])],
                    &[&axis],
                    &[&axis],
                    7,
                )
                .into_part(),
                bare_alternative(&Identifier::new("plain")),
            ],
        )],
    );
    let configuration = configure(
        &space,
        [
            (inner, int(2)),
            (choice, chosen(&realization)),
            (knob, int(1)),
        ],
    );
    (space, configuration)
}

#[test]
fn a_foreign_variable_has_a_foreign_wire_form_and_a_plain_one_has_none() {
    let knob = TileKnob::part(&Identifier::new("knob"), int_param(&[1]), &[]);
    let plain = int_variable(&Identifier::new("v"), &[1]);

    let foreign = VariableData::of(&knob).expect("has a wire form");
    let none = VariableData::of(&plain).expect("has a wire form");

    assert_eq!(foreign.foreign().map(Foreign::type_id), Some(TILE_KNOB));
    assert!(none.foreign().is_none(), "{none:?}");
}

#[test]
fn a_foreign_alternative_has_a_foreign_wire_form_and_a_plain_one_has_none() {
    let realization = Realization::new(&Identifier::new("r"), Vec::new(), &[], &[], 1).into_part();
    let plain = bare_alternative(&Identifier::new("a"));

    let foreign = AlternativeData::of(&realization).expect("has a wire form");
    let none = AlternativeData::of(&plain).expect("has a wire form");

    assert_eq!(foreign.foreign().map(Foreign::type_id), Some(REALIZATION));
    assert!(none.foreign().is_none(), "{none:?}");
}

#[test]
fn a_space_with_foreign_parts_builds_back_through_a_resolver() {
    let (space, _) = build_foreign_space();
    let solver = Solver::new();

    let built = SpaceData::of(&space)
        .expect("has a wire form")
        .build(&ImplementorResolver, &ParamContext::new(&solver))
        .expect("builds");

    assert_eq!(built, space);
    assert!(built.is_structurally_equivalent(&space).expect("compares"));
}

#[test]
fn a_configuration_with_foreign_parts_builds_back_through_a_resolver() {
    let (_, configuration) = build_foreign_space();
    let solver = ground_solver();

    let built = ConfigurationData::of(&configuration)
        .expect("has a wire form")
        .build(&ImplementorResolver, &ParamContext::new(&solver))
        .expect("builds");

    assert_eq!(built, configuration);
    assert!(
        built
            .is_structurally_equivalent(&configuration)
            .expect("compares")
    );
}

#[test]
fn a_space_with_foreign_parts_refuses_a_resolver_that_does_not_know_them() {
    let (space, _) = build_foreign_space();
    let solver = Solver::new();

    let result = SpaceData::of(&space)
        .expect("has a wire form")
        .build(&NoForeign, &ParamContext::new(&solver));

    let Err(BuildError::Foreign(ForeignError::Unresolved { type_id })) = result else {
        panic!("a foreign part is refused: {result:?}");
    };
    assert_eq!(type_id, TILE_KNOB);
}

#[test]
fn a_configuration_with_foreign_parts_refuses_a_resolver_that_does_not_know_them() {
    let (_, configuration) = build_foreign_space();
    let solver = ground_solver();

    let result = ConfigurationData::of(&configuration)
        .expect("has a wire form")
        .build(&NoForeign, &ParamContext::new(&solver));

    let Err(BuildError::Foreign(ForeignError::Unresolved { type_id })) = result else {
        panic!("a foreign part is refused: {result:?}");
    };
    assert_eq!(type_id, TILE_KNOB);
}

#[test]
fn a_space_deserializes_no_foreign_part() {
    let (space, _) = build_foreign_space();
    let text = serde_json::to_string(&space).expect("encodes");

    let error = serde_json::from_str::<Space>(&text).expect_err("a foreign part is refused");

    assert!(error.to_string().contains("`test.tile_knob`"), "{error}");
}

#[test]
fn a_configuration_deserializes_no_foreign_part() {
    let (_, configuration) = build_foreign_space();
    let text = serde_json::to_string(&configuration).expect("encodes");

    let error =
        serde_json::from_str::<Configuration>(&text).expect_err("a foreign part is refused");

    assert!(error.to_string().contains("`test.tile_knob`"), "{error}");
}

// -- decoding goes through constructors -------------------------------------

/// Return the JSON of a space whose second variable repeats the name of
/// its first, and that name.
fn build_space_with_a_repeated_name() -> (serde_json::Value, Identifier) {
    let space = Space::new(
        restored(63_330, "space"),
        vec![
            build_variable_part(63_331, "x", 63_332, "p"),
            build_variable_part(63_333, "y", 63_334, "p"),
        ],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
    .expect("the space is valid");
    let mut value = serde_json::to_value(&space).expect("encodes");
    let first = value["variables"][0]["plain"]["identifier"].clone();
    value["variables"][1]["plain"]["identifier"] = first;
    (value, restored(63_331, "x"))
}

#[test]
fn a_space_payload_with_a_repeated_name_fails_to_decode() {
    let (value, _) = build_space_with_a_repeated_name();

    let error = serde_json::from_value::<Space>(value).expect_err("a repeated name is refused");

    assert!(
        error.to_string().contains("is used more than once"),
        "{error}"
    );
}

#[test]
fn a_space_payload_with_a_repeated_name_builds_to_a_duplicate_name_error() {
    let (value, x) = build_space_with_a_repeated_name();
    let data: SpaceData = serde_json::from_value(value).expect("the shape reads");
    let solver = Solver::new();

    let result = data.build(&NoForeign, &ParamContext::new(&solver));

    let Err(BuildError::Invalid(error)) = result else {
        panic!("a repeated name is refused: {result:?}");
    };
    let Some(SpaceError::DuplicateName { name }) = error.downcast_ref::<SpaceError>() else {
        panic!("a duplicate name: {error:?}");
    };
    assert_eq!(name, &x);
}

/// Return the JSON of a configuration of [`pinned`]'s space whose value
/// for `t` is 9, outside its domain.
fn build_configuration_outside_the_domain() -> serde_json::Value {
    let pinned = build_pinned();
    let configuration = configure(&pinned.space, [(pinned.t.clone(), int(1))]);
    let mut value = serde_json::to_value(&configuration).expect("encodes");
    value["entries"][0]["value"] = json!({"int": "9"});
    value
}

#[test]
fn a_configuration_payload_with_a_value_outside_the_domain_fails_to_decode() {
    let value = build_configuration_outside_the_domain();

    serde_json::from_value::<Configuration>(value).expect_err("the value is refused");
}

#[test]
fn a_configuration_payload_with_a_value_outside_the_domain_builds_to_an_assignment_problem() {
    let data: ConfigurationData =
        serde_json::from_value(build_configuration_outside_the_domain()).expect("the shape reads");
    let solver = ground_solver();

    let result = data.build(&NoForeign, &ParamContext::new(&solver));

    let Err(BuildError::Invalid(error)) = result else {
        panic!("a value outside the domain is refused: {result:?}");
    };
    let Some(errors) = error.downcast_ref::<ConfigurationErrors>() else {
        panic!("configuration problems: {error:?}");
    };
    let [ConfigurationError::Assignment { variable, .. }] = errors.errors() else {
        panic!("one assignment problem: {errors:?}");
    };
    assert_eq!(variable, &restored(63_341, "t"));
}

#[rstest]
#[case::variable("/variables/0/plain")]
#[case::space("")]
#[case::condition("/conditions/0")]
#[case::forbidden("/forbidden/0")]
fn a_space_payload_refuses_an_unknown_field(#[case] pointer: &str) {
    let value = add_extra_field(
        serde_json::to_value(build_conditioned_space()).expect("encodes"),
        pointer,
    );

    let error =
        serde_json::from_value::<SpaceData>(value).expect_err("an unknown field is refused");

    assert!(
        error.to_string().contains("unknown field `extra`"),
        "{error}"
    );
}

#[rstest]
#[case::configuration("")]
#[case::entry("/entries/0")]
#[case::space("/space")]
fn a_configuration_payload_refuses_an_unknown_field(#[case] pointer: &str) {
    let pinned = build_pinned();
    let configuration = configure(&pinned.space, [(pinned.t.clone(), int(1))]);
    let value = add_extra_field(
        serde_json::to_value(&configuration).expect("encodes"),
        pointer,
    );

    let error = serde_json::from_value::<ConfigurationData>(value)
        .expect_err("an unknown field is refused");

    assert!(
        error.to_string().contains("unknown field `extra`"),
        "{error}"
    );
}

#[test]
fn a_plain_variable_payload_refuses_an_unknown_field() {
    let value = add_extra_field(
        serde_json::to_value(build_plain_variable(63_300, "v", 63_301, "p")).expect("encodes"),
        "",
    );

    let error =
        serde_json::from_value::<PlainVariable>(value).expect_err("an unknown field is refused");

    assert!(
        error.to_string().contains("unknown field `extra`"),
        "{error}"
    );
}

// -- restore semantics ------------------------------------------------------

/// A param observer that counts a solver's refusal as undecided.
struct SolveIsUndecided;

impl ParamObserver for SolveIsUndecided {
    fn notify(&self, _event: &ParamEvent<'_>) {}

    fn is_undecidable(&self, error: &ConstraintError) -> bool {
        matches!(error, ConstraintError::Solve(_))
    }
}

/// Return the space of one variable over the non-negative integers
/// constrained by `p <= 10`, and the variable's name.
fn build_bounded_space() -> (Space, Identifier) {
    let name = Identifier::new("n");
    let variable = Identifier::new("p");
    let solver = ground_solver();
    let param = Param::new(
        ParamDomain::from(IntegerDomain::new(
            Sign::NonNegative,
            ZeroInclusion::Included,
        )),
        variable.clone(),
        vec![at_most(&variable, 10)],
        &ParamContext::new(&solver),
    )
    .expect("the param is valid");
    let space = space_of(
        &Identifier::new("bounded"),
        vec![plain_variable(&name, param)],
        Vec::new(),
    );
    (space, name)
}

#[test]
fn building_a_configuration_refuses_a_value_its_params_equation_leaves_undecided() {
    let (space, name) = build_bounded_space();
    let solver = Solver::new();
    let observer = SolveIsUndecided;
    let context = ParamContext::new(&solver).with_observer(&observer);

    let result = Configuration::new(&space, [(name.clone(), int(3))], &context);

    let Err(errors) = result else {
        panic!("an undecided equation is refused: {result:?}");
    };
    let [
        ConfigurationError::Assignment {
            variable,
            error: AssignmentError::UnverifiedConstraint { .. },
        },
    ] = errors.errors()
    else {
        panic!("one unverified constraint: {errors:?}");
    };
    assert_eq!(variable, &name);
}

#[test]
fn restoring_a_configuration_accepts_a_value_its_params_equation_leaves_undecided() {
    let (space, name) = build_bounded_space();
    let configuration = configure(&space, [(name, int(3))]);
    let data = ConfigurationData::of(&configuration).expect("has a wire form");
    let solver = Solver::new();
    let observer = SolveIsUndecided;
    let context = ParamContext::new(&solver).with_observer(&observer);

    let restored = data
        .build(&ImplementorResolver, &context)
        .expect("a restored value is not verified again");

    assert_eq!(restored, configuration);
}

// -- the ground simplifier of a configuration's own Deserialize -------------

/// Return a space whose variable `y` is active while `x >= 2`, and the
/// names `x` and `y`.
fn build_gated_space() -> (Space, Identifier, Identifier) {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let space = Space::new(
        Identifier::new("gated"),
        vec![
            plain_variable(&x, natural_param()),
            int_variable(&y, &[1, 2]),
        ],
        Vec::new(),
        vec![condition(&y, [at_least(&x, 2)])],
        Vec::new(),
    )
    .expect("the space is valid");
    (space, x, y)
}

#[test]
fn a_configuration_deserializes_with_a_condition_its_equation_decides() {
    let (space, x, y) = build_gated_space();
    let configuration = configure(&space, [(x, int(3)), (y, int(1))]);
    let text = serde_json::to_string(&configuration).expect("encodes");

    let decoded: Configuration = serde_json::from_str(&text).expect("decodes");

    assert_eq!(decoded, configuration);
}

#[test]
fn a_configuration_payload_giving_a_value_to_a_decision_its_condition_leaves_inactive_fails_to_decode()
 {
    let (space, x, y) = build_gated_space();
    let configuration = configure(&space, [(x, int(3)), (y, int(1))]);
    let mut value = serde_json::to_value(&configuration).expect("encodes");
    value["entries"][0]["value"] = json!({"int": "1"});

    serde_json::from_value::<Configuration>(value).expect_err("an inactive decision is refused");
}

// -- configuration keys -----------------------------------------------------

/// A space whose variable `m` takes the name of the alternative `a`, which
/// the space binds, or the free name `f`: its names are `s`, `m`, `c`, `a`.
struct Mirrored {
    space: Space,
    m: Identifier,
    c: Identifier,
    a: Identifier,
    f: Identifier,
}

/// Return a [`Mirrored`] space, its names fresh.
fn build_mirrored() -> Mirrored {
    let [m, c, a, f] = ["m", "c", "a", "f"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("s"),
        vec![plain_variable(
            &m,
            categorical(vec![
                Value::Identifier(a.clone()),
                Value::Identifier(f.clone()),
            ]),
        )],
        vec![choice_of(&c, vec![bare_alternative(&a)])],
        Vec::new(),
        Vec::new(),
    )
    .expect("the space is valid");
    Mirrored { space, m, c, a, f }
}

#[test]
fn configuration_key_serializes_its_entries_in_canonical_order() {
    let pinned = build_pinned();
    let configuration = configure(
        &pinned.space,
        [
            (pinned.t.clone(), int(1)),
            (pinned.c.clone(), chosen(&pinned.a)),
        ],
    );

    assert_pinned(
        &configuration.key(),
        r#"{"entries":[{"value":{"leaf":{"int":"1"}}},{"alternative":{"index":0}},{"unassigned":{}}]}"#,
    );
}

#[test]
fn configuration_key_writes_an_inactive_decision() {
    let (space, x, _) = build_gated_space();
    let configuration = configure(&space, [(x, int(0))]);

    assert_pinned(
        &configuration.key(),
        r#"{"entries":[{"value":{"leaf":{"int":"0"}}},{"inactive":{}}]}"#,
    );
}

#[test]
fn configuration_key_writes_a_bound_identifier_as_its_position() {
    let mirrored = build_mirrored();
    let configuration = configure(&mirrored.space, [(mirrored.m.clone(), chosen(&mirrored.a))]);

    assert_pinned(
        &configuration.key(),
        r#"{"entries":[{"value":{"bound":{"position":3}}},{"unassigned":{}}]}"#,
    );
}

#[test]
fn configuration_key_writes_a_free_identifier_as_itself() {
    let mirrored = build_mirrored();
    let configuration = configure(&mirrored.space, [(mirrored.m.clone(), chosen(&mirrored.f))]);
    let text = format!(
        r#"{{"entries":[{{"value":{{"leaf":{{"identifier":{}}}}}}},{{"unassigned":{{}}}}]}}"#,
        build_id_text(mirrored.f.id(), mirrored.f.name_hint()),
    );

    assert_pinned(&configuration.key(), &text);
}

#[rstest]
#[case::bound(true)]
#[case::free(false)]
fn configuration_key_round_trips_through_json_and_postcard(#[case] bound: bool) {
    let mirrored = build_mirrored();
    let value = if bound { &mirrored.a } else { &mirrored.f };
    let configuration = configure(
        &mirrored.space,
        [
            (mirrored.m.clone(), chosen(value)),
            (mirrored.c.clone(), chosen(&mirrored.a)),
        ],
    );

    check_serde_round_trip(&configuration.key()).expect("the key round-trips");
}

#[test]
fn keys_of_relabeled_configurations_stay_equal_after_a_round_trip() {
    let [left, right] = [build_mirrored(), build_mirrored()];
    let left_key = configure(&left.space, [(left.m.clone(), chosen(&left.a))]).key();
    let right_key = configure(&right.space, [(right.m.clone(), chosen(&right.a))]).key();

    let decoded: ConfigurationKey =
        serde_json::from_str(&serde_json::to_string(&left_key).expect("encodes")).expect("decodes");

    assert_eq!(decoded, right_key);
}

#[test]
fn configuration_key_refuses_an_unknown_entry() {
    serde_json::from_str::<ConfigurationKey>(r#"{"entries":[{"chosen":{}}]}"#)
        .expect_err("an unknown entry is refused");
}
