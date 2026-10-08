//! Tests for the resolver registry (`fhy_core::search_space::wire::
//! ResolverRegistry` and `RegistryResolver`): a function per type id and
//! family, the refusal of an unregistered id, the repeated and reserved
//! ids, `merge`, the resolver handed to a variable's function, and two
//! crates' registries composed to decode one space.

use fhy_core::constraint::{CustomConstraint, OpaqueValue};
use fhy_core::diagnostic::Note;
use fhy_core::foreign::{BuildError, Foreign, ForeignError, NoForeign, Part, Resolve};
use fhy_core::identifier::Identifier;
use fhy_core::param::wire::ParamData;
use fhy_core::param::{CustomDomain, Param, ParamContext};
use fhy_core::search_space::wire::{
    ChoiceData, ConfigurationData, ResolverRegistry, SearchSpaceResolver, SpaceData, VariableData,
};
use fhy_core::search_space::{
    Alternative, Configuration, PlainAlternative, PlainVariable, RegistryError, Space, Variable,
};
use fhy_core::solver::Solver;
use rstest::rstest;
use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::support::constraint::int;
use crate::support::foreign::{WireCustom, WireDomain, WireToken};
use crate::support::search_space::{
    REALIZATION, Realization, TILE_KNOB, TileKnob, bare_alternative, choice_of, chosen, configure,
    ground_solver, int_param, int_variable, space_of,
};

// -- helpers ----------------------------------------------------------------

/// The five families of parts a registry holds functions for.
#[derive(Debug, Clone, Copy)]
enum Family {
    Variable,
    Alternative,
    Opaque,
    Constraint,
    Domain,
}

/// The payload of a [`TileKnob`]'s foreign part, its param still wire data.
#[derive(Deserialize)]
struct KnobPayload {
    name: Identifier,
    param: ParamData,
    notes: Vec<Note>,
    index_symbols: Vec<Identifier>,
}

/// The payload of a [`Realization`]'s foreign part, its parts still wire
/// data.
#[derive(Deserialize)]
struct RealizationPayload {
    name: Identifier,
    variables: Vec<VariableData>,
    choices: Vec<ChoiceData>,
    notes: Vec<Note>,
    axes: Vec<Identifier>,
    order: Vec<Identifier>,
    tag: i64,
}

/// Return the failure of a payload `foreign` carries that its function
/// cannot read.
fn failed(
    foreign: &Foreign,
    error: impl std::error::Error + Send + Sync + 'static,
) -> ForeignError {
    ForeignError::Failed {
        type_id: foreign.type_id().to_owned(),
        source: Box::new(error),
    }
}

/// Return the payload of `foreign` read as JSON.
fn read_payload<T: DeserializeOwned>(foreign: &Foreign) -> Result<T, ForeignError> {
    serde_json::from_str(foreign.data()).map_err(|error| failed(foreign, error))
}

/// Return the refusal of a part inside `foreign`: the foreign error it
/// holds, so that an unresolved inner part is reported by its own type id.
fn refuse_inner(foreign: &Foreign, error: BuildError) -> ForeignError {
    match error {
        BuildError::Foreign(inner) => inner,
        other => failed(foreign, other),
    }
}

/// Build the tile knob of `foreign`, its param through `resolver`.
fn build_knob(
    foreign: &Foreign,
    resolver: &dyn SearchSpaceResolver,
    context: &ParamContext<'_>,
) -> Result<TileKnob, ForeignError> {
    let payload: KnobPayload = read_payload(foreign)?;
    let param = payload
        .param
        .build(resolver, context)
        .map_err(|error| refuse_inner(foreign, error))?;
    Ok(TileKnob {
        name: payload.name,
        param,
        notes: payload.notes,
        index_symbols: payload.index_symbols,
    })
}

/// A [`VariableResolverFn`](fhy_core::search_space::wire::VariableResolverFn)
/// reading a [`TileKnob`].
fn resolve_knob(
    foreign: &Foreign,
    resolver: &dyn SearchSpaceResolver,
    context: &ParamContext<'_>,
) -> Result<Part<dyn Variable>, ForeignError> {
    Ok(Part::new(build_knob(foreign, resolver, context)?))
}

/// Return the text that names the solver of `context`, so a function can
/// stamp the context it was handed on what it builds.
fn name_solver(context: &ParamContext<'_>) -> String {
    format!("{:p}", context.solver())
}

/// A variable function like [`resolve_knob`] that adds a note naming the
/// solver of the context it was handed.
fn resolve_stamped_knob(
    foreign: &Foreign,
    resolver: &dyn SearchSpaceResolver,
    context: &ParamContext<'_>,
) -> Result<Part<dyn Variable>, ForeignError> {
    let mut knob = build_knob(foreign, resolver, context)?;
    knob.notes.push(Note::with_other_kind(name_solver(context)));
    Ok(Part::new(knob))
}

/// An [`AlternativeResolverFn`](fhy_core::search_space::wire::AlternativeResolverFn)
/// reading a [`Realization`], its variables and choices through `resolver`.
fn resolve_realization(
    foreign: &Foreign,
    resolver: &dyn SearchSpaceResolver,
    context: &ParamContext<'_>,
) -> Result<Part<dyn Alternative>, ForeignError> {
    let payload: RealizationPayload = read_payload(foreign)?;
    let variables = payload
        .variables
        .into_iter()
        .map(|variable| variable.build(resolver, context))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| refuse_inner(foreign, error))?;
    let choices = payload
        .choices
        .into_iter()
        .map(|choice| choice.build(resolver, context))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| refuse_inner(foreign, error))?;
    Ok(Part::new(Realization {
        name: payload.name,
        variables,
        choices,
        notes: payload.notes,
        axes: payload.axes,
        order: payload.order,
        tag: payload.tag,
    }))
}

/// An opaque-value function reading a [`WireToken`] from its payload's
/// digits, whatever its type id.
fn resolve_token(foreign: &Foreign) -> Result<Part<dyn OpaqueValue>, ForeignError> {
    let payload = foreign
        .data()
        .parse()
        .map_err(|error| failed(foreign, error))?;
    Ok(Part::new(WireToken(payload)))
}

/// A custom-constraint function reading a [`WireCustom`] labelled by its
/// payload, whatever its type id.
#[expect(
    clippy::unnecessary_wraps,
    reason = "the signature is that of a `*ResolverFn`"
)]
fn resolve_custom(foreign: &Foreign) -> Result<Part<dyn CustomConstraint>, ForeignError> {
    Ok(Part::new(WireCustom(foreign.data().to_owned())))
}

/// A custom-domain function reading a [`WireDomain`] labelled by its
/// payload, whatever its type id.
#[expect(
    clippy::unnecessary_wraps,
    reason = "the signature is that of a `*ResolverFn`"
)]
fn resolve_domain(foreign: &Foreign) -> Result<Part<dyn CustomDomain>, ForeignError> {
    Ok(Part::new(WireDomain(foreign.data().to_owned())))
}

/// Return `registry` with `type_id` registered in `family`, to the
/// function of the helpers above that reads that family's sample.
fn register(
    registry: ResolverRegistry,
    family: Family,
    type_id: &str,
) -> Result<ResolverRegistry, RegistryError> {
    match family {
        Family::Variable => registry.with_variable_kind(type_id, resolve_knob),
        Family::Alternative => registry.with_alternative_kind(type_id, resolve_realization),
        Family::Opaque => registry.with_opaque_value(type_id, resolve_token),
        Family::Constraint => registry.with_custom_constraint(type_id, resolve_custom),
        Family::Domain => registry.with_custom_domain(type_id, resolve_domain),
    }
}

/// Return the registry holding each of `entries`.
///
/// # Panics
///
/// Panics if an entry is refused.
fn build_registry(entries: &[(Family, &str)]) -> ResolverRegistry {
    entries
        .iter()
        .fold(ResolverRegistry::new(), |registry, &(family, type_id)| {
            register(registry, family, type_id).expect("the entry is accepted")
        })
}

/// A foreign part of one family, and the text a part built from it shows.
struct Sample {
    foreign: Foreign,
    label: String,
}

/// Return a sample of `family` whose foreign part has the type id
/// `type_id`: what the family's function of [`register`] builds from it
/// shows as `label`.
fn build_sample(family: Family, type_id: &str) -> Sample {
    match family {
        Family::Variable => {
            let knob = TileKnob::part(
                &Identifier::new("knob"),
                int_param(&[1, 2]),
                &[&Identifier::new("index")],
            );
            let data = knob.get().to_foreign().expect("has a wire form");
            Sample {
                foreign: Foreign::new(type_id, data.data()),
                label: format!("{knob:?}"),
            }
        }
        Family::Alternative => {
            let realization = Realization::new(
                &Identifier::new("realization"),
                vec![int_variable(&Identifier::new("inner"), &[1, 2])],
                &[&Identifier::new("axis")],
                &[],
                3,
            )
            .into_part();
            let data = realization.get().to_foreign().expect("has a wire form");
            Sample {
                foreign: Foreign::new(type_id, data.data()),
                label: format!("{realization:?}"),
            }
        }
        Family::Opaque => {
            let part: Part<dyn OpaqueValue> = Part::new(WireToken(5));
            Sample {
                foreign: Foreign::new(type_id, "5"),
                label: format!("{part:?}"),
            }
        }
        Family::Constraint => {
            let part: Part<dyn CustomConstraint> = Part::new(WireCustom("c".to_owned()));
            Sample {
                foreign: Foreign::new(type_id, "c"),
                label: format!("{part:?}"),
            }
        }
        Family::Domain => {
            let part: Part<dyn CustomDomain> = Part::new(WireDomain("d".to_owned()));
            Sample {
                foreign: Foreign::new(type_id, "d"),
                label: format!("{part:?}"),
            }
        }
    }
}

/// Return the text of the part `resolver` builds from `foreign` in
/// `family`.
fn resolve_label<R: SearchSpaceResolver + ?Sized>(
    family: Family,
    resolver: &R,
    foreign: &Foreign,
) -> Result<String, ForeignError> {
    match family {
        Family::Variable => Resolve::<Part<dyn Variable>>::resolve(resolver, foreign)
            .map(|part| format!("{part:?}")),
        Family::Alternative => Resolve::<Part<dyn Alternative>>::resolve(resolver, foreign)
            .map(|part| format!("{part:?}")),
        Family::Opaque => Resolve::<Part<dyn OpaqueValue>>::resolve(resolver, foreign)
            .map(|part| format!("{part:?}")),
        Family::Constraint => Resolve::<Part<dyn CustomConstraint>>::resolve(resolver, foreign)
            .map(|part| format!("{part:?}")),
        Family::Domain => Resolve::<Part<dyn CustomDomain>>::resolve(resolver, foreign)
            .map(|part| format!("{part:?}")),
    }
}

/// Return the type id of the `Unresolved` refusal of `result`.
///
/// # Panics
///
/// Panics if `result` is not that refusal.
fn unresolved_id(result: Result<String, ForeignError>) -> String {
    let Err(ForeignError::Unresolved { type_id }) = result else {
        panic!("expected Unresolved, got {result:?}");
    };
    type_id
}

/// Return the type id of the repeated-id error of `result`.
///
/// # Panics
///
/// Panics if `result` is not that error.
fn repeated_id(result: Result<ResolverRegistry, RegistryError>) -> String {
    let Err(RegistryError::RepeatedTypeId { type_id }) = result else {
        panic!("expected RepeatedTypeId, got {result:?}");
    };
    type_id
}

// -- an empty registry ------------------------------------------------------

/// Test that a registry from `new` refuses a part of every family with
/// `Unresolved` naming its type id, as `NoForeign` does.
#[rstest]
fn new_registry_refuses_a_part_of_every_family(
    #[values(
        Family::Variable,
        Family::Alternative,
        Family::Opaque,
        Family::Constraint,
        Family::Domain
    )]
    family: Family,
) {
    let registry = ResolverRegistry::new();
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let foreign = build_sample(family, "test.anything").foreign;

    let result = resolve_label(family, &registry.resolver(&context), &foreign);

    let expected = resolve_label(family, &NoForeign, &foreign);
    assert_eq!(unresolved_id(result), "test.anything");
    assert_eq!(unresolved_id(expected), "test.anything");
}

/// Test that a registry from `default` refuses a part of every family with
/// `Unresolved` naming its type id, as `NoForeign` does.
#[rstest]
fn default_registry_refuses_a_part_of_every_family(
    #[values(
        Family::Variable,
        Family::Alternative,
        Family::Opaque,
        Family::Constraint,
        Family::Domain
    )]
    family: Family,
) {
    let registry = ResolverRegistry::default();
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let foreign = build_sample(family, "test.anything").foreign;

    let result = resolve_label(family, &registry.resolver(&context), &foreign);

    assert_eq!(unresolved_id(result), "test.anything");
}

// -- one function per family ------------------------------------------------

/// Test that a function registered under a type id builds the part of that
/// id through the registry's resolver, in each family.
#[rstest]
fn registered_function_builds_the_part_of_its_type_id(
    #[values(
        Family::Variable,
        Family::Alternative,
        Family::Opaque,
        Family::Constraint,
        Family::Domain
    )]
    family: Family,
) {
    let registry = build_registry(&[(family, "test.part")]);
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let sample = build_sample(family, "test.part");

    let result = resolve_label(family, &registry.resolver(&context), &sample.foreign);

    assert_eq!(result.expect("the registered id resolves"), sample.label);
}

/// Test that a registry holding one type id in a family still refuses
/// another id of that family with `Unresolved`.
#[rstest]
fn registered_function_leaves_another_type_id_unresolved(
    #[values(
        Family::Variable,
        Family::Alternative,
        Family::Opaque,
        Family::Constraint,
        Family::Domain
    )]
    family: Family,
) {
    let registry = build_registry(&[(family, "test.part")]);
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let foreign = build_sample(family, "test.other").foreign;

    let result = resolve_label(family, &registry.resolver(&context), &foreign);

    assert_eq!(unresolved_id(result), "test.other");
}

/// Test that a type id registered in one family does not resolve a part of
/// another family.
#[test]
fn registered_function_does_not_resolve_a_part_of_another_family() {
    let registry = build_registry(&[(Family::Opaque, "test.part")]);
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let foreign = build_sample(Family::Domain, "test.part").foreign;

    let result = resolve_label(Family::Domain, &registry.resolver(&context), &foreign);

    assert_eq!(unresolved_id(result), "test.part");
}

// -- repeated and reserved ids ----------------------------------------------

/// Test that registering a type id twice in one family reports
/// `RepeatedTypeId` naming it.
#[rstest]
fn registering_a_type_id_twice_in_one_family_is_repeated(
    #[values(
        Family::Variable,
        Family::Alternative,
        Family::Opaque,
        Family::Constraint,
        Family::Domain
    )]
    family: Family,
) {
    let registry = build_registry(&[(family, "test.part")]);

    let result = register(registry, family, "test.part");

    assert_eq!(repeated_id(result), "test.part");
}

/// Test that one type id registered in each of the five families is
/// accepted and resolves in each.
#[rstest]
fn the_same_type_id_in_different_families_is_accepted(
    #[values(
        Family::Variable,
        Family::Alternative,
        Family::Opaque,
        Family::Constraint,
        Family::Domain
    )]
    family: Family,
) {
    let registry = build_registry(&[
        (Family::Variable, "shared"),
        (Family::Alternative, "shared"),
        (Family::Opaque, "shared"),
        (Family::Constraint, "shared"),
        (Family::Domain, "shared"),
    ]);
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let sample = build_sample(family, "shared");

    let result = resolve_label(family, &registry.resolver(&context), &sample.foreign);

    assert_eq!(result.expect("the id resolves"), sample.label);
}

/// Test that the plain variable's own kind is refused as a variable kind
/// with `ReservedTypeId`.
#[test]
fn with_variable_kind_refuses_the_plain_variables_kind() {
    let result = ResolverRegistry::new().with_variable_kind(PlainVariable::KIND, resolve_knob);

    let Err(RegistryError::ReservedTypeId { type_id }) = result else {
        panic!("expected ReservedTypeId, got {result:?}");
    };
    assert_eq!(type_id, PlainVariable::KIND);
}

/// Test that the plain alternative's own kind is refused as an alternative
/// kind with `ReservedTypeId`.
#[test]
fn with_alternative_kind_refuses_the_plain_alternatives_kind() {
    let result =
        ResolverRegistry::new().with_alternative_kind(PlainAlternative::KIND, resolve_realization);

    let Err(RegistryError::ReservedTypeId { type_id }) = result else {
        panic!("expected ReservedTypeId, got {result:?}");
    };
    assert_eq!(type_id, PlainAlternative::KIND);
}

// -- merge ------------------------------------------------------------------

/// Test that disjoint registries merge into one that resolves the part of
/// each side: the left side holds a variable, an opaque value and a
/// domain, the right side a variable, an alternative and a constraint.
#[rstest]
#[case::left_variable(Family::Variable, "left")]
#[case::right_variable(Family::Variable, "right")]
#[case::left_opaque(Family::Opaque, "left")]
#[case::left_domain(Family::Domain, "left")]
#[case::right_alternative(Family::Alternative, "right")]
#[case::right_constraint(Family::Constraint, "right")]
fn merge_of_disjoint_registries_resolves_a_part_of_each_side(
    #[case] family: Family,
    #[case] type_id: &str,
) {
    let left = build_registry(&[
        (Family::Variable, "left"),
        (Family::Opaque, "left"),
        (Family::Domain, "left"),
    ]);
    let right = build_registry(&[
        (Family::Variable, "right"),
        (Family::Alternative, "right"),
        (Family::Constraint, "right"),
    ]);
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let sample = build_sample(family, type_id);

    let merged = left.merge(right).expect("disjoint registries merge");

    let result = resolve_label(family, &merged.resolver(&context), &sample.foreign);
    assert_eq!(result.expect("a held id resolves"), sample.label);
}

/// Test that a merge conflict reports the first id both sides hold, the
/// families in the order variables, alternatives, opaque values, custom
/// constraints, custom domains, then the ids ascending.
#[rstest]
#[case::variables_ascending(
    &[(Family::Variable, "b"), (Family::Variable, "a")],
    &[(Family::Variable, "a"), (Family::Variable, "b")],
    "a"
)]
#[case::variable_before_alternative(
    &[(Family::Variable, "z"), (Family::Alternative, "a")],
    &[(Family::Alternative, "a"), (Family::Variable, "z")],
    "z"
)]
#[case::alternative_before_opaque(
    &[(Family::Opaque, "a"), (Family::Alternative, "z")],
    &[(Family::Alternative, "z"), (Family::Opaque, "a")],
    "z"
)]
#[case::opaque_before_domain(
    &[(Family::Opaque, "o"), (Family::Domain, "d")],
    &[(Family::Domain, "d"), (Family::Opaque, "o")],
    "o"
)]
#[case::opaque_before_constraint(
    &[(Family::Constraint, "a"), (Family::Opaque, "z")],
    &[(Family::Opaque, "z"), (Family::Constraint, "a")],
    "z"
)]
#[case::constraint_before_domain(
    &[(Family::Domain, "a"), (Family::Constraint, "k")],
    &[(Family::Constraint, "k"), (Family::Domain, "a")],
    "k"
)]
#[case::only_the_shared_id_conflicts(
    &[(Family::Variable, "a"), (Family::Variable, "b")],
    &[(Family::Variable, "b"), (Family::Variable, "c")],
    "b"
)]
fn merge_reports_the_first_repeated_id(
    #[case] left: &[(Family, &str)],
    #[case] right: &[(Family, &str)],
    #[case] expected: &str,
) {
    let left = build_registry(left);
    let right = build_registry(right);

    let result = left.merge(right);

    assert_eq!(repeated_id(result), expected);
}

// -- the resolver a variable's function is handed ---------------------------

/// Return the space of one tile knob whose param is over a custom domain,
/// and its name.
fn build_custom_domain_space() -> (Space, Identifier) {
    let name = Identifier::new("knob");
    let solver = Solver::new();
    let param = Param::new(
        WireDomain::build("d"),
        Identifier::new("p"),
        Vec::new(),
        &ParamContext::new(&solver),
    )
    .expect("the param is valid");
    let space = space_of(
        &Identifier::new("custom"),
        vec![TileKnob::part(&name, param, &[])],
        Vec::new(),
    );
    (space, name)
}

/// Test that a variable function is handed the registry's resolver for its
/// param, so a custom domain registered in the same registry decodes.
#[test]
fn variable_function_resolves_its_params_custom_domain_through_the_registry() {
    let (space, _) = build_custom_domain_space();
    let registry = build_registry(&[
        (Family::Variable, TILE_KNOB),
        (Family::Domain, "test.domain"),
    ]);
    let solver = Solver::new();
    let context = ParamContext::new(&solver);

    let built = SpaceData::of(&space)
        .expect("has a wire form")
        .build(&registry.resolver(&context), &context)
        .expect("the domain is registered");

    assert_eq!(built, space);
}

/// Test that a variable whose param's custom domain is not registered
/// fails the build with `Unresolved` naming the domain's type id.
#[test]
fn variable_function_without_the_params_domain_registered_fails_unresolved() {
    let (space, _) = build_custom_domain_space();
    let registry = build_registry(&[(Family::Variable, TILE_KNOB)]);
    let solver = Solver::new();
    let context = ParamContext::new(&solver);

    let result = SpaceData::of(&space)
        .expect("has a wire form")
        .build(&registry.resolver(&context), &context);

    let Err(BuildError::Foreign(ForeignError::Unresolved { type_id })) = result else {
        panic!("expected the domain unresolved, got {result:?}");
    };
    assert_eq!(type_id, "test.domain");
}

/// Test that a variable function is handed the context given to
/// `ResolverRegistry::resolver`: the solver of each context lent to the
/// registry is the one the function stamps on the knob it builds.
#[test]
fn variable_function_is_handed_the_context_the_resolver_was_lent() {
    let registry = ResolverRegistry::new()
        .with_variable_kind("test.stamped", resolve_stamped_knob)
        .expect("the id is free");
    let sample = build_sample(Family::Variable, "test.stamped");
    let [first_solver, second_solver] = [Solver::new(), Solver::new()];
    let first = ParamContext::new(&first_solver);
    let second = ParamContext::new(&second_solver);

    let stamps = [&first, &second].map(|context| {
        let part =
            Resolve::<Part<dyn Variable>>::resolve(&registry.resolver(context), &sample.foreign)
                .expect("the knob resolves");
        let notes: Vec<_> = part
            .get()
            .notes()
            .iter()
            .map(|note| note.message().to_owned())
            .collect();
        notes
    });

    assert_eq!(
        stamps,
        [
            vec![format!("{:p}", &first_solver)],
            vec![format!("{:p}", &second_solver)]
        ]
    );
}

// -- copying the resolver and cloning the registry --------------------------

/// Test that a `RegistryResolver` is `Copy`: the original and its copies
/// each resolve the same part.
#[test]
fn registry_resolver_copies_resolve_the_same_part() {
    let registry = build_registry(&[(Family::Opaque, "test.part")]);
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let sample = build_sample(Family::Opaque, "test.part");
    let resolver = registry.resolver(&context);

    let copy = resolver;
    let another = resolver;

    let labels = [resolver, copy, another]
        .map(|resolver| resolve_label(Family::Opaque, &resolver, &sample.foreign));
    for label in labels {
        assert_eq!(label.expect("the id resolves"), sample.label);
    }
}

/// Test that a clone of a registry resolves the parts of the original,
/// after the original is dropped.
#[rstest]
fn cloned_registry_resolves_the_same_parts_after_the_original_is_dropped(
    #[values(
        Family::Variable,
        Family::Alternative,
        Family::Opaque,
        Family::Constraint,
        Family::Domain
    )]
    family: Family,
) {
    let registry = build_registry(&[(family, "test.part")]);
    let clone = registry.clone();
    drop(registry);
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let sample = build_sample(family, "test.part");

    let result = resolve_label(family, &clone.resolver(&context), &sample.foreign);

    assert_eq!(result.expect("the clone resolves"), sample.label);
}

// -- two crates' registries composed ----------------------------------------

/// The type id of the custom domain the first crate registers.
const CRATE_DOMAIN: &str = "test.domain";

/// Return the first "crate"'s registry: the tile knob and the custom
/// domain its param uses.
fn export_knob_crate() -> ResolverRegistry {
    ResolverRegistry::new()
        .with_variable_kind(TILE_KNOB, resolve_knob)
        .expect("the knob kind is free")
        .with_custom_domain(CRATE_DOMAIN, resolve_domain)
        .expect("the domain is free")
}

/// Return the second "crate"'s registry: the realization.
fn export_realization_crate() -> ResolverRegistry {
    ResolverRegistry::new()
        .with_alternative_kind(REALIZATION, resolve_realization)
        .expect("the realization kind is free")
}

/// Return the param over the custom domain `d`.
fn build_domain_param() -> Param {
    let solver = Solver::new();
    Param::new(
        WireDomain::build("d"),
        Identifier::new("p"),
        Vec::new(),
        &ParamContext::new(&solver),
    )
    .expect("the param is valid")
}

/// Return a space holding a tile knob over a custom domain and a choice
/// whose realization holds a plain variable and another such knob, and a
/// configuration of it.
fn build_composed_space() -> (Space, Configuration) {
    let [top, inner_knob, plain, choice, realization, axis, other] =
        ["top", "inner_knob", "plain", "c", "r", "axis", "other"].map(Identifier::new);
    let space = space_of(
        &Identifier::new("composed"),
        vec![TileKnob::part(&top, build_domain_param(), &[])],
        vec![choice_of(
            &choice,
            vec![
                Realization::new(
                    &realization,
                    vec![
                        int_variable(&plain, &[1, 2]),
                        TileKnob::part(&inner_knob, build_domain_param(), &[&axis]),
                    ],
                    &[&axis],
                    &[&axis],
                    7,
                )
                .into_part(),
                bare_alternative(&other),
            ],
        )],
    );
    let configuration = configure(
        &space,
        [
            (top, int(4)),
            (choice, chosen(&realization)),
            (plain, int(2)),
            (inner_knob, int(8)),
        ],
    );
    (space, configuration)
}

/// Test that the merge of two crates' registries decodes, through JSON, a
/// space holding the kinds of both: the realization holds a knob, so the
/// registry itself is handed to the realization's function.
#[test]
fn merged_crate_registries_decode_a_space_holding_both_crates_kinds() {
    let (space, _) = build_composed_space();
    let registry = export_knob_crate()
        .merge(export_realization_crate())
        .expect("the crates share no id");
    let text =
        serde_json::to_string(&SpaceData::of(&space).expect("has a wire form")).expect("encodes");
    let solver = ground_solver();
    let context = ParamContext::new(&solver);

    let data: SpaceData = serde_json::from_str(&text).expect("decodes");
    let built = data
        .build(&registry.resolver(&context), &context)
        .expect("both crates' kinds resolve");

    assert_eq!(built, space);
}

/// Test that the merge of two crates' registries decodes, through JSON, a
/// configuration of a space holding the kinds of both.
#[test]
fn merged_crate_registries_decode_a_configuration_of_both_crates_kinds() {
    let (_, configuration) = build_composed_space();
    let registry = export_realization_crate()
        .merge(export_knob_crate())
        .expect("the crates share no id");
    let text =
        serde_json::to_string(&ConfigurationData::of(&configuration).expect("has a wire form"))
            .expect("encodes");
    let solver = ground_solver();
    let context = ParamContext::new(&solver);

    let data: ConfigurationData = serde_json::from_str(&text).expect("decodes");
    let built = data
        .build(&registry.resolver(&context), &context)
        .expect("both crates' kinds resolve");

    assert_eq!(built, configuration);
}

/// Test that the first crate's registry alone fails a space with the
/// second crate's kind, `Unresolved` naming the realization.
#[test]
fn knob_crate_alone_fails_on_the_realization() {
    let (space, _) = build_composed_space();
    let registry = export_knob_crate();
    let solver = ground_solver();
    let context = ParamContext::new(&solver);

    let result = SpaceData::of(&space)
        .expect("has a wire form")
        .build(&registry.resolver(&context), &context);

    let Err(BuildError::Foreign(ForeignError::Unresolved { type_id })) = result else {
        panic!("expected the realization unresolved, got {result:?}");
    };
    assert_eq!(type_id, REALIZATION);
}

/// Test that the second crate's registry alone fails a space with the
/// first crate's kind, `Unresolved` naming the tile knob.
#[test]
fn realization_crate_alone_fails_on_the_tile_knob() {
    let (space, _) = build_composed_space();
    let registry = export_realization_crate();
    let solver = ground_solver();
    let context = ParamContext::new(&solver);

    let result = SpaceData::of(&space)
        .expect("has a wire form")
        .build(&registry.resolver(&context), &context);

    let Err(BuildError::Foreign(ForeignError::Unresolved { type_id })) = result else {
        panic!("expected the tile knob unresolved, got {result:?}");
    };
    assert_eq!(type_id, TILE_KNOB);
}

/// Test that a configuration of the composed space fails with only the
/// second crate's registry, `Unresolved` naming the tile knob.
#[test]
fn realization_crate_alone_fails_a_configuration_on_the_tile_knob() {
    let (_, configuration) = build_composed_space();
    let registry = export_realization_crate();
    let solver = ground_solver();
    let context = ParamContext::new(&solver);

    let result = ConfigurationData::of(&configuration)
        .expect("has a wire form")
        .build(&registry.resolver(&context), &context);

    let Err(BuildError::Foreign(ForeignError::Unresolved { type_id })) = result else {
        panic!("expected the tile knob unresolved, got {result:?}");
    };
    assert_eq!(type_id, TILE_KNOB);
}

// -- the error text ---------------------------------------------------------

/// Test that the two registry errors read as pinned, one line each. A pin
/// of a new type's text: it holds on the stub, where `Display` is already
/// implemented.
#[rstest]
#[case::repeated(
    RegistryError::RepeatedTypeId { type_id: "x".to_owned() },
    r#"the type id "x" is registered twice"#
)]
#[case::reserved(
    RegistryError::ReservedTypeId { type_id: "x".to_owned() },
    r#"the type id "x" is the search space's own"#
)]
fn registry_error_displays_its_pinned_text(#[case] error: RegistryError, #[case] text: &str) {
    let shown = error.to_string();

    assert_eq!(shown, text);
}
