//! Tests for a search space over a categorical variable whose categories
//! are tuples, as a tile shape is: counting, enumerating, sampling and
//! replaying it, its configuration keys and traces on the wire, and a
//! forbidden clause naming a tuple.

use std::collections::BTreeSet;

use fhy_core::constraint::Value;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{Cardinality, Configuration, ConfigurationError, RandomOracle, Space};
use num_bigint::BigUint;
use rstest::rstest;

use crate::support::constraint::int;
use crate::support::param::in_set;
use crate::support::search::with_context;
use crate::support::search_space::{
    categorical, forbidden, plain_variable, space_of, try_configure,
};
use crate::support::serde::check_serde_round_trip;

/// Return the tile shape `(rows, columns)`.
fn shape(rows: i64, columns: i64) -> Value {
    Value::Tuple(vec![int(rows), int(columns)])
}

/// Return the shapes `(4, 4)`, `(8, 8)` and `(16, 16)`.
fn build_shapes() -> Vec<Value> {
    vec![shape(4, 4), shape(8, 8), shape(16, 16)]
}

/// Return the space of one variable `tile` over [`build_shapes`], and the
/// variable's name.
fn build_tile_space() -> (Space, Identifier) {
    let tile = Identifier::new("tile");
    let space = space_of(
        &Identifier::new("tiles"),
        vec![plain_variable(&tile, categorical(build_shapes()))],
        Vec::new(),
    );
    (space, tile)
}

/// Return the texts of the values `configurations` give `name`, sorted.
fn collect_values(configurations: &[Configuration], name: &Identifier) -> BTreeSet<String> {
    configurations
        .iter()
        .map(|configuration| {
            configuration
                .value(name)
                .expect("the variable is assigned")
                .to_string()
        })
        .collect()
}

/// Test a space over three tuple categories counts exactly three
/// configurations and enumerates one per shape.
#[test]
fn space_over_tuple_categories_counts_and_enumerates_each_shape() {
    let (space, tile) = build_tile_space();

    let (count, configurations) = with_context(|context| {
        let count = space.cardinality(context, 100).expect("the count succeeds");
        let configurations: Vec<Configuration> = space
            .enumerate(context)
            .collect::<Result<_, _>>()
            .expect("the enumeration succeeds");
        (count, configurations)
    });

    assert_eq!(count, Cardinality::Exact(BigUint::from(3_u32)));
    assert_eq!(configurations.len(), 3);
    let expected: BTreeSet<String> = build_shapes().iter().map(ToString::to_string).collect();
    assert_eq!(collect_values(&configurations, &tile), expected);
}

/// Test sampling a space over tuple categories draws one of the shapes,
/// and the run's trace replays into the same configuration.
#[rstest]
#[case::seed_0(0)]
#[case::seed_1(1)]
#[case::seed_2(2)]
#[case::seed_3(3)]
fn space_sample_over_tuple_categories_draws_a_shape_and_replays(#[case] seed: u64) {
    let (space, tile) = build_tile_space();

    let (sampled, replayed) = with_context(|context| {
        let recorded = space
            .sample(&mut RandomOracle::new(seed), context)
            .expect("every shape is admissible");
        let sampled = recorded
            .configuration()
            .expect("a run over a space")
            .clone();
        let replayed = space
            .replay(recorded.trace(), context)
            .expect("the trace replays");
        (sampled, replayed)
    });

    let value = sampled.value(&tile).expect("the variable is assigned");
    assert!(build_shapes().contains(value), "{value:?}");
    assert_eq!(replayed, sampled);
}

/// Test the keys of configurations over tuple categories are equal for one
/// shape and differ for two, across two spaces built apart.
#[test]
fn configuration_key_over_tuple_categories_tells_the_shapes_apart() {
    let (left, left_tile) = build_tile_space();
    let (right, right_tile) = build_tile_space();

    let small = try_configure(&left, [(left_tile.clone(), shape(4, 4))]).expect("a shape");
    let large = try_configure(&left, [(left_tile, shape(8, 8))]).expect("a shape");
    let other_large = try_configure(&right, [(right_tile, shape(8, 8))]).expect("a shape");

    assert_ne!(small.key(), large.key());
    assert_eq!(large.key(), other_large.key());
}

/// Test a configuration refuses a tuple that is no category, and a leaf.
#[rstest]
#[case::other_shape(shape(5, 5))]
#[case::transposed_prefix(Value::Tuple(vec![int(4)]))]
#[case::leaf(int(4))]
fn configuration_new_refuses_a_value_that_is_no_shape(#[case] value: Value) {
    let (space, tile) = build_tile_space();

    let result = try_configure(&space, [(tile.clone(), value)]);

    let errors = result.expect_err("no shape");
    let [ConfigurationError::Assignment { variable, .. }] = errors.errors() else {
        panic!("expected one assignment error, got {errors:?}");
    };
    assert_eq!(variable, &tile);
}

/// Test the key of a configuration over a tuple category round-trips
/// through JSON and postcard.
#[test]
fn configuration_key_over_a_tuple_category_round_trips() {
    let (space, tile) = build_tile_space();
    let configuration = try_configure(&space, [(tile, shape(16, 16))]).expect("a shape");

    let result = check_serde_round_trip(&configuration.key());

    result.unwrap_or_else(|failure| panic!("{failure}"));
}

/// Test the trace of a run over tuple categories round-trips through JSON
/// and postcard.
#[test]
fn trace_over_a_tuple_category_round_trips() {
    let (space, _) = build_tile_space();
    let recorded = with_context(|context| space.sample(&mut RandomOracle::new(5), context))
        .expect("every shape is admissible");

    let result = check_serde_round_trip(recorded.trace());

    result.unwrap_or_else(|failure| panic!("{failure}"));
}

/// Test a forbidden clause `tile in {(8, 8)}` removes that shape from the
/// enumeration and refuses a configuration choosing it.
#[test]
fn forbidden_clause_over_a_tuple_member_removes_that_shape() {
    let tile = Identifier::new("tile");
    let space = Space::new(
        Identifier::new("tiles"),
        vec![plain_variable(&tile, categorical(build_shapes()))],
        Vec::new(),
        Vec::new(),
        vec![forbidden([in_set(&tile, [shape(8, 8)])])],
    )
    .expect("the space is valid");

    let configurations: Vec<Configuration> = with_context(|context| {
        space
            .enumerate(context)
            .collect::<Result<_, _>>()
            .expect("the enumeration succeeds")
    });
    let refused = try_configure(&space, [(tile.clone(), shape(8, 8))]);

    let expected: BTreeSet<String> = [shape(4, 4), shape(16, 16)]
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(collect_values(&configurations, &tile), expected);
    let errors = refused.expect_err("the shape is forbidden");
    assert!(
        matches!(
            errors.errors(),
            [ConfigurationError::Forbidden { index: 0 }]
        ),
        "{errors:?}"
    );
}
