//! A measurement of what a mutation and one-entry-at-a-time building cost on
//! a candidate table. It prints times; it asserts none.

use std::num::NonZeroU32;
use std::time::Instant;

use fhy_core::search_space::{Configuration, RandomOracle, Rng, Space};

use crate::support::mutation::{build_flat_space, build_table_space};
use crate::support::search::with_context;

/// The table sizes a mutation is timed at.
const MUTATION_ENTRIES: [usize; 4] = [10, 50, 100, 200];

/// The number of decisions a configuration is built one entry at a time
/// over.
const BUILDING_DECISIONS: usize = 3_400;

/// Return a complete configuration of `space`, sampled with `seed`.
fn sample_configuration(space: &Space, seed: u64) -> Configuration {
    with_context(|context| space.sample(&mut RandomOracle::new(seed), context))
        .expect("the table is sampled")
        .configuration()
        .expect("a run over a space")
        .clone()
}

/// Time `Space::mutate` at 10, 50, 100 and 200 table entries and building
/// a configuration of 3 400 decisions one entry at a time, and print
/// the times.
///
/// Not a test: run it in a release build with `cargo test --release -p
/// fhy-core --test it search_space_timing -- --ignored --nocapture`. The
/// targets are a mutation of a 200-entry table under 100 ms and the
/// building under 50 ms.
#[test]
#[ignore = "a timing, not a check: run it in a release build with --nocapture"]
#[expect(clippy::print_stdout, reason = "the timing is the output")]
fn search_space_timing() {
    for entries in MUTATION_ENTRIES {
        let space = build_table_space(entries);
        let configuration = sample_configuration(&space, 1);
        let started = Instant::now();
        let mutated = with_context(|context| {
            space.mutate(
                &configuration,
                &mut Rng::new(2),
                context,
                NonZeroU32::new(16).expect("positive"),
            )
        });
        let elapsed = started.elapsed();
        println!(
            "mutate at {entries} entries ({} decisions): {elapsed:?}, {}",
            configuration.entries().len(),
            if mutated.is_ok() { "ok" } else { "refused" },
        );
    }

    let space = build_flat_space(BUILDING_DECISIONS);
    let target = sample_configuration(&space, 3);
    let steps: Vec<_> = space
        .decision_order()
        .iter()
        .filter_map(|name| {
            target
                .value(name)
                .map(|value| (name.clone(), value.clone()))
        })
        .collect();
    let started = Instant::now();
    let built = with_context(|context| {
        let mut configuration =
            Configuration::new(&space, [], context).expect("the empty configuration is valid");
        for (name, value) in &steps {
            configuration = configuration
                .with_entry(name.clone(), value.clone(), context)
                .expect("each entry is accepted");
        }
        configuration
    });
    println!(
        "one entry at a time, {} decisions: {:?}, equal to the target: {}",
        steps.len(),
        started.elapsed(),
        built == target,
    );
}
