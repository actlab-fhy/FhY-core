//! User-story tests for `fhy_core::testing`, reached through the `testing`
//! feature as a downstream crate would reach it.

use std::thread;

use fhy_core::identifier::Identifier;
use fhy_core::testing::DeterministicIdentifierScope;

/// A reduction split into a partial and a final accumulator.
#[derive(Debug, PartialEq)]
struct LoweredReduction {
    input: Identifier,
    partial: Identifier,
    total: Identifier,
}

/// Stand in for code under test that creates identifiers internally.
fn lower_reduction(input: &Identifier) -> LoweredReduction {
    LoweredReduction {
        input: input.clone(),
        partial: Identifier::new("partial"),
        total: Identifier::new("total"),
    }
}

/// Test a test compares a graph whose identifiers the code under test
/// created with a graph the test builds itself.
#[test]
fn a_test_compares_a_lowered_graph_with_one_it_builds_itself() {
    let input = Identifier::new("stories-reduction-input");
    let _scope = DeterministicIdentifierScope::enter();

    let lowered = lower_reduction(&input);

    let expected = LoweredReduction {
        input,
        partial: Identifier::new("partial"),
        total: Identifier::new("total"),
    };
    assert_eq!(lowered, expected);
}

/// Test worker threads that join the test's scope build the same graph as
/// the test thread.
#[test]
fn worker_threads_that_join_the_scope_build_the_same_graph() {
    let input = Identifier::new("stories-parallel-input");
    let scope = DeterministicIdentifierScope::enter();
    let handle = scope.share();

    let lowered: Vec<LoweredReduction> = thread::scope(|threads| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                threads.spawn(|| {
                    let _joined = handle.enter();
                    lower_reduction(&input)
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect()
    });

    let expected = lower_reduction(&input);
    assert!(
        lowered.iter().all(|graph| *graph == expected),
        "{lowered:?}"
    );
}
