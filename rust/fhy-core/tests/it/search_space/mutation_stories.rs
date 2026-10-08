//! Tests for `Space::mutate`'s search: which decisions it finds
//! changeable, how many times it asks a variable's hooks, and the seeded
//! outputs it keeps.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use fhy_core::constraint::{Outcome, Value};
use fhy_core::expression::Expression;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{Configuration, RandomOracle, Rng, Space, TraceError};

use crate::support::constraint::{TestCustom, int};
use crate::support::mutation::{
    build_counting_space, build_table_space, build_unreachable_alternative_space, describe_entries,
};
use crate::support::search::{
    attempts, build_tiling_space, permutation_param, tiling_entries, with_context,
};
use crate::support::search_space::{
    bare_alternative, choice_of, chosen, configure, forbidden, int_variable, natural_param,
    plain_alternative, plain_variable, space_of,
};

/// Test a mutation refuses a choice's other alternative when no completion
/// of it exists, though the search for one outlasts the search budget: the
/// alternative holds eleven two-valued variables (2048 paths) and then a
/// variable that admits no value.
#[test]
fn space_mutate_refuses_an_alternative_without_a_completion_beyond_the_search_budget() {
    let (space, choice, empty) = build_unreachable_alternative_space();
    let configuration = configure(&space, [(choice, chosen(&empty))]);

    let result = with_context(|context| {
        space.mutate(&configuration, &mut Rng::new(0), context, attempts(4))
    });

    assert!(
        matches!(result, Err(TraceError::NothingToMutate)),
        "{result:?}"
    );
}

/// Return the space of one choice `c` between `a`, holding the variable `x`
/// over `values`, the alternatives `extra` holding nothing, and `u`,
/// holding the variable `n` over the unbounded natural numbers, with the
/// names of `c`, `a`, `x` and `u`.
fn build_unbounded_alternative_space(
    values: &[i64],
    extra: &[&str],
) -> (Space, Identifier, Identifier, Identifier, Identifier) {
    let [name, c, a, x, u, n] = ["unbounded", "c", "a", "x", "u", "n"].map(Identifier::new);
    let mut alternatives = vec![plain_alternative(
        &a,
        vec![int_variable(&x, values)],
        Vec::new(),
    )];
    alternatives.extend(
        extra
            .iter()
            .map(|name| bare_alternative(&Identifier::new(name))),
    );
    alternatives.push(plain_alternative(
        &u,
        vec![plain_variable(&n, natural_param())],
        Vec::new(),
    ));
    let space = space_of(&name, Vec::new(), vec![choice_of(&c, alternatives)]);
    (space, c, a, x, u)
}

/// Test a mutation never moves to an alternative holding a variable with no
/// finite domain, whatever the seed: no value of its variable can be drawn,
/// so it is no option, and every seed mutates to another one.
#[test]
fn space_mutate_never_moves_to_an_alternative_holding_an_unbounded_variable() {
    let (space, choice, alternative, variable, unbounded) =
        build_unbounded_alternative_space(&[1, 2, 3], &["b"]);
    let configuration = configure(
        &space,
        [(choice.clone(), chosen(&alternative)), (variable, int(2))],
    );

    for seed in 0..64 {
        let mutated = with_context(|context| {
            space.mutate(&configuration, &mut Rng::new(seed), context, attempts(16))
        })
        .unwrap_or_else(|error| panic!("seed {seed} mutates, got {error:?}"));
        let configuration = mutated.configuration().expect("a run over a space");
        assert_ne!(
            configuration.value(&choice),
            Some(&chosen(&unbounded)),
            "seed {seed} moved to the unbounded alternative"
        );
    }
}

/// Test a mutation whose only other option is an alternative holding a
/// variable with no finite domain finds nothing to mutate, for every seed.
#[test]
fn space_mutate_finds_nothing_when_the_only_other_alternative_is_unbounded() {
    let (space, choice, alternative, variable, _) = build_unbounded_alternative_space(&[1], &[]);
    let configuration = configure(&space, [(choice, chosen(&alternative)), (variable, int(1))]);

    for seed in 0..16 {
        let result = with_context(|context| {
            space.mutate(&configuration, &mut Rng::new(seed), context, attempts(16))
        });
        assert!(
            matches!(result, Err(TraceError::NothingToMutate)),
            "seed {seed}: {result:?}"
        );
    }
}

/// Test a mutation asks each variable's search-domain hook a number of
/// times linear in the decisions, not once per decision per candidate
/// value: hook calls per mutation are linear in the decisions.
#[test]
fn space_mutate_asks_each_variable_search_domain_a_linear_number_of_times() {
    let asks = Arc::new(AtomicUsize::new(0));
    let space = build_counting_space(100, &asks);
    let decisions = space.decisions().len();
    let sampled = with_context(|context| space.sample(&mut RandomOracle::new(3), context))
        .expect("the space is sampled");
    let configuration = sampled.configuration().expect("a run over a space");
    asks.store(0, Ordering::SeqCst);

    let mutated = with_context(|context| {
        space.mutate(configuration, &mut Rng::new(11), context, attempts(16))
    })
    .expect("a mutation");

    let asked = asks.load(Ordering::SeqCst);
    assert!(
        asked <= 20 * decisions,
        "{asked} asks for {decisions} decisions"
    );
    assert!(
        mutated
            .configuration()
            .is_some_and(Configuration::is_complete)
    );
}

/// The number of variables of the logged-clause space.
const LOGGED_VARIABLES: usize = 40;

/// Return the space of `count` top-level variables over `{1, 2, 3}`, each
/// named in a forbidden clause of its own that never holds and logs each
/// evaluation to `log`.
fn build_logged_clause_space(count: usize, log: &Arc<Mutex<Vec<String>>>) -> Space {
    let names: Vec<Identifier> = (0..count)
        .map(|index| Identifier::new(&format!("v{index}")))
        .collect();
    let clauses = names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            forbidden([TestCustom::build(
                &format!("clause{index}"),
                Expression::from(name),
                Outcome::Violated,
                log,
            )])
        })
        .collect();
    Space::new(
        Identifier::new("logged"),
        names
            .iter()
            .map(|name| int_variable(name, &[1, 2, 3]))
            .collect(),
        Vec::new(),
        Vec::new(),
        clauses,
    )
    .expect("the space is valid")
}

/// Test a mutation evaluates the forbidden clauses a number of times
/// linear in the decisions: a step re-checks only the clauses naming what
/// it changed, and the search for what may change stays inside each
/// clause's component, instead of re-checking every clause at every step
/// of a walk of the whole space per candidate value.
#[test]
fn space_mutate_evaluates_forbidden_clauses_a_linear_number_of_times() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let space = build_logged_clause_space(LOGGED_VARIABLES, &log);
    let sampled = with_context(|context| space.sample(&mut RandomOracle::new(3), context))
        .expect("the space is sampled");
    let configuration = sampled.configuration().expect("a run over a space");
    log.lock().unwrap_or_else(PoisonError::into_inner).clear();

    let mutated = with_context(|context| {
        space.mutate(configuration, &mut Rng::new(5), context, attempts(16))
    })
    .expect("a mutation");

    let evaluations = log.lock().unwrap_or_else(PoisonError::into_inner).len();
    assert!(
        evaluations <= 20 * LOGGED_VARIABLES,
        "{evaluations} clause evaluations for {LOGGED_VARIABLES} variables"
    );
    assert!(
        mutated
            .configuration()
            .is_some_and(Configuration::is_complete)
    );
}

// ---------------------------------------------------------------------------
// Seeded outputs that stay as they are
// ---------------------------------------------------------------------------

/// What a seeded mutation gives, per seed from 0: the mutated entries and
/// the generator's next number afterwards.
type Pins<'a> = &'a [(&'a [&'a str], u64)];

/// Assert that mutating `configuration` of `space` with each seed from 0
/// gives `pins`.
fn assert_pinned(space: &Space, configuration: &Configuration, pins: Pins<'_>) {
    for (seed, (entries, next)) in (0..).zip(pins) {
        let (observed, after) = observe_mutation(space, configuration, seed);
        assert_eq!(observed, *entries, "the entries of seed {seed}");
        assert_eq!(after, *next, "the generator after seed {seed}");
    }
}

/// Return what mutating `configuration` of `space` with the seed `seed`
/// gives: the mutated entries, and the generator's next number after the
/// mutation, which pins every draw the mutation made.
fn observe_mutation(space: &Space, configuration: &Configuration, seed: u64) -> (Vec<String>, u64) {
    let mut rng = Rng::new(seed);
    let mutated =
        with_context(|context| space.mutate(configuration, &mut rng, context, attempts(16)))
            .expect("a mutation");
    (
        describe_entries(mutated.configuration().expect("a run over a space")),
        rng.next_u64(),
    )
}

/// Return the space of one choice `c` between `a`, holding the variable
/// `x` over `{1, 2, 3}`, and `b`, holding nothing, with the names of `c`,
/// `a` and `x`.
fn build_choice_space() -> (Space, Identifier, Identifier, Identifier) {
    let [name, c, a, x, b] = ["choosing", "c", "a", "x", "b"].map(Identifier::new);
    let space = space_of(
        &name,
        Vec::new(),
        vec![choice_of(
            &c,
            vec![
                plain_alternative(&a, vec![int_variable(&x, &[1, 2, 3])], Vec::new()),
                bare_alternative(&b),
            ],
        )],
    );
    (space, c, a, x)
}

/// Test the tiling space's seeded mutations are the pinned ones: per seed,
/// the mutated entries and the generator's next number afterwards.
#[test]
fn space_mutate_of_the_tiling_space_keeps_its_pinned_seeded_output() {
    let tiling = build_tiling_space();
    let configuration = configure(
        &tiling.space,
        tiling_entries(&tiling, &(1, tiling.a.clone(), Some(1))),
    );

    assert_pinned(
        &tiling.space,
        &configuration,
        &[
            (&["t=1", "c=a", "x=2"], 487_617_019_471_545_679),
            (&["t=1", "c=b"], 17_911_839_290_282_890_590),
            (&["t=1", "c=b"], 10_987_583_248_141_275_951),
            (&["t=2", "c=a"], 11_307_387_092_600_937_729),
            (&["t=1", "c=b"], 15_847_914_186_252_977_247),
            (&["t=1", "c=b"], 4_292_726_422_858_613_063),
        ],
    );
}

/// Test a choice space's seeded mutations, from its alternative holding a
/// variable, are the pinned ones.
#[test]
fn space_mutate_of_a_choice_space_keeps_its_pinned_seeded_output() {
    let (space, choice, alternative, variable) = build_choice_space();
    let configuration = configure(&space, [(choice, chosen(&alternative)), (variable, int(2))]);

    assert_pinned(
        &space,
        &configuration,
        &[
            (&["c=a", "x=1"], 487_617_019_471_545_679),
            (&["c=a", "x=3"], 17_911_839_290_282_890_590),
            (&["c=a", "x=3"], 10_987_583_248_141_275_951),
            (&["c=b"], 11_307_387_092_600_937_729),
            (&["c=b"], 15_847_914_186_252_977_247),
            (&["c=b"], 4_292_726_422_858_613_063),
        ],
    );
}

/// Test the seeded mutations of a space holding an ordering and a variable
/// are the pinned ones.
#[test]
fn space_mutate_of_an_ordering_space_keeps_its_pinned_seeded_output() {
    let [name, order, k, i, j, l] = ["ordered", "order", "k", "i", "j", "l"].map(Identifier::new);
    let space = space_of(
        &name,
        vec![
            plain_variable(&order, permutation_param(&[&i, &j, &l])),
            int_variable(&k, &[1, 2, 3]),
        ],
        Vec::new(),
    );
    let configuration = configure(
        &space,
        [
            (order, Value::Tuple([&i, &j, &l].map(chosen).to_vec())),
            (k, int(2)),
        ],
    );

    assert_pinned(
        &space,
        &configuration,
        &[
            (&["order=(i, j, l)", "k=1"], 487_617_019_471_545_679),
            (&["order=(i, j, l)", "k=3"], 17_911_839_290_282_890_590),
            (&["order=(i, j, l)", "k=3"], 10_987_583_248_141_275_951),
            (&["order=(i, l, j)", "k=2"], 11_307_387_092_600_937_729),
            (&["order=(i, l, j)", "k=2"], 15_847_914_186_252_977_247),
            (&["order=(i, l, j)", "k=2"], 4_292_726_422_858_613_063),
        ],
    );
}

/// The entries of the three-entry table's sampled starting point.
const TABLE_START: &str = "entry0=option1 size=1 order=3 flag=2 entry1=option1 size=2 order=2 flag=1 entry2=option0 size=4 order=1 flag=2";

/// The table's seeded mutations, per seed from 0: the mutated entries and
/// the generator's next number afterwards.
const TABLE_PINS: [(&str, u64); 6] = [
    (
        "entry0=option1 size=1 order=3 flag=2 entry1=option1 size=2 order=2 flag=1 entry2=option0 size=4 order=2 flag=2",
        487_617_019_471_545_679,
    ),
    (
        "entry0=option1 size=1 order=3 flag=2 entry1=option1 size=2 order=3 flag=1 entry2=option0 size=4 order=1 flag=2",
        17_911_839_290_282_890_590,
    ),
    (
        "entry0=option1 size=1 order=3 flag=2 entry1=option1 size=2 order=2 flag=2 entry2=option0 size=4 order=1 flag=2",
        10_987_583_248_141_275_951,
    ),
    (
        "entry0=option1 size=6 order=3 flag=2 entry1=option1 size=2 order=2 flag=1 entry2=option0 size=4 order=1 flag=2",
        11_307_387_092_600_937_729,
    ),
    (
        "entry0=option1 size=1 order=3 flag=2 entry1=option1 size=8 order=2 flag=1 entry2=option0 size=4 order=1 flag=2",
        15_847_914_186_252_977_247,
    ),
    (
        "entry0=option1 size=1 order=3 flag=2 entry1=option3 size=2 order=1 flag=1 entry2=option0 size=4 order=1 flag=2",
        7_020_995_479_949_754_436,
    ),
];

/// Test the seeded mutations of a three-entry candidate table, which
/// change a bounded integer, a categorical or an entry's alternative, are
/// the pinned ones.
#[test]
fn space_mutate_of_a_table_space_keeps_its_pinned_seeded_output() {
    let table = build_table_space(3);
    let sampled = with_context(|context| table.sample(&mut RandomOracle::new(7), context))
        .expect("the table is sampled");
    let configuration = sampled.configuration().expect("a run over a space");

    let observed: Vec<(String, u64)> = (0..)
        .take(TABLE_PINS.len())
        .map(|seed| {
            let (entries, next) = observe_mutation(&table, configuration, seed);
            (entries.join(" "), next)
        })
        .collect();

    assert_eq!(
        describe_entries(configuration).join(" "),
        TABLE_START,
        "the pinned starting point"
    );
    let expected: Vec<(String, u64)> = TABLE_PINS
        .iter()
        .map(|&(entries, next)| (entries.to_owned(), next))
        .collect();
    assert_eq!(observed, expected);
}
