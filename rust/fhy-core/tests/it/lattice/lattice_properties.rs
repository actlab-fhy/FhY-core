//! Properties of `PartiallyOrderedSet` and `Lattice`.
//!
//! The order agrees with the transitive closure of the added orders over
//! random DAGs, and the lattice laws hold over lattices built by
//! construction (powersets, divisor sets, chains and products of chains),
//! against an order predicate of each family that never asks the lattice,
//! as `tests/test_poset_properties.py` and `tests/test_lattice_properties.py`
//! check them in Python.

use fhy_core::lattice::{Lattice, PartiallyOrderedSet};
use proptest::prelude::*;

/// Return a random DAG: a node count, and edges from a smaller node to a
/// larger one, in a shuffled order.
fn dag_strategy() -> impl Strategy<Value = (usize, Vec<(usize, usize)>)> {
    (2_usize..=8).prop_flat_map(|count| {
        let candidates: Vec<(usize, usize)> = (0..count)
            .flat_map(|lower| (lower + 1..count).map(move |upper| (lower, upper)))
            .collect();
        let length = candidates.len();
        (
            Just(count),
            proptest::collection::vec(any::<bool>(), length).prop_flat_map(move |chosen| {
                let edges: Vec<(usize, usize)> = candidates
                    .iter()
                    .zip(chosen)
                    .filter_map(|(edge, keep)| keep.then_some(*edge))
                    .collect();
                Just(edges).prop_shuffle()
            }),
        )
    })
}

/// Return the reflexive and transitive closure of `edges` over `count`
/// nodes, as a matrix.
fn closure(count: usize, edges: &[(usize, usize)]) -> Vec<Vec<bool>> {
    let mut reach = vec![vec![false; count]; count];
    for (node, row) in reach.iter_mut().enumerate() {
        row[node] = true;
    }
    for &(lower, upper) in edges {
        reach[lower][upper] = true;
    }
    for middle in 0..count {
        for lower in 0..count {
            for upper in 0..count {
                if reach[lower][middle] && reach[middle][upper] {
                    reach[lower][upper] = true;
                }
            }
        }
    }
    reach
}

/// Return the poset of the DAG.
fn build_poset(count: usize, edges: &[(usize, usize)]) -> PartiallyOrderedSet<usize> {
    let mut poset = PartiallyOrderedSet::new();
    for node in 0..count {
        poset.add_element(node).expect("each node is new");
    }
    for (lower, upper) in edges {
        poset.add_order(lower, upper).expect("the DAG is acyclic");
    }
    poset
}

/// A lattice built by construction, with its elements and an order
/// predicate that never asks the lattice.
struct LatticeCase {
    lattice: Lattice<Vec<u32>>,
    elements: Vec<Vec<u32>>,
    is_at_most: fn(&[u32], &[u32]) -> bool,
}

/// Return the case of `elements` under `is_at_most`, ordered by every pair
/// the predicate relates.
fn build_case(elements: Vec<Vec<u32>>, is_at_most: fn(&[u32], &[u32]) -> bool) -> LatticeCase {
    let mut lattice = Lattice::new();
    for element in &elements {
        lattice
            .add_element(element.clone())
            .expect("each element is new");
    }
    for lower in &elements {
        for upper in &elements {
            if lower != upper && is_at_most(lower, upper) {
                lattice
                    .add_order(lower, upper)
                    .expect("the predicate is an order");
            }
        }
    }
    LatticeCase {
        lattice,
        elements,
        is_at_most,
    }
}

/// Return the powerset of `0..size`, each subset as a bit mask, ordered by
/// inclusion.
fn powerset(size: u32) -> LatticeCase {
    let elements = (0..1_u32 << size).map(|mask| vec![mask]).collect();
    build_case(elements, |lower, upper| lower[0] & !upper[0] == 0)
}

/// Return the divisors of `number`, ordered by divisibility.
fn divisors(number: u32) -> LatticeCase {
    let elements = (1..=number)
        .filter(|d| number % d == 0)
        .map(|d| vec![d])
        .collect();
    build_case(elements, |lower, upper| upper[0] % lower[0] == 0)
}

/// Return the chain `0..length`.
fn chain(length: u32) -> LatticeCase {
    build_case((0..length).map(|i| vec![i]).collect(), |lower, upper| {
        lower[0] <= upper[0]
    })
}

/// Return the product of two chains, ordered componentwise.
fn product(left: u32, right: u32) -> LatticeCase {
    let elements = (0..left)
        .flat_map(|i| (0..right).map(move |j| vec![i, j]))
        .collect();
    build_case(elements, |lower, upper| {
        lower[0] <= upper[0] && lower[1] <= upper[1]
    })
}

fn lattice_case_strategy() -> impl Strategy<Value = u32> {
    0_u32..12
}

/// Return the case numbered `index` of the fixed families.
fn lattice_case(index: u32) -> LatticeCase {
    match index {
        0..=2 => powerset(index + 1),
        3 => divisors(6),
        4 => divisors(12),
        5 => divisors(30),
        6 => divisors(36),
        7..=9 => chain(index - 5),
        10 => product(2, 3),
        _ => product(3, 3),
    }
}

/// Return the brute-force meet of `x` and `y`: the greatest lower bound
/// under the case's predicate.
fn reference_meet<'a>(case: &'a LatticeCase, x: &[u32], y: &[u32]) -> &'a Vec<u32> {
    let is_at_most = case.is_at_most;
    let lower_bounds: Vec<&Vec<u32>> = case
        .elements
        .iter()
        .filter(|z| is_at_most(z, x) && is_at_most(z, y))
        .collect();
    lower_bounds
        .iter()
        .find(|z| lower_bounds.iter().all(|w| is_at_most(w, z)))
        .expect("a lattice has a meet")
}

/// Return the brute-force join of `x` and `y`.
fn reference_join<'a>(case: &'a LatticeCase, x: &[u32], y: &[u32]) -> &'a Vec<u32> {
    let is_at_most = case.is_at_most;
    let upper_bounds: Vec<&Vec<u32>> = case
        .elements
        .iter()
        .filter(|z| is_at_most(x, z) && is_at_most(y, z))
        .collect();
    upper_bounds
        .iter()
        .find(|z| upper_bounds.iter().all(|w| is_at_most(z, w)))
        .expect("a lattice has a join")
}

proptest! {
    #[test]
    fn order_agrees_with_the_closure_of_the_added_orders((count, edges) in dag_strategy()) {
        let poset = build_poset(count, &edges);
        let reach = closure(count, &edges);
        for (lower, row) in reach.iter().enumerate() {
            for (upper, &holds) in row.iter().enumerate() {
                prop_assert_eq!(poset.is_at_most(&lower, &upper), Ok(holds));
            }
        }
    }

    #[test]
    fn iteration_is_a_topological_order((count, edges) in dag_strategy()) {
        let poset = build_poset(count, &edges);
        let order: Vec<usize> = poset.iter().copied().collect();
        prop_assert_eq!(order.len(), count);
        for (lower, upper) in &edges {
            let lower_at = order.iter().position(|node| node == lower);
            let upper_at = order.iter().position(|node| node == upper);
            prop_assert!(lower_at < upper_at);
        }
    }

    #[test]
    fn reversing_an_order_is_refused_exactly_when_it_holds(
        (count, edges) in dag_strategy(),
        first in 0_usize..8,
        second in 0_usize..8,
    ) {
        let (first, second) = (first % count, second % count);
        let reach = closure(count, &edges);
        let mut poset = build_poset(count, &edges);
        prop_assert_eq!(poset.add_order(&second, &first).is_err(), reach[first][second]);
    }

    #[test]
    fn is_lattice_exactly_when_no_bound_is_missing((count, edges) in dag_strategy()) {
        let mut lattice = Lattice::new();
        for node in 0..count {
            lattice.add_element(node).expect("each node is new");
        }
        for (lower, upper) in &edges {
            lattice.add_order(lower, upper).expect("the DAG is acyclic");
        }
        prop_assert_eq!(lattice.is_lattice(), lattice.missing_bounds().next().is_none());
    }

    #[test]
    fn meet_and_join_are_the_bounds_of_the_family_order(index in lattice_case_strategy()) {
        let case = lattice_case(index);
        prop_assert!(case.lattice.is_lattice());
        for x in &case.elements {
            for y in &case.elements {
                prop_assert_eq!(case.lattice.meet(x, y), Ok(Some(reference_meet(&case, x, y))));
                prop_assert_eq!(case.lattice.join(x, y), Ok(Some(reference_join(&case, x, y))));
            }
        }
    }

    #[test]
    fn meet_and_join_are_commutative_associative_idempotent_and_absorbing(
        index in lattice_case_strategy(),
    ) {
        let case = lattice_case(index);
        let lattice = &case.lattice;
        let meet = |x: &Vec<u32>, y: &Vec<u32>| lattice.meet(x, y).expect("members").expect("a lattice").clone();
        let join = |x: &Vec<u32>, y: &Vec<u32>| lattice.join(x, y).expect("members").expect("a lattice").clone();
        for x in &case.elements {
            prop_assert_eq!(&meet(x, x), x);
            prop_assert_eq!(&join(x, x), x);
            for y in &case.elements {
                prop_assert_eq!(meet(x, y), meet(y, x));
                prop_assert_eq!(join(x, y), join(y, x));
                prop_assert_eq!(&meet(x, &join(x, y)), x);
                prop_assert_eq!(&join(x, &meet(x, y)), x);
                for z in &case.elements {
                    prop_assert_eq!(meet(&meet(x, y), z), meet(x, &meet(y, z)));
                    prop_assert_eq!(join(&join(x, y), z), join(x, &join(y, z)));
                }
            }
        }
    }
}
