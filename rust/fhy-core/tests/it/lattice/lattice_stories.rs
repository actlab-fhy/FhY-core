//! Tests for `Lattice`: meets, joins, `is_lattice` and the missing bounds
//! over the lattices and non-lattices of `tests/test_lattice.py`.

use fhy_core::lattice::{Lattice, MissingBound, OrderError};

/// Return the lattice of `elements` with the orders `orders`.
fn build_lattice<T: Eq + std::hash::Hash + Clone + std::fmt::Debug>(
    elements: &[T],
    orders: &[(T, T)],
) -> Lattice<T> {
    let mut lattice = Lattice::new();
    for element in elements {
        lattice
            .add_element(element.clone())
            .expect("each element is new");
    }
    for (lower, upper) in orders {
        lattice
            .add_order(lower, upper)
            .expect("each order is acyclic");
    }
    lattice
}

/// The subsets of `{x, y, z}`, ordered by inclusion.
fn build_subsets_of_xyz() -> Lattice<&'static str> {
    build_lattice(
        &["0", "x", "y", "z", "xy", "xz", "yz", "xyz"],
        &[
            ("0", "x"),
            ("0", "y"),
            ("0", "z"),
            ("x", "xy"),
            ("x", "xz"),
            ("y", "xy"),
            ("y", "yz"),
            ("z", "xz"),
            ("z", "yz"),
            ("xy", "xyz"),
            ("xz", "xyz"),
            ("yz", "xyz"),
        ],
    )
}

/// Two minimal elements both below two maximal ones: no pair across a
/// level has a meet or a join.
fn build_crown() -> Lattice<i32> {
    build_lattice(&[1, 2, 3, 4], &[(1, 3), (1, 4), (2, 3), (2, 4)])
}

/// The crown over a bottom element: every pair has a meet, but `1` and `2`
/// have two minimal upper bounds.
fn build_crown_over_a_bottom() -> Lattice<i32> {
    build_lattice(
        &[0, 1, 2, 3, 4],
        &[(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4)],
    )
}

#[test]
fn empty_lattice_is_a_lattice_and_holds_nothing() {
    let lattice: Lattice<i32> = Lattice::new();

    assert!(!lattice.contains(&1));
    assert!(lattice.is_lattice());
    assert_eq!(lattice.missing_bounds().count(), 0);
}

#[test]
fn empty_lattice_refuses_a_non_member() {
    let lattice: Lattice<i32> = Lattice::new();

    assert_eq!(lattice.meet(&1, &1), Err(OrderError::NotAMember(1)));
    assert_eq!(lattice.join(&1, &1), Err(OrderError::NotAMember(1)));
}

#[test]
fn singleton_lattice_meets_and_joins_its_element_with_itself() {
    let lattice = build_lattice(&[1], &[]);

    assert!(lattice.contains(&1));
    assert_eq!(lattice.meet(&1, &1), Ok(Some(&1)));
    assert_eq!(lattice.join(&1, &1), Ok(Some(&1)));
    assert!(lattice.is_lattice());
}

#[test]
fn two_element_chain_meets_at_its_bottom_and_joins_at_its_top() {
    let lattice = build_lattice(&[1, 2], &[(1, 2)]);

    assert_eq!(lattice.meet(&1, &2), Ok(Some(&1)));
    assert_eq!(lattice.meet(&2, &1), Ok(Some(&1)));
    assert_eq!(lattice.join(&1, &2), Ok(Some(&2)));
    assert_eq!(lattice.join(&2, &2), Ok(Some(&2)));
    assert!(lattice.is_lattice());
}

#[test]
fn ordering_through_the_lattice_refuses_a_cycle() {
    let mut lattice = build_lattice(&[1, 2], &[(1, 2)]);

    assert_eq!(
        lattice.add_order(&2, &1),
        Err(OrderError::WouldCycle { lower: 2, upper: 1 })
    );
    assert_eq!(lattice.add_element(1), Err(OrderError::AlreadyAMember(1)));
}

#[test]
fn chain_meets_at_the_minimum_and_joins_at_the_maximum() {
    let elements: Vec<i32> = (1..=10).collect();
    let orders: Vec<(i32, i32)> = (1..10).map(|i| (i, i + 1)).collect();
    let lattice = build_lattice(&elements, &orders);

    assert_eq!(lattice.meet(&3, &5), Ok(Some(&3)));
    assert_eq!(lattice.meet(&6, &4), Ok(Some(&4)));
    assert_eq!(lattice.join(&3, &5), Ok(Some(&5)));
    assert_eq!(lattice.join(&6, &4), Ok(Some(&6)));
    assert!(lattice.is_lattice());
}

#[test]
fn subsets_meet_at_the_intersection() {
    let lattice = build_subsets_of_xyz();

    assert_eq!(lattice.meet(&"x", &"y"), Ok(Some(&"0")));
    assert_eq!(lattice.meet(&"x", &"xy"), Ok(Some(&"x")));
    assert_eq!(lattice.meet(&"x", &"z"), Ok(Some(&"0")));
    assert_eq!(lattice.meet(&"xz", &"yz"), Ok(Some(&"z")));
    assert_eq!(lattice.meet(&"xy", &"xyz"), Ok(Some(&"xy")));
}

#[test]
fn subsets_join_at_the_union() {
    let lattice = build_subsets_of_xyz();

    assert_eq!(lattice.join(&"x", &"y"), Ok(Some(&"xy")));
    assert_eq!(lattice.join(&"x", &"xy"), Ok(Some(&"xy")));
    assert_eq!(lattice.join(&"x", &"z"), Ok(Some(&"xz")));
    assert_eq!(lattice.join(&"xz", &"yz"), Ok(Some(&"xyz")));
    assert_eq!(lattice.join(&"xy", &"xyz"), Ok(Some(&"xyz")));
    assert!(lattice.is_lattice());
}

#[test]
fn crown_has_no_meet_below_and_no_join_above() {
    let lattice = build_crown();

    assert_eq!(lattice.join(&3, &4), Ok(None));
    assert_eq!(lattice.join(&1, &2), Ok(None));
    assert_eq!(lattice.meet(&3, &4), Ok(None));
    assert!(!lattice.is_lattice());
}

#[test]
fn several_minimal_upper_bounds_mean_no_join() {
    let lattice = build_crown_over_a_bottom();

    assert_eq!(lattice.join(&1, &2), Ok(None));
    assert_eq!(lattice.meet(&1, &2), Ok(Some(&0)));
    assert_eq!(lattice.meet(&3, &4), Ok(None));
    assert!(!lattice.is_lattice());
}

#[test]
fn meet_and_join_refuse_a_non_member_first_argument_first() {
    let lattice = build_lattice(&[1, 2], &[(1, 2)]);

    assert_eq!(lattice.meet(&3, &4), Err(OrderError::NotAMember(3)));
    assert_eq!(lattice.join(&1, &4), Err(OrderError::NotAMember(4)));
}

#[test]
fn missing_bounds_list_each_ordered_pair_meet_first() {
    let lattice = build_crown_over_a_bottom();

    let missing: Vec<MissingBound<'_, i32>> = lattice.missing_bounds().collect();

    assert_eq!(
        missing,
        [
            MissingBound::Join(&1, &2),
            MissingBound::Join(&2, &1),
            MissingBound::Meet(&3, &4),
            MissingBound::Join(&3, &4),
            MissingBound::Meet(&4, &3),
            MissingBound::Join(&4, &3),
        ]
    );
}

#[test]
fn missing_bounds_of_the_crown_name_every_pair_across_the_levels() {
    let lattice = build_crown();

    let missing: Vec<MissingBound<'_, i32>> = lattice.missing_bounds().collect();

    assert_eq!(
        missing,
        [
            MissingBound::Meet(&1, &2),
            MissingBound::Join(&1, &2),
            MissingBound::Meet(&2, &1),
            MissingBound::Join(&2, &1),
            MissingBound::Meet(&3, &4),
            MissingBound::Join(&3, &4),
            MissingBound::Meet(&4, &3),
            MissingBound::Join(&4, &3),
        ]
    );
}

#[test]
fn poset_view_reports_the_elements_in_iteration_order() {
    let lattice = build_subsets_of_xyz();

    assert_eq!(lattice.poset().len(), 8);
    assert_eq!(lattice.poset().iter().next(), Some(&"0"));
    assert_eq!(lattice.poset().iter().last(), Some(&"xyz"));
}
