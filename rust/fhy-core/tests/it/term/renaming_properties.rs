//! Property tests for `AlphaRenaming` with binder frames: correspondence is
//! symmetric under the inverse renaming, agrees with a de Bruijn reading of
//! the frames, and leaving a frame undoes entering it; the hash agrees with
//! equality; and pairing two binder lists accepts exactly the lists of one
//! length that repeat no identifier.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use fhy_core::identifier::Identifier;
use fhy_core::term::AlphaRenaming;
use proptest::prelude::*;

use crate::support::hashing::hash_of;

/// The identifiers the generated renamings map among, few enough that
/// frames and the free renaming often share keys and images.
static POOL: LazyLock<[Identifier; 4]> = LazyLock::new(|| {
    [
        Identifier::new("p0"),
        Identifier::new("p1"),
        Identifier::new("p2"),
        Identifier::new("p3"),
    ]
});

/// Return the injective map keeping, of `pairs` of pool indices, each pair
/// whose key and image no earlier kept pair has.
fn build_injective_map(pairs: &[(usize, usize)]) -> HashMap<Identifier, Identifier> {
    let mut map = HashMap::new();
    let mut images = HashSet::new();
    for &(key, image) in pairs {
        let (key, image) = (&POOL[key], &POOL[image]);
        if !map.contains_key(key) && images.insert(image.clone()) {
            map.insert(key.clone(), image.clone());
        }
    }
    map
}

fn invert(map: &HashMap<Identifier, Identifier>) -> HashMap<Identifier, Identifier> {
    map.iter()
        .map(|(key, image)| (image.clone(), key.clone()))
        .collect()
}

fn build_renaming(
    free: &HashMap<Identifier, Identifier>,
    frames: &[HashMap<Identifier, Identifier>],
) -> AlphaRenaming {
    let mut renaming = AlphaRenaming::try_new(free.clone()).expect("the map is injective");
    for frame in frames {
        renaming
            .enter_binder(frame.clone())
            .expect("the frame is injective");
    }
    renaming
}

/// Return the innermost-first index of the frame that binds `identifier`
/// by `is_bound_by`, its de Bruijn index, or `None` when no frame binds it.
fn find_binding_frame(
    frames: &[HashMap<Identifier, Identifier>],
    identifier: &Identifier,
    is_bound_by: impl Fn(&HashMap<Identifier, Identifier>, &Identifier) -> bool,
) -> Option<usize> {
    frames
        .iter()
        .rev()
        .position(|frame| is_bound_by(frame, identifier))
}

/// Return whether `left` corresponds to `right` in the reading of the
/// frames as binders: both are bound by the same frame and paired there, or
/// neither is bound and the free renaming relates them.
fn is_corresponding_by_binding_frames(
    free: &HashMap<Identifier, Identifier>,
    frames: &[HashMap<Identifier, Identifier>],
    left: &Identifier,
    right: &Identifier,
) -> bool {
    let left_frame = find_binding_frame(frames, left, |frame, identifier| {
        frame.contains_key(identifier)
    });
    let right_frame = find_binding_frame(frames, right, |frame, identifier| {
        frame.values().any(|image| image == identifier)
    });
    match (left_frame, right_frame) {
        (Some(left_level), Some(right_level)) => {
            left_level == right_level && frames[frames.len() - 1 - left_level][left] == *right
        }
        (None, None) => match free.get(left) {
            Some(image) => image == right,
            None => !free.values().any(|image| image == right) && left == right,
        },
        _ => false,
    }
}

/// Return `map` rebuilt by inserting its pairs in the reverse of its
/// iteration order.
fn reverse_insertion(map: &HashMap<Identifier, Identifier>) -> HashMap<Identifier, Identifier> {
    let mut pairs: Vec<_> = map
        .iter()
        .map(|(key, image)| (key.clone(), image.clone()))
        .collect();
    pairs.reverse();
    let mut rebuilt = HashMap::with_capacity(pairs.len());
    for (key, image) in pairs {
        rebuilt.insert(key, image);
    }
    rebuilt
}

fn build_pairs_strategy() -> impl Strategy<Value = Vec<(usize, usize)>> {
    prop::collection::vec((0..POOL.len(), 0..POOL.len()), 0..4)
}

fn build_frames_strategy() -> impl Strategy<Value = Vec<Vec<(usize, usize)>>> {
    prop::collection::vec(build_pairs_strategy(), 0..4)
}

proptest! {
    /// Test `left` corresponds to `right` exactly when `right` corresponds
    /// to `left` under the renaming with every frame and the free renaming
    /// inverted.
    #[test]
    fn alpha_renaming_is_corresponding_is_symmetric_under_the_inverse(
        free_pairs in build_pairs_strategy(),
        frame_pairs in build_frames_strategy(),
        left in 0..POOL.len(),
        right in 0..POOL.len(),
    ) {
        let free = build_injective_map(&free_pairs);
        let frames: Vec<_> = frame_pairs.iter().map(|pairs| build_injective_map(pairs)).collect();
        let inverse_frames: Vec<_> = frames.iter().map(invert).collect();
        let renaming = build_renaming(&free, &frames);
        let inverse = build_renaming(&invert(&free), &inverse_frames);

        let forward = renaming.is_corresponding(&POOL[left], &POOL[right]);
        let backward = inverse.is_corresponding(&POOL[right], &POOL[left]);

        prop_assert_eq!(forward, backward);
    }

    /// Test correspondence agrees with reading the frames as binders: two
    /// bound identifiers correspond when one frame binds and pairs both, and
    /// two free ones when the free renaming relates them.
    #[test]
    fn alpha_renaming_is_corresponding_agrees_with_the_binding_frames(
        free_pairs in build_pairs_strategy(),
        frame_pairs in build_frames_strategy(),
        left in 0..POOL.len(),
        right in 0..POOL.len(),
    ) {
        let free = build_injective_map(&free_pairs);
        let frames: Vec<_> = frame_pairs.iter().map(|pairs| build_injective_map(pairs)).collect();
        let renaming = build_renaming(&free, &frames);

        let corresponding = renaming.is_corresponding(&POOL[left], &POOL[right]);

        prop_assert_eq!(
            corresponding,
            is_corresponding_by_binding_frames(&free, &frames, &POOL[left], &POOL[right])
        );
    }

    /// Test leaving a frame just entered restores the renaming.
    #[test]
    fn alpha_renaming_leave_binder_undoes_enter_binder(
        free_pairs in build_pairs_strategy(),
        frame_pairs in build_frames_strategy(),
        entered_pairs in build_pairs_strategy(),
    ) {
        let free = build_injective_map(&free_pairs);
        let frames: Vec<_> = frame_pairs.iter().map(|pairs| build_injective_map(pairs)).collect();
        let before = build_renaming(&free, &frames);
        let mut renaming = before.clone();
        renaming
            .enter_binder(build_injective_map(&entered_pairs))
            .expect("the frame is injective");

        prop_assert!(renaming.leave_binder());
        prop_assert_eq!(renaming, before);
    }

    /// Test two renamings built from the same pairs, in opposite orders,
    /// are equal and hash alike, and that renamings the generator builds
    /// equal hash alike.
    #[test]
    fn alpha_renaming_hash_agrees_with_equality(
        free_pairs in build_pairs_strategy(),
        frame_pairs in build_frames_strategy(),
        other_free_pairs in build_pairs_strategy(),
        other_frame_pairs in build_frames_strategy(),
    ) {
        let free = build_injective_map(&free_pairs);
        let frames: Vec<_> = frame_pairs.iter().map(|pairs| build_injective_map(pairs)).collect();
        let reordered_free: HashMap<_, _> = reverse_insertion(&free);
        let reordered_frames: Vec<HashMap<_, _>> = frames
            .iter()
            .map(reverse_insertion)
            .collect();
        let renaming = build_renaming(&free, &frames);
        let reordered = build_renaming(&reordered_free, &reordered_frames);
        let other = build_renaming(
            &build_injective_map(&other_free_pairs),
            &other_frame_pairs.iter().map(|pairs| build_injective_map(pairs)).collect::<Vec<_>>(),
        );

        prop_assert_eq!(&renaming, &reordered);
        prop_assert_eq!(hash_of(&renaming), hash_of(&reordered));
        if renaming == other {
            prop_assert_eq!(hash_of(&renaming), hash_of(&other));
        }
    }

    /// Test `enter_binders` accepts exactly two lists of one length that
    /// repeat no identifier, and then pushes the frame of their pairs.
    #[test]
    fn alpha_renaming_enter_binders_accepts_exactly_the_lists_without_repeats(
        left in prop::collection::vec(0..POOL.len(), 0..4),
        right in prop::collection::vec(0..POOL.len(), 0..4),
    ) {
        let left: Vec<Identifier> = left.into_iter().map(|index| POOL[index].clone()).collect();
        let right: Vec<Identifier> = right.into_iter().map(|index| POOL[index].clone()).collect();
        let is_distinct = |list: &[Identifier]| list.iter().collect::<HashSet<_>>().len() == list.len();
        let mut renaming = AlphaRenaming::default();

        let result = renaming.enter_binders(&left, &right);

        let pairs = left.len() == right.len() && is_distinct(&left) && is_distinct(&right);
        prop_assert_eq!(result.is_ok(), pairs);
        if pairs {
            let mut expected = AlphaRenaming::default();
            expected
                .enter_binder(left.iter().cloned().zip(right.iter().cloned()).collect::<HashMap<_, _>>())
                .expect("distinct lists pair injectively");
            prop_assert_eq!(renaming, expected);
        } else {
            prop_assert_eq!(renaming, AlphaRenaming::default());
        }
    }
}
