//! Tests for `Rng`: the stream SplitMix64 gives for a seed, the draws
//! `below`, `below_big` and `shuffle` take from it, `split`, cloning, and
//! the wire form a stream resumes from.
//!
//! The pinned numbers come from an independent reference implementation of
//! the documented algorithms (Vigna's SplitMix64, Lemire's method, the
//! limb-drawing rejection and Fisher-Yates from the last index down); the
//! first number of seed 0 is SplitMix64's published `0xe220a8397b1dcdaf`.

use std::num::NonZeroU64;

use fhy_core::search_space::Rng;
use num_bigint::BigUint;
use rstest::rstest;

/// Return the next `count` numbers of `rng`.
fn draw(rng: &mut Rng, count: usize) -> Vec<u64> {
    (0..count).map(|_| rng.next_u64()).collect()
}

/// Return `count` draws of `rng.below(bound)`.
fn draw_below(rng: &mut Rng, bound: u64, count: usize) -> Vec<u64> {
    let bound = NonZeroU64::new(bound).expect("a positive bound");
    (0..count).map(|_| rng.below(bound)).collect()
}

/// Test the stream of seed 0 is SplitMix64's.
#[test]
fn rng_stream_of_seed_zero_is_splitmix64s() {
    let mut rng = Rng::new(0);

    let numbers = draw(&mut rng, 8);

    assert_eq!(
        numbers,
        [
            0xe220_a839_7b1d_cdaf,
            0x6e78_9e6a_a1b9_65f4,
            0x06c4_5d18_8009_454f,
            0xf88b_b8a8_724c_81ec,
            0x1b39_896a_51a8_749b,
            0x53cb_9f0c_747e_a2ea,
            0x2c82_9abe_1f45_32e1,
            0xc584_133a_c916_ab3c,
        ]
    );
}

/// Test the streams of seed 1 and of the largest seed, whose state wraps.
#[rstest]
#[case::one(1, [0x910a_2dec_8902_5cc1, 0xbeeb_8da1_658e_ec67, 0xf893_a2ee_fb32_555e])]
#[case::largest(u64::MAX, [0xe4d9_7177_1b65_2c20, 0xe99f_f867_dbf6_82c9, 0x382f_f84c_b272_81e9])]
fn rng_stream_of_a_seed_is_pinned(#[case] seed: u64, #[case] expected: [u64; 3]) {
    let mut rng = Rng::new(seed);

    let numbers = draw(&mut rng, 3);

    assert_eq!(numbers, expected);
}

/// Test `below` maps each number of the stream to its pinned draw.
#[rstest]
#[case::one(1, vec![0, 0, 0, 0, 0, 0, 0, 0])]
#[case::three(3, vec![2, 0, 0, 1, 0, 2, 0, 2])]
#[case::six(6, vec![4, 0, 1, 2, 0, 5, 1, 4])]
#[case::largest(
    u64::MAX,
    vec![
        13_679_457_532_755_275_412,
        2_949_826_092_126_892_290,
        5_139_283_748_462_763_857,
        6_349_198_060_258_255_763,
        701_532_786_141_963_249,
        16_015_981_125_662_989_061,
        4_028_864_712_777_624_924,
        14_769_051_326_987_775_907,
    ]
)]
fn rng_below_draws_are_pinned(#[case] bound: u64, #[case] expected: Vec<u64>) {
    let mut rng = Rng::new(42);

    let draws = draw_below(&mut rng, bound, 8);

    assert_eq!(draws, expected);
}

/// Test `below` draws a thousand values as pinned, from a third seed.
#[test]
fn rng_below_a_thousand_draws_are_pinned() {
    let mut rng = Rng::new(123);

    let draws = draw_below(&mut rng, 1000, 10);

    assert_eq!(draws, [706, 976, 859, 686, 686, 667, 999, 482, 619, 140]);
}

/// Test `below` rejects the numbers whose low half falls under the
/// threshold, drawing again: a bound just above `2^63` rejects often, and
/// the stream after six draws is where 117 numbers, not six, leave it.
#[test]
fn rng_below_rejects_and_draws_again_below_the_threshold() {
    let mut rng = Rng::new(0);

    let draws = draw_below(&mut rng, (1 << 63) + 1, 6);

    assert_eq!(
        draws,
        [
            243_808_509_735_772_839,
            8_954_805_688_390_271_222,
            980_875_101_213_047_373,
            1_603_648_013_000_153_456,
            7_116_260_932_800_173_470,
            2_266_080_580_496_311_649,
        ]
    );
    assert_eq!(rng.next_u64(), 0xf3b8_488c_368c_b0a6);
}

/// Test `below` never reaches its bound, over many draws of small bounds.
#[test]
fn rng_below_stays_under_its_bound() {
    let mut rng = Rng::new(9);

    let out_of_range = (1..=64_u64)
        .flat_map(|bound| {
            draw_below(&mut rng, bound, 200)
                .into_iter()
                .map(move |draw| (draw, bound))
        })
        .find(|(draw, bound)| draw >= bound);

    assert_eq!(out_of_range, None);
}

/// Test `below(6)` is uniform: over 60 000 draws of one fixed seed, the
/// chi-square statistic of the six counts stays under 20.52, the 0.001
/// critical value with five degrees of freedom.
#[test]
fn rng_below_six_is_uniform_over_a_fixed_seed() {
    let mut rng = Rng::new(2024);
    let mut counts = [0_u64; 6];

    for draw in draw_below(&mut rng, 6, 60_000) {
        counts[usize::try_from(draw).expect("a small draw")] += 1;
    }

    let expected = 10_000.0;
    let statistic: f64 = counts
        .iter()
        .map(|&count| {
            let difference = f64::from(u32::try_from(count).expect("a small count")) - expected;
            difference * difference / expected
        })
        .sum();
    assert!(statistic < 20.52, "chi-square {statistic} over {counts:?}");
}

/// Test `below_big` draws numbers wider than a limb as pinned, taking two
/// numbers of the stream per draw when none is rejected.
#[test]
fn rng_below_big_draws_wide_numbers_as_pinned() {
    let mut rng = Rng::new(42);
    let bound = BigUint::from(10_u32).pow(30);

    let draws: Vec<BigUint> = (0..4).map(|_| rng.below_big(&bound)).collect();

    let expected: Vec<BigUint> = [
        "292897267883935188527843339925",
        "811006502546476498273989829618",
        "646668284273496457662110203349",
        "259944956782528464820904019686",
    ]
    .iter()
    .map(|text| text.parse().expect("a decimal"))
    .collect();
    assert_eq!(draws, expected);
    assert_eq!(rng.next_u64(), 0xaa47_e31c_02e7_8edc);
}

/// Test `below_big` of `2^64` takes two limbs and keeps one bit of the
/// second.
#[test]
fn rng_below_big_of_two_to_the_sixty_fourth_keeps_one_bit_of_its_top_limb() {
    let mut rng = Rng::new(42);
    let bound = BigUint::from(1_u8) << 64_u32;

    let draws: Vec<BigUint> = (0..4).map(|_| rng.below_big(&bound)).collect();

    let expected: Vec<BigUint> = [
        5_139_283_748_462_763_858_u64,
        701_532_786_141_963_250,
        4_028_864_712_777_624_925,
        6_270_620_877_612_482_005,
    ]
    .into_iter()
    .map(BigUint::from)
    .collect();
    assert_eq!(draws, expected);
}

/// Test `below_big(1)` answers 0, drawing one number per attempt and
/// rejecting the ones whose kept bit is set.
#[test]
fn rng_below_big_of_one_answers_zero_and_rejects_odd_numbers() {
    let mut rng = Rng::new(42);

    let draws: Vec<BigUint> = (0..3)
        .map(|_| rng.below_big(&BigUint::from(1_u8)))
        .collect();

    assert_eq!(draws, vec![BigUint::from(0_u8); 3]);
    assert_eq!(rng.next_u64(), 0xde44_31fa_3c80_db06);
}

/// Test `below_big` refuses a zero bound with its documented panic.
#[test]
#[should_panic(expected = "the bound of below_big must be positive")]
fn rng_below_big_panics_on_a_zero_bound() {
    let mut rng = Rng::new(1);

    let _never = rng.below_big(&BigUint::from(0_u8));
}

/// Test `shuffle` moves the items as pinned.
#[rstest]
#[case::ten_from_seven(7, 10, vec![9, 5, 8, 6, 1, 2, 4, 7, 0, 3])]
#[case::four_from_zero(0, 4, vec![2, 0, 1, 3])]
#[case::four_from_one(1, 4, vec![0, 1, 3, 2])]
fn rng_shuffle_is_pinned(#[case] seed: u64, #[case] length: u32, #[case] expected: Vec<u32>) {
    let mut rng = Rng::new(seed);
    let mut items: Vec<u32> = (0..length).collect();

    rng.shuffle(&mut items);

    assert_eq!(items, expected);
}

/// Test `shuffle` draws nothing for zero or one item.
#[rstest]
#[case::empty(0)]
#[case::single(1)]
fn rng_shuffle_of_at_most_one_item_draws_nothing(#[case] length: u32) {
    let mut rng = Rng::new(5);
    let mut items: Vec<u32> = (0..length).collect();

    rng.shuffle(&mut items);

    assert_eq!(items, (0..length).collect::<Vec<_>>());
    assert_eq!(rng.next_u64(), Rng::new(5).next_u64());
}

/// Test `split` seeds a child with the parent's next number, and the parent
/// continues after it.
#[test]
fn rng_split_seeds_the_child_with_the_next_number() {
    let mut parent = Rng::new(7);

    let mut child = parent.split();

    assert_eq!(
        draw(&mut child, 3),
        [
            0xb8b4_c297_7eab_ce45,
            0xa653_05fd_338e_c8fe,
            0x8ca3_cbb6_ca63_129b
        ]
    );
    assert_eq!(
        draw(&mut parent, 2),
        [0x044c_3cd7_f43c_661c, 0xe698_4080_bab1_2a02]
    );
}

/// Test a clone continues the same stream, independently of the original.
#[test]
fn rng_clone_continues_the_same_stream() {
    let mut rng = Rng::new(11);
    let _skipped = draw(&mut rng, 3);

    let mut copy = rng.clone();

    assert_eq!(draw(&mut copy, 4), draw(&mut rng, 4));
}

/// Test the wire form names the algorithm and the state, and a decoded
/// generator resumes the stream where it was written.
#[test]
fn rng_round_trips_through_json_and_resumes_its_stream() {
    let mut rng = Rng::new(0);
    let _skipped = draw(&mut rng, 2);

    let text = serde_json::to_string(&rng).expect("a generator serializes");
    let mut decoded: Rng = serde_json::from_str(&text).expect("the text decodes");

    let value: serde_json::Value = serde_json::from_str(&text).expect("the text is JSON");
    assert_eq!(value["algorithm"], "splitmix64");
    assert_eq!(draw(&mut decoded, 6), draw(&mut rng, 6));
}

/// Test decoding refuses a generator of another algorithm.
#[test]
fn rng_decoding_refuses_another_algorithm() {
    let text = r#"{"algorithm": "mt19937", "state": 5}"#;

    let result = serde_json::from_str::<Rng>(text);

    let error = result.expect_err("another algorithm is refused");
    assert!(error.to_string().contains("mt19937"), "{error}");
}

/// Test the generator's name is the one its wire form records.
#[test]
fn rng_algorithm_is_splitmix64() {
    let text = serde_json::to_string(&Rng::new(3)).expect("a generator serializes");

    let value: serde_json::Value = serde_json::from_str(&text).expect("the text is JSON");

    assert_eq!(value["algorithm"], Rng::ALGORITHM);
}
