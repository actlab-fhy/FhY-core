//! Tests for the mapping between function sorts and core data types,
//! mirroring `tests/types/checking/test_sort_compatibility.py`.

use fhy_core::expression::FunctionSort;
use fhy_core::types::CoreDataType;

/// The core data types each sort admits.
fn admitted(sort: FunctionSort) -> Vec<CoreDataType> {
    CoreDataType::all()
        .filter(|core| match sort {
            FunctionSort::Bool => *core == CoreDataType::Bool,
            FunctionSort::Nat => core.is_unsigned(),
            FunctionSort::Int => core.is_integral(),
            FunctionSort::Real => core.is_integral() || core.is_real_float(),
        })
        .collect()
}

#[test]
fn each_sort_admits_exactly_its_family() {
    use CoreDataType::{
        Bool, Float, Float16, Float32, Float64, Int, Int8, Int16, Int32, Int64, Uint, Uint8,
        Uint16, Uint32,
    };
    assert_eq!(admitted(FunctionSort::Bool), [Bool]);
    assert_eq!(admitted(FunctionSort::Nat), [Uint, Uint8, Uint16, Uint32]);
    assert_eq!(
        admitted(FunctionSort::Int),
        [Uint, Int, Uint8, Uint16, Uint32, Int8, Int16, Int32, Int64]
    );
    assert_eq!(
        admitted(FunctionSort::Real),
        [
            Uint, Int, Float, Uint8, Uint16, Uint32, Int8, Int16, Int32, Int64, Float16, Float32,
            Float64
        ]
    );
}

#[test]
fn compatibility_agrees_with_the_families_over_every_pair() {
    for sort in [
        FunctionSort::Bool,
        FunctionSort::Nat,
        FunctionSort::Int,
        FunctionSort::Real,
    ] {
        let family = admitted(sort);
        for core in CoreDataType::all() {
            assert_eq!(
                core.is_compatible_with_sort(sort),
                family.contains(&core),
                "{core} {sort}"
            );
        }
    }
}

#[test]
fn complex_numbers_satisfy_no_sort() {
    for core in [
        CoreDataType::Complex32,
        CoreDataType::Complex64,
        CoreDataType::Complex128,
    ] {
        for sort in [
            FunctionSort::Bool,
            FunctionSort::Nat,
            FunctionSort::Int,
            FunctionSort::Real,
        ] {
            assert!(!core.is_compatible_with_sort(sort));
        }
    }
}

#[test]
fn each_sort_s_result_type_is_concrete_and_compatible() {
    let results = [
        (FunctionSort::Bool, CoreDataType::Bool),
        (FunctionSort::Nat, CoreDataType::Uint32),
        (FunctionSort::Int, CoreDataType::Int64),
        (FunctionSort::Real, CoreDataType::Float64),
    ];
    for (sort, core) in results {
        assert_eq!(CoreDataType::of_sort(sort), core);
        assert!(!core.is_weak());
        assert!(core.is_compatible_with_sort(sort));
    }
}
