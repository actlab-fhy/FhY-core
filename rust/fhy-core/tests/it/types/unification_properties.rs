//! Properties of binding, substitution and unification over random
//! templated arrays.

use crate::support::hashing::hash_of;
use crate::support::types::{array, literal_dimension, template};

use fhy_core::expression::Expression;
use fhy_core::identifier::Identifier;
use fhy_core::types::{CoreDataType, DataType, Dimension, Type, TypeUnificationEnvironment};
use proptest::prelude::*;

/// Return a random concrete array: a sized core data type over up to four
/// literal extents.
fn concrete_array_strategy() -> impl Strategy<Value = (CoreDataType, Vec<u32>)> {
    let data_types = prop::sample::select(vec![
        CoreDataType::Uint8,
        CoreDataType::Int16,
        CoreDataType::Int32,
        CoreDataType::Float32,
        CoreDataType::Complex64,
        CoreDataType::Bool,
    ]);
    (data_types, prop::collection::vec(1_u32..64, 0..4))
}

/// Return the pattern of a concrete array's rank: a template data type over
/// shape variables, where `shared` makes every other dimension reuse the
/// previous variable.
fn pattern_for(rank: usize, shared: bool) -> (Type, Vec<Identifier>) {
    let t = Identifier::new("T");
    let mut variables: Vec<Identifier> = Vec::new();
    let mut shape = Vec::new();
    for position in 0..rank {
        let variable = if shared && position % 2 == 1 {
            variables[position - 1].clone()
        } else {
            Identifier::new("N")
        };
        shape.push(Dimension::Expression(Expression::from(variable.clone())));
        variables.push(variable);
    }
    (array(template(&t), shape), variables)
}

fn concrete(data_type: CoreDataType, extents: &[u32]) -> Type {
    array(
        DataType::Primitive(data_type),
        extents.iter().map(|&extent| literal_dimension(extent)),
    )
}

proptest! {
    #[test]
    fn binding_then_substituting_gives_back_the_actual((data_type, extents) in concrete_array_strategy()) {
        let actual = concrete(data_type, &extents);
        let (pattern, _) = pattern_for(extents.len(), false);

        let environment = pattern.bind_template(&actual, &TypeUnificationEnvironment::new()).expect("binds");

        prop_assert_eq!(pattern.substitute_template(&environment).expect("substitutes"), actual);
    }

    #[test]
    fn a_shared_shape_variable_binds_exactly_when_its_extents_agree(
        (data_type, extents) in concrete_array_strategy(),
    ) {
        let actual = concrete(data_type, &extents);
        let (pattern, _) = pattern_for(extents.len(), true);
        let agree = extents.chunks(2).all(|pair| pair.len() == 1 || pair[0] == pair[1]);

        let result = pattern.bind_template(&actual, &TypeUnificationEnvironment::new());

        prop_assert_eq!(result.is_ok(), agree);
    }

    #[test]
    fn unification_binds_on_either_side_and_its_result_matches_both(
        (data_type, extents) in concrete_array_strategy(),
    ) {
        let actual = concrete(data_type, &extents);
        let (pattern, _) = pattern_for(extents.len(), false);
        let empty = TypeUnificationEnvironment::new();

        let (forward, forward_environment) = pattern.unify(&actual, &empty).expect("unifies");
        let (backward, backward_environment) = actual.unify(&pattern, &empty).expect("unifies");

        prop_assert_eq!(&forward, &actual);
        prop_assert_eq!(&backward, &actual);
        prop_assert_eq!(&forward_environment, &backward_environment);
        prop_assert_eq!(pattern.substitute_template(&forward_environment).expect("substitutes"), actual);
    }

    #[test]
    fn equal_types_hash_alike(
        (data_type, extents) in concrete_array_strategy(),
    ) {
        let left = concrete(data_type, &extents);
        let right = concrete(data_type, &extents);

        prop_assert_eq!(&left, &right);
        prop_assert_eq!(hash_of(&left), hash_of(&right));
        prop_assert!(left.is_structurally_equivalent(&right));
    }
}
