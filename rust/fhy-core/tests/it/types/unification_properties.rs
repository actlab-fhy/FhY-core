//! Properties of binding, substitution and unification over random
//! templated arrays.

use crate::support::hashing::hash_of;
use crate::support::types::Equivalent;
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
        prop_assert!(left.is_equivalent(&right));
    }
}

proptest! {
    /// The environment's tables persist: a sequence of `with_*` calls, some
    /// replacing a binding, agrees at every step with a map built afresh,
    /// and leaves every earlier environment as it was.
    #[test]
    fn an_environment_agrees_with_a_map_of_its_bindings(
        operations in prop::collection::vec((0_usize..12, 0_i64..5), 0..80),
    ) {
        let names: Vec<Identifier> = (0..12).map(|index| Identifier::new(&format!("N{index}"))).collect();
        let mut model: std::collections::HashMap<Identifier, Expression> = std::collections::HashMap::new();
        let mut history = vec![(TypeUnificationEnvironment::new(), model.clone())];
        for (name, value) in operations {
            let (environment, _) = history.last().expect("an environment");
            let next = environment.with_expression_binding(names[name].clone(), Expression::from(value));
            model.insert(names[name].clone(), Expression::from(value));
            history.push((next, model.clone()));
        }

        for (environment, model) in &history {
            let rebuilt = TypeUnificationEnvironment::from_bindings(
                std::collections::HashMap::new(),
                std::collections::HashMap::new(),
                model.clone(),
            );
            prop_assert_eq!(environment.expression_bindings().len(), model.len());
            for name in &names {
                prop_assert_eq!(environment.expression_binding(name), model.get(name));
            }
            let listed: std::collections::HashMap<Identifier, Expression> = environment
                .expression_bindings()
                .map(|(identifier, value)| (identifier.clone(), value.clone()))
                .collect();
            prop_assert_eq!(&listed, model);
            prop_assert_eq!(environment, &rebuilt);
            prop_assert_eq!(hash_of(environment), hash_of(&rebuilt));
            prop_assert!(environment.is_structurally_equivalent(&rebuilt).expect("no extension"));
        }
    }
}

/// Return `expression` substituted as the definition states it, by
/// recursion: each bound variable becomes its binding substituted in turn,
/// and a variable already on `chain` stays.
fn reference_substitution(
    expression: &Expression,
    environment: &TypeUnificationEnvironment,
    chain: &mut Vec<Identifier>,
) -> Expression {
    let mut replacements = std::collections::HashMap::new();
    for identifier in expression.free_identifiers() {
        if chain.contains(&identifier) {
            continue;
        }
        let Some(bound) = environment.expression_binding(&identifier) else {
            continue;
        };
        chain.push(identifier.clone());
        let replacement = reference_substitution(bound, environment, chain);
        chain.pop();
        replacements.insert(identifier, replacement);
    }
    if replacements.is_empty() {
        return expression.clone();
    }
    expression.substitute(&replacements).expect("no piecewise")
}

/// Return the sum of the variables `terms` names plus `constant`.
fn sum_of(names: &[Identifier], terms: &[usize], constant: i64) -> Expression {
    terms.iter().fold(Expression::from(constant), |sum, index| {
        sum + Expression::from(names[*index].clone())
    })
}

proptest! {
    /// Substituting through the bindings, cycles included, gives the form
    /// the recursive definition gives, however the memo reuses forms.
    #[test]
    fn substitution_agrees_with_its_recursive_definition(
        bindings in prop::collection::vec(
            prop::option::of((prop::collection::vec(0_usize..6, 0..3), 0_i64..3)),
            6,
        ),
        root in prop::collection::vec(0_usize..6, 1..4),
    ) {
        let names: Vec<Identifier> = (0..6).map(|index| Identifier::new(&format!("N{index}"))).collect();
        let environment = bindings.iter().enumerate().fold(
            TypeUnificationEnvironment::new(),
            |environment, (index, binding)| match binding {
                Some((terms, constant)) => environment
                    .with_expression_binding(names[index].clone(), sum_of(&names, terms, *constant)),
                None => environment,
            },
        );
        let expression = sum_of(&names, &root, 0);
        let pattern = array(DataType::Primitive(CoreDataType::Int32), [Dimension::Expression(expression.clone())]);

        let substituted = pattern.substitute_template(&environment).expect("substitutes");

        let expected = array(
            DataType::Primitive(CoreDataType::Int32),
            [Dimension::Expression(reference_substitution(&expression, &environment, &mut Vec::new()))],
        );
        prop_assert_eq!(substituted, expected);
    }
}
