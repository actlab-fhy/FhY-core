//! Decision rules of params not otherwise pinned by a suite shared with
//! Python: both spellings of a bound, the registry of a context, categorical
//! subsets, assignment equivalence per value kind, and three brute-force
//! comparisons: the interval hull, finite and integer intersection, and
//! numeric feasibility and subset against a real solver.

use fhy_core::constraint::{Binding, Bindings, Constraint, EquationConstraint, Outcome, Value};
use fhy_core::expression::registry::{FunctionRegistry, NativeConstant};
use fhy_core::expression::{BigInt, Decimal, Expression, FunctionName, FunctionSort, LiteralValue};
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    BoundSide, CategoricalDomain, Inclusivity, IntegerDomain, IntervalIntegerDomain, Operand,
    OrdinalDomain, Param, ParamAssignment, ParamBuildError, ParamContext, ParamDomain, Sign,
    ValueCheck, ZeroInclusion,
};
use fhy_core::solver::{SatResult, Solver};
use proptest::prelude::*;
use rstest::rstest;

use crate::support::constraint::{TestOpaque, int, text};
use crate::support::param::{
    RecordingParamObserver, at_least, at_most, boolean, context, float, in_set, not_in_set,
    real_solver, reference, scripted_solver,
};

/// Return whether `param` holds `value` valid, decided.
fn admits(param: &Param, value: i64, context: &ParamContext<'_>) -> Option<bool> {
    let environment = param
        .environment(Binding::Value(int(value)), &Bindings::new())
        .ok()?;
    match param.check_value(&environment, context).ok()? {
        ValueCheck::Valid => Some(true),
        ValueCheck::Undecided { .. } => None,
        _ => Some(false),
    }
}

// ---------------------------------------------------------------------------
// Both spellings of a bound
// ---------------------------------------------------------------------------

/// Return the bound `x <cmp> k` in the spelling `x <cmp> k` or `k <inverse>
/// x`.
fn bound(x: &Identifier, upper: bool, k: i64, reversed: bool) -> Constraint {
    let variable = reference(x);
    let literal = Expression::from(k);
    let expression = match (upper, reversed) {
        (true, false) => variable.less_equal(literal),
        (true, true) => literal.greater_equal(variable),
        (false, false) => variable.greater_equal(literal),
        (false, true) => literal.less_equal(variable),
    };
    Constraint::from(EquationConstraint::new(expression))
}

/// `x <= 5` and `5 >= x` bound an interval param alike, and so do `x >= 1`
/// and `1 <= x`: interval arithmetic reads either spelling.
#[rstest]
#[case::plain(false)]
#[case::reversed(true)]
fn a_bound_reads_the_same_in_either_spelling(#[case] reversed: bool) {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let x = Identifier::new("x");
    let param = Param::new(
        ParamDomain::from(IntervalIntegerDomain::new(
            Inclusivity::Inclusive,
            Sign::Any,
            ZeroInclusion::Included,
        )),
        x.clone(),
        [bound(&x, false, 1, reversed), bound(&x, true, 5, reversed)],
        &context,
    )
    .expect("two bounds");

    let negated = param.checked_neg(&context).expect("an interval");
    let shifted = param
        .checked_add(&Operand::Integer(BigInt::from(10)), &context)
        .expect("an interval");

    for value in -8..=18 {
        assert_eq!(
            admits(&negated, value, &context),
            Some((-5..=-1).contains(&value)),
            "-x at {value}"
        );
        assert_eq!(
            admits(&shifted, value, &context),
            Some((11..=15).contains(&value)),
            "x + 10 at {value}"
        );
    }
}

// ---------------------------------------------------------------------------
// The registry of a context
// ---------------------------------------------------------------------------

#[test]
fn a_context_with_a_registry_knows_its_native_constants() {
    let solver = Solver::new();
    let mut registry = FunctionRegistry::new();
    let tau = registry
        .register_constant(
            NativeConstant::new(
                FunctionName::new("tau_p").expect("a name"),
                FunctionSort::Int,
                7,
            )
            .expect("an int"),
        )
        .expect("registers");
    let bare = ParamContext::new(&solver);
    let with_registry = ParamContext::new(&solver).with_registry(&registry);
    let integer = || ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included));

    assert!(bare.registry().is_none());
    assert!(std::ptr::eq(
        with_registry.registry().expect("a registry"),
        &raw const registry
    ));
    assert!(!bare.is_native_constant(&tau));
    assert!(with_registry.is_native_constant(&tau));
    Param::new(integer(), tau.clone(), [], &bare).expect("a param without a registry");
    assert!(matches!(
        Param::new(integer(), tau.clone(), [], &with_registry),
        Err(ParamBuildError::NativeConstantVariable(variable)) if variable == tau
    ));
}

// ---------------------------------------------------------------------------
// Categorical subsets
// ---------------------------------------------------------------------------

fn categorical(values: &[&str]) -> ParamDomain {
    ParamDomain::from(
        CategoricalDomain::new(values.iter().map(|value| text(value)).collect()).expect("values"),
    )
}

#[rstest]
#[case::fewer_categories(categorical(&["a", "b"]), categorical(&["b", "c", "a"]), true)]
#[case::the_same_in_another_order(categorical(&["b", "a"]), categorical(&["a", "b"]), true)]
#[case::more_categories(categorical(&["a", "b", "c"]), categorical(&["a", "b"]), false)]
#[case::disjoint(categorical(&["a"]), categorical(&["b"]), false)]
#[case::against_an_ordinal(
    categorical(&["a"]),
    ParamDomain::from(OrdinalDomain::new(vec![text("a")]).expect("values")),
    false
)]
#[case::type_strict(
    ParamDomain::from(CategoricalDomain::new(vec![int(1)]).expect("values")),
    ParamDomain::from(CategoricalDomain::new(vec![boolean(true), text("1")]).expect("values")),
    false
)]
fn a_categorical_value_set_is_a_subset_of_one_holding_its_categories(
    #[case] own: ParamDomain,
    #[case] other: ParamDomain,
    #[case] expected: bool,
) {
    let solver = Solver::new();

    assert_eq!(
        own.is_value_set_subset(&other, &ParamContext::new(&solver))
            .ok(),
        Some(expected)
    );
}

// ---------------------------------------------------------------------------
// Assignment equivalence per value kind
// ---------------------------------------------------------------------------

fn decimal(text: &str) -> Value {
    Value::Decimal(text.parse::<Decimal>().expect("a decimal"))
}

/// Two assignments of one param are structurally equivalent exactly when
/// their values are equal type-strictly: of one kind, at every depth, so
/// `1` and `True` differ inside a tuple, the zeros are equal, and a NaN
/// equals nothing; `==` differs only for the NaN.
#[rstest]
#[case::bool(boolean(true), boolean(true), true)]
#[case::int(int(3), int(3), true)]
#[case::int_against_float(int(5), float(5.0), false)]
#[case::zeros(float(-0.0), float(0.0), true)]
#[case::nan(float(f64::NAN), float(f64::NAN), false)]
#[case::decimal(decimal("0.1"), decimal("0.10"), true)]
#[case::decimal_against_float(decimal("0.5"), float(0.5), false)]
#[case::str(text("a"), text("a"), true)]
#[case::tuple_of_zeros(Value::Tuple(vec![float(-0.0)]), Value::Tuple(vec![float(0.0)]), true)]
#[case::one_against_true_in_a_tuple(
    Value::Tuple(vec![int(1)]),
    Value::Tuple(vec![boolean(true)]),
    false
)]
#[case::tuple_order(Value::Tuple(vec![int(1), int(2)]), Value::Tuple(vec![int(2), int(1)]), false)]
#[case::frozenset_order(
    Value::FrozenSet(vec![int(1), int(2)]),
    Value::FrozenSet(vec![int(2), int(1), int(1)]),
    true
)]
#[case::one_against_true_in_a_frozenset(
    Value::FrozenSet(vec![int(1)]),
    Value::FrozenSet(vec![boolean(true)]),
    false
)]
#[case::opaque(
    TestOpaque::token(1).into_value(),
    TestOpaque::token(1).into_value(),
    true
)]
#[case::other_opaque(
    TestOpaque::token(1).into_value(),
    TestOpaque::token(2).into_value(),
    false
)]
fn assignment_equivalence_compares_values_type_strictly(
    #[case] left: Value,
    #[case] right: Value,
    #[case] equivalent: bool,
) {
    let solver = Solver::new();
    let param = Param::new(
        ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included)),
        Identifier::new("p"),
        [],
        &ParamContext::new(&solver),
    )
    .expect("a param");
    let is_nan = matches!(left, Value::Float(value) if value.is_nan());
    let left = ParamAssignment::new_unvalidated(param.clone(), left);
    let right = ParamAssignment::new_unvalidated(param, right);

    assert_eq!(left.is_structurally_equivalent(&right), equivalent);
    assert_eq!(right.is_structurally_equivalent(&left), equivalent);
    assert_eq!(left == right, equivalent || is_nan);
}

// ---------------------------------------------------------------------------
// Brute-force comparisons
// ---------------------------------------------------------------------------

/// The interval an interval param's bounds give, and its restriction.
#[derive(Debug, Clone, Copy)]
struct IntervalSpec {
    lower: Option<i64>,
    upper: Option<i64>,
    lower_inclusive: bool,
    upper_inclusive: bool,
    /// `Some(zero_included)` for a non-negative domain.
    natural: Option<bool>,
    prefer_inclusive: bool,
}

/// An interval operation's name and its arithmetic on values.
type Operation = (&'static str, fn(i64, i64) -> i64);

/// The half-width of the window unbounded sides sample.
const WINDOW: i64 = 60;

impl IntervalSpec {
    /// Return the least and greatest value the param admits, if bounded.
    fn effective(&self) -> (Option<i64>, Option<i64>) {
        let mut lower = self.lower.map(|lower| {
            if self.lower_inclusive {
                lower
            } else {
                lower + 1
            }
        });
        let upper = self.upper.map(|upper| {
            if self.upper_inclusive {
                upper
            } else {
                upper - 1
            }
        });
        if let Some(zero) = self.natural {
            let floor = i64::from(!zero);
            lower = Some(lower.map_or(floor, |lower| lower.max(floor)));
        }
        (lower, upper)
    }

    /// Return the values the param admits within the window.
    fn samples(&self) -> Vec<i64> {
        let (lower, upper) = self.effective();
        (lower.unwrap_or(-WINDOW)..=upper.unwrap_or(WINDOW)).collect()
    }

    fn build(&self, context: &ParamContext<'_>) -> Result<Param, String> {
        let (sign, zero) = match self.natural {
            Some(zero) => (Sign::NonNegative, ZeroInclusion::included_if(zero)),
            None => (Sign::Any, ZeroInclusion::Included),
        };
        let mut param = Param::new(
            ParamDomain::from(IntervalIntegerDomain::new(
                Inclusivity::inclusive_if(self.prefer_inclusive),
                sign,
                zero,
            )),
            Identifier::new("p"),
            Vec::new(),
            context,
        )
        .map_err(|error| error.to_string())?;
        if let Some(lower) = self.lower {
            param = param
                .with_bound(
                    &LiteralValue::Int(BigInt::from(lower)),
                    BoundSide::Lower,
                    self.lower_inclusive,
                    context,
                )
                .map_err(|error| error.to_string())?;
        }
        if let Some(upper) = self.upper {
            param = param
                .with_bound(
                    &LiteralValue::Int(BigInt::from(upper)),
                    BoundSide::Upper,
                    self.upper_inclusive,
                    context,
                )
                .map_err(|error| error.to_string())?;
        }
        Ok(param)
    }
}

fn interval_spec() -> impl Strategy<Value = IntervalSpec> {
    (
        prop::option::of(-4_i64..4),
        prop::option::of(-4_i64..6),
        any::<bool>(),
        any::<bool>(),
        prop::option::of(any::<bool>()),
        any::<bool>(),
    )
        .prop_map(
            |(lower, upper, lower_inclusive, upper_inclusive, natural, prefer_inclusive)| {
                let mut spec = IntervalSpec {
                    lower,
                    upper,
                    lower_inclusive,
                    upper_inclusive,
                    natural,
                    prefer_inclusive,
                };
                if let Some(zero) = natural {
                    // A natural domain's bound literal must be admissible.
                    if let Some(lower) = spec.lower {
                        let least = i64::from(zero != lower_inclusive);
                        spec.lower = Some(lower.max(least));
                    }
                    if let Some(upper) = spec.upper {
                        let least = i64::from(!zero) + i64::from(!upper_inclusive);
                        if upper < least {
                            spec.upper = Some(least + 3);
                        }
                    }
                }
                spec
            },
        )
        .prop_filter("a non-empty interval", |spec| match spec.effective() {
            (Some(lower), Some(upper)) => lower <= upper,
            _ => true,
        })
}

/// A finite param's side: its values and set constraints.
#[derive(Debug, Clone)]
struct FiniteSpec {
    values: Vec<i64>,
    sets: Vec<(Vec<i64>, bool)>,
}

impl FiniteSpec {
    fn build(&self, ordinal: bool, context: &ParamContext<'_>) -> Param {
        let x = Identifier::new("x");
        let values: Vec<Value> = self.values.iter().copied().map(int).collect();
        let domain = if ordinal {
            ParamDomain::from(OrdinalDomain::new(values).expect("values"))
        } else {
            ParamDomain::from(CategoricalDomain::new(values).expect("values"))
        };
        let constraints: Vec<Constraint> = self
            .sets
            .iter()
            .map(|(values, is_in)| {
                let values = values.iter().copied().map(int);
                if *is_in {
                    in_set(&x, values)
                } else {
                    not_in_set(&x, values)
                }
            })
            .collect();
        Param::new(domain, x, constraints, context).expect("set constraints")
    }

    fn holds(&self, value: i64) -> bool {
        self.values.contains(&value)
            && self
                .sets
                .iter()
                .all(|(values, is_in)| values.contains(&value) == *is_in)
    }
}

fn finite_spec() -> impl Strategy<Value = FiniteSpec> {
    (
        prop::collection::btree_set(0_i64..8, 1..6),
        prop::collection::vec((prop::collection::vec(0_i64..8, 0..4), any::<bool>()), 0..3),
    )
        .prop_map(|(values, sets)| FiniteSpec {
            values: values.into_iter().collect(),
            sets,
        })
}

/// A numeric param's side for the real solver: bounds, set constraints of
/// which the first may hold a member of another kind, and the naturals.
#[derive(Debug, Clone)]
struct NumericSpec {
    lower: i64,
    upper: i64,
    sets: Vec<(Vec<i64>, bool)>,
    natural: bool,
    extra: Option<Value>,
}

impl NumericSpec {
    fn build(&self, name: &str, context: &ParamContext<'_>) -> Param {
        let x = Identifier::new(name);
        let mut constraints = vec![at_least(&x, self.lower), at_most(&x, self.upper)];
        for (index, (values, is_in)) in self.sets.iter().enumerate() {
            let mut values: Vec<Value> = values.iter().copied().map(int).collect();
            if index == 0 {
                values.extend(self.extra.clone());
            }
            constraints.push(if *is_in {
                in_set(&x, values)
            } else {
                not_in_set(&x, values)
            });
        }
        let sign = if self.natural {
            Sign::NonNegative
        } else {
            Sign::Any
        };
        Param::new(
            ParamDomain::from(IntegerDomain::new(sign, ZeroInclusion::Included)),
            x,
            constraints,
            context,
        )
        .expect("bounds and set constraints")
    }

    fn holds(&self, value: i64) -> bool {
        (!self.natural || value >= 0)
            && self.lower <= value
            && value <= self.upper
            && self
                .sets
                .iter()
                .all(|(values, is_in)| values.contains(&value) == *is_in)
    }
}

fn numeric_spec() -> impl Strategy<Value = NumericSpec> {
    (
        -4_i64..4,
        -4_i64..6,
        prop::collection::vec(
            (prop::collection::vec(-4_i64..6, 0..3), any::<bool>()),
            0..3,
        ),
        any::<bool>(),
        prop::option::of(prop_oneof![
            Just(Value::Float(1.0)),
            Just(Value::Bool(true)),
            Just(Value::Float(0.5)),
        ]),
    )
        .prop_map(|(lower, upper, sets, natural, extra)| NumericSpec {
            lower,
            upper,
            sets,
            natural,
            extra,
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// The interval hull of `+`, `-`, `*`, reversed `-` and negation, over
    /// unbounded, natural and exclusive operands, admits exactly the
    /// values between the least and greatest result of the operands'
    /// values.
    #[test]
    fn the_interval_hull_agrees_with_brute_force(left in interval_spec(), right in interval_spec()) {
        let (solver, _smt) = scripted_solver(SatResult::Sat);
        let observer = RecordingParamObserver::default();
        let context = context(&solver, &observer);
        let (Ok(left_param), Ok(right_param)) = (left.build(&context), right.build(&context)) else {
            return Err(TestCaseError::reject("a spec the builder refuses"));
        };
        let operand = Operand::Param(right_param);
        let operations: [Operation; 4] = [
            ("add", |a, b| a + b),
            ("sub", |a, b| a - b),
            ("mul", |a, b| a * b),
            ("rsub", |a, b| b - a),
        ];
        let (left_samples, right_samples) = (left.samples(), right.samples());
        for (name, apply) in operations {
            let result = match name {
                "add" => left_param.checked_add(&operand, &context),
                "sub" => left_param.checked_sub(&operand, &context),
                "mul" => left_param.checked_mul(&operand, &context),
                _ => left_param.checked_reverse_sub(&operand, &context),
            }
            .map_err(|error| TestCaseError::fail(format!("{name}: {error}")))?;
            let results: Vec<i64> = left_samples
                .iter()
                .flat_map(|&a| right_samples.iter().map(move |&b| apply(a, b)))
                .collect();
            let least = *results.iter().min().expect("a sample");
            let greatest = *results.iter().max().expect("a sample");
            for value in -20..=20 {
                prop_assert_eq!(
                    admits(&result, value, &context),
                    Some((least..=greatest).contains(&value)),
                    "{} of {:?} and {:?} at {}", name, left, right, value
                );
            }
        }
        let negated = left_param
            .checked_neg(&context)
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let least = *left_samples.iter().min().expect("a sample");
        let greatest = *left_samples.iter().max().expect("a sample");
        for value in -20..=20 {
            prop_assert_eq!(
                admits(&negated, value, &context),
                Some(least <= -value && -value <= greatest),
                "neg of {:?} at {}", left, value
            );
        }
    }

    /// The union and intersection of finite params admit exactly the
    /// values either or both admit, and intersection succeeds in either
    /// order alike.
    #[test]
    fn finite_union_and_intersection_agree_with_brute_force(
        left in finite_spec(),
        right in finite_spec(),
        ordinal in any::<bool>(),
    ) {
        let (solver, _smt) = scripted_solver(SatResult::Sat);
        let observer = RecordingParamObserver::default();
        let context = context(&solver, &observer);
        let (left_param, right_param) = (left.build(ordinal, &context), right.build(ordinal, &context));

        let union: Vec<i64> = (0..8).filter(|&value| left.holds(value) || right.holds(value)).collect();
        match left_param.union(&right_param, Identifier::new("u"), &context) {
            Ok(param) => for value in 0..8 {
                prop_assert_eq!(admits(&param, value, &context), Some(union.contains(&value)), "union at {}", value);
            },
            Err(error) => prop_assert!(union.is_empty(), "union failed: {}", error),
        }
        let intersection: Vec<i64> = (0..8).filter(|&value| left.holds(value) && right.holds(value)).collect();
        let forward = left_param.intersection(&right_param, Identifier::new("i"), &context);
        match &forward {
            Ok(param) => for value in 0..8 {
                prop_assert_eq!(admits(param, value, &context), Some(intersection.contains(&value)), "intersection at {}", value);
            },
            Err(error) => prop_assert!(intersection.is_empty(), "intersection failed: {}", error),
        }
        let backward = right_param.intersection(&left_param, Identifier::new("i"), &context);
        prop_assert_eq!(forward.is_ok(), backward.is_ok());
    }

    /// The intersection of two integer params, one possibly natural, with
    /// bounds and an excluded set, admits exactly the values both admit.
    #[test]
    fn integer_intersection_agrees_with_brute_force(
        (lower_1, upper_1) in (-4_i64..4).prop_flat_map(|lower| (Just(lower), lower..6)),
        (lower_2, upper_2) in (-4_i64..4).prop_flat_map(|lower| (Just(lower), lower..6)),
        excluded in prop::collection::vec(-4_i64..6, 0..3),
        natural in any::<bool>(),
    ) {
        let (solver, _smt) = scripted_solver(SatResult::Sat);
        let observer = RecordingParamObserver::default();
        let context = context(&solver, &observer);
        let (x, y) = (Identifier::new("x"), Identifier::new("y"));
        let sign = if natural { Sign::NonNegative } else { Sign::Any };
        let left = Param::new(
            ParamDomain::from(IntegerDomain::new(sign, ZeroInclusion::Included)),
            x.clone(),
            [at_least(&x, lower_1), at_most(&x, upper_1)],
            &context,
        )
        .expect("bounds");
        let mut constraints = vec![at_least(&y, lower_2), at_most(&y, upper_2)];
        if !excluded.is_empty() {
            constraints.push(not_in_set(&y, excluded.iter().copied().map(int)));
        }
        let right = Param::new(
            ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included)),
            y,
            constraints,
            &context,
        )
        .expect("bounds and a set");

        let expected: Vec<i64> = (-6..8)
            .filter(|&value| {
                (!natural || value >= 0)
                    && (lower_1..=upper_1).contains(&value)
                    && (lower_2..=upper_2).contains(&value)
                    && !excluded.contains(&value)
            })
            .collect();
        match left.intersection(&right, Identifier::new("i"), &context) {
            Ok(param) => for value in -6..8 {
                prop_assert_eq!(admits(&param, value, &context), Some(expected.contains(&value)), "at {}", value);
            },
            Err(error) => prop_assert!(expected.is_empty(), "intersection failed: {} but expected {:?}", error, expected),
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Numeric feasibility and subset, asked of a real solver, are sound:
    /// every decided answer agrees with brute force over the window. Runs
    /// on the z3 backend under the `z3` feature, and otherwise on the
    /// executable `FHY_SMT_SOLVER` names, as
    /// `system_satisfiability_agrees_with_brute_force` does.
    #[test]
    fn numeric_feasibility_and_subset_are_sound(left in numeric_spec(), right in numeric_spec()) {
        let Some(solver) = real_solver() else {
            return Ok(());
        };
        let context = ParamContext::new(&solver);
        let (left_param, right_param) = (left.build("x", &context), right.build("y", &context));

        let feasible = (-6..8).any(|value| left.holds(value));
        match left_param.check_feasibility(&context) {
            Ok(Outcome::Satisfied) => prop_assert!(feasible, "claims feasible: {:?}", left),
            Ok(Outcome::Violated) => prop_assert!(!feasible, "claims infeasible: {:?}", left),
            Ok(Outcome::Undecided) => {}
            Err(error) => prop_assert!(false, "feasibility error {} for {:?}", error, left),
        }
        let subset = (-6..8).filter(|&value| left.holds(value)).all(|value| right.holds(value));
        match left_param.check_subset(&right_param, &context) {
            Ok(Outcome::Satisfied) => prop_assert!(subset, "claims subset: {:?} of {:?}", left, right),
            Ok(Outcome::Violated) => prop_assert!(!subset, "claims no subset: {:?} of {:?}", left, right),
            Ok(Outcome::Undecided) => {}
            Err(error) => prop_assert!(false, "subset error {} for {:?} {:?}", error, left, right),
        }
    }
}
