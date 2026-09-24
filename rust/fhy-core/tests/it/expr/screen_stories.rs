//! Tests for the Boolean-position screen `BooleanScreen`, its
//! `check_logical_operands` and `check_predicate`, and for the error they
//! report, over trees and over DAGs sharing their subtrees.
//!
//! Public API only (`fhy_core::expr`). Calls of built-in functions take
//! their result sorts from the catalogue (`floor` returns an integer, `sqrt`
//! a real, `nand` a Boolean). Native constants take their sorts from a
//! test-local [`SortLookup`] holding the real constants `pi`, `e`, `inf` and
//! `nan`, so the tests exercise the same sorts a populated registry
//! reports.

use crate::support::expression as expression_support;
use crate::support::stack as stack_support;

use std::cell::Cell;
use std::collections::HashMap;

use expression_support::{
    build_call_or_panic, build_decimal_literal, build_deep_conjunction, build_identifier,
    build_literal, build_piecewise_or_panic,
};
use fhy_core::expr::builtins::BuiltinFunction;
use fhy_core::expr::{
    BooleanPosition, BooleanScreen, Expression, FunctionName, FunctionSort, LogicalOperation,
    NoRegisteredSorts, NonBooleanLogicalOperandError, SortLookup, SymbolType, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use rstest::rstest;
use stack_support::{SMALL_STACK_DEPTH, run_on_small_stack};

/// Names of the real-valued built-in constants.
const REAL_CONSTANT_NAMES: [&str; 4] = ["pi", "e", "inf", "nan"];

/// The sorts of the built-in constants, plus any constants a test adds.
#[derive(Debug)]
struct BuiltinSorts {
    constants: HashMap<Identifier, FunctionSort>,
}

impl BuiltinSorts {
    /// Mint a canonical identifier for each real-valued built-in constant.
    fn new() -> Self {
        let constants = REAL_CONSTANT_NAMES
            .iter()
            .map(|name| (Identifier::new(name), FunctionSort::Real))
            .collect();
        Self { constants }
    }

    /// Add a constant of `sort` named `name` and return its canonical
    /// identifier.
    fn add_constant(&mut self, name: &str, sort: FunctionSort) -> Identifier {
        let identifier = Identifier::new(name);
        self.constants.insert(identifier.clone(), sort);
        identifier
    }

    /// Return the canonical identifier of the constant `name`.
    fn find_constant(&self, name: &str) -> Identifier {
        self.constants
            .keys()
            .find(|identifier| identifier.name_hint() == name)
            .unwrap_or_else(|| panic!("no constant named {name}"))
            .clone()
    }

    /// Return a reference to the canonical identifier of the constant
    /// `name`.
    fn reference_constant(&self, name: &str) -> Expression {
        Expression::from(self.find_constant(name))
    }
}

impl SortLookup for BuiltinSorts {
    fn native_constant_sort(&self, identifier: &Identifier) -> Option<FunctionSort> {
        self.constants.get(identifier).copied()
    }
}

/// A lookup knowing the result sorts of the named functions `real_valued`
/// (a real) and `predicate` (a Boolean), and no constant.
#[derive(Debug)]
struct NamedSorts;

impl SortLookup for NamedSorts {
    fn call_result_sort(&self, name: &FunctionName) -> Option<FunctionSort> {
        match name.as_str() {
            "real_valued" => Some(FunctionSort::Real),
            "predicate" => Some(FunctionSort::Bool),
            _ => None,
        }
    }
}

/// The two screens.
#[derive(Debug, Clone, Copy)]
enum Screen {
    LogicalOperands,
    Predicate,
}

impl Screen {
    /// Run the screen over `expression` with the given bindings, declared
    /// types, and sorts.
    fn run_with(
        self,
        expression: &Expression,
        environment: &HashMap<Identifier, Expression>,
        symbol_types: &HashMap<Identifier, SymbolType>,
        sorts: &dyn SortLookup,
    ) -> Result<(), NonBooleanLogicalOperandError> {
        let screen = BooleanScreen::new()
            .with_sorts(sorts)
            .with_environment(environment)
            .with_symbol_types(symbol_types);
        match self {
            Self::LogicalOperands => screen.check_logical_operands(expression),
            Self::Predicate => screen.check_predicate(expression),
        }
    }

    /// Run the screen over `expression` with nothing bound or declared and
    /// the built-in sorts.
    fn run(self, expression: &Expression) -> Result<(), NonBooleanLogicalOperandError> {
        self.run_with(
            expression,
            &HashMap::new(),
            &HashMap::new(),
            &BuiltinSorts::new(),
        )
    }
}

/// Return the refusal `result` holds, failing the test if it passed.
fn expect_refusal(
    result: Result<(), NonBooleanLogicalOperandError>,
) -> NonBooleanLogicalOperandError {
    result.expect_err("the screen refuses the expression")
}

/// Assert the refusal names `operand` at `position` under `parent`.
fn assert_refusal(
    error: &NonBooleanLogicalOperandError,
    operand: &Expression,
    parent: &Expression,
    position: BooleanPosition,
) {
    assert_eq!(error.operand(), operand, "operand of {error:?}");
    assert_eq!(
        error.parent(),
        Some((parent, position)),
        "parent of {error:?}"
    );
}

/// Assert the refusal names `operand` as the root of a predicate, with no
/// parent.
fn assert_root_refusal(error: &NonBooleanLogicalOperandError, operand: &Expression) {
    assert_eq!(error.operand(), operand, "operand of {error:?}");
    assert_eq!(error.parent(), None, "parent of {error:?}");
}

/// Return the position of the refused operand within its parent, or `None`
/// for a predicate root.
fn find_position(error: &NonBooleanLogicalOperandError) -> Option<BooleanPosition> {
    error.parent().map(|(_, position)| position)
}

/// Return the conjunction `left && right`.
fn build_and(left: &Expression, right: &Expression) -> Expression {
    left.and(right)
}

/// Return the disjunction `left || right`.
fn build_or(left: &Expression, right: &Expression) -> Expression {
    left.or(right)
}

/// A way to put an operand in a Boolean position, with the position it
/// lands in.
#[derive(Debug, Clone, Copy)]
enum Placement {
    AndLeft,
    OrRight,
    Negated,
    CaseCondition,
}

impl Placement {
    /// Put `operand` in the placement's Boolean position.
    fn place(self, operand: &Expression) -> Expression {
        match self {
            Self::AndLeft => build_and(operand, &build_literal(true)),
            Self::OrRight => build_or(&build_literal(false), operand),
            Self::Negated => !operand,
            Self::CaseCondition => {
                build_piecewise_or_panic([(operand, &build_literal(1))], build_literal(2))
            }
        }
    }

    /// Return the position the placement puts its operand in.
    fn position(self) -> BooleanPosition {
        match self {
            Self::AndLeft => BooleanPosition::LogicalOperand {
                operation: LogicalOperation::And,
                operand_index: 0,
            },
            Self::OrRight => BooleanPosition::LogicalOperand {
                operation: LogicalOperation::Or,
                operand_index: 1,
            },
            Self::Negated => BooleanPosition::NegatedOperand,
            Self::CaseCondition => BooleanPosition::CaseCondition { case_index: 0 },
        }
    }
}

// =============================================================================
// check_logical_operands: numbers under connectives
// =============================================================================

/// Test a connective over a number is refused, naming the first numeric
/// operand and the connective.
#[rstest]
#[case::and(build_and(&build_literal(2), &build_literal(4)), build_literal(2), LogicalOperation::And, 0)]
#[case::or(build_or(&build_literal(2), &build_literal(4)), build_literal(2), LogicalOperation::Or, 0)]
#[case::and_one_numeric_operand(build_and(&build_literal(true), &build_literal(4)), build_literal(4), LogicalOperation::And, 1)]
#[case::and_floats(build_and(&build_literal(1.5), &build_literal(2.5)), build_literal(1.5), LogicalOperation::And, 0)]
#[case::and_decimals(build_and(&build_decimal_literal("2"), &build_decimal_literal("4")), build_decimal_literal("2"), LogicalOperation::And, 0)]
#[case::and_arithmetic_operand(build_and(&(build_literal(1) + 2), &build_literal(true)), build_literal(1) + 2, LogicalOperation::And, 0)]
#[case::last_of_four(
    Expression::any([build_literal(true), build_literal(false), build_literal(true), build_literal(9)]),
    build_literal(9),
    LogicalOperation::Or,
    3
)]
fn validate_logical_operands_rejects_a_numeric_connective_operand(
    #[case] expression: Expression,
    #[case] operand: Expression,
    #[case] operation: LogicalOperation,
    #[case] operand_index: usize,
) {
    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    assert_refusal(
        &error,
        &operand,
        &expression,
        BooleanPosition::LogicalOperand {
            operation,
            operand_index,
        },
    );
}

/// Test the refusal of a number among many operands of one logical node
/// names the operand's index within the node.
#[rstest]
#[case::first(0)]
#[case::middle(2)]
#[case::last(4)]
fn boolean_screen_reports_the_operand_index_of_a_logical_operand(#[case] numeric_index: usize) {
    let operands: Vec<Expression> = (0..5)
        .map(|index| {
            if index == numeric_index {
                build_literal(7)
            } else {
                build_identifier(&format!("p{index}")).1
            }
        })
        .collect();
    let expression = Expression::all(&operands);

    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    assert!(Expression::ptr_eq(
        error.operand(),
        &operands[numeric_index]
    ));
    assert_eq!(
        find_position(&error),
        Some(BooleanPosition::LogicalOperand {
            operation: LogicalOperation::And,
            operand_index: numeric_index,
        })
    );
}

/// Test a negation of a number is refused, arithmetic negation included.
#[rstest]
#[case::literal(build_literal(2))]
#[case::negation(-build_literal(1))]
fn validate_logical_operands_rejects_a_numeric_negated_operand(#[case] operand: Expression) {
    let expression = !&operand;

    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    assert_refusal(
        &error,
        &operand,
        &expression,
        BooleanPosition::NegatedOperand,
    );
}

/// Test the refusal names the connective node and the numeric operand.
#[test]
fn validate_logical_operands_error_names_the_connective_and_the_operand() {
    let expression = build_or(&build_literal(2), &build_literal(4));

    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    assert_refusal(
        &error,
        &build_literal(2),
        &expression,
        BooleanPosition::LogicalOperand {
            operation: LogicalOperation::Or,
            operand_index: 0,
        },
    );
}

/// Test a numeric operand below a well-typed root is still found, with the
/// inner connective as its parent.
#[test]
fn validate_logical_operands_descends_past_the_root() {
    let (_, x) = build_identifier("x");
    let nested = build_and(&build_literal(2), &build_literal(4));
    let expression = build_and(&x.greater(0), &nested);

    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    assert_refusal(
        &error,
        &build_literal(2),
        &nested,
        BooleanPosition::LogicalOperand {
            operation: LogicalOperation::And,
            operand_index: 0,
        },
    );
}

/// Test a node's own Boolean-position operands are checked before its
/// children are walked.
#[test]
fn validate_logical_operands_checks_operands_before_descending() {
    let inner = build_and(&build_literal(2), &build_literal(3));
    let expression = build_and(&inner, &build_literal(4));

    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    assert_refusal(
        &error,
        &build_literal(4),
        &expression,
        BooleanPosition::LogicalOperand {
            operation: LogicalOperation::And,
            operand_index: 1,
        },
    );
}

/// Test a piecewise whose every branch is numeric is a numeric operand.
#[test]
fn validate_logical_operands_rejects_an_all_numeric_piecewise_operand() {
    let (_, x) = build_identifier("x");
    let numeric = build_piecewise_or_panic([(&x.greater(0), &build_literal(1))], build_literal(2));
    let expression = build_and(&numeric, &build_literal(true));

    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    assert_refusal(
        &error,
        &numeric,
        &expression,
        BooleanPosition::LogicalOperand {
            operation: LogicalOperation::And,
            operand_index: 0,
        },
    );
}

/// Test a piecewise whose branches are Booleans passes.
#[test]
fn validate_logical_operands_accepts_a_boolean_valued_piecewise_operand() {
    let (_, x) = build_identifier("x");
    let boolean = build_piecewise_or_panic(
        [(&x.greater(0), &build_literal(true))],
        build_literal(false),
    );

    let result = Screen::LogicalOperands.run(&build_and(&boolean, &build_literal(true)));

    assert_eq!(result, Ok(()));
}

/// Test a piecewise in a Boolean position with one numeric branch is
/// refused at that branch.
#[rstest]
#[case::value(build_literal(2), build_literal(true), BooleanPosition::CaseValue { case_index: 0 })]
#[case::otherwise(build_literal(true), build_literal(2), BooleanPosition::Otherwise)]
fn validate_logical_operands_rejects_a_piecewise_operand_with_one_numeric_branch(
    #[case] value: Expression,
    #[case] otherwise: Expression,
    #[case] position: BooleanPosition,
) {
    let (_, x) = build_identifier("x");
    let mixed = build_piecewise_or_panic([(&x.greater(0), &value)], &otherwise);

    let error = expect_refusal(Screen::LogicalOperands.run(&!&mixed));

    assert_refusal(&error, &build_literal(2), &mixed, position);
}

/// Test a branch shared by a case value and the otherwise branch is
/// reported at its first position, the case value.
#[test]
fn validate_logical_operands_reports_a_shared_branch_at_its_first_position() {
    let (_, x) = build_identifier("x");
    let shared = build_literal(2);
    let mixed = Expression::piecewise(
        [
            (x.greater(0), shared.clone()),
            (x.less(0), build_literal(true)),
        ],
        &shared,
    )
    .expect("a valid piecewise");

    let error = expect_refusal(Screen::LogicalOperands.run(&!&mixed));

    assert_refusal(
        &error,
        &shared,
        &mixed,
        BooleanPosition::CaseValue { case_index: 0 },
    );
}

/// Test every case condition of a piecewise in a Boolean position is checked
/// before any of its case values: a numeric later condition is reported
/// ahead of a numeric earlier value.
#[rstest]
#[case::logical_operands(Screen::LogicalOperands)]
#[case::predicate(Screen::Predicate)]
fn validate_checks_every_case_condition_before_any_case_value(#[case] screen: Screen) {
    let (_, x) = build_identifier("x");
    let numeric_condition = &x + 1;
    let mixed = Expression::piecewise(
        [
            (x.greater(0), build_literal(2)),
            (numeric_condition.clone(), build_literal(true)),
        ],
        build_literal(true),
    )
    .expect("a valid piecewise");
    let expression = match screen {
        Screen::LogicalOperands => !&mixed,
        Screen::Predicate => mixed.clone(),
    };

    let error = expect_refusal(screen.run(&expression));

    assert_refusal(
        &error,
        &numeric_condition,
        &mixed,
        BooleanPosition::CaseCondition { case_index: 1 },
    );
}

/// Test a numeric piecewise compared as a number passes both screens.
#[rstest]
#[case::logical_operands(Screen::LogicalOperands)]
#[case::predicate(Screen::Predicate)]
fn validate_accepts_a_numeric_piecewise_compared_as_a_number(#[case] screen: Screen) {
    let (_, x) = build_identifier("x");
    let numeric = build_piecewise_or_panic([(&x.greater(0), &build_literal(2))], build_literal(3));
    let expression = build_and(&numeric.greater(1), &build_literal(true));

    let result = screen.run(&expression);

    assert_eq!(result, Ok(()));
}

/// Test a call whose result sort is an integer or a real is refused in a
/// Boolean position.
#[rstest]
fn validate_logical_operands_rejects_a_numeric_result_call(
    #[values("floor", "sqrt")] function_name: &str,
    #[values(
        Placement::AndLeft,
        Placement::OrRight,
        Placement::Negated,
        Placement::CaseCondition
    )]
    placement: Placement,
) {
    let call = build_call_or_panic(function_name, &[build_literal(1.5)]);
    let expression = placement.place(&call);

    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    assert_refusal(&error, &call, &expression, placement.position());
}

/// Test a call of a named function nothing knows the sort of is not
/// refused.
#[test]
fn validate_logical_operands_accepts_a_call_the_lookup_does_not_know() {
    let call = build_call_or_panic("f", &[build_literal(1.5)]);
    let expression = build_and(&call, &build_literal(true));

    let result = BooleanScreen::new().check_logical_operands(&expression);

    assert_eq!(result, Ok(()));
}

/// Test a built-in function's catalogue result sort decides its call, with
/// no lookup registering it: `floor(1.5) && true` is refused and
/// `nand(true, true) && true` passes.
#[test]
fn boolean_screen_knows_builtin_result_sorts_without_a_lookup() {
    let floor = Expression::call(BuiltinFunction::Floor, [1.5]);
    let nand = Expression::call(
        BuiltinFunction::Nand,
        [build_literal(true), build_literal(true)],
    );
    let refused = build_and(&floor, &build_literal(true));
    let accepted = build_and(&nand, &build_literal(true));

    let refusal = BooleanScreen::new().check_logical_operands(&refused);
    let acceptance = BooleanScreen::new().check_logical_operands(&accepted);

    assert_refusal(
        &expect_refusal(refusal),
        &floor,
        &refused,
        BooleanPosition::LogicalOperand {
            operation: LogicalOperation::And,
            operand_index: 0,
        },
    );
    assert_eq!(acceptance, Ok(()));
}

/// Test a named function's call is judged by the result sort the lookup
/// reports for its name.
#[rstest]
#[case::real_result("real_valued", true)]
#[case::boolean_result("predicate", false)]
#[case::unknown("unknown", false)]
fn validate_logical_operands_asks_the_lookup_for_a_named_call(
    #[case] name: &str,
    #[case] is_refused: bool,
) {
    let call = build_call_or_panic(name, &[build_literal(1)]);
    let expression = !&call;

    let result = BooleanScreen::new()
        .with_sorts(&NamedSorts)
        .check_logical_operands(&expression);

    assert_eq!(result.is_err(), is_refused, "{result:?}");
}

/// Test operands the screen cannot prove numeric pass.
#[rstest]
#[case::boolean_literals(build_and(&build_literal(true), &build_literal(false)))]
#[case::boolean_literal_negation(!build_literal(true))]
#[case::unbound_identifiers(build_and(&build_identifier("p").1, &build_identifier("q").1))]
#[case::comparisons({ let x = build_identifier("x").1; build_and(&x.greater(0), &x.less(5)) })]
#[case::boolean_call(build_and(&build_call_or_panic("nand", &[build_literal(true), build_literal(true)]), &build_literal(true)))]
#[case::nested_connective(build_and(&!build_identifier("p").1, &build_literal(true)))]
fn validate_logical_operands_accepts_an_operand_it_cannot_prove_numeric(
    #[case] expression: Expression,
) {
    let result = Screen::LogicalOperands.run(&expression);

    assert_eq!(result, Ok(()));
}

// =============================================================================
// check_logical_operands: bindings and declared types
// =============================================================================

/// Test an identifier bound to a number is screened as that number.
#[test]
fn validate_logical_operands_screens_an_identifier_bound_to_a_number() {
    let (p, p_reference) = build_identifier("p");
    let (q, q_reference) = build_identifier("q");
    let expression = build_and(&p_reference, &q_reference);
    let environment = HashMap::from([(p, build_literal(2)), (q, build_literal(4))]);

    let error = expect_refusal(Screen::LogicalOperands.run_with(
        &expression,
        &environment,
        &HashMap::new(),
        &BuiltinSorts::new(),
    ));

    assert_refusal(
        &error,
        &p_reference,
        &expression,
        BooleanPosition::LogicalOperand {
            operation: LogicalOperation::And,
            operand_index: 0,
        },
    );
}

/// Test an identifier bound to a Boolean passes.
#[test]
fn validate_logical_operands_accepts_an_identifier_bound_to_a_boolean() {
    let (p, p_reference) = build_identifier("p");
    let expression = build_and(&p_reference, &build_literal(true));

    let result = Screen::LogicalOperands.run_with(
        &expression,
        &HashMap::from([(p, build_literal(false))]),
        &HashMap::new(),
        &BuiltinSorts::new(),
    );

    assert_eq!(result, Ok(()));
}

/// Test a bound value is judged without applying another binding to it.
#[test]
fn validate_logical_operands_does_not_chain_environment_bindings() {
    let (p, p_reference) = build_identifier("p");
    let (q, q_reference) = build_identifier("q");
    let expression = build_and(&p_reference, &build_literal(true));
    let environment = HashMap::from([(p, q_reference), (q, build_literal(2))]);

    let result = Screen::LogicalOperands.run_with(
        &expression,
        &environment,
        &HashMap::new(),
        &BuiltinSorts::new(),
    );

    assert_eq!(result, Ok(()));
}

/// Test an arithmetic case condition is refused.
#[test]
fn validate_logical_operands_rejects_a_numeric_piecewise_condition() {
    let (_, x) = build_identifier("x");
    let condition = &x + 1;
    let expression = build_piecewise_or_panic([(&condition, &build_literal(5))], build_literal(0));

    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    assert_refusal(
        &error,
        &condition,
        &expression,
        BooleanPosition::CaseCondition { case_index: 0 },
    );
}

/// Test the refusal names the piecewise and its condition.
#[test]
fn validate_logical_operands_error_names_the_piecewise_and_its_condition() {
    let (_, x) = build_identifier("x");
    let condition = &x * 2;
    let expression = build_piecewise_or_panic([(&condition, &build_literal(5))], build_literal(0));

    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    assert_refusal(
        &error,
        &condition,
        &expression,
        BooleanPosition::CaseCondition { case_index: 0 },
    );
}

/// Test a native real constant is refused in each Boolean position.
#[rstest]
fn validate_logical_operands_rejects_a_native_constant_in_a_boolean_position(
    #[values("pi", "e", "inf", "nan")] constant_name: &str,
    #[values(
        Placement::AndLeft,
        Placement::OrRight,
        Placement::Negated,
        Placement::CaseCondition
    )]
    placement: Placement,
) {
    let sorts = BuiltinSorts::new();
    let constant = sorts.reference_constant(constant_name);
    let expression = placement.place(&constant);

    let error = expect_refusal(Screen::LogicalOperands.run_with(
        &expression,
        &HashMap::new(),
        &HashMap::new(),
        &sorts,
    ));

    assert_refusal(&error, &constant, &expression, placement.position());
}

/// Test a Boolean binding for a constant's identifier does not make it
/// Boolean.
#[test]
fn validate_logical_operands_reads_a_constant_by_its_sort_not_a_binding() {
    let sorts = BuiltinSorts::new();
    let pi_identifier = sorts.find_constant("pi");
    let pi = Expression::from(pi_identifier.clone());
    let expression = build_and(&pi, &build_literal(true));

    let error = expect_refusal(Screen::LogicalOperands.run_with(
        &expression,
        &HashMap::from([(pi_identifier, build_literal(true))]),
        &HashMap::new(),
        &sorts,
    ));

    assert_eq!(error.operand(), &pi);
}

/// Test the value a constant's identifier is bound to is not walked:
/// `pi + 1` passes with `pi` bound to `!(3)`, whose negated operand is a
/// number.
#[test]
fn validate_logical_operands_does_not_walk_a_binding_of_a_constant() {
    let sorts = BuiltinSorts::new();
    let pi_identifier = sorts.find_constant("pi");
    let expression = Expression::from(pi_identifier.clone()) + 1;

    let result = Screen::LogicalOperands.run_with(
        &expression,
        &HashMap::from([(pi_identifier, !build_literal(3))]),
        &HashMap::new(),
        &sorts,
    );

    assert_eq!(result, Ok(()));
}

/// Test a Boolean constant passes as a Boolean operand.
#[test]
fn validate_logical_operands_accepts_a_boolean_native_constant() {
    let mut sorts = BuiltinSorts::new();
    let always = sorts.add_constant("always", FunctionSort::Bool);
    let expression = build_and(&Expression::from(always), &build_literal(true));

    let result =
        Screen::LogicalOperands.run_with(&expression, &HashMap::new(), &HashMap::new(), &sorts);

    assert_eq!(result, Ok(()));
}

/// Test a constant outside a Boolean position, and an identifier merely
/// named after a constant, pass.
#[rstest]
#[case::compared_constant(|sorts: &BuiltinSorts| {
    build_and(&sorts.reference_constant("pi").greater(3), &build_literal(true))
})]
#[case::namesake_identifier(|_: &BuiltinSorts| {
    let (_, namesake) = build_identifier("pi");
    build_and(&namesake, &build_literal(true))
})]
fn validate_logical_operands_accepts_a_constant_outside_a_boolean_position(
    #[case] build: fn(&BuiltinSorts) -> Expression,
) {
    let sorts = BuiltinSorts::new();
    let expression = build(&sorts);

    let result =
        Screen::LogicalOperands.run_with(&expression, &HashMap::new(), &HashMap::new(), &sorts);

    assert_eq!(result, Ok(()));
}

/// Test an identifier declared an integer or a real is refused in each
/// Boolean position.
#[rstest]
fn validate_logical_operands_rejects_an_identifier_declared_numeric(
    #[values(SymbolType::Int, SymbolType::Real)] symbol_type: SymbolType,
    #[values(Placement::AndLeft, Placement::Negated, Placement::CaseCondition)]
    placement: Placement,
) {
    let (x, reference) = build_identifier("x");
    let expression = placement.place(&reference);

    let error = expect_refusal(Screen::LogicalOperands.run_with(
        &expression,
        &HashMap::new(),
        &HashMap::from([(x, symbol_type)]),
        &BuiltinSorts::new(),
    ));

    assert_refusal(&error, &reference, &expression, placement.position());
}

/// Test an identifier declared Boolean, or not declared, passes.
#[rstest]
#[case::declared_bool(Some(SymbolType::Bool))]
#[case::undeclared(None)]
fn validate_logical_operands_accepts_an_identifier_not_declared_numeric(
    #[case] symbol_type: Option<SymbolType>,
) {
    let (x, reference) = build_identifier("x");
    let symbol_types: HashMap<Identifier, SymbolType> = symbol_type
        .map(|symbol_type| (x, symbol_type))
        .into_iter()
        .collect();

    let result = Screen::LogicalOperands.run_with(
        &build_and(&reference, &build_literal(true)),
        &HashMap::new(),
        &symbol_types,
        &BuiltinSorts::new(),
    );

    assert_eq!(result, Ok(()));
}

/// Test a bound identifier is judged by its value, not its declared type.
#[test]
fn validate_logical_operands_reads_a_binding_ahead_of_a_declared_type() {
    let (x, reference) = build_identifier("x");

    let result = Screen::LogicalOperands.run_with(
        &build_and(&reference, &build_literal(true)),
        &HashMap::from([(x.clone(), build_literal(false))]),
        &HashMap::from([(x, SymbolType::Int)]),
        &BuiltinSorts::new(),
    );

    assert_eq!(result, Ok(()));
}

/// Test an identifier a binding brings in is judged by its declared type.
#[test]
fn validate_logical_operands_reads_the_type_a_binding_brings_in() {
    let (p, p_reference) = build_identifier("p");
    let (q, q_reference) = build_identifier("q");
    let expression = build_and(&p_reference, &build_literal(true));

    let error = expect_refusal(Screen::LogicalOperands.run_with(
        &expression,
        &HashMap::from([(p, q_reference)]),
        &HashMap::from([(q, SymbolType::Int)]),
        &BuiltinSorts::new(),
    ));

    assert_eq!(error.operand(), &p_reference);
}

/// Test an identifier condition bound to a number is refused.
#[test]
fn validate_logical_operands_screens_a_case_condition_bound_to_a_number() {
    let (c, condition) = build_identifier("c");
    let expression = build_piecewise_or_panic([(&condition, &build_literal(1))], build_literal(0));

    let error = expect_refusal(Screen::LogicalOperands.run_with(
        &expression,
        &HashMap::from([(c, build_literal(1))]),
        &HashMap::new(),
        &BuiltinSorts::new(),
    ));

    assert_refusal(
        &error,
        &condition,
        &expression,
        BooleanPosition::CaseCondition { case_index: 0 },
    );
}

/// Test an unbound or Boolean-bound condition passes beside numeric values.
#[test]
fn validate_logical_operands_accepts_an_unprovable_case_condition() {
    let (c, condition) = build_identifier("c");
    let expression = build_piecewise_or_panic([(&condition, &build_literal(1))], build_literal(0));

    let unbound = Screen::LogicalOperands.run(&expression);
    let bound = Screen::LogicalOperands.run_with(
        &expression,
        &HashMap::from([(c, build_literal(true))]),
        &HashMap::new(),
        &BuiltinSorts::new(),
    );

    assert_eq!(unbound, Ok(()));
    assert_eq!(bound, Ok(()));
}

/// Test an operand bound to a piecewise with one numeric branch is refused
/// at that branch.
#[rstest]
#[case::value(build_literal(1), build_literal(true), BooleanPosition::CaseValue { case_index: 0 })]
#[case::otherwise(build_literal(true), build_literal(1), BooleanPosition::Otherwise)]
fn validate_logical_operands_screens_a_bound_piecewise_with_a_mixed_branch(
    #[case] value: Expression,
    #[case] otherwise: Expression,
    #[case] position: BooleanPosition,
) {
    let (x, reference) = build_identifier("x");
    let mixed = build_piecewise_or_panic([(&build_literal(false), &value)], &otherwise);

    let error = expect_refusal(Screen::LogicalOperands.run_with(
        &!&reference,
        &HashMap::from([(x, mixed.clone())]),
        &HashMap::new(),
        &BuiltinSorts::new(),
    ));

    assert_refusal(&error, &build_literal(1), &mixed, position);
}

/// Test an identifier compared as a number may be bound to a numeric
/// piecewise.
#[test]
fn validate_logical_operands_accepts_a_bound_piecewise_in_a_numeric_position() {
    let (x, reference) = build_identifier("x");
    let numeric = build_piecewise_or_panic(
        [(&build_literal(false), &build_literal(1))],
        build_literal(2),
    );

    let result = Screen::LogicalOperands.run_with(
        &reference.greater(0),
        &HashMap::from([(x, numeric)]),
        &HashMap::new(),
        &BuiltinSorts::new(),
    );

    assert_eq!(result, Ok(()));
}

// =============================================================================
// check_predicate: the root is a Boolean position
// =============================================================================

/// Test a numeric root is refused as a predicate root.
#[rstest]
#[case::int_literal(build_literal(2))]
#[case::float_literal(build_literal(1.5))]
#[case::decimal_literal(build_decimal_literal("2.5"))]
#[case::arithmetic_node(build_literal(1) + 2)]
#[case::negation(-build_literal(1))]
#[case::integer_result_call(build_call_or_panic("floor", &[build_literal(1.5)]))]
#[case::real_result_call(build_call_or_panic("sqrt", &[build_literal(4.0)]))]
fn validate_predicate_rejects_a_numeric_root(#[case] expression: Expression) {
    let error = expect_refusal(Screen::Predicate.run(&expression));

    assert_root_refusal(&error, &expression);
}

/// Test a native real constant as the root is refused.
#[test]
fn validate_predicate_rejects_a_native_constant_root() {
    let sorts = BuiltinSorts::new();
    let pi = sorts.reference_constant("pi");

    let error =
        expect_refusal(Screen::Predicate.run_with(&pi, &HashMap::new(), &HashMap::new(), &sorts));

    assert_root_refusal(&error, &pi);
}

/// Test a root piecewise has its branches screened: a numeric otherwise
/// branch is refused.
#[test]
fn validate_predicate_rejects_a_piecewise_root_with_a_numeric_branch() {
    let (_, c) = build_identifier("c");
    let expression = build_piecewise_or_panic([(&c, &build_literal(true))], build_literal(2));

    let error = expect_refusal(Screen::Predicate.run(&expression));

    assert_refusal(
        &error,
        &build_literal(2),
        &expression,
        BooleanPosition::Otherwise,
    );
}

/// Test a root identifier declared an integer or a real is refused.
#[rstest]
#[case::int(SymbolType::Int)]
#[case::real(SymbolType::Real)]
fn validate_predicate_rejects_a_root_identifier_declared_numeric(#[case] symbol_type: SymbolType) {
    let (x, reference) = build_identifier("x");

    let error = expect_refusal(Screen::Predicate.run_with(
        &reference,
        &HashMap::new(),
        &HashMap::from([(x, symbol_type)]),
        &BuiltinSorts::new(),
    ));

    assert_root_refusal(&error, &reference);
}

/// Test a root identifier bound to a number is refused.
#[test]
fn validate_predicate_rejects_a_root_identifier_bound_to_a_number() {
    let (x, reference) = build_identifier("x");

    let error = expect_refusal(Screen::Predicate.run_with(
        &reference,
        &HashMap::from([(x, build_literal(2))]),
        &HashMap::new(),
        &BuiltinSorts::new(),
    ));

    assert_root_refusal(&error, &reference);
}

/// Test a Boolean root, or one the screen cannot prove numeric, passes.
#[rstest]
#[case::bool_literal(build_literal(true))]
#[case::comparison(build_literal(1).greater(0))]
#[case::connective(build_and(&build_literal(true), &build_literal(false)))]
#[case::undeclared_unbound_identifier(build_identifier("p").1)]
#[case::boolean_piecewise(build_piecewise_or_panic([(&build_identifier("c").1, &build_literal(true))], build_literal(false)))]
#[case::boolean_result_call(build_call_or_panic("nand", &[build_literal(true), build_literal(true)]))]
#[case::unknown_call(build_call_or_panic("totally_unregistered_function", &[build_literal(true)]))]
fn validate_predicate_accepts_a_boolean_or_unprovable_root(#[case] expression: Expression) {
    let result = Screen::Predicate.run(&expression);

    assert_eq!(result, Ok(()));
}

/// Test a root identifier declared Boolean passes.
#[test]
fn validate_predicate_accepts_a_root_identifier_declared_bool() {
    let (x, reference) = build_identifier("x");

    let result = Screen::Predicate.run_with(
        &reference,
        &HashMap::new(),
        &HashMap::from([(x, SymbolType::Bool)]),
        &BuiltinSorts::new(),
    );

    assert_eq!(result, Ok(()));
}

/// Test a numeric operand under a connective root is still found.
#[test]
fn validate_predicate_still_screens_a_nested_boolean_position() {
    let expression = build_and(&build_literal(2), &build_literal(4));

    let error = expect_refusal(Screen::Predicate.run(&expression));

    assert_refusal(
        &error,
        &build_literal(2),
        &expression,
        BooleanPosition::LogicalOperand {
            operation: LogicalOperation::And,
            operand_index: 0,
        },
    );
}

/// Test a root identifier bound to a piecewise with a numeric case value is
/// refused at that value.
#[test]
fn validate_predicate_screens_a_bound_piecewise_with_a_mixed_branch() {
    let (x, reference) = build_identifier("x");
    let mixed = build_piecewise_or_panic(
        [(&build_literal(false), &build_literal(1))],
        build_literal(true),
    );

    let error = expect_refusal(Screen::Predicate.run_with(
        &reference,
        &HashMap::from([(x, mixed.clone())]),
        &HashMap::new(),
        &BuiltinSorts::new(),
    ));

    assert_refusal(
        &error,
        &build_literal(1),
        &mixed,
        BooleanPosition::CaseValue { case_index: 0 },
    );
}

/// Test a root identifier bound to an all-Boolean piecewise passes.
#[test]
fn validate_predicate_accepts_a_bound_well_typed_boolean_piecewise() {
    let (x, reference) = build_identifier("x");
    let well_typed = build_piecewise_or_panic(
        [(&build_literal(false), &build_literal(false))],
        build_literal(true),
    );

    let result = Screen::Predicate.run_with(
        &reference,
        &HashMap::from([(x, well_typed)]),
        &HashMap::new(),
        &BuiltinSorts::new(),
    );

    assert_eq!(result, Ok(()));
}

// =============================================================================
// The error and the empty lookup
// =============================================================================

/// Test the message for each position is one short line naming the
/// position, never the expressions.
#[rstest]
#[case::negated(
    BooleanPosition::NegatedOperand,
    "the operand of a logical not provably denotes a number but sits in a boolean position"
)]
#[case::logical(
    BooleanPosition::LogicalOperand { operation: LogicalOperation::And, operand_index: 0 },
    "operand 0 of a logical and provably denotes a number but sits in a boolean position"
)]
#[case::case_condition(
    BooleanPosition::CaseCondition { case_index: 0 },
    "the condition of piecewise case 0 provably denotes a number but sits in a boolean position"
)]
#[case::case_value(
    BooleanPosition::CaseValue { case_index: 0 },
    "the value of piecewise case 0 provably denotes a number but sits in a boolean position"
)]
#[case::otherwise(
    BooleanPosition::Otherwise,
    "the otherwise branch of a piecewise provably denotes a number but sits in a boolean position"
)]
fn non_boolean_logical_operand_error_display_describes_the_position(
    #[case] position: BooleanPosition,
    #[case] expected: &str,
) {
    let number = -build_literal(7);
    let (expression, screen) = match position {
        BooleanPosition::NegatedOperand => (!&number, Screen::LogicalOperands),
        BooleanPosition::LogicalOperand { .. } => (
            build_and(&number, &build_literal(true)),
            Screen::LogicalOperands,
        ),
        BooleanPosition::CaseCondition { .. } => (
            build_piecewise_or_panic([(&number, &build_literal(true))], build_literal(true)),
            Screen::LogicalOperands,
        ),
        BooleanPosition::CaseValue { .. } => (
            build_piecewise_or_panic([(&build_literal(true), &number)], build_literal(true)),
            Screen::Predicate,
        ),
        BooleanPosition::Otherwise => (
            build_piecewise_or_panic([(&build_literal(true), &build_literal(true))], &number),
            Screen::Predicate,
        ),
        _ => unreachable!("every position has a case"),
    };

    let error = expect_refusal(screen.run(&expression));

    assert_eq!(find_position(&error), Some(position));
    assert_eq!(error.to_string(), expected);
}

/// Test the message for a predicate root, which has no parent.
#[test]
fn non_boolean_logical_operand_error_display_describes_a_predicate_root() {
    let error = expect_refusal(Screen::Predicate.run(&-build_literal(7)));

    assert_eq!(error.to_string(), "the predicate provably denotes a number");
}

/// Test each Boolean position displays as the phrase naming it.
#[rstest]
#[case::negated(BooleanPosition::NegatedOperand, "the operand of a logical not")]
#[case::and(
    BooleanPosition::LogicalOperand { operation: LogicalOperation::And, operand_index: 2 },
    "operand 2 of a logical and"
)]
#[case::or(
    BooleanPosition::LogicalOperand { operation: LogicalOperation::Or, operand_index: 0 },
    "operand 0 of a logical or"
)]
#[case::case_condition(
    BooleanPosition::CaseCondition { case_index: 3 },
    "the condition of piecewise case 3"
)]
#[case::case_value(BooleanPosition::CaseValue { case_index: 1 }, "the value of piecewise case 1")]
#[case::otherwise(BooleanPosition::Otherwise, "the otherwise branch of a piecewise")]
fn boolean_position_display_names_the_position(
    #[case] position: BooleanPosition,
    #[case] expected: &str,
) {
    let text = position.to_string();

    assert_eq!(text, expected);
}

/// Test a refusal has no parent exactly when its operand is the root of a
/// predicate: a numeric root under the predicate screen, and never under
/// the operand screen.
#[rstest]
#[case::numeric_root(Screen::Predicate, build_literal(2), true)]
#[case::negated_number(Screen::Predicate, !build_literal(2), false)]
#[case::conjunction_of_numbers(Screen::Predicate, build_and(&build_literal(2), &build_literal(3)), false)]
#[case::operands_screen(Screen::LogicalOperands, !build_literal(2), false)]
fn boolean_screen_error_parent_is_none_only_at_a_predicate_root(
    #[case] screen: Screen,
    #[case] expression: Expression,
    #[case] is_root: bool,
) {
    let error = expect_refusal(screen.run(&expression));

    assert_eq!(error.parent().is_none(), is_root, "{error:?}");
    assert_eq!(Expression::ptr_eq(error.operand(), &expression), is_root);
}

/// Test a bare screen answers as one told nothing through empty maps and
/// the empty lookup, over refused and accepted fixtures.
#[test]
fn boolean_screen_new_knows_nothing() {
    let (x, x_reference) = build_identifier("x");
    let fixtures = [
        Expression::all([build_literal(2), build_literal(true)]),
        Expression::all([x_reference.clone(), build_literal(true)]),
        !&x_reference,
        build_piecewise_or_panic([(x_reference.less(1), 1)], 2),
        !build_call_or_panic("f", &[build_literal(1)]),
        !Expression::call(BuiltinFunction::Floor, [1.5]),
        x_reference.clone(),
    ];
    let environment: HashMap<Identifier, Expression> = HashMap::new();
    let symbol_types: HashMap<Identifier, SymbolType> = HashMap::new();
    let told_nothing = BooleanScreen::new()
        .with_sorts(&NoRegisteredSorts)
        .with_environment(&environment)
        .with_symbol_types(&symbol_types);

    for fixture in &fixtures {
        let bare = BooleanScreen::new();

        assert_eq!(
            bare.check_logical_operands(fixture),
            told_nothing.check_logical_operands(fixture),
            "{fixture}"
        );
        assert_eq!(
            bare.check_predicate(fixture),
            told_nothing.check_predicate(fixture),
            "{fixture}"
        );
    }
    assert_eq!(
        BooleanScreen::default().check_predicate(&Expression::from(x)),
        Ok(())
    );
}

/// Test a closure declares the symbol types, as a map does.
#[test]
fn boolean_screen_accepts_a_closure_for_symbol_types() {
    let (n, n_reference) = build_identifier("n");
    let (_, p_reference) = build_identifier("p");
    let declare_n_integer = |identifier: &Identifier| (identifier == &n).then_some(SymbolType::Int);
    let expression = Expression::all([p_reference.clone(), n_reference.clone()]);

    let screen = BooleanScreen::new().with_symbol_types(&declare_n_integer);
    let result = screen.check_logical_operands(&expression);

    assert_refusal(
        &expect_refusal(result),
        &n_reference,
        &expression,
        BooleanPosition::LogicalOperand {
            operation: LogicalOperation::And,
            operand_index: 1,
        },
    );
    assert_eq!(screen.check_predicate(&p_reference), Ok(()));
}

/// A lookup counting the named-function result sorts asked of it, and
/// knowing none.
#[derive(Debug, Default)]
struct CountingSorts {
    call_result_sorts: Cell<usize>,
}

impl SortLookup for CountingSorts {
    fn call_result_sort(&self, _name: &FunctionName) -> Option<FunctionSort> {
        self.call_result_sorts.set(self.call_result_sorts.get() + 1);
        None
    }
}

/// Test a predicate screen of a chain of piecewise nodes, each the otherwise
/// branch of the one above and all in Boolean position, judges each node
/// once: the named calls in the case values are looked up at most twice
/// each, not once per enclosing level.
#[test]
fn boolean_screen_judges_each_nested_piecewise_once() {
    const DEPTH: usize = 2000;
    let (_, x) = build_identifier("x");
    let mut chain = build_literal(true);
    for level in 0..DEPTH {
        let value = build_call_or_panic("f", &[build_literal(1)]);
        chain = build_piecewise_or_panic([(x.less(level), value)], chain);
    }
    let sorts = CountingSorts::default();

    let result = BooleanScreen::new()
        .with_sorts(&sorts)
        .check_predicate(&chain);

    assert_eq!(result, Ok(()));
    let lookups = sorts.call_result_sorts.get();
    assert!(lookups <= 2 * DEPTH, "{lookups} lookups for {DEPTH} levels");
    assert!(lookups >= DEPTH, "{lookups} lookups for {DEPTH} levels");
}

/// Test a screen's `Debug` writes its type name, not its lookups.
#[test]
fn boolean_screen_debug_writes_the_type_name() {
    let text = format!("{:?}", BooleanScreen::new());

    assert_eq!(text, "BooleanScreen { .. }");
}

/// Test the empty lookup knows no constant and no function.
#[test]
fn no_registered_sorts_knows_nothing() {
    let (x, _) = build_identifier("x");

    assert_eq!(NoRegisteredSorts.native_constant_sort(&x), None);
    let f = FunctionName::try_new("f").expect("a user function name");

    assert_eq!(NoRegisteredSorts.call_result_sort(&f), None);
}

/// Test the screen takes a boxed lookup through a trait object.
#[test]
fn validate_logical_operands_takes_a_trait_object_lookup() {
    let sorts: Box<dyn SortLookup> = Box::new(NamedSorts);
    let call = build_call_or_panic("real_valued", &[build_literal(2.0)]);
    let expression = !&call;

    let result = BooleanScreen::new()
        .with_sorts(&*sorts)
        .check_logical_operands(&expression);

    assert_eq!(expect_refusal(result).operand(), &call);
}

// =============================================================================
// Shared subtrees
// =============================================================================

/// The number of levels of the DAGs below: they have more than `2^64`
/// occurrences, which no walk visiting every occurrence finishes.
const DAG_LEVELS: usize = 64;

/// Return `x(k+1) = xk && xk` from `x0 = leaf`, `levels` conjunctions deep,
/// both operands of each conjunction one shared node.
fn build_doubling_conjunction(leaf: &Expression, levels: usize) -> Expression {
    let mut dag = leaf.clone();
    for _ in 0..levels {
        dag = dag.and(&dag);
    }
    dag
}

/// Return `x(k+1) = {xk if condition; xk otherwise}` from `x0 = leaf`,
/// `levels` piecewise nodes deep, the value and the otherwise branch of
/// each one shared node.
fn build_doubling_piecewise(
    condition: &Expression,
    leaf: &Expression,
    levels: usize,
) -> Expression {
    let mut dag = leaf.clone();
    for _ in 0..levels {
        dag = build_piecewise_or_panic([(condition, &dag)], &dag);
    }
    dag
}

/// Test both screens pass a doubling conjunction DAG over an undeclared
/// identifier.
#[rstest]
#[case::logical_operands(Screen::LogicalOperands)]
#[case::predicate(Screen::Predicate)]
fn validate_passes_a_doubling_conjunction_dag(#[case] screen: Screen) {
    let (_, p) = build_identifier("p");
    let dag = build_doubling_conjunction(&p, DAG_LEVELS);

    let result = screen.run(&dag);

    assert_eq!(result, Ok(()));
}

/// Test both screens find a number beside a shared doubling conjunction
/// DAG walked before it: `d && (d && 3)`.
#[rstest]
#[case::logical_operands(Screen::LogicalOperands)]
#[case::predicate(Screen::Predicate)]
fn validate_refuses_a_number_beside_a_shared_doubling_dag(#[case] screen: Screen) {
    let (_, p) = build_identifier("p");
    let dag = build_doubling_conjunction(&p, DAG_LEVELS);
    let number = build_literal(3);
    let parent = dag.and(&number);
    let expression = dag.and(&parent);

    let error = expect_refusal(screen.run(&expression));

    assert!(Expression::ptr_eq(error.operand(), &number));
    let (found, position) = error.parent().expect("a conjunction parents the number");
    assert!(Expression::ptr_eq(found, &parent));
    assert_eq!(
        position,
        BooleanPosition::LogicalOperand {
            operation: LogicalOperation::And,
            operand_index: 1,
        }
    );
}

/// Test a doubling piecewise DAG whose leaf is a number is proven numeric
/// as a predicate root and as a negated operand.
#[test]
fn validate_proves_a_doubling_piecewise_dag_numeric() {
    let (_, q) = build_identifier("q");
    let dag = build_doubling_piecewise(&q, &build_literal(1), DAG_LEVELS);
    let negation = !&dag;

    let root_error = expect_refusal(Screen::Predicate.run(&dag));
    let negated_error = expect_refusal(Screen::LogicalOperands.run(&negation));

    assert!(Expression::ptr_eq(root_error.operand(), &dag));
    assert_eq!(find_position(&root_error), None);
    assert!(Expression::ptr_eq(negated_error.operand(), &dag));
    assert_eq!(
        find_position(&negated_error),
        Some(BooleanPosition::NegatedOperand)
    );
}

/// Test a doubling piecewise DAG whose leaf is an undeclared identifier
/// passes as a predicate, every branch of it sitting in a Boolean position.
#[test]
fn validate_predicate_passes_a_doubling_piecewise_dag_over_an_identifier() {
    let (_, q) = build_identifier("q");
    let (_, r) = build_identifier("r");
    let dag = build_doubling_piecewise(&q, &r, DAG_LEVELS);

    let result = Screen::Predicate.run(&dag);

    assert_eq!(result, Ok(()));
}

/// Test an identifier occurring throughout a doubling DAG and bound to a
/// DAG is screened through its bound value: a Boolean value passes, and a
/// value with a number beside a shared subtree is refused there.
#[test]
fn validate_logical_operands_screens_a_bound_dag_at_a_shared_identifier() {
    let (b, b_reference) = build_identifier("b");
    let (_, p) = build_identifier("p");
    let expression = build_doubling_conjunction(&b_reference, DAG_LEVELS);
    let boolean = build_doubling_conjunction(&p, DAG_LEVELS);
    let number = build_literal(3);
    let parent = boolean.and(&number);
    let numeric = boolean.and(&parent);
    let sorts = BuiltinSorts::new();

    let passed = Screen::LogicalOperands.run_with(
        &expression,
        &HashMap::from([(b.clone(), boolean.clone())]),
        &HashMap::new(),
        &sorts,
    );
    let refused = Screen::LogicalOperands.run_with(
        &expression,
        &HashMap::from([(b, numeric)]),
        &HashMap::new(),
        &sorts,
    );

    assert_eq!(passed, Ok(()));
    let error = expect_refusal(refused);
    assert!(Expression::ptr_eq(error.operand(), &number));
    assert!(
        error
            .parent()
            .is_some_and(|(found, _)| Expression::ptr_eq(found, &parent))
    );
}

// =============================================================================
// Deep trees
// =============================================================================

/// Test a Boolean conjunction [`SMALL_STACK_DEPTH`] levels deep passes,
/// and one with a number at the bottom is refused there by both screens, on
/// a small thread stack.
#[test]
fn validate_walks_a_deep_conjunction_on_a_small_stack() {
    run_on_small_stack(|| {
        let (_, p) = build_identifier("p");
        let boolean = build_deep_conjunction(&p, SMALL_STACK_DEPTH);
        let number = build_literal(3);
        let numeric = build_deep_conjunction(&number, SMALL_STACK_DEPTH);

        let boolean_result = Screen::LogicalOperands.run(&boolean);
        let predicate_result = Screen::Predicate.run(&boolean);
        let numeric_error = expect_refusal(Screen::LogicalOperands.run(&numeric));
        let predicate_error = expect_refusal(Screen::Predicate.run(&numeric));

        assert_eq!(boolean_result, Ok(()));
        assert_eq!(predicate_result, Ok(()));
        assert!(Expression::ptr_eq(numeric_error.operand(), &number));
        assert!(Expression::ptr_eq(predicate_error.operand(), &number));
    });
}

/// Test a piecewise nested [`SMALL_STACK_DEPTH`] levels deep along its
/// otherwise branches, with numeric values and a numeric bottom, is proven
/// numeric by both screens on a small thread stack: as a negated operand and
/// as a predicate root.
#[test]
fn validate_proves_a_deep_piecewise_numeric_on_a_small_stack() {
    run_on_small_stack(|| {
        let (_, p) = build_identifier("p");
        let mut piecewise = build_literal(3);
        for _ in 0..SMALL_STACK_DEPTH {
            piecewise = build_piecewise_or_panic([(&p, build_literal(1))], piecewise);
        }
        let negation = !&piecewise;

        let negated_error = expect_refusal(Screen::LogicalOperands.run(&negation));
        let root_error = expect_refusal(Screen::Predicate.run(&piecewise));

        assert!(Expression::ptr_eq(negated_error.operand(), &piecewise));
        assert_eq!(
            find_position(&negated_error),
            Some(BooleanPosition::NegatedOperand)
        );
        assert!(Expression::ptr_eq(root_error.operand(), &piecewise));
        assert_eq!(find_position(&root_error), None);
    });
}

/// Test the unary operation of a negated operand is a logical negation.
#[test]
fn validate_logical_operands_negated_operand_parent_is_a_negation() {
    let expression = !build_literal(3);

    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    let (parent, position) = error.parent().expect("a negation parents its operand");
    assert_eq!(
        parent,
        &Expression::new_unary(UnaryOperation::LogicalNot, build_literal(3))
    );
    assert_eq!(position, BooleanPosition::NegatedOperand);
}

/// Test both screens pass a doubling conjunction DAG
/// [`SMALL_STACK_DEPTH`] levels deep, and prove a doubling piecewise DAG
/// that deep over a number numeric, on a small thread stack.
#[test]
fn validate_walks_deep_doubling_dags_on_a_small_stack() {
    run_on_small_stack(|| {
        let (_, p) = build_identifier("p");
        let conjunction = build_doubling_conjunction(&p, SMALL_STACK_DEPTH);
        let piecewise = build_doubling_piecewise(&p, &build_literal(1), SMALL_STACK_DEPTH);

        let operands_result = Screen::LogicalOperands.run(&conjunction);
        let predicate_result = Screen::Predicate.run(&conjunction);
        let root_error = expect_refusal(Screen::Predicate.run(&piecewise));

        assert_eq!(operands_result, Ok(()));
        assert_eq!(predicate_result, Ok(()));
        assert!(Expression::ptr_eq(root_error.operand(), &piecewise));
    });
}
