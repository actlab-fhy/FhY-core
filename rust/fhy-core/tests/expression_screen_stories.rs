//! Tests for the Boolean-position screens `validate_logical_operands` and
//! `validate_predicate`, and for the error they report.
//!
//! Public API only (`fhy_core::symbolic::expression`). Calls and native
//! constants take their sorts from a test-local [`SortLookup`] holding the
//! sorts of the built-in functions and constants the tests name (`floor`
//! returns an integer, `sqrt` a real, `nand` a Boolean; `pi`, `e`, `inf`
//! and `nan` are real constants), so the tests exercise the same sorts a
//! populated function registry reports.

#[path = "common/expression.rs"]
pub mod expression_support;

use std::collections::HashMap;
use std::thread;

use expression_support::{
    DEEP_TREE_DEPTH, WALK_STACK_BYTES, build_call_or_panic, build_deep_conjunction, build_deep_sum,
    build_identifier, build_literal, build_piecewise_or_panic, build_text_literal,
    run_on_large_stack,
};
use fhy_core::identifier::Identifier;
use fhy_core::symbolic::expression::{
    BinaryOperation, BooleanPosition, Expression, FunctionSort, NoRegisteredSorts,
    NonBooleanLogicalOperandError, SortLookup, UnaryOperation, build_logical_and, build_logical_or,
    build_piecewise, validate_logical_operands, validate_predicate,
};
use fhy_core::symbolic::symbol_type::SymbolType;
use rstest::rstest;

/// The tail of every refusal message naming a parent node.
const ILL_TYPED: &str = "which provably denotes a number; the expression is ill-typed and no \
                         symbolic backend lowers it faithfully";

/// Depth of the tree whose refusal is displayed on a small stack: far
/// deeper than a renderer recursing once per level could reach.
const DISPLAY_TREE_DEPTH: usize = 100_000;

/// Stack size of the thread a deep refusal is displayed on.
const SMALL_STACK_BYTES: usize = 128 << 10;

/// Names of the real-valued built-in constants.
const REAL_CONSTANT_NAMES: [&str; 4] = ["pi", "e", "inf", "nan"];

/// The sorts of the built-in functions and constants the tests name, plus
/// any constants a test adds.
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

    fn call_result_sort(&self, function_name: &str) -> Option<FunctionSort> {
        match function_name {
            "floor" => Some(FunctionSort::Int),
            "sqrt" | "max" => Some(FunctionSort::Real),
            "nand" => Some(FunctionSort::Bool),
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
        match self {
            Self::LogicalOperands => {
                validate_logical_operands(expression, environment, symbol_types, sorts)
            }
            Self::Predicate => validate_predicate(expression, environment, symbol_types, sorts),
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
    parent: Option<&Expression>,
    position: BooleanPosition,
) {
    assert_eq!(error.operand(), operand, "operand of {error:?}");
    assert_eq!(error.parent(), parent, "parent of {error:?}");
    assert_eq!(error.position(), position, "position of {error:?}");
}

/// Return the conjunction `left && right`.
fn build_and(left: &Expression, right: &Expression) -> Expression {
    build_logical_and([left, right]).expect("two operands")
}

/// Return the disjunction `left || right`.
fn build_or(left: &Expression, right: &Expression) -> Expression {
    build_logical_or([left, right]).expect("two operands")
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
            Self::Negated => operand.logical_not(),
            Self::CaseCondition => {
                build_piecewise_or_panic([(operand, &build_literal(1))], build_literal(2))
            }
        }
    }

    /// Return the position the placement puts its operand in.
    fn position(self) -> BooleanPosition {
        match self {
            Self::AndLeft => BooleanPosition::LogicalOperand {
                operation: BinaryOperation::LogicalAnd,
            },
            Self::OrRight => BooleanPosition::LogicalOperand {
                operation: BinaryOperation::LogicalOr,
            },
            Self::Negated => BooleanPosition::NegatedOperand,
            Self::CaseCondition => BooleanPosition::CaseCondition { case_index: 0 },
        }
    }
}

// =============================================================================
// validate_logical_operands: numbers under connectives
// =============================================================================

/// Test a connective over a number is refused, naming the first numeric
/// operand and the connective.
#[rstest]
#[case::and(build_and(&build_literal(2), &build_literal(4)), build_literal(2), BinaryOperation::LogicalAnd)]
#[case::or(build_or(&build_literal(2), &build_literal(4)), build_literal(2), BinaryOperation::LogicalOr)]
#[case::and_one_numeric_operand(build_and(&build_literal(true), &build_literal(4)), build_literal(4), BinaryOperation::LogicalAnd)]
#[case::and_floats(build_and(&build_literal(1.5), &build_literal(2.5)), build_literal(1.5), BinaryOperation::LogicalAnd)]
#[case::and_texts(build_and(&build_text_literal("2"), &build_text_literal("4")), build_text_literal("2"), BinaryOperation::LogicalAnd)]
#[case::and_arithmetic_operand(build_and(&(build_literal(1) + 2), &build_literal(true)), build_literal(1) + 2, BinaryOperation::LogicalAnd)]
fn validate_logical_operands_rejects_a_numeric_connective_operand(
    #[case] expression: Expression,
    #[case] operand: Expression,
    #[case] operation: BinaryOperation,
) {
    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    assert_refusal(
        &error,
        &operand,
        Some(&expression),
        BooleanPosition::LogicalOperand { operation },
    );
}

/// Test a negation of a number is refused, arithmetic negation included.
#[rstest]
#[case::literal(build_literal(2))]
#[case::negation(-build_literal(1))]
fn validate_logical_operands_rejects_a_numeric_negated_operand(#[case] operand: Expression) {
    let expression = operand.logical_not();

    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    assert_refusal(
        &error,
        &operand,
        Some(&expression),
        BooleanPosition::NegatedOperand,
    );
}

/// Test the message names the connective node and the numeric operand.
#[test]
fn validate_logical_operands_error_names_the_connective_and_the_operand() {
    let expression = build_or(&build_literal(2), &build_literal(4));

    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    assert_eq!(
        error.to_string(),
        format!("(2 || 4) applies the Boolean connective logical_or to the operand 2, {ILL_TYPED}")
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
        Some(&nested),
        BooleanPosition::LogicalOperand {
            operation: BinaryOperation::LogicalAnd,
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
        Some(&expression),
        BooleanPosition::LogicalOperand {
            operation: BinaryOperation::LogicalAnd,
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
        Some(&expression),
        BooleanPosition::LogicalOperand {
            operation: BinaryOperation::LogicalAnd,
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

    let error = expect_refusal(Screen::LogicalOperands.run(&mixed.logical_not()));

    assert_refusal(&error, &build_literal(2), Some(&mixed), position);
}

/// Test a branch shared by a case value and the otherwise branch is
/// reported at its first position, the case value.
#[test]
fn validate_logical_operands_reports_a_shared_branch_at_its_first_position() {
    let (_, x) = build_identifier("x");
    let shared = build_literal(2);
    let mixed = build_piecewise(
        [
            (x.greater(0), shared.clone()),
            (x.less(0), build_literal(true)),
        ],
        &shared,
    )
    .expect("a valid piecewise");

    let error = expect_refusal(Screen::LogicalOperands.run(&mixed.logical_not()));

    assert_refusal(
        &error,
        &shared,
        Some(&mixed),
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
    let mixed = build_piecewise(
        [
            (x.greater(0), build_literal(2)),
            (numeric_condition.clone(), build_literal(true)),
        ],
        build_literal(true),
    )
    .expect("a valid piecewise");
    let expression = match screen {
        Screen::LogicalOperands => mixed.logical_not(),
        Screen::Predicate => mixed.clone(),
    };

    let error = expect_refusal(screen.run(&expression));

    assert_refusal(
        &error,
        &numeric_condition,
        Some(&mixed),
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

    assert_refusal(&error, &call, Some(&expression), placement.position());
}

/// Test a call nothing knows the sort of is not refused.
#[test]
fn validate_logical_operands_accepts_a_call_the_lookup_does_not_know() {
    let call = build_call_or_panic("floor", &[build_literal(1.5)]);
    let expression = build_and(&call, &build_literal(true));

    let result = validate_logical_operands(
        &expression,
        &HashMap::new(),
        &HashMap::new(),
        &NoRegisteredSorts,
    );

    assert_eq!(result, Ok(()));
}

/// Test operands the screen cannot prove numeric pass.
#[rstest]
#[case::boolean_literals(build_and(&build_literal(true), &build_literal(false)))]
#[case::boolean_literal_negation(build_literal(true).logical_not())]
#[case::unbound_identifiers(build_and(&build_identifier("p").1, &build_identifier("q").1))]
#[case::comparisons({ let x = build_identifier("x").1; build_and(&x.greater(0), &x.less(5)) })]
#[case::boolean_call(build_and(&build_call_or_panic("nand", &[build_literal(true), build_literal(true)]), &build_literal(true)))]
#[case::nested_connective(build_and(&build_identifier("p").1.logical_not(), &build_literal(true)))]
fn validate_logical_operands_accepts_an_operand_it_cannot_prove_numeric(
    #[case] expression: Expression,
) {
    let result = Screen::LogicalOperands.run(&expression);

    assert_eq!(result, Ok(()));
}

// =============================================================================
// validate_logical_operands: bindings and declared types
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
        Some(&expression),
        BooleanPosition::LogicalOperand {
            operation: BinaryOperation::LogicalAnd,
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
        Some(&expression),
        BooleanPosition::CaseCondition { case_index: 0 },
    );
}

/// Test the message names the piecewise and its condition.
#[test]
fn validate_logical_operands_error_names_the_piecewise_and_its_condition() {
    let (x_identifier, x) = build_identifier("x");
    let condition = &x * 2;
    let expression = build_piecewise_or_panic([(&condition, &build_literal(5))], build_literal(0));

    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    let id = x_identifier.id();
    assert_eq!(
        error.to_string(),
        format!(
            "{{5 if (x::{id} * 2); 0 otherwise}} takes (x::{id} * 2) as the condition of case 0, \
             {ILL_TYPED}"
        )
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

    assert_refusal(&error, &constant, Some(&expression), placement.position());
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

    assert_refusal(&error, &reference, Some(&expression), placement.position());
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
        Some(&expression),
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
        &reference.logical_not(),
        &HashMap::from([(x, mixed.clone())]),
        &HashMap::new(),
        &BuiltinSorts::new(),
    ));

    assert_refusal(&error, &build_literal(1), Some(&mixed), position);
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
// validate_predicate: the root is a Boolean position
// =============================================================================

/// Test a numeric root is refused as a predicate root.
#[rstest]
#[case::int_literal(build_literal(2))]
#[case::float_literal(build_literal(1.5))]
#[case::decimal_text_literal(build_text_literal("2.5"))]
#[case::arithmetic_node(build_literal(1) + 2)]
#[case::negation(-build_literal(1))]
#[case::integer_result_call(build_call_or_panic("floor", &[build_literal(1.5)]))]
#[case::real_result_call(build_call_or_panic("sqrt", &[build_literal(4.0)]))]
fn validate_predicate_rejects_a_numeric_root(#[case] expression: Expression) {
    let error = expect_refusal(Screen::Predicate.run(&expression));

    assert_refusal(&error, &expression, None, BooleanPosition::PredicateRoot);
}

/// Test a native real constant as the root is refused.
#[test]
fn validate_predicate_rejects_a_native_constant_root() {
    let sorts = BuiltinSorts::new();
    let pi = sorts.reference_constant("pi");

    let error =
        expect_refusal(Screen::Predicate.run_with(&pi, &HashMap::new(), &HashMap::new(), &sorts));

    assert_refusal(&error, &pi, None, BooleanPosition::PredicateRoot);
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
        Some(&expression),
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

    assert_refusal(&error, &reference, None, BooleanPosition::PredicateRoot);
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

    assert_refusal(&error, &reference, None, BooleanPosition::PredicateRoot);
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
        Some(&expression),
        BooleanPosition::LogicalOperand {
            operation: BinaryOperation::LogicalAnd,
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
        Some(&mixed),
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

/// Test the message for each position.
#[rstest]
#[case::negated(BooleanPosition::NegatedOperand)]
#[case::logical(BooleanPosition::LogicalOperand { operation: BinaryOperation::LogicalAnd })]
#[case::case_condition(BooleanPosition::CaseCondition { case_index: 0 })]
#[case::case_value(BooleanPosition::CaseValue { case_index: 0 })]
#[case::otherwise(BooleanPosition::Otherwise)]
#[case::predicate_root(BooleanPosition::PredicateRoot)]
fn non_boolean_logical_operand_error_display_describes_the_position(
    #[case] position: BooleanPosition,
) {
    let number = -build_literal(7);
    let (expression, screen) = match position {
        BooleanPosition::NegatedOperand => (number.logical_not(), Screen::LogicalOperands),
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
        BooleanPosition::PredicateRoot => (number.clone(), Screen::Predicate),
        _ => unreachable!("every position has a case"),
    };
    let expected = match position {
        BooleanPosition::NegatedOperand => format!(
            "(!(-7)) applies the Boolean connective logical_not to the operand (-7), {ILL_TYPED}"
        ),
        BooleanPosition::LogicalOperand { .. } => format!(
            "((-7) && True) applies the Boolean connective logical_and to the operand (-7), \
             {ILL_TYPED}"
        ),
        BooleanPosition::CaseCondition { .. } => format!(
            "{{True if (-7); True otherwise}} takes (-7) as the condition of case 0, {ILL_TYPED}"
        ),
        BooleanPosition::CaseValue { .. } => format!(
            "{{(-7) if True; True otherwise}} takes (-7) as the value of case 0, {ILL_TYPED}"
        ),
        BooleanPosition::Otherwise => format!(
            "{{True if True; (-7) otherwise}} takes (-7) as its otherwise branch, {ILL_TYPED}"
        ),
        BooleanPosition::PredicateRoot => String::from(
            "(-7) is used as a predicate but provably denotes a number; the expression is \
             ill-typed and no symbolic backend lowers it faithfully",
        ),
        _ => unreachable!("every position has a case"),
    };

    let error = expect_refusal(screen.run(&expression));

    assert_eq!(error.position(), position);
    assert_eq!(error.to_string(), expected);
}

/// Test the message renders identifiers with their ids, as `name::id`.
#[test]
fn non_boolean_logical_operand_error_display_writes_identifier_ids() {
    let (x_identifier, x) = build_identifier("x");
    let expression = -&x;

    let error = expect_refusal(Screen::Predicate.run(&expression));

    let id = x_identifier.id();
    assert_eq!(
        error.to_string(),
        format!(
            "(-x::{id}) is used as a predicate but provably denotes a number; the expression is \
             ill-typed and no symbolic backend lowers it faithfully"
        )
    );
}

/// Test displaying a refusal whose parent and operand are
/// [`DISPLAY_TREE_DEPTH`] levels deep completes on a small stack.
#[test]
fn non_boolean_logical_operand_error_display_writes_a_deep_tree_on_a_small_stack() {
    let operand = build_deep_sum(&build_literal(0), DISPLAY_TREE_DEPTH);
    let expression = operand.logical_not();
    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    let handle = thread::Builder::new()
        .stack_size(SMALL_STACK_BYTES)
        .spawn(move || (error.to_string(), error))
        .expect("the display thread spawns");
    let (text, _error) = match handle.join() {
        Ok(result) => result,
        Err(payload) => std::panic::resume_unwind(payload),
    };

    let operand_text = format!(
        "{}0{}",
        "(".repeat(DISPLAY_TREE_DEPTH),
        " + 1)".repeat(DISPLAY_TREE_DEPTH)
    );
    assert_eq!(
        text,
        format!(
            "(!{operand_text}) applies the Boolean connective logical_not to the operand \
             {operand_text}, {ILL_TYPED}"
        )
    );
}

/// Test the empty lookup knows no constant and no function.
#[test]
fn no_registered_sorts_knows_nothing() {
    let (x, _) = build_identifier("x");

    assert_eq!(NoRegisteredSorts.native_constant_sort(&x), None);
    assert_eq!(NoRegisteredSorts.call_result_sort("floor"), None);
}

/// Test the screens take a boxed lookup through a trait object.
#[test]
fn validate_logical_operands_takes_a_trait_object_lookup() {
    let sorts: Box<dyn SortLookup> = Box::new(BuiltinSorts::new());
    let call = build_call_or_panic("sqrt", &[build_literal(2.0)]);
    let expression = call.logical_not();

    let result = validate_logical_operands(&expression, &HashMap::new(), &HashMap::new(), &*sorts);

    assert_eq!(expect_refusal(result).operand(), &call);
}

// =============================================================================
// Deep trees
// =============================================================================

/// Test a Boolean conjunction thousands of levels deep passes, and one with a
/// number at the bottom is refused there.
#[test]
fn validate_logical_operands_walks_a_deep_conjunction() {
    run_on_large_stack(WALK_STACK_BYTES, || {
        let (_, p) = build_identifier("p");
        let boolean = build_deep_conjunction(&p, DEEP_TREE_DEPTH);
        let number = build_literal(3);
        let numeric = build_deep_conjunction(&number, DEEP_TREE_DEPTH);

        let boolean_result = Screen::LogicalOperands.run(&boolean);
        let numeric_error = expect_refusal(Screen::LogicalOperands.run(&numeric));
        let predicate_error = expect_refusal(Screen::Predicate.run(&numeric));

        assert_eq!(boolean_result, Ok(()));
        assert_eq!(numeric_error.operand(), &number);
        assert_eq!(predicate_error.operand(), &number);
    });
}

/// Test the unary operation of a negated operand is a logical negation.
#[test]
fn validate_logical_operands_negated_operand_parent_is_a_negation() {
    let expression = build_literal(3).logical_not();

    let error = expect_refusal(Screen::LogicalOperands.run(&expression));

    let parent = error.parent().expect("a negation parents its operand");
    assert_eq!(
        parent,
        &Expression::new_unary(UnaryOperation::LogicalNot, build_literal(3))
    );
}
