//! Stories for the SMT-LIB2 lowering, [`SmtScript::lower`]: the text of
//! each kind of node, the exact rationals of literals, `to_real`, the floor
//! encodings, powers by squaring, the logic, symbols, sharing, depth, and
//! every refusal.

use std::collections::HashMap;

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::{
    BigInt, Expression, FunctionName, LiteralValue, NoRegisteredSorts, SymbolType,
};
use fhy_core::identifier::Identifier;
use fhy_core::solver::{Logic, LoweringError, SmtScript};
use rstest::rstest;

use crate::support::expression::{build_identifier, build_literal};
use crate::support::solver::{build_symbol_types, quoted_symbol};
use crate::support::stack::{SMALL_STACK_DEPTH, run_on_small_stack};

/// Return the script of `expression`, failing the test on a refusal.
fn lower(expression: &Expression, symbol_types: &HashMap<Identifier, SymbolType>) -> SmtScript {
    SmtScript::lower(expression, symbol_types, &NoRegisteredSorts).expect("the expression lowers")
}

/// Return the refusal of `expression`, failing the test if it lowers.
fn refuse(
    expression: &Expression,
    symbol_types: &HashMap<Identifier, SymbolType>,
) -> LoweringError {
    SmtScript::lower(expression, symbol_types, &NoRegisteredSorts)
        .expect_err("the expression is refused")
}

/// Return the text of the one assertion of `script`, without `(assert` and
/// its closing parenthesis.
fn asserted(script: &SmtScript) -> String {
    let text = script.to_string();
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| line.starts_with("(assert "))
        .collect();
    assert_eq!(lines.len(), 1, "one assertion in {text}");
    lines[0]
        .strip_prefix("(assert ")
        .and_then(|line| line.strip_suffix(')'))
        .expect("an assert line")
        .to_owned()
}

/// Return the asserted term of `expression` over identifiers all of `sort`.
fn asserted_over(expression: &Expression, sort: SymbolType) -> String {
    let symbol_types = |_: &Identifier| Some(sort);
    asserted(&SmtScript::lower(expression, &symbol_types, &NoRegisteredSorts).expect("lowers"))
}

/// Return a decimal literal parsed from `text`.
fn build_decimal(text: &str) -> Expression {
    build_literal(LiteralValue::parse_text(text).expect("a decimal text"))
}

// ---------------------------------------------------------------------------
// Scripts, declarations and symbols
// ---------------------------------------------------------------------------

#[test]
fn script_sets_the_logic_declares_each_constant_and_asserts_the_predicate() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let expression = x_reference.greater(y_reference);
    let symbol_types = build_symbol_types(&[(&y, SymbolType::Int), (&x, SymbolType::Int)]);

    let script = lower(&expression, &symbol_types);

    assert_eq!(
        script.to_string(),
        format!(
            "(set-logic QF_LIA)\n(declare-const {x} Int)\n(declare-const {y} Int)\n(assert (> {x} {y}))\n",
            x = quoted_symbol(&x),
            y = quoted_symbol(&y),
        )
    );
    assert_eq!(script.logic(), Logic::QfLia);
    assert_eq!(script.value_sort(), None);
}

#[test]
fn declarations_pair_each_identifier_with_its_symbol_and_sort_ordered_by_id() {
    let (b, b_reference) = build_identifier("b");
    let (r, r_reference) = build_identifier("r");
    let (n, n_reference) = build_identifier("n");
    let expression = Expression::all([b_reference, r_reference.greater(0.5), n_reference.less(3)]);
    let symbol_types = build_symbol_types(&[
        (&n, SymbolType::Int),
        (&b, SymbolType::Bool),
        (&r, SymbolType::Real),
    ]);

    let script = lower(&expression, &symbol_types);
    let declarations: Vec<(u64, String, SymbolType)> = script
        .declarations()
        .iter()
        .map(|declaration| {
            (
                declaration.identifier().id(),
                declaration.symbol().to_owned(),
                declaration.sort(),
            )
        })
        .collect();

    assert_eq!(
        declarations,
        vec![
            (b.id(), format!("b_{}", b.id()), SymbolType::Bool),
            (r.id(), format!("r_{}", r.id()), SymbolType::Real),
            (n.id(), format!("n_{}", n.id()), SymbolType::Int),
        ]
    );
    assert!(
        script
            .to_string()
            .contains(&format!("(declare-const |b_{}| Bool)", b.id()))
    );
    assert!(
        script
            .to_string()
            .contains(&format!("(declare-const |r_{}| Real)", r.id()))
    );
}

#[test]
fn symbol_replaces_the_characters_a_quoted_symbol_cannot_hold() {
    let (odd, reference) = build_identifier(r"a|b\c d");
    let script = lower(&reference, &build_symbol_types(&[(&odd, SymbolType::Bool)]));

    assert_eq!(
        script.declarations()[0].symbol(),
        format!("a_b_c d_{}", odd.id())
    );
    assert_eq!(asserted(&script), format!("|a_b_c d_{}|", odd.id()));
}

#[rstest]
#[case::nul("a\0b", "a_b")]
#[case::bell("a\u{7}b", "a_b")]
#[case::newline_and_tab("a\nb\tc", "a_b_c")]
#[case::delete("a\u{7f}b", "a_b")]
#[case::c1_control("a\u{85}b", "a_b")]
fn a_name_hint_with_control_characters_lowers_to_printable_text(
    #[case] name_hint: &str,
    #[case] sanitized: &str,
) {
    let (odd, reference) = build_identifier(name_hint);
    let script = lower(&reference, &build_symbol_types(&[(&odd, SymbolType::Bool)]));
    let symbol = format!("{sanitized}_{}", odd.id());

    assert_eq!(script.declarations()[0].symbol(), symbol);
    assert_eq!(asserted(&script), format!("|{symbol}|"));
    assert!(
        !script
            .to_string()
            .chars()
            .any(|character| character.is_control() && character != '\n'),
        "no control character but the line breaks reaches the script"
    );
}

#[test]
fn identifiers_sharing_a_name_hint_get_distinct_symbols() {
    let (first, first_reference) = build_identifier("x");
    let (second, second_reference) = build_identifier("x");
    let expression = first_reference.less(second_reference);
    let symbol_types = build_symbol_types(&[(&first, SymbolType::Int), (&second, SymbolType::Int)]);

    assert_eq!(
        asserted(&lower(&expression, &symbol_types)),
        format!("(< {} {})", quoted_symbol(&first), quoted_symbol(&second))
    );
}

#[test]
fn numeric_expression_is_named_by_the_value_constant() {
    let (x, reference) = build_identifier("x");
    let script = lower(
        &(reference + 1),
        &build_symbol_types(&[(&x, SymbolType::Int)]),
    );

    assert_eq!(script.value_sort(), Some(SymbolType::Int));
    assert_eq!(
        script.to_string(),
        format!(
            "(set-logic QF_LIA)\n(declare-const {x} Int)\n(declare-const value Int)\n(assert (= value (+ {x} 1)))\n",
            x = quoted_symbol(&x)
        )
    );
    assert_eq!(
        script.declarations().len(),
        1,
        "value is not an identifier's constant"
    );
}

// ---------------------------------------------------------------------------
// Literals
// ---------------------------------------------------------------------------

#[rstest]
#[case::true_literal(true, "true")]
#[case::false_literal(false, "false")]
fn boolean_literal_is_true_or_false(#[case] value: bool, #[case] text: &str) {
    let script = lower(&build_literal(value), &HashMap::new());

    assert_eq!(asserted(&script), text);
    assert_eq!(script.logic(), Logic::All, "only booleans");
}

#[rstest]
#[case::positive(BigInt::from(5), "5")]
#[case::zero(BigInt::from(0), "0")]
#[case::negative(BigInt::from(-5), "(- 5)")]
#[case::big(BigInt::from(10).pow(30), "1000000000000000000000000000000")]
fn integer_literal_is_a_numeral_negated_when_negative(#[case] value: BigInt, #[case] text: &str) {
    let script = lower(&build_literal(value), &HashMap::new());

    assert_eq!(script.value_sort(), Some(SymbolType::Int));
    assert_eq!(asserted(&script), format!("(= value {text})"));
}

#[rstest]
#[case::tenth(0.1, "(/ 3602879701896397.0 36028797018963968.0)")]
#[case::large(1e16, "10000000000000000.0")]
#[case::half(0.5, "(/ 1.0 2.0)")]
#[case::negative(-2.5, "(- (/ 5.0 2.0))")]
#[case::whole(3.0, "3.0")]
#[case::negative_whole(-3.0, "(- 3.0)")]
#[case::zero(0.0, "0.0")]
#[case::negative_zero(-0.0, "0.0")]
fn float_literal_is_its_exact_binary_rational(#[case] value: f64, #[case] text: &str) {
    let script = lower(&build_literal(value), &HashMap::new());

    assert_eq!(script.value_sort(), Some(SymbolType::Real));
    assert_eq!(asserted(&script), format!("(= value {text})"));
}

#[rstest]
#[case::tenth("0.1", "(/ 1.0 10.0)")]
#[case::reduced("2.50", "(/ 5.0 2.0)")]
#[case::whole("100.0", "100.0")]
#[case::small("0.004", "(/ 1.0 250.0)")]
fn decimal_literal_is_its_exact_decimal_rational(#[case] text: &str, #[case] term: &str) {
    let script = lower(&build_decimal(text), &HashMap::new());

    assert_eq!(asserted(&script), format!("(= value {term})"));
}

#[test]
fn non_finite_float_is_refused() {
    let literal = build_literal(f64::NAN);

    assert_eq!(
        refuse(&literal.clone().equals(1.0), &HashMap::new()),
        LoweringError::NonFiniteLiteral(literal)
    );
}

// ---------------------------------------------------------------------------
// Arithmetic and to_real
// ---------------------------------------------------------------------------

#[rstest]
#[case::sum(|x: Expression, y: Expression| x + y, "(+ X Y)")]
#[case::difference(|x: Expression, y: Expression| x - y, "(- X Y)")]
#[case::product(|x: Expression, y: Expression| x * y, "(* X Y)")]
#[case::negation(|x: Expression, _: Expression| -x, "(- X)")]
#[case::plus(|x: Expression, _: Expression| x.positive(), "X")]
fn arithmetic_is_the_smt_lib2_operator(
    #[case] build: fn(Expression, Expression) -> Expression,
    #[case] template: &str,
) {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let expression = build(x_reference, y_reference);
    let symbol_types = |_: &Identifier| Some(SymbolType::Int);
    let script = SmtScript::lower(&expression, &symbol_types, &NoRegisteredSorts).expect("lowers");

    assert_eq!(
        asserted(&script),
        format!(
            "(= value {})",
            template
                .replace('X', &quoted_symbol(&x))
                .replace('Y', &quoted_symbol(&y))
        )
    );
}

#[test]
fn integer_side_meeting_a_real_is_converted_with_to_real_in_every_position() {
    let (n, n_reference) = build_identifier("n");
    let (r, r_reference) = build_identifier("r");
    let (b, b_reference) = build_identifier("b");
    let symbol_types = build_symbol_types(&[
        (&n, SymbolType::Int),
        (&r, SymbolType::Real),
        (&b, SymbolType::Bool),
    ]);
    let (n_symbol, r_symbol, b_symbol) = (quoted_symbol(&n), quoted_symbol(&r), quoted_symbol(&b));
    let branches = Expression::piecewise([(b_reference, n_reference.clone())], r_reference.clone())
        .expect("a piecewise");

    assert_eq!(
        asserted(&lower(
            &(n_reference.clone() + r_reference.clone()),
            &symbol_types
        )),
        format!("(= value (+ (to_real {n_symbol}) {r_symbol}))")
    );
    assert_eq!(
        asserted(&lower(
            &n_reference.clone().less(r_reference.clone()),
            &symbol_types
        )),
        format!("(< (to_real {n_symbol}) {r_symbol})")
    );
    assert_eq!(
        asserted(&lower(&n_reference.equals(r_reference), &symbol_types)),
        format!("(= (to_real {n_symbol}) {r_symbol})")
    );
    assert_eq!(
        asserted(&lower(&branches, &symbol_types)),
        format!("(= value (ite {b_symbol} (to_real {n_symbol}) {r_symbol}))")
    );
}

#[test]
fn integer_literal_in_a_real_position_is_a_real_numeral() {
    let (r, reference) = build_identifier("r");
    let symbol_types = build_symbol_types(&[(&r, SymbolType::Real)]);
    let script = lower(&(reference.clone() + 1).greater(-2), &symbol_types);

    assert_eq!(
        asserted(&script),
        format!("(> (+ {r} 1.0) (- 2.0))", r = quoted_symbol(&r))
    );
    assert_eq!(script.logic(), Logic::QfLra);
}

#[test]
fn division_is_exact_over_the_reals() {
    let (n, n_reference) = build_identifier("n");
    let (r, r_reference) = build_identifier("r");
    let symbol_types = build_symbol_types(&[(&n, SymbolType::Int), (&r, SymbolType::Real)]);
    let integer_division = lower(&(n_reference.clone() / 2), &symbol_types);
    let real_division = lower(&(r_reference.clone() / r_reference), &symbol_types);

    assert_eq!(
        asserted(&integer_division),
        format!("(= value (/ (to_real {}) 2.0))", quoted_symbol(&n))
    );
    assert_eq!(integer_division.value_sort(), Some(SymbolType::Real));
    assert_eq!(
        integer_division.logic(),
        Logic::All,
        "to_real mixes the sorts"
    );
    assert_eq!(
        asserted(&real_division),
        format!("(= value (/ {r} {r}))", r = quoted_symbol(&r))
    );
    assert_eq!(real_division.logic(), Logic::QfNra);
    let _ = n_reference;
}

#[test]
fn integer_floor_division_divides_by_the_sign_of_the_divisor() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let (xs, ys) = (quoted_symbol(&x), quoted_symbol(&y));

    assert_eq!(
        asserted_over(&x_reference.clone().floor_divide(3), SymbolType::Int),
        format!("(= value (div {xs} 3))")
    );
    assert_eq!(
        asserted_over(&x_reference.clone().floor_divide(-3), SymbolType::Int),
        format!("(= value (div (- {xs}) 3))")
    );
    assert_eq!(
        asserted_over(&x_reference.floor_divide(y_reference), SymbolType::Int),
        format!("(= value (ite (> {ys} 0) (div {xs} {ys}) (div (- {xs}) (- {ys}))))")
    );
}

#[test]
fn integer_floor_modulo_takes_the_sign_of_the_divisor() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let (xs, ys) = (quoted_symbol(&x), quoted_symbol(&y));

    assert_eq!(
        asserted_over(&x_reference.clone().floor_mod(3), SymbolType::Int),
        format!("(= value (mod {xs} 3))")
    );
    assert_eq!(
        asserted_over(&x_reference.clone().floor_mod(-3), SymbolType::Int),
        format!("(= value (- (mod (- {xs}) 3)))")
    );
    assert_eq!(
        asserted_over(&x_reference.floor_mod(y_reference), SymbolType::Int),
        format!("(= value (ite (> {ys} 0) (mod {xs} {ys}) (- (mod (- {xs}) (- {ys})))))")
    );
}

#[test]
fn real_floor_operations_go_through_to_int() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let (xs, ys) = (quoted_symbol(&x), quoted_symbol(&y));
    let floor_division = SmtScript::lower(
        &x_reference.clone().floor_divide(y_reference.clone()),
        &|_: &Identifier| Some(SymbolType::Real),
        &NoRegisteredSorts,
    )
    .expect("lowers");

    assert_eq!(
        asserted(&floor_division),
        format!("(= value (to_real (to_int (/ {xs} {ys}))))")
    );
    assert_eq!(floor_division.value_sort(), Some(SymbolType::Real));
    assert_eq!(
        asserted_over(&x_reference.floor_mod(y_reference), SymbolType::Real),
        format!("(= value (- {xs} (* {ys} (to_real (to_int (/ {xs} {ys}))))))")
    );
}

#[test]
fn integer_floor_division_by_a_real_literal_is_a_real_floor() {
    let (x, reference) = build_identifier("x");

    assert_eq!(
        asserted_over(&reference.floor_divide(2.5), SymbolType::Int),
        format!(
            "(= value (to_real (to_int (/ (to_real {}) (/ 5.0 2.0)))))",
            quoted_symbol(&x)
        )
    );
}

// ---------------------------------------------------------------------------
// Powers
// ---------------------------------------------------------------------------

#[rstest]
#[case::first(1, "X")]
#[case::square(2, "(* X X)")]
#[case::cube(3, "(* X (* X X))")]
#[case::fourth(4, "(let ((t!1 (* X X))) (* t!1 t!1))")]
fn power_by_an_integer_literal_is_a_product_by_squaring(
    #[case] exponent: i64,
    #[case] template: &str,
) {
    let (x, reference) = build_identifier("x");
    let script = lower(
        &reference.power(exponent),
        &build_symbol_types(&[(&x, SymbolType::Int)]),
    );
    let symbol = quoted_symbol(&x);

    let expected = if template.starts_with("(let") {
        format!("(let ((t!1 (* {symbol} {symbol}))) (= value (* t!1 t!1)))")
    } else {
        format!("(= value {})", template.replace('X', &symbol))
    };
    assert_eq!(asserted(&script), expected);
    assert_eq!(script.value_sort(), Some(SymbolType::Int));
}

#[test]
fn power_keeps_the_sort_of_a_real_base_and_is_nonlinear() {
    let (r, reference) = build_identifier("r");
    let script = lower(
        &reference.power(2).greater(1),
        &build_symbol_types(&[(&r, SymbolType::Real)]),
    );

    assert_eq!(
        asserted(&script),
        format!("(> (* {r} {r}) 1.0)", r = quoted_symbol(&r))
    );
    assert_eq!(script.logic(), Logic::QfNra);
}

#[test]
fn power_by_squaring_is_logarithmic_in_the_exponent() {
    let (x, reference) = build_identifier("x");
    let script = lower(
        &reference.power(BigInt::from(1) << 40_u32).greater(0),
        &build_symbol_types(&[(&x, SymbolType::Int)]),
    );

    assert!(script.to_string().len() < 4_000, "{}", script);
}

#[rstest]
#[case::zero(build_literal(0))]
#[case::negative(build_literal(-2))]
#[case::float(build_literal(2.0))]
#[case::variable(build_identifier("n").1)]
fn power_by_another_exponent_is_refused(#[case] exponent: Expression) {
    let (_, reference) = build_identifier("x");
    let power = reference.power(exponent);

    let error = SmtScript::lower(
        &power,
        &|_: &Identifier| Some(SymbolType::Int),
        &NoRegisteredSorts,
    )
    .expect_err("refused");

    assert_eq!(error, LoweringError::UnsupportedPower(power));
}

// ---------------------------------------------------------------------------
// Comparisons, connectives and piecewise
// ---------------------------------------------------------------------------

#[rstest]
#[case::equal(Expression::equals, "=")]
#[case::not_equal(Expression::not_equals, "distinct")]
#[case::less(Expression::less, "<")]
#[case::less_equal(Expression::less_equal, "<=")]
#[case::greater(Expression::greater, ">")]
#[case::greater_equal(Expression::greater_equal, ">=")]
fn comparison_is_the_smt_lib2_predicate(
    #[case] compare: fn(&Expression, Expression) -> Expression,
    #[case] symbol: &str,
) {
    let (x, reference) = build_identifier("x");

    assert_eq!(
        asserted_over(&compare(&reference, build_literal(1)), SymbolType::Int),
        format!("({symbol} {} 1)", quoted_symbol(&x))
    );
}

#[test]
fn connectives_are_n_ary_and_or_and_not() {
    let (a, a_reference) = build_identifier("a");
    let (b, b_reference) = build_identifier("b");
    let (c, c_reference) = build_identifier("c");
    let (sa, sb, sc) = (quoted_symbol(&a), quoted_symbol(&b), quoted_symbol(&c));

    assert_eq!(
        asserted_over(
            &Expression::all([
                a_reference.clone(),
                b_reference.clone(),
                c_reference.clone()
            ]),
            SymbolType::Bool
        ),
        format!("(and {sa} {sb} {sc})")
    );
    assert_eq!(
        asserted_over(
            &Expression::any([a_reference.clone(), !b_reference.clone()]),
            SymbolType::Bool
        ),
        format!("(or {sa} (not {sb}))")
    );
    assert_eq!(
        asserted_over(&a_reference.equals(b_reference), SymbolType::Bool),
        format!("(= {sa} {sb})")
    );
    let _ = c_reference;
}

#[test]
fn piecewise_is_a_right_folded_ite_whose_first_match_wins() {
    let (x, reference) = build_identifier("x");
    let piecewise = Expression::piecewise(
        [
            (reference.clone().less(0), build_literal(-1)),
            (reference.clone().equals(0), build_literal(0)),
        ],
        build_literal(1),
    )
    .expect("a piecewise");
    let xs = quoted_symbol(&x);

    assert_eq!(
        asserted_over(&piecewise.greater(reference), SymbolType::Int),
        format!("(> (ite (< {xs} 0) (- 1) (ite (= {xs} 0) 0 1)) {xs})")
    );
}

// ---------------------------------------------------------------------------
// Logics
// ---------------------------------------------------------------------------

#[rstest]
#[case::linear_integer(SymbolType::Int, false, Logic::QfLia)]
#[case::linear_real(SymbolType::Real, false, Logic::QfLra)]
#[case::nonlinear_integer(SymbolType::Int, true, Logic::QfNia)]
#[case::nonlinear_real(SymbolType::Real, true, Logic::QfNra)]
fn logic_is_the_narrowest_that_fits(
    #[case] sort: SymbolType,
    #[case] is_nonlinear: bool,
    #[case] logic: Logic,
) {
    let (_, x_reference) = build_identifier("x");
    let (_, y_reference) = build_identifier("y");
    let term = if is_nonlinear {
        x_reference * y_reference
    } else {
        x_reference * 3 + y_reference
    };
    let comparison = if sort == SymbolType::Int {
        term.greater(0)
    } else {
        term.greater(0.0)
    };
    let script = SmtScript::lower(
        &comparison,
        &|_: &Identifier| Some(sort),
        &NoRegisteredSorts,
    )
    .expect("lowers");

    assert_eq!(script.logic(), logic);
}

#[test]
fn logic_is_all_for_mixed_numeric_sorts() {
    let (n, n_reference) = build_identifier("n");
    let (r, r_reference) = build_identifier("r");
    let symbol_types = build_symbol_types(&[(&n, SymbolType::Int), (&r, SymbolType::Real)]);

    assert_eq!(
        lower(&n_reference.less(r_reference), &symbol_types).logic(),
        Logic::All
    );
}

#[test]
fn logic_is_linear_only_for_numeral_coefficients_and_divisors() {
    let (x, reference) = build_identifier("x");
    let symbol_types = build_symbol_types(&[(&x, SymbolType::Real)]);
    let scaled = build_literal(-2.5) * reference.clone() / 4.0;
    let scaled_by_a_sum = (build_literal(2.0) + 1.0) * reference.clone();
    let divided_by_a_sum = reference.clone() / (build_literal(4.0) + 1.0);

    assert_eq!(
        lower(&scaled.greater(reference.clone()), &symbol_types).logic(),
        Logic::QfLra
    );
    assert_eq!(
        lower(&scaled_by_a_sum.greater(0.0), &symbol_types).logic(),
        Logic::QfNra,
        "solvers read a product of a sum as nonlinear"
    );
    assert_eq!(
        lower(&divided_by_a_sum.greater(0.0), &symbol_types).logic(),
        Logic::QfNra
    );
}

#[test]
fn logic_display_is_its_smt_lib2_name() {
    let names: Vec<String> = [
        Logic::QfLia,
        Logic::QfLra,
        Logic::QfNia,
        Logic::QfNra,
        Logic::Lia,
        Logic::Lra,
        Logic::Nia,
        Logic::Nra,
        Logic::All,
    ]
    .iter()
    .map(ToString::to_string)
    .collect();

    assert_eq!(
        names,
        [
            "QF_LIA", "QF_LRA", "QF_NIA", "QF_NRA", "LIA", "LRA", "NIA", "NRA", "ALL"
        ]
    );
}

// ---------------------------------------------------------------------------
// Sharing and depth
// ---------------------------------------------------------------------------

#[test]
fn shared_subterm_is_written_once_under_a_let() {
    let (x, reference) = build_identifier("x");
    let shared = reference + 1;
    let expression = (shared.clone() * shared).greater(0);
    let xs = quoted_symbol(&x);

    assert_eq!(
        asserted_over(&expression, SymbolType::Int),
        format!("(let ((t!1 (+ {xs} 1))) (> (* t!1 t!1) 0))")
    );
}

#[test]
fn shared_dag_lowers_into_a_script_linear_in_its_distinct_nodes() {
    let (x, reference) = build_identifier("x");
    let mut dag = reference;
    for _ in 0..64 {
        dag = dag.clone() + dag;
    }

    let script = lower(
        &dag.greater(0),
        &build_symbol_types(&[(&x, SymbolType::Int)]),
    );

    assert!(script.to_string().len() < 10_000);
    assert!(script.to_string().contains("t!63"));
}

#[test]
fn deep_tree_lowers_and_prints_on_a_small_stack() {
    let length = run_on_small_stack(|| {
        let (x, reference) = build_identifier("x");
        let mut chain = reference;
        for _ in 0..SMALL_STACK_DEPTH {
            chain = chain + 1;
        }
        let symbol_types = build_symbol_types(&[(&x, SymbolType::Int)]);
        let script =
            SmtScript::lower(&chain.greater(0), &symbol_types, &NoRegisteredSorts).expect("lowers");
        script.to_string().len()
    });

    assert!(length > SMALL_STACK_DEPTH * 4);
}

// ---------------------------------------------------------------------------
// Refusals, in their order
// ---------------------------------------------------------------------------

#[test]
fn missing_symbol_types_are_refused_first_naming_every_identifier_by_id() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let pi = Expression::from(BuiltinConstant::Pi.identifier().clone());
    let expression = Expression::all([
        build_literal(2),
        y_reference.less(x_reference).and(pi.greater(3)),
    ]);

    assert_eq!(
        refuse(&expression, &HashMap::new()),
        LoweringError::MissingSymbolTypes(vec![x, y])
    );
}

#[test]
fn ill_typed_expression_is_refused_before_a_native_constant() {
    let pi = Expression::from(BuiltinConstant::Pi.identifier().clone());
    let expression = Expression::all([build_literal(2), pi.greater(3)]);

    assert!(matches!(
        refuse(&expression, &HashMap::new()),
        LoweringError::IllTyped(_)
    ));
}

#[test]
fn native_constants_are_refused_before_the_nodes() {
    let pi = BuiltinConstant::Pi.identifier().clone();
    let e = BuiltinConstant::E.identifier().clone();
    let (b, b_reference) = build_identifier("b");
    let expression =
        (Expression::from(e.clone()) + b_reference).greater(Expression::from(pi.clone()));

    assert_eq!(
        refuse(&expression, &build_symbol_types(&[(&b, SymbolType::Bool)])),
        LoweringError::NativeConstants(vec![pi, e])
    );
}

#[rstest]
#[case::sum(|b: Expression| b + 1)]
#[case::negation(|b: Expression| -b)]
#[case::ordering(|b: Expression| b.less(1))]
#[case::equality(|b: Expression| b.equals(1))]
fn boolean_meeting_a_number_is_refused(#[case] build: fn(Expression) -> Expression) {
    let (b, reference) = build_identifier("b");
    let node = build(reference);

    assert_eq!(
        refuse(&node, &build_symbol_types(&[(&b, SymbolType::Bool)])),
        LoweringError::SortMismatch(node)
    );
}

#[test]
fn piecewise_mixing_a_boolean_and_a_number_is_refused() {
    let (b, reference) = build_identifier("b");
    let mixed = Expression::piecewise([(reference, build_literal(1))], build_literal(false))
        .expect("a piecewise");

    assert_eq!(
        refuse(&mixed, &build_symbol_types(&[(&b, SymbolType::Bool)])),
        LoweringError::SortMismatch(mixed)
    );
}

#[rstest]
#[case::composed_builtin(Expression::call(BuiltinFunction::Relu, [build_literal(1)]), r#"smt-lib2 has no term for a call of the built-in "relu"; inline it first with inline_functions"#)]
#[case::native_builtin(Expression::call(BuiltinFunction::Exp, [build_literal(1)]), r#"smt-lib2 has no term for a call of the native built-in "exp""#)]
#[case::named(Expression::call(FunctionName::new("f").expect("a name"), [build_literal(1)]), r#"smt-lib2 has no term for a call of "f"; a user function must be inlined first with inline_functions"#)]
fn call_is_refused_naming_the_callee(#[case] call: Expression, #[case] text: &str) {
    let error = refuse(&call.clone().greater(0), &HashMap::new());

    assert_eq!(error, LoweringError::Call(call));
    assert_eq!(error.to_string(), text);
}

#[test]
fn lowering_errors_display_one_lowercase_line() {
    let (x, x_reference) = build_identifier("x");
    let pi = BuiltinConstant::Pi.identifier().clone();
    let node = x_reference.clone() + 1;

    assert_eq!(
        LoweringError::MissingSymbolTypes(vec![x.clone()]).to_string(),
        format!(
            "symbol_types is missing entries for identifiers: x::{}",
            x.id()
        )
    );
    assert_eq!(
        LoweringError::NativeConstants(vec![pi]).to_string(),
        "smt-lib2 has no term for the native constants pi::48"
    );
    assert_eq!(
        LoweringError::NonFiniteLiteral(build_literal(f64::INFINITY)).to_string(),
        "smt-lib2 has no term for the non-finite float inf"
    );
    assert_eq!(
        LoweringError::SortMismatch(node.clone()).to_string(),
        "a boolean and a number meet in the node (x + 1)"
    );
    assert_eq!(
        LoweringError::UnsupportedPower(x_reference.power(0)).to_string(),
        "smt-lib2 has no term for the power (x ** 0), whose exponent is not an integer literal \
         of at least one"
    );
}

/// Test a lowering error on a depth-20 doubling DAG displays in bounded
/// size: each variant holding a node writes it up to a budget of node
/// occurrences (R2-010), not once per path.
#[test]
fn a_lowering_error_on_a_depth_20_dag_displays_in_bounded_size() {
    let (x, reference) = build_identifier("x");
    let dag = crate::support::expression::build_doubling_dag(&reference, 20);
    let boolean = build_literal(true);
    let mismatch = &dag + &boolean;

    let refused = refuse(
        &mismatch.greater(0),
        &build_symbol_types(&[(&x, SymbolType::Int)]),
    );

    assert_eq!(refused, LoweringError::SortMismatch(mismatch));
    for error in [
        refused,
        LoweringError::Call(dag.clone()),
        LoweringError::UnsupportedPower(dag.clone()),
        LoweringError::NonFiniteLiteral(dag),
    ] {
        let text = error.to_string();
        assert!(text.len() < 4096, "{} bytes", text.len());
        assert!(text.contains('…'), "{text}");
    }
}
