//! The text and the source of every variant of the `constraint` errors.

use fhy_core::constraint::{ConstraintError, MemberError, UnusableBindingReason, Value};
use fhy_core::expression::{BooleanScreen, Expression, LiteralValue, PiecewiseError};
use fhy_core::solver::{QueryKind, SolveError};
use rstest::rstest;

use crate::support::constraint::{int, member, text};
use crate::support::error_text::{Source, assert_error_text, fixed, test_error};

fn unusable(reason: UnusableBindingReason) -> ConstraintError {
    ConstraintError::UnusableBinding {
        identifier: fixed(7, "x"),
        reason,
    }
}

#[rstest]
#[case::not_a_literal(
    unusable(UnusableBindingReason::NotALiteral),
    "the binding of x::7 is neither an expression nor a literal".to_owned(),
    Source::None
)]
#[case::unparsable_text(
    unusable(UnusableBindingReason::UnparsableText(
        LiteralValue::parse_text("abc").expect_err("no literal"),
    )),
    "the binding of x::7 cannot be lifted into a literal".to_owned(),
    Source::LiteralText
)]
#[case::not_member_shaped(
    unusable(UnusableBindingReason::NotMemberShaped),
    "the binding of x::7 is neither an expression nor a value that could be a member".to_owned(),
    Source::None
)]
#[case::unhashable(
    unusable(UnusableBindingReason::Unhashable(test_error())),
    "the binding of x::7 is unhashable, so its membership cannot be checked".to_owned(),
    Source::TestValue
)]
#[case::ill_typed(
    ConstraintError::IllTyped(
        BooleanScreen::new()
            .check_predicate(&Expression::literal(1))
            .expect_err("a number is no predicate"),
    ),
    "the predicate is ill-typed".to_owned(),
    Source::NonBooleanOperand
)]
#[case::non_boolean_result(
    ConstraintError::NonBooleanResult {
        predicate: Expression::from(fixed(7, "x")).greater(1),
        result: Expression::literal(2),
    },
    "the predicate (x > 1) simplified to the literal 2, which is not a boolean, so it denotes a \
     number"
        .to_owned(),
    Source::None
)]
#[case::unliftable_string(
    ConstraintError::UnliftableMember(member(text("a"))),
    "the string member \"a\" cannot be converted to an expression: membership is type-strict, \
     but literal equality would canonicalize the string against numeric members"
        .to_owned(),
    Source::None
)]
#[case::unliftable_tuple(
    ConstraintError::UnliftableMember(member(Value::Tuple(vec![int(1)]))),
    "conversion of type tuple to an expression is not supported".to_owned(),
    Source::None
)]
#[case::solve(
    ConstraintError::Solve(SolveError::NoCapableBackend(QueryKind::Simplification)),
    "the solver refused or failed the question".to_owned(),
    Source::Solve
)]
#[case::missing_symbol_types(
    ConstraintError::MissingSymbolTypes(vec![fixed(7, "x"), fixed(8, "y")]),
    "symbol_types is missing an entry for free identifier(s): x::7, y::8".to_owned(),
    Source::None
)]
#[case::substitution(
    ConstraintError::Substitution(PiecewiseError::NoCases),
    "substituting the bindings failed".to_owned(),
    Source::Piecewise
)]
#[case::custom(
    ConstraintError::Custom(test_error()),
    "a custom constraint failed".to_owned(),
    Source::TestValue
)]
fn constraint_error_text(
    #[case] error: ConstraintError,
    #[case] text: String,
    #[case] source: Source,
) {
    assert_error_text(&error, &text, source);
}

#[rstest]
#[case::nan(
    MemberError::Nan,
    "a member is, or holds, a NaN, which is unequal to itself and could never match a bound \
     value",
    Source::None
)]
#[case::decimal(
    MemberError::Decimal,
    "a member is, or holds, a decimal, which is no member kind",
    Source::None
)]
#[case::not_member_shaped(
    MemberError::NotMemberShaped { type_name: "Handle".to_owned() },
    "a member is, or holds, a value of type Handle, which is no member kind",
    Source::None
)]
#[case::ordering_key(
    MemberError::OrderingKey { type_name: "Token".to_owned(), source: test_error() },
    "the ordering key of a member of type Token failed",
    Source::TestValue
)]
fn member_error_text(#[case] error: MemberError, #[case] text: &str, #[case] source: Source) {
    assert_error_text(&error, text, source);
}
