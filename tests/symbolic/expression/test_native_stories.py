"""End-to-end user-story and integration tests for native functions.

Each test mirrors a real caller path through the public surface
(registry, inliner, evaluator, type-checker, sympy bridge) for native
functions, native constants, and sorts. The cases above unit-test
boundaries individually; the cases here exercise them in combination.
"""

import math

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    call,
    convert_expression_to_sympy_expression,
    convert_sympy_expression_to_expression,
    evaluate_expression,
    inline_functions,
    register_native_function,
)
from fhy_core.symbolic.solver import simplify_expression
from fhy_core.types import (
    CoreDataType,
    FhYCoreTypeError,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
)
from fhy_core.types.checking import synthesize_expression_type

from .conftest import mock_identifier

pytestmark = pytest.mark.integration


def _scalar(core_data_type: CoreDataType) -> NumericalType:
    return NumericalType(PrimitiveDataType(core_data_type))


def _no_identifiers(identifier: Identifier) -> tuple[Type, TypeQualifier]:
    raise KeyError(identifier.name_hint)


# =============================================================================
# User story: native function call evaluates to a literal
# =============================================================================


def test_user_story_native_call_evaluates_to_literal_result() -> None:
    """Test ``exp(0.0)`` evaluates to a literal ``1.0``."""
    expression = call("exp", LiteralExpression(0.0))

    evaluated = evaluate_expression(expression)

    assert isinstance(evaluated, LiteralExpression)
    assert evaluated.value == math.exp(0.0)


def test_user_story_evaluate_ceil_of_float_yields_int() -> None:
    """Test ``ceil(2.5)`` evaluates to an ``int``-valued literal ``3``."""
    expression = call("ceil", LiteralExpression(2.5))

    evaluated = evaluate_expression(expression)

    assert isinstance(evaluated, LiteralExpression)
    assert evaluated.value == 3
    assert type(evaluated.value) is int


# =============================================================================
# User story: clamp-of-exp under literal bounds
# =============================================================================


def test_user_story_evaluate_clamp_of_exp_under_literal_bounds() -> None:
    """Test ``clamp(exp(2.0), 0, 5)`` inlines + evaluates the exp + simplifies to 5.0.

    The caller wants the saturated clamp result. The evaluator folds
    ``exp(2.0)`` to a literal; the surrounding ``min`` / ``max``
    structure is then simplified via sympy to the saturated value.
    """
    expression = call("clamp", call("exp", LiteralExpression(2.0)), 0, 5)

    inlined = inline_functions(expression)
    evaluated = evaluate_expression(inlined)
    simplified = simplify_expression(evaluated)

    assert isinstance(simplified, LiteralExpression)
    # `math.exp(2.0) > 5`, so the clamp result is 5.
    assert float(simplified.value) == 5.0


# =============================================================================
# User story: symbolic exp stays symbolic under the evaluator
# =============================================================================


def test_user_story_symbolic_exp_stays_symbolic_under_evaluation() -> None:
    """Test ``exp(x)`` evaluates unchanged when ``x`` is a free identifier."""
    x = mock_identifier("x", 0)
    expression = call("exp", x)

    evaluated = evaluate_expression(expression)

    assert isinstance(evaluated, CallExpression)
    assert evaluated.function_name == "exp"
    assert evaluated.arguments[0].is_structurally_equivalent(IdentifierExpression(x))


# =============================================================================
# User story: sort error on a boolean argument to a real-typed native
# =============================================================================


def test_user_story_real_native_rejects_boolean_argument() -> None:
    """Test calling ``exp(True)`` raises a type error citing the function name."""
    expression = call("exp", LiteralExpression(True))

    with pytest.raises(FhYCoreTypeError, match="exp"):
        synthesize_expression_type(expression, _no_identifiers)


# =============================================================================
# User story: constant reference in a sympy round-trip
# =============================================================================


def test_user_story_sin_of_pi_round_trips_through_sympy_to_zero() -> None:
    """Test ``sin(pi)`` lowers + simplifies through sympy and lifts back to ``0``."""
    expression = call("sin", mock_identifier("pi", 0))

    simplified = simplify_expression(expression)

    assert isinstance(simplified, LiteralExpression)
    assert simplified.value == 0


# =============================================================================
# User story: pipeline inline -> evaluate -> simplify for sigmoid(0)
# =============================================================================


def test_user_story_pipeline_sigmoid_zero_simplifies_to_half() -> None:
    """Test the inline / evaluate / simplify pipeline yields ``0.5``."""
    expression = call("sigmoid", LiteralExpression(0.0))

    inlined = inline_functions(expression)
    evaluated = evaluate_expression(inlined)
    simplified = simplify_expression(evaluated)

    assert isinstance(simplified, LiteralExpression)
    assert float(simplified.value) == 0.5


# =============================================================================
# User story: type a ceil expression and observe the INT result sort
# =============================================================================


def test_user_story_type_synthesis_for_ceil_returns_int64() -> None:
    """Test ``ceil(2.5)`` synthesizes to concrete `INT64`, not a float type."""
    expression = call("ceil", LiteralExpression(2.5))

    result_type, _ = synthesize_expression_type(expression, _no_identifiers)

    assert result_type.is_structurally_equivalent(_scalar(CoreDataType.INT64))


# =============================================================================
# User story: register a custom native function and use it
# =============================================================================


def test_user_story_register_custom_native_and_evaluate_under_literal(
    function_registry_snapshot: None,
) -> None:
    """Test registering a custom native function and evaluating a literal call.

    Caller wants to bind a Python helper into the IR. They register it
    with a sort signature; the call site type-checks and the evaluator
    folds literal arguments through the helper.
    """

    def power_of_two(exponent: int) -> int:
        return 1 << exponent

    register_native_function(
        "test_story_pow2",
        parameter_sorts=[FunctionSort.NAT],
        result_sort=FunctionSort.NAT,
        implementation=power_of_two,
    )

    expression = call("test_story_pow2", LiteralExpression(5))
    result_type, _ = synthesize_expression_type(expression, _no_identifiers)
    evaluated = evaluate_expression(expression)

    assert result_type.is_structurally_equivalent(_scalar(CoreDataType.UINT32))
    assert isinstance(evaluated, LiteralExpression)
    assert evaluated.value == 32


# =============================================================================
# User story: evaluator preserves outer arithmetic over a folded native call
# =============================================================================


def test_user_story_outer_arithmetic_preserved_around_folded_native_call() -> None:
    """Test ``exp(0.0) + 1`` evaluates to ``1.0 + 1`` (outer add intact).

    The evaluator's narrow scope means the outer ``+`` is preserved
    even though both operands are literals after folding.
    """
    expression = BinaryExpression(
        BinaryOperation.ADD,
        call("exp", LiteralExpression(0.0)),
        LiteralExpression(1),
    )

    evaluated = evaluate_expression(expression)

    assert isinstance(evaluated, BinaryExpression)
    assert evaluated.operation == BinaryOperation.ADD
    assert isinstance(evaluated.left, LiteralExpression)
    assert evaluated.left.value == 1.0


# =============================================================================
# User story: sympy round-trip of a tree mixing native calls and constants
# =============================================================================


def test_user_story_native_call_with_constant_round_trips_through_sympy() -> None:
    """Test ``cos(pi)`` round-trips through sympy lower / lift and folds to ``-1``."""
    expression = call("cos", mock_identifier("pi", 0))

    lowered = convert_expression_to_sympy_expression(expression)
    lifted = convert_sympy_expression_to_expression(lowered)

    # `sympy.cos(sympy.pi)` simplifies eagerly to `-1`; the lifted form
    # is the integer literal `-1` (constant folding already happened in
    # sympy before lifting).
    assert isinstance(lifted, LiteralExpression)
    assert lifted.value == -1
