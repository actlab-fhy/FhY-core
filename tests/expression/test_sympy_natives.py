"""Sympy lowering / lifting tests for native functions and constants.

The sympy bridge gains a per-name mapping that lowers each seeded
``NativeFunction`` (``exp``, ``log``, ``sin``, ...) to its sympy
counterpart and lifts the same sympy operators back to a
``CallExpression`` with the seeded name. Native constants (``pi``,
``e``) lower to ``sympy.pi`` / ``sympy.E`` and lift back to an
``IdentifierExpression`` whose identifier name matches the seeded
constant.

Unmapped native calls raise ``TypeError`` from the lowering pass.
Expression-bodied calls still require the caller to ``inline_functions``
first; that rejection stays.
"""

import pytest
import sympy  # type: ignore[import-untyped]

from fhy_core.expression import (
    CallExpression,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    call,
    convert_expression_to_sympy_expression,
    convert_sympy_expression_to_expression,
    register_function,
    register_native_function,
)
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import PassExecutionError

from .conftest import mock_identifier

# =============================================================================
# Native function lowering (IR -> sympy)
# =============================================================================


@pytest.mark.parametrize(
    "native_name, sympy_operator",
    [
        ("exp", sympy.exp),
        ("log", sympy.log),
        ("log2", lambda x: sympy.log(x, 2)),
        ("log10", lambda x: sympy.log(x, 10)),
        ("sqrt", sympy.sqrt),
        ("sin", sympy.sin),
        ("cos", sympy.cos),
        ("tan", sympy.tan),
        ("arcsin", sympy.asin),
        ("arccos", sympy.acos),
        ("arctan", sympy.atan),
        ("sinh", sympy.sinh),
        ("cosh", sympy.cosh),
        ("tanh", sympy.tanh),
        ("erf", sympy.erf),
        ("floor", sympy.floor),
        ("ceil", sympy.ceiling),
    ],
)
def test_seeded_native_function_lowers_to_sympy_counterpart(
    native_name: str, sympy_operator: object
) -> None:
    """Test each seeded native lowers to its sympy operator over a symbolic arg."""
    x = mock_identifier("x", 0)
    expression = call(native_name, x)

    lowered = convert_expression_to_sympy_expression(expression)

    expected = sympy_operator(sympy.Symbol("x_0"))  # type: ignore[operator]
    assert lowered == expected


# =============================================================================
# Native function lifting (sympy -> IR)
# =============================================================================


@pytest.mark.parametrize(
    "native_name, sympy_operator",
    [
        ("exp", sympy.exp),
        ("sqrt", sympy.sqrt),
        ("sin", sympy.sin),
        ("cos", sympy.cos),
        ("tan", sympy.tan),
        ("arcsin", sympy.asin),
        ("arccos", sympy.acos),
        ("arctan", sympy.atan),
        ("sinh", sympy.sinh),
        ("cosh", sympy.cosh),
        ("tanh", sympy.tanh),
        ("erf", sympy.erf),
        ("floor", sympy.floor),
        ("ceil", sympy.ceiling),
    ],
)
def test_sympy_operator_lifts_back_to_native_call(
    native_name: str, sympy_operator: object
) -> None:
    """Test a sympy operator application lifts back to a `CallExpression`."""
    sympy_expression = sympy_operator(sympy.Symbol("x_0"))  # type: ignore[operator]

    lifted = convert_sympy_expression_to_expression(sympy_expression)

    assert isinstance(lifted, CallExpression)
    assert lifted.function_name == native_name


def test_log_lifts_back_to_call_log() -> None:
    """Test the natural-log sympy operator lifts back to a ``log`` call.

    ``sympy.log(x)`` (single-arg form) is natural log; this lifts to the
    seeded ``log`` native.
    """
    sympy_expression = sympy.log(sympy.Symbol("x_0"))

    lifted = convert_sympy_expression_to_expression(sympy_expression)

    assert isinstance(lifted, CallExpression)
    assert lifted.function_name == "log"


# =============================================================================
# Round-trip through sympy
# =============================================================================


def test_native_call_round_trips_through_sympy_unchanged() -> None:
    """Test a `call("exp", x)` lowers to sympy and lifts back to the same structure."""
    x = mock_identifier("x", 0)
    original = call("exp", x)

    lowered = convert_expression_to_sympy_expression(original)
    lifted = convert_sympy_expression_to_expression(lowered)

    assert lifted.is_structurally_equivalent(original)


def test_nested_native_call_round_trips_through_sympy() -> None:
    """Test a nested chain `sin(sqrt(x))` round-trips through sympy."""
    x = mock_identifier("x", 0)
    original = call("sin", call("sqrt", x))

    lowered = convert_expression_to_sympy_expression(original)
    lifted = convert_sympy_expression_to_expression(lowered)

    assert lifted.is_structurally_equivalent(original)


# =============================================================================
# Unmapped native call raises
# =============================================================================


def _identity_implementation(value: object) -> object:
    return value


def test_unmapped_native_function_call_lowering_raises_typed_error(
    function_registry_snapshot: None,
) -> None:
    """Test lowering an unmapped native call raises ``PassExecutionError``."""
    register_native_function(
        "test_sympy_unmapped_native",
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        implementation=_identity_implementation,  # type: ignore[arg-type]
    )
    x = mock_identifier("x", 0)
    expression = call("test_sympy_unmapped_native", x)

    with pytest.raises(PassExecutionError, match="test_sympy_unmapped_native"):
        convert_expression_to_sympy_expression(expression)


# =============================================================================
# Expression-bodied function call lowering still requires inlining
# =============================================================================


def test_expression_bodied_call_lowering_still_raises(
    function_registry_snapshot: None,
) -> None:
    """Test a `RegisteredFunction` call still raises during sympy lowering.

    The caller must inline expression-bodied calls before lowering;
    only native calls are handled by the new mapping.
    """
    parameter = Identifier("p")
    register_function(
        "test_sympy_expr_bodied",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )
    expression = call("test_sympy_expr_bodied", LiteralExpression(1.0))

    with pytest.raises(PassExecutionError, match="test_sympy_expr_bodied"):
        convert_expression_to_sympy_expression(expression)


# =============================================================================
# Native constants lower / lift
# =============================================================================


def test_pi_constant_reference_lowers_to_sympy_pi() -> None:
    """Test an `IdentifierExpression(Identifier("pi"))` lowers to `sympy.pi`."""
    expression = IdentifierExpression(Identifier("pi"))

    lowered = convert_expression_to_sympy_expression(expression)

    assert lowered == sympy.pi


def test_e_constant_reference_lowers_to_sympy_e() -> None:
    """Test an `IdentifierExpression(Identifier("e"))` lowers to `sympy.E`."""
    expression = IdentifierExpression(Identifier("e"))

    lowered = convert_expression_to_sympy_expression(expression)

    assert lowered == sympy.E


def test_sympy_pi_lifts_back_to_pi_identifier_reference() -> None:
    """Test `sympy.pi` lifts to an `IdentifierExpression` named ``pi``."""
    lifted = convert_sympy_expression_to_expression(sympy.pi)

    assert isinstance(lifted, IdentifierExpression)
    assert lifted.identifier.name_hint == "pi"


def test_sympy_e_lifts_back_to_e_identifier_reference() -> None:
    """Test `sympy.E` lifts to an `IdentifierExpression` named ``e``."""
    lifted = convert_sympy_expression_to_expression(sympy.E)

    assert isinstance(lifted, IdentifierExpression)
    assert lifted.identifier.name_hint == "e"


def test_native_call_with_constant_argument_round_trips_through_sympy() -> None:
    """Test `sin(pi)` round-trips: lowers, simplifies, lifts back to literal.

    Sympy simplifies ``sin(pi)`` to ``0``; the lifted expression is the
    integer literal ``0`` (not a constant reference any more).
    """
    expression = call("sin", Identifier("pi"))

    lowered = convert_expression_to_sympy_expression(expression)
    simplified = sympy.simplify(lowered)
    lifted = convert_sympy_expression_to_expression(simplified)

    assert isinstance(lifted, LiteralExpression)
    assert lifted.value == 0
