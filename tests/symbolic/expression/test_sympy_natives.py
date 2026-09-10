"""Sympy lowering / lifting tests for native functions and constants.

The sympy bridge maps each seeded ``NativeFunction`` (``exp``, ``log``,
``sin``, ...) to its sympy counterpart when lowering, and lifts the
same sympy operators back to a ``CallExpression`` with the seeded
name. Native constants (``pi``, ``e``, ``inf``, ``nan``) are referenced
through the canonical identifier the registry minted for them; those
references lower to ``sympy.pi``, ``sympy.E``, ``sympy.oo``, and
``sympy.nan`` and lift back to an ``IdentifierExpression`` carrying
the same canonical identifier, with ``-oo`` lifting as the negated
``inf``. An identifier that merely shares a constant's name is an
ordinary free variable and lowers to a sympy ``Symbol``.

Unmapped native calls raise ``TypeError`` from the lowering pass.
Expression-bodied calls require the caller to ``inline_functions``
first.
"""

import math

import pytest
import sympy  # type: ignore[import-untyped]

from fhy_core.pass_infrastructure import PassExecutionError
from fhy_core.symbolic.expression import (
    CallExpression,
    Expression,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    call,
    convert_expression_to_sympy_expression,
    convert_sympy_expression_to_expression,
    get_native_constant_identifier,
    register_function,
    register_native_function,
)
from fhy_core.symbolic.solver import simplify_expression

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
    parameter = mock_identifier("p", 0)
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
    """Test the canonical ``pi`` identifier lowers to `sympy.pi`."""
    expression = IdentifierExpression(get_native_constant_identifier("pi"))

    lowered = convert_expression_to_sympy_expression(expression)

    assert lowered == sympy.pi


def test_e_constant_reference_lowers_to_sympy_e() -> None:
    """Test the canonical ``e`` identifier lowers to `sympy.E`."""
    expression = IdentifierExpression(get_native_constant_identifier("e"))

    lowered = convert_expression_to_sympy_expression(expression)

    assert lowered == sympy.E


@pytest.mark.parametrize("constant_name", ["pi", "e", "inf", "nan"])
def test_identifier_merely_named_like_a_constant_lowers_to_a_symbol(
    constant_name: str,
) -> None:
    """Test an identifier that only shares a constant's name lowers to a symbol.

    Constant resolution keys on the canonical identifier the registry
    minted, not on ``name_hint``, so a caller's own variable called
    ``pi`` stays a substitutable sympy ``Symbol`` instead of being
    replaced by the constant's value.
    """
    variable = mock_identifier(constant_name, 512)
    expression = IdentifierExpression(variable)

    lowered = convert_expression_to_sympy_expression(expression)

    assert isinstance(lowered, sympy.Symbol)
    assert lowered.name == f"{constant_name}_512"


def test_sympy_pi_lifts_back_to_the_canonical_pi_identifier() -> None:
    """Test `sympy.pi` lifts to the registry's canonical ``pi`` identifier."""
    lifted = convert_sympy_expression_to_expression(sympy.pi)

    assert isinstance(lifted, IdentifierExpression)
    assert lifted.identifier == get_native_constant_identifier("pi")


def test_sympy_e_lifts_back_to_the_canonical_e_identifier() -> None:
    """Test `sympy.E` lifts to the registry's canonical ``e`` identifier."""
    lifted = convert_sympy_expression_to_expression(sympy.E)

    assert isinstance(lifted, IdentifierExpression)
    assert lifted.identifier == get_native_constant_identifier("e")


@pytest.mark.parametrize(
    "constant_name, sympy_value",
    [
        pytest.param("inf", sympy.oo, id="inf"),
        pytest.param("nan", sympy.nan, id="nan"),
    ],
)
def test_non_finite_constant_reference_lowers_to_its_sympy_value(
    constant_name: str, sympy_value: sympy.Expr
) -> None:
    """Test the canonical ``inf``/``nan`` identifier lowers to `oo`/`nan`."""
    expression = IdentifierExpression(get_native_constant_identifier(constant_name))

    lowered = convert_expression_to_sympy_expression(expression)

    assert lowered == sympy_value


@pytest.mark.parametrize(
    "sympy_value, constant_name",
    [
        pytest.param(sympy.oo, "inf", id="oo"),
        pytest.param(sympy.nan, "nan", id="nan"),
    ],
)
def test_sympy_non_finite_value_lifts_back_to_the_canonical_identifier(
    sympy_value: sympy.Expr, constant_name: str
) -> None:
    """Test `sympy.oo`/`sympy.nan` lift to the canonical ``inf``/``nan`` identifier."""
    lifted = convert_sympy_expression_to_expression(sympy_value)

    assert isinstance(lifted, IdentifierExpression)
    assert lifted.identifier == get_native_constant_identifier(constant_name)


def test_sympy_negative_infinity_lifts_as_the_negated_canonical_inf() -> None:
    """Test `-oo` lifts as ``NEGATE`` over the canonical ``inf`` identifier.

    SymPy folds a negated infinity into an atom of its own, which has no
    registered constant; the IR spells it as the negation of ``inf``.
    """
    lifted = convert_sympy_expression_to_expression(-sympy.oo)

    expected = UnaryExpression(
        UnaryOperation.NEGATE,
        IdentifierExpression(get_native_constant_identifier("inf")),
    )
    assert lifted.is_structurally_equivalent(expected)


@pytest.mark.parametrize("constant_name", ["inf", "nan"])
def test_simplify_returns_a_non_finite_constant_as_its_canonical_identifier(
    constant_name: str,
) -> None:
    """Test simplifying ``c + 0`` for ``inf``/``nan`` yields the constant itself.

    SymPy folds the sum to ``oo`` or ``nan``, which lifts back to the
    identifier the expression started from, so simplification keeps the
    reference just as it does for ``pi`` and ``e``.
    """
    constant = IdentifierExpression(get_native_constant_identifier(constant_name))

    result = simplify_expression(constant + LiteralExpression(0))

    assert result.is_structurally_equivalent(constant)


def test_simplify_negated_inf_returns_the_negated_canonical_identifier() -> None:
    """Test simplifying ``-inf`` round-trips through `-oo` to the same tree."""
    infinity = IdentifierExpression(get_native_constant_identifier("inf"))

    result = simplify_expression(-infinity)

    assert result.is_structurally_equivalent(
        UnaryExpression(UnaryOperation.NEGATE, infinity)
    )


@pytest.mark.parametrize(
    "value, expected_name, is_negated",
    [
        pytest.param(math.inf, "inf", False, id="inf"),
        pytest.param(-math.inf, "inf", True, id="negative_inf"),
        pytest.param(math.nan, "nan", False, id="nan"),
    ],
)
def test_simplify_non_finite_float_literal_lifts_to_the_native_constant(
    value: float, expected_name: str, is_negated: bool
) -> None:
    """Test a non-finite ``float`` literal simplifies to the matching constant.

    SymPy holds one ``oo`` and one ``nan`` rather than a float form of
    each, so the lift has no literal to return; the registered constant
    denotes the same value.
    """
    constant = IdentifierExpression(get_native_constant_identifier(expected_name))
    expected: Expression = (
        UnaryExpression(UnaryOperation.NEGATE, constant) if is_negated else constant
    )

    result = simplify_expression(LiteralExpression(value) + LiteralExpression(0))

    assert result.is_structurally_equivalent(expected)


def test_native_call_with_constant_argument_round_trips_through_sympy() -> None:
    """Test `sin(pi)` round-trips: lowers, simplifies, lifts back to literal.

    Sympy simplifies ``sin(pi)`` to ``0``; the lifted expression is the
    integer literal ``0`` (not a constant reference any more).
    """
    expression = call("sin", get_native_constant_identifier("pi"))

    lowered = convert_expression_to_sympy_expression(expression)
    simplified = sympy.simplify(lowered)
    lifted = convert_sympy_expression_to_expression(simplified)

    assert isinstance(lifted, LiteralExpression)
    assert lifted.value == 0
