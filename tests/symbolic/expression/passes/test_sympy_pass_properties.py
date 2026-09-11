"""Hypothesis property tests for the SymPy bridge.

Covers the SymPy round trip -- semantic (numeric and boolean gate trees,
NumPy oracle) and structural (the SymPy-stable subset) -- and the
cross-bridge substitution contract between ``Expression.substitute`` and
``substitute_sympy_expression_variables``.

The semantic round trip and both substitution properties draw
division-free trees: lowering ``FLOOR_DIVIDE``/``MODULO`` can
auto-evaluate to a ``Rational`` that lifts to an exact-decimal string
literal the NumPy oracle then refuses (the same finding
``tests/symbolic/test_solver_properties.py`` pins with a strict xfail).
They enable calls, but restricted to :data:`SYMPY_STABLE_CALL_FUNCTIONS`
(``floor``, ``ceil``): the other member of
``INTEGER_RESULT_NATIVE_FUNCTIONS``, ``round``, cannot be lifted back
from SymPy, the gap ``test_round_call_structural_round_trip_is_unsupported``
below pins. ``build_numeric_expression_strategy`` and
``build_boolean_expression_strategy`` thread ``include_division``,
``include_calls``, and ``native_functions`` through every subtree they
draw (including a piecewise condition or value), so setting them once at
the root is enough to govern the whole tree. The structural round trip
uses the already-restricted ``build_sympy_stable_expression_strategy``
instead, which excludes arithmetic ``BinaryExpression`` outright (SymPy
re-associates and folds those) and only ever calls ``floor``/``ceil``.
"""

from typing import Final

import pytest

pytest.importorskip("hypothesis")
pytest.importorskip("numpy")

from hypothesis import example, given
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    call,
    convert_expression_to_sympy_expression,
    convert_sympy_expression_to_expression,
    evaluate_expression_with_numpy,
    make_binary_expression,
    piecewise,
    substitute_sympy_expression_variables,
)

from ....strategies.expressions import (
    SYMPY_STABLE_CALL_FUNCTIONS,
    build_any_sort_expression_strategy,
    build_integer_environment_strategy,
    build_numeric_expression_strategy,
    build_sympy_stable_expression_strategy,
    evaluate_with_python,
)
from ....strategies.identifiers import build_identifier_pool

pytestmark = pytest.mark.property

_POOL: Final[tuple[Identifier, ...]] = build_identifier_pool(3)
_V0, _V1 = _POOL[0], _POOL[1]


# =============================================================================
# Semantic round trip through SymPy
# =============================================================================


# include_division=False: a symbolic FLOOR_DIVIDE/MODULO can lower and
# auto-evaluate to a Rational such as 1/5, which lifts to the
# exact-decimal string literal "0.2" that evaluate_expression_with_numpy
# refuses with StringLiteralPrecisionError (no binary float equals 0.2
# exactly). See the module docstring.
@given(
    expression=build_any_sort_expression_strategy(
        _POOL,
        6,
        include_division=False,
        include_calls=True,
        native_functions=SYMPY_STABLE_CALL_FUNCTIONS,
        include_piecewise=True,
    ),
    environment=build_integer_environment_strategy(_POOL),
)
def test_sympy_round_trip_preserves_evaluation(
    expression: Expression, environment: dict[Identifier, int]
) -> None:
    """Test lifting the lowering of a tree evaluates identically to the tree.

    Oracle: ``evaluate_expression_with_numpy``, applied to both the
    original and the round-tripped expression.
    """
    lowered = convert_expression_to_sympy_expression(expression)
    lifted = convert_sympy_expression_to_expression(lowered)

    original_value = evaluate_expression_with_numpy(expression, environment)
    lifted_value = evaluate_expression_with_numpy(lifted, environment)
    if isinstance(original_value, bool):
        assert bool(lifted_value) == bool(original_value)
    else:
        assert int(lifted_value) == int(original_value)


# =============================================================================
# Structural round trip on the SymPy-stable subset
# =============================================================================


# Hand-picked cases naming natives outside the strategy's own vocabulary
# (SYMPY_STABLE_CALL_FUNCTIONS): both round-trip structurally with no
# simplification involved, for the same reason the drawn shapes do.
_EXP_OF_IDENTIFIER: Final[Expression] = call("exp", IdentifierExpression(_V0))
_SIN_OF_SQRT_OF_IDENTIFIER: Final[Expression] = call(
    "sin", call("sqrt", IdentifierExpression(_V0))
)


@example(expression=_EXP_OF_IDENTIFIER)
@example(expression=_SIN_OF_SQRT_OF_IDENTIFIER)
@given(expression=build_sympy_stable_expression_strategy(_POOL))
def test_sympy_round_trip_is_structurally_stable(expression: Expression) -> None:
    """Test lifting the lowering of a SymPy-stable tree reconstructs it exactly.

    Oracle: structural equivalence. ``build_sympy_stable_expression_strategy``
    is restricted to the shapes SymPy neither re-associates, folds, nor
    auto-evaluates (see its own docstring); the two pinned examples above
    additionally cover natives outside that strategy's
    ``SYMPY_STABLE_CALL_FUNCTIONS`` (``exp``, and a nested ``sin(sqrt(x))``),
    which round-trip structurally for the same reason.
    """
    lowered = convert_expression_to_sympy_expression(expression)
    lifted = convert_sympy_expression_to_expression(lowered)

    assert lifted.is_structurally_equivalent(expression)


# =============================================================================
# Known SymPy lifting finding -- "round" has no lift dispatch entry
# =============================================================================


@pytest.mark.xfail(
    strict=True,
    reason=(
        "convert_sympy_expression_to_expression has no dispatch entry for "
        "SymPy's opaque round() Function node: lifting "
        "call('round', IdentifierExpression(v)) raises PassExecutionError "
        "wrapping TypeError: Unsupported expression type: round. "
        "SYMPY_STABLE_CALL_FUNCTIONS therefore lists only floor and ceil."
    ),
)
@given(identifier=st.sampled_from(_POOL))
def test_round_call_structural_round_trip_is_unsupported(
    identifier: Identifier,
) -> None:
    """Test ``call("round", v)`` fails to round-trip: lifting raises ``TypeError``."""
    expression = call("round", IdentifierExpression(identifier))

    lowered = convert_expression_to_sympy_expression(expression)
    lifted = convert_sympy_expression_to_expression(lowered)

    assert lifted.is_structurally_equivalent(expression)


# =============================================================================
# Expression.substitute is simultaneous and agrees with the SymPy bridge
# =============================================================================


@st.composite
def _draw_substitution_case(
    draw: st.DrawFn,
    *,
    include_piecewise: bool,
) -> tuple[Expression, Expression, Expression, dict[Identifier, int]]:
    """Draw ``e``, replacements ``e1``/``e2`` for ``v0``/``v1``, and an environment.

    ``e1`` and ``e2`` are drawn over the same pool as ``e`` (not a
    disjoint set of fresh variables), so a replacement can itself
    reference ``v0`` or ``v1`` and simultaneity actually matters:
    substituting ``{v0: e1, v1: e2}`` must not let ``e1`` see ``v1``'s
    replacement or vice versa.

    ``include_piecewise`` is a parameter, not always-on, because of the
    SymPy-bridge lift gap
    ``test_substitute_agrees_with_sympy_bridge_substitution_on_self_referential_piecewise``
    below pins: the plain-Python oracle test has no such gap and always
    draws piecewise, but the SymPy-bridge test excludes it.
    """
    # Division stays off: see the module docstring for the
    # StringLiteralPrecisionError finding this excludes.
    numeric_strategy = build_numeric_expression_strategy(
        _POOL,
        6,
        include_division=False,
        include_calls=True,
        native_functions=SYMPY_STABLE_CALL_FUNCTIONS,
        include_piecewise=include_piecewise,
    )
    e = draw(numeric_strategy)
    e1 = draw(numeric_strategy)
    e2 = draw(numeric_strategy)
    environment = draw(build_integer_environment_strategy(_POOL))
    return e, e1, e2, environment


@given(case=_draw_substitution_case(include_piecewise=True))
def test_substitute_matches_simultaneous_python_semantics(
    case: tuple[Expression, Expression, Expression, dict[Identifier, int]],
) -> None:
    """Test ``substitute`` agrees with evaluating under a rebound environment.

    Oracle: ``evaluate_with_python``. ``e.substitute({v0: e1, v1: e2})``
    evaluated under ``env`` must equal ``e`` evaluated under ``env``
    updated with ``v0 -> eval(e1, env)`` and ``v1 -> eval(e2, env)``,
    both computed from the *original* ``env`` (simultaneity). No SymPy
    bridge is involved, so piecewise stays enabled.
    """
    e, e1, e2, environment = case
    replacements = {_V0: e1, _V1: e2}

    substituted = e.substitute(replacements)

    rebound_environment = dict(environment)
    rebound_environment[_V0] = evaluate_with_python(e1, environment)
    rebound_environment[_V1] = evaluate_with_python(e2, environment)
    assert evaluate_with_python(substituted, environment) == evaluate_with_python(
        e, rebound_environment
    )


# include_piecewise=False: see
# test_substitute_agrees_with_sympy_bridge_substitution_on_self_referential_piecewise
# below for the pinned gap a piecewise-holding replacement can trigger
# through this test's SymPy bridge.
@given(case=_draw_substitution_case(include_piecewise=False))
def test_substitute_agrees_with_sympy_bridge_substitution(
    case: tuple[Expression, Expression, Expression, dict[Identifier, int]],
) -> None:
    """Test the IR and SymPy-bridge substitutions evaluate identically.

    Oracle: ``evaluate_expression_with_numpy``, comparing
    ``lift(substitute_sympy_expression_variables(lower(e), s))`` against
    ``e.substitute(s)`` under the same environment.
    """
    e, e1, e2, environment = case
    replacements = {_V0: e1, _V1: e2}

    lowered = convert_expression_to_sympy_expression(e)
    substituted_sympy = substitute_sympy_expression_variables(lowered, replacements)
    lifted = convert_sympy_expression_to_expression(substituted_sympy)
    ir_substituted = e.substitute(replacements)

    assert int(evaluate_expression_with_numpy(lifted, environment)) == int(
        evaluate_expression_with_numpy(ir_substituted, environment)
    )


# =============================================================================
# Second SymPy substitution finding: a piecewise-holding replacement whose
# own condition reuses the substituted identifier lifts to an unsupported
# boolean node. Covered by a separate, strict xfail below rather than
# folded into the general substitution property above, which excludes
# piecewise from its replacements for exactly this reason.
# =============================================================================


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Substituting an identifier with a numeric replacement that itself "
        "holds a piecewise whose own condition references that same "
        "identifier makes substitute_sympy_expression_variables's sympy.subs "
        "fold the resulting Eq(0, Piecewise(...)) into a boolean ITE(...) "
        "node; convert_bool's dispatch table (Not/And/Or/Xor/Nor/Nand/...) "
        "has no entry for ITE, so lifting raises PassExecutionError wrapping "
        "TypeError: Unsupported boolean expression type: ITE. Minimal "
        "example: target = piecewise((0 == v, 0), otherwise=1), replacement "
        "for v = piecewise((0 == v, 0), otherwise=v)."
    ),
)
@given(identifier=st.sampled_from(_POOL))
def test_substitute_agrees_with_sympy_bridge_substitution_on_self_referential_piecewise(
    identifier: Identifier,
) -> None:
    """Test a self-referential piecewise replacement fails to lift back."""
    condition = make_binary_expression(
        BinaryOperation.EQUAL, LiteralExpression(0), identifier
    )
    zero_case = (condition, LiteralExpression(0))
    target = piecewise(zero_case, otherwise=LiteralExpression(1))
    replacement = piecewise(zero_case, otherwise=identifier)
    replacements = {identifier: replacement}
    environment = {identifier: 0}

    lowered = convert_expression_to_sympy_expression(target)
    substituted_sympy = substitute_sympy_expression_variables(lowered, replacements)
    lifted = convert_sympy_expression_to_expression(substituted_sympy)
    ir_substituted = target.substitute(replacements)

    assert int(evaluate_expression_with_numpy(lifted, environment)) == int(
        evaluate_expression_with_numpy(ir_substituted, environment)
    )
