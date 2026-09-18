"""Hypothesis property tests for the SymPy bridge.

Covers the SymPy round trip -- semantic (numeric and boolean gate trees,
NumPy oracle) and structural (the SymPy-stable subset) -- and the
cross-bridge substitution contract between ``Expression.substitute`` and
``substitute_sympy_expression_variables``.

Every tree the semantic round trip and the substitution properties draw
may hold Boolean identifiers from a pool of their own:
as piecewise conditions, as Boolean piecewise branches, and as operands
of connectives and of ``==``/``!=`` between Booleans, so a Boolean
piecewise reaches every Boolean position the bridge rewrites. The
environments bind each Boolean identifier to a bool, and the
substitution properties replace one Boolean identifier with a Boolean
tree, which may itself be a Boolean piecewise.

The semantic round trip and both substitution properties draw
division-free trees. Lowering ``FLOOR_DIVIDE`` to ``floor(a / b)`` lets
SymPy distribute the divisor over a sum, so ``(v0 * v0 + v0) // -5``
lifts as ``floor(-v0**2 / 5 - v0 / 5)``. The lifted coefficients are
exact integer ``DIVIDE`` nodes, but the NumPy oracle computes them by
float true division, and ``floor`` turns the rounding error into an
off-by-one: ``-23`` rather than ``-22`` at ``v0 = -11``.
They enable calls, restricted to :data:`SYMPY_STABLE_CALL_FUNCTIONS`
(``floor``, ``ceil``, ``round``), the natives whose SymPy node lifts
back to the same native name.
``build_any_sort_expression_strategy`` and
``draw_simultaneous_substitution_case`` thread every option through
every subtree they draw (including a piecewise condition or value, and
every replacement), so setting it once is enough to govern the whole
tree. The structural round trip
uses the already-restricted ``build_sympy_stable_expression_strategy``
instead, which excludes arithmetic ``BinaryExpression`` outright (SymPy
re-associates and folds those) and only ever calls those same natives.
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
    build_gate_environment_strategy,
    build_sympy_stable_expression_strategy,
    draw_simultaneous_substitution_case,
    evaluate_with_python,
)
from ....strategies.identifiers import (
    build_boolean_identifier_pool,
    build_identifier_pool,
)

pytestmark = pytest.mark.property

_POOL: Final[tuple[Identifier, ...]] = build_identifier_pool(3)
_BOOLEAN_POOL: Final[tuple[Identifier, ...]] = build_boolean_identifier_pool(2)
_V0: Final = _POOL[0]

_SubstitutionCase = tuple[
    Expression, dict[Identifier, Expression], dict[Identifier, int | bool]
]


# =============================================================================
# Semantic round trip through SymPy
# =============================================================================


_CONSTANT_BOOLEAN_PIECEWISE_CONJUNCTION: Final[Expression] = make_binary_expression(
    BinaryOperation.LOGICAL_AND,
    piecewise(
        (
            make_binary_expression(BinaryOperation.EQUAL, 0, _V0),
            LiteralExpression(False),
        ),
        otherwise=LiteralExpression(False),
    ),
    LiteralExpression(False),
)


# include_division=False: see the module docstring for the off-by-one a
# lifted FLOOR_DIVIDE can evaluate to under the NumPy oracle.
@example(
    expression=_CONSTANT_BOOLEAN_PIECEWISE_CONJUNCTION,
    environment={**dict.fromkeys(_POOL, 0), **dict.fromkeys(_BOOLEAN_POOL, False)},
)
@given(
    expression=build_any_sort_expression_strategy(
        _POOL,
        6,
        boolean_identifiers=_BOOLEAN_POOL,
        include_division=False,
        include_calls=True,
        native_functions=SYMPY_STABLE_CALL_FUNCTIONS,
        include_piecewise=True,
    ),
    environment=build_gate_environment_strategy(_POOL, _BOOLEAN_POOL),
)
def test_sympy_round_trip_preserves_evaluation(
    expression: Expression, environment: dict[Identifier, int | bool]
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
# "round" round-trips structurally
# =============================================================================


@given(identifier=st.sampled_from(_POOL))
def test_round_call_round_trips_structurally(
    identifier: Identifier,
) -> None:
    """Test ``call("round", v)`` lifts back to the same call.

    Oracle: structural equivalence. ``round`` is the one native with no
    SymPy operator of its own, so it lowers to a SymPy function the
    bridge defines; lifting maps that node back to the native name like
    every other member of ``SYMPY_STABLE_CALL_FUNCTIONS``.
    """
    expression = call("round", IdentifierExpression(identifier))

    lowered = convert_expression_to_sympy_expression(expression)
    lifted = convert_sympy_expression_to_expression(lowered)

    assert lifted.is_structurally_equivalent(expression)


# =============================================================================
# Expression.substitute is simultaneous and agrees with the SymPy bridge
# =============================================================================


# Division stays off: see the module docstring.
_SUBSTITUTION_CASES: Final = draw_simultaneous_substitution_case(
    _POOL,
    _BOOLEAN_POOL,
    6,
    include_division=False,
    include_calls=True,
    native_functions=SYMPY_STABLE_CALL_FUNCTIONS,
    include_piecewise=True,
)


@given(case=_SUBSTITUTION_CASES)
def test_substitute_matches_simultaneous_python_semantics(
    case: _SubstitutionCase,
) -> None:
    """Test ``substitute`` agrees with evaluating under a rebound environment.

    Oracle: ``evaluate_with_python``. ``e.substitute(s)`` evaluated under
    ``env`` must equal ``e`` evaluated under ``env`` with each replaced
    identifier ``v`` rebound to ``eval(s[v], env)``, every rebinding
    computed from the *original* ``env`` (simultaneity). The replacements
    cover two integer identifiers and one Boolean identifier, and each is
    drawn over the same pools, so a replacement may reference an
    identifier that is itself replaced. No SymPy bridge is involved.
    """
    expression, replacements, environment = case

    substituted = expression.substitute(replacements)

    rebound_environment = dict(environment)
    for identifier, replacement in replacements.items():
        rebound_environment[identifier] = evaluate_with_python(replacement, environment)
    assert evaluate_with_python(substituted, environment) == evaluate_with_python(
        expression, rebound_environment
    )


@given(case=_SUBSTITUTION_CASES)
def test_substitute_agrees_with_sympy_bridge_substitution(
    case: _SubstitutionCase,
) -> None:
    """Test the IR and SymPy-bridge substitutions evaluate identically.

    Oracle: ``evaluate_expression_with_numpy``, comparing
    ``lift(substitute_sympy_expression_variables(lower(e), s))`` against
    ``e.substitute(s)`` under the same environment. A Boolean identifier
    ``s`` replaces with a Boolean piecewise reaches the bridge's Boolean
    positions only once substituted.
    """
    expression, replacements, environment = case

    lowered = convert_expression_to_sympy_expression(expression)
    substituted_sympy = substitute_sympy_expression_variables(lowered, replacements)
    lifted = convert_sympy_expression_to_expression(substituted_sympy)
    ir_substituted = expression.substitute(replacements)

    assert int(evaluate_expression_with_numpy(lifted, environment)) == int(
        evaluate_expression_with_numpy(ir_substituted, environment)
    )


# =============================================================================
# A self-referential piecewise replacement puts a piecewise inside a case
# condition and still lifts. Kept as its own property, alongside the drawn
# cases above, because the shape needs a replacement whose piecewise
# condition reuses the very identifier it replaces.
# =============================================================================


@given(identifier=st.sampled_from(_POOL))
def test_substitute_agrees_with_sympy_bridge_substitution_on_self_referential_piecewise(
    identifier: Identifier,
) -> None:
    """Test a self-referential piecewise replacement substitutes identically.

    Oracle: ``evaluate_expression_with_numpy``, comparing the
    SymPy-bridge substitution against ``Expression.substitute``.
    Replacing ``v`` in ``piecewise((0 == v, 0), otherwise=1)`` with
    ``piecewise((0 == v, 0), otherwise=v)`` turns the case condition into
    ``Eq(0, Piecewise(...))``, a comparison with a piecewise operand, and
    the substituted node must still lift to a piecewise that evaluates
    like the IR substitution.
    """
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
