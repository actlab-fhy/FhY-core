"""Hypothesis property tests for the SymPy bridge (P4, P5).

Covers the SymPy round trip -- semantic (numeric and boolean gate trees,
NumPy oracle) and structural (the SymPy-stable subset) -- and the
cross-bridge substitution contract between ``Expression.substitute`` and
``substitute_sympy_expression_variables``.

The semantic round trip (P4a) and both substitution properties (P5)
exclude native calls, division, and piecewise nodes from their numeric
gate trees, and exclude piecewise from their substitution subtrees, for
the same reasons ``tests/symbolic/test_solver_properties.py`` narrows
its own strategies: lowering a call to an expression-bodied
``RegisteredFunction`` (``sign``) raises before the round trip even
starts, and lowering ``FLOOR_DIVIDE``/``MODULO`` can auto-evaluate to a
``Rational`` that lifts to an exact-decimal string literal the NumPy
oracle then refuses. P4b uses the already-restricted
``build_sympy_stable_expression_strategy`` instead, which excludes
arithmetic ``BinaryExpression`` outright (SymPy re-associates and folds
those) and only ever calls ``floor``/``ceil``.
"""

from typing import Final

import pytest

pytest.importorskip("hypothesis")
pytest.importorskip("numpy")

from hypothesis import example, given
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    UnaryOperation,
    call,
    convert_expression_to_sympy_expression,
    convert_sympy_expression_to_expression,
    evaluate_expression_with_numpy,
    make_binary_expression,
    make_unary_expression,
    substitute_sympy_expression_variables,
)

from ....strategies.expressions import (
    COMPARISON_OPERATIONS,
    LOGICAL_BINARY_OPERATIONS,
    build_integer_environment_strategy,
    build_numeric_expression_strategy,
    build_sympy_stable_expression_strategy,
    evaluate_with_python,
)
from ....strategies.identifiers import build_identifier_pool
from ....strategies.literals import build_boolean_literal_strategy

pytestmark = pytest.mark.property

_POOL: Final[tuple[Identifier, ...]] = build_identifier_pool(3)
_V0, _V1 = _POOL[0], _POOL[1]


def _build_call_free_numeric_strategy(
    identifiers: tuple[Identifier, ...], max_leaves: int = 6
) -> st.SearchStrategy[Expression]:
    """Return a numeric gate-tree strategy with no call, division, or piecewise.

    The three are excluded for the reasons the module docstring gives:
    a call can raise before lowering even completes, and division can
    auto-evaluate to a ``Rational`` the NumPy oracle then refuses.
    Excluding piecewise closes a leak the shared strategy package does
    not expose a parameter for: its boolean-condition subtree always
    draws a full-default (call- and division-including) numeric operand
    for a comparison, regardless of this module's own restriction.
    """
    return build_numeric_expression_strategy(
        identifiers,
        max_leaves,
        include_division=False,
        include_calls=False,
        include_piecewise=False,
    )


def _build_call_free_boolean_strategy(
    identifiers: tuple[Identifier, ...], max_leaves: int = 4
) -> st.SearchStrategy[Expression]:
    """Return a boolean gate-tree strategy whose comparisons never call or divide.

    Built with ``st.recursive`` rather than the shared leaf-budget
    machinery, mirroring ``test_solver_properties.py``'s strategy of the
    same shape and for the same reason.
    """
    numeric = _build_call_free_numeric_strategy(identifiers, max_leaves=3)
    comparisons = st.builds(
        make_binary_expression,
        st.sampled_from(COMPARISON_OPERATIONS),
        numeric,
        numeric,
    )
    base = st.one_of(build_boolean_literal_strategy(), comparisons)

    def _extend(
        children: st.SearchStrategy[Expression],
    ) -> st.SearchStrategy[Expression]:
        negations = children.map(
            lambda operand: make_unary_expression(UnaryOperation.LOGICAL_NOT, operand)
        )
        binary = st.builds(
            make_binary_expression,
            st.sampled_from(LOGICAL_BINARY_OPERATIONS),
            children,
            children,
        )
        return st.one_of(negations, binary)

    return st.recursive(base, _extend, max_leaves=max_leaves)


# =============================================================================
# P4a: semantic round trip through SymPy
# =============================================================================


@given(
    expression=st.one_of(
        _build_call_free_numeric_strategy(_POOL),
        _build_call_free_boolean_strategy(_POOL),
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
# P4b: structural round trip on the SymPy-stable subset
# =============================================================================


# Welded from tests/symbolic/expression/test_sympy_natives.py, deleted there
# in the same change: both round-trip structurally with no simplification
# involved.
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
    auto-evaluates (see its own docstring); the two welded examples
    additionally cover natives outside that strategy's
    ``SYMPY_STABLE_CALL_FUNCTIONS`` (``exp``, and a nested ``sin(sqrt(x))``),
    which round-trip structurally for the same reason.
    """
    lowered = convert_expression_to_sympy_expression(expression)
    lifted = convert_sympy_expression_to_expression(lowered)

    assert lifted.is_structurally_equivalent(expression)


# =============================================================================
# Second SymPy lifting finding (not the "round" gap below): a native call
# on a native-constant argument that SymPy evaluates in closed form at
# lowering time, before the lift even runs, does not round-trip
# structurally. Welded from test_sympy_natives.py::
# test_native_call_with_constant_argument_round_trips_through_sympy
# (deleted there in the same change).
# =============================================================================


# =============================================================================
# Addendum: known SymPy lifting finding -- "round" has no lift dispatch entry
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
# P5: Expression.substitute is simultaneous and agrees with the SymPy bridge
# =============================================================================


@st.composite
def _draw_substitution_case(
    draw: st.DrawFn,
) -> tuple[Expression, Expression, Expression, dict[Identifier, int]]:
    """Draw ``e``, replacements ``e1``/``e2`` for ``v0``/``v1``, and an environment.

    ``e1`` and ``e2`` are drawn over the same pool as ``e`` (not a
    disjoint set of fresh variables), so a replacement can itself
    reference ``v0`` or ``v1`` and simultaneity actually matters:
    substituting ``{v0: e1, v1: e2}`` must not let ``e1`` see ``v1``'s
    replacement or vice versa.
    """
    e = draw(_build_call_free_numeric_strategy(_POOL))
    e1 = draw(_build_call_free_numeric_strategy(_POOL))
    e2 = draw(_build_call_free_numeric_strategy(_POOL))
    environment = draw(build_integer_environment_strategy(_POOL))
    return e, e1, e2, environment


@given(case=_draw_substitution_case())
def test_substitute_matches_simultaneous_python_semantics(
    case: tuple[Expression, Expression, Expression, dict[Identifier, int]],
) -> None:
    """Test ``substitute`` agrees with evaluating under a rebound environment.

    Oracle: ``evaluate_with_python``. ``e.substitute({v0: e1, v1: e2})``
    evaluated under ``env`` must equal ``e`` evaluated under ``env``
    updated with ``v0 -> eval(e1, env)`` and ``v1 -> eval(e2, env)``,
    both computed from the *original* ``env`` (simultaneity).
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


@given(case=_draw_substitution_case())
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
