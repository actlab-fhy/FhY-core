"""Hypothesis property tests for the expression type checker.

Checks that `check_expression_type(e, t, lookup)` succeeds and reproduces
`t` when `t` is exactly what `synthesize_expression_type(e, lookup)` just
returned, for numeric and boolean gate-grammar trees over a pool of
identifiers all bound to a concrete `int32` scalar. A numeric tree with
no identifier and no call anywhere synthesizes a *weak*, unresolved-width
type (plain `uint`, `int`, or `float`); the round trip holds for those
too, since a literal checked against a weak expected type keeps the weak
type it synthesizes on its own instead of resolving to a width. The
bare-literal case is pinned on its own at the bottom of this file.
"""

import pytest

pytest.importorskip("hypothesis")

from collections.abc import Callable

from hypothesis import given

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    Expression,
    LiteralExpression,
)
from fhy_core.types import (
    CoreDataType,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
    is_structurally_equivalent,
)
from fhy_core.types.checking.type_checker import (
    check_expression_type,
    synthesize_expression_type,
)

from ...strategies.expressions import (
    build_boolean_expression_strategy,
    build_numeric_expression_strategy,
)
from ...strategies.identifiers import build_identifier_pool

pytestmark = pytest.mark.property

_POOL = build_identifier_pool(4)
_MAX_LEAVES = 8
_INT32_SCALAR: Type = NumericalType(PrimitiveDataType(CoreDataType.INT32), [])


def build_pool_lookup(
    bindings: dict[Identifier, tuple[Type, TypeQualifier]],
) -> Callable[[Identifier], tuple[Type, TypeQualifier]]:
    """Build a `get_identifier_type` callable bound to `bindings`.

    Mirrors the lookup `make_identifier_checker`
    (`tests/types/checking/conftest.py`) builds internally: an unknown
    identifier raises `KeyError`, which `ExpressionTypeChecker` catches
    and frames as a type error.
    """

    def lookup(identifier: Identifier) -> tuple[Type, TypeQualifier]:
        if identifier in bindings:
            return bindings[identifier]
        raise KeyError(identifier.name_hint)

    return lookup


_LOOKUP = build_pool_lookup(dict.fromkeys(_POOL, (_INT32_SCALAR, TypeQualifier.PARAM)))


@given(build_numeric_expression_strategy(_POOL, _MAX_LEAVES, include_calls=True))
def test_synthesize_then_check_round_trips_for_numeric_gate_trees(
    expression: Expression,
) -> None:
    """Test checking a numeric gate tree against its own synthesized type succeeds.

    Oracle: `synthesize_expression_type` and `check_expression_type`
    should agree once the expected type given to `check` is exactly what
    `synthesize` already produced.
    """
    synthesized_type, _ = synthesize_expression_type(expression, _LOOKUP)
    assert isinstance(synthesized_type, NumericalType)
    assert isinstance(synthesized_type.data_type, PrimitiveDataType)
    assert synthesized_type.data_type.core_data_type != CoreDataType.BOOL

    checked_type, _ = check_expression_type(expression, synthesized_type, _LOOKUP)
    assert is_structurally_equivalent(checked_type, synthesized_type)


@given(build_boolean_expression_strategy(_POOL, _MAX_LEAVES))
def test_synthesize_then_check_round_trips_for_boolean_gate_trees(
    expression: Expression,
) -> None:
    """Test checking a boolean gate tree against its own synthesized type succeeds.

    Oracle: same round trip as the numeric case; a boolean gate tree is
    always concrete, since `BOOL` has no weak counterpart.
    """
    synthesized_type, _ = synthesize_expression_type(expression, _LOOKUP)
    assert isinstance(synthesized_type, NumericalType)
    assert isinstance(synthesized_type.data_type, PrimitiveDataType)
    assert synthesized_type.data_type.core_data_type == CoreDataType.BOOL

    checked_type, _ = check_expression_type(expression, synthesized_type, _LOOKUP)
    assert is_structurally_equivalent(checked_type, synthesized_type)


def test_synthesize_then_check_round_trips_for_a_bare_weak_literal() -> None:
    """Test the round trip for the smallest tree whose synthesized type is weak.

    A bare literal is the shape
    `test_synthesize_then_check_round_trips_for_numeric_gate_trees` above
    shrinks to: `LiteralExpression(0)` synthesizes plain `uint`, and
    checking it against that weak type returns `uint` rather than
    resolving to a concrete width.
    """
    expression: Expression = LiteralExpression(0)
    synthesized_type, _ = synthesize_expression_type(expression, _LOOKUP)
    checked_type, _ = check_expression_type(expression, synthesized_type, _LOOKUP)
    assert is_structurally_equivalent(checked_type, synthesized_type)
