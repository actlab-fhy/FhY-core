"""Hypothesis property tests for the expression pretty-printer.

Covers cheap invariants of ``pformat_expression``: a DICT
serialization round trip formats identically to the original tree, every
free identifier's name hint appears somewhere in the formatted output,
and the Python ``ExpressionPrettyFormatter`` renders the core's text.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given

from fhy_core.symbolic.expression import Expression, pformat_expression
from fhy_core.symbolic.expression.pprint import ExpressionPrettyFormatter

from ...strategies.identifiers import build_identifier_pool
from ...strategies.structural_expressions import build_structural_expression_strategy

pytestmark = pytest.mark.property

_POOL = build_identifier_pool(3)


@given(build_structural_expression_strategy(_POOL))
def test_dict_round_trip_formats_identically(expression: Expression) -> None:
    """Test a DICT round trip pretty-prints to the same string as the original."""
    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())

    assert pformat_expression(restored) == pformat_expression(expression)


@given(build_structural_expression_strategy(_POOL))
def test_every_free_identifier_name_hint_appears_in_the_output(
    expression: Expression,
) -> None:
    """Test every free identifier's name_hint is a substring of the formatted output."""
    formatted = pformat_expression(expression)

    for identifier in expression.get_free_identifiers():
        assert identifier.name_hint in formatted


@given(build_structural_expression_strategy(_POOL))
def test_pretty_formatter_agrees_with_the_core_text(expression: Expression) -> None:
    """Test the Python formatter renders the core's text under every option."""
    for show_id in (False, True):
        for functional in (False, True):
            formatter = ExpressionPrettyFormatter(
                is_id_shown=show_id, is_printed_functional=functional
            )
            assert formatter(expression) == pformat_expression(
                expression, show_id=show_id, functional=functional
            )
