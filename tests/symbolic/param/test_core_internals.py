"""Tests for private helpers in `fhy_core.symbolic.param`.

The helpers exercised here cover validation paths that the public-API tests
cannot easily reach because the public constructors and validators reject
malformed inputs before they propagate. Each test calls the private helper
directly.
"""

from typing import Any

import pytest

from fhy_core.symbolic.param.values import ParamError, serialize_wrapped_leaf_value

from .conftest import mock_identifier

# =============================================================================
# `serialize_wrapped_leaf_value`
# =============================================================================


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="bool"),
        pytest.param(1, id="int"),
        pytest.param(1.5, id="float"),
        pytest.param("text", id="str"),
        pytest.param(mock_identifier("x", 0), id="serializable"),
    ],
)
def test_serialize_wrapped_leaf_value_accepts_each_supported_type(
    value: Any,
) -> None:
    """Test the helper serializes each supported leaf-value type without raising."""
    serialize_wrapped_leaf_value(value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([1, 2, 3], id="list"),
        pytest.param({1: 2}, id="dict"),
        pytest.param(object(), id="opaque-object"),
    ],
)
def test_serialize_wrapped_leaf_value_rejects_unsupported_type(
    value: Any,
) -> None:
    """Test the helper raises `ParamError` for a value of an unsupported type."""
    with pytest.raises(ParamError, match="serializable leaf"):
        serialize_wrapped_leaf_value(value)


# =============================================================================
# Constraint ordering
#
# `Param` constraint ordering goes through `Constraint.build_ordering_key`,
# covered directly under `tests/symbolic/constraint/**`.
# `test_scope_attachment.py::test_param_constraint_tuple_matches_build_ordering_key_order`  # noqa: E501
# pins the same ordering property at the `Param` level.
# =============================================================================
