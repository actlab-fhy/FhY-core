"""Tests for private helpers in `fhy_core.symbolic.param`.

The helpers exercised here cover validation paths that the public-API tests
cannot easily reach because the public constructors and validators reject
malformed inputs before they propagate. Each test calls the private helper
directly.
"""

from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.param.values import ParamError, serialize_wrapped_leaf_value

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
        pytest.param(Identifier("x"), id="serializable"),
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
# `_constraint_structural_ordering_key` (design: D3)
#
# This private helper is deleted: `param/core.py` no longer builds its own
# JSON-based constraint ordering key. Constraint ordering now goes through
# the constraint module's public `build_constraint_ordering_key`, which is a
# structural key (not a serialized-dict key), so the dict-key-order-
# independence property this section used to pin via a duck-typed stub no
# longer applies to anything in this module. `build_constraint_ordering_key`
# has its own coverage in `tests/symbolic/constraint/**`, and
# `test_scope_attachment.py::test_param_constraint_tuple_matches_build_constraint_ordering_key_order`  # noqa: E501
# pins the same property at the `Param` level.
# =============================================================================
