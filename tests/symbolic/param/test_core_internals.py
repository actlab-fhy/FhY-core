"""Tests for private helpers in `fhy_core.symbolic.param`.

The helpers exercised here cover validation paths that the public-API tests
cannot easily reach because the public constructors and validators reject
malformed inputs before they propagate. Each test calls the private helper
directly.
"""

from typing import Any

import pytest

from fhy_core.symbolic.param.core import _constraint_structural_ordering_key
from fhy_core.symbolic.param.values import ParamError, serialize_wrapped_leaf_value

from .conftest import SerializableEqualHashable

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
        pytest.param(SerializableEqualHashable(1), id="serializable"),
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
# `_constraint_structural_ordering_key`
#
# Tests this private helper directly: the public `Param.is_structurally_equivalent`
# path cannot easily reach the key-order-independence behavior pinned here.
# =============================================================================


class _StubConstraintWithOrderedDict:
    """Minimal stub whose ``serialize_to_dict`` yields a configured key order."""

    def __init__(self, items: tuple[tuple[str, Any], ...]) -> None:
        self._items = items

    def serialize_to_dict(self) -> dict[str, Any]:
        return dict(self._items)


def test_constraint_structural_ordering_key_is_stable_across_dict_key_orders() -> None:
    """Test the helper returns the same key regardless of serialized-dict key order.

    Two stub constraints whose serialized dicts contain the same items in
    opposite insertion order must produce identical ordering keys. Otherwise
    `Param.is_structurally_equivalent` can pair constraints incorrectly when
    constraint serialization rearranges keys.
    """
    forward = _StubConstraintWithOrderedDict(
        (("variable", {"name": "x"}), ("operation", "GE"), ("value", 0))
    )
    reverse = _StubConstraintWithOrderedDict(
        (("value", 0), ("operation", "GE"), ("variable", {"name": "x"}))
    )

    # The structural stubs satisfy the duck-typed contract the helper consumes.
    forward_key = _constraint_structural_ordering_key(forward)  # type: ignore[arg-type]
    reverse_key = _constraint_structural_ordering_key(reverse)  # type: ignore[arg-type]

    assert forward_key == reverse_key
