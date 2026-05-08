"""Tests numeric type predicates."""

import pytest

from fhy_core.utils.numeric_utils import is_strict_int


@pytest.mark.parametrize(
    "value",
    [0, 1, -1, 2**62, -(2**62)],
    ids=["zero", "one", "minus-one", "large-positive", "large-negative"],
)
def test_is_strict_int_accepts_plain_ints(value: int) -> None:
    """Test `is_strict_int` returns True for plain integers."""
    assert is_strict_int(value) is True


@pytest.mark.parametrize("value", [True, False], ids=["true", "false"])
def test_is_strict_int_rejects_booleans(value: bool) -> None:
    """Test `is_strict_int` returns False for booleans."""
    assert is_strict_int(value) is False


@pytest.mark.parametrize(
    "value",
    [None, 1.0, "1", b"1", [1], (1,), object()],
    ids=["none", "float", "str", "bytes", "list", "tuple", "object"],
)
def test_is_strict_int_rejects_non_integers(value: object) -> None:
    """Test `is_strict_int` returns False for non-integer values."""
    assert is_strict_int(value) is False
