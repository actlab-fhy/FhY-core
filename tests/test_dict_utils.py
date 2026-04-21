"""Tests the general dictionary utilities."""

from frozendict import frozendict

from fhy_core.utils.dict_utils import invert_dict, invert_frozen_dict


def test_invert_dict() -> None:
    """Test that the dictionary inversion works."""
    test_dict = {"a": 1, "b": 2, "c": 3}
    inverted_dict = invert_dict(test_dict)
    assert inverted_dict == {1: "a", 2: "b", 3: "c"}


def test_invert_frozen_dict() -> None:
    """Test that the frozen dictionary inversion works."""
    test_dict = {"a": 1, "b": 2, "c": 3}
    frozen_dict = frozendict(test_dict)
    inverted_dict = invert_frozen_dict(frozen_dict)
    assert inverted_dict == {1: "a", 2: "b", 3: "c"}
