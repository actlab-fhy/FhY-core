"""Tests the string utilities module."""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.utils.str_utils import format_comma_separated_list


@pytest.mark.parametrize(
    "items, add_space, str_func, expected",
    [
        ([1, 2, 3], True, str, "1, 2, 3"),
        ([1, 2, 3], False, str, "1,2,3"),
        ([], True, str, ""),
        ([1], True, str, "1"),
        ([1, 2, 3], False, lambda x: f"{x} -", "1 -,2 -,3 -"),
    ],
)
def test_format_comma_separated_list(
    items: list[Any],
    add_space: bool,
    str_func: Callable[[Any], str],
    expected: str,
) -> None:
    """Test various cases for format_comma_separated_list."""
    result = format_comma_separated_list(items, add_space=add_space, str_func=str_func)
    assert result == expected


@pytest.mark.parametrize(
    "items, str_func, expected",
    [
        (["a", "b"], repr, "a, b"),
        (["a", 1, "b", 2], lambda x: f"<{x}>", "a, <1>, b, <2>"),
    ],
    ids=["only_strings", "strings_mixed_with_non_strings"],
)
def test_format_comma_separated_list_inserts_string_items_without_str_func(
    items: list[Any],
    str_func: Callable[[Any], str],
    expected: str,
) -> None:
    """Test string items are joined as they are and only non-strings use str_func."""
    result = format_comma_separated_list(items, str_func=str_func)
    assert result == expected


def test_format_comma_separated_list_keeps_string_items_unquoted_by_default() -> None:
    """Test string items are not passed through the default repr str_func."""
    assert format_comma_separated_list(["a", "b"]) == "a, b"
