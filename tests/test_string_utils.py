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
