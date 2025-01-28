"""Tests the string utilities module."""

from fhy_core.utils.str_utils import format_comma_separated_list


def test_separate_list_of_numbers_with_comma():
    """Test separating a list of integers with commas and spaces."""
    items = [1, 2, 3]
    expected = "1, 2, 3"
    result = format_comma_separated_list(items)
    assert result == expected


def test_separate_list_of_numbers_with_comma_without_space():
    """Test separating a list of integers with commas and no spaces."""
    items = [1, 2, 3]
    expected = "1,2,3"
    result = format_comma_separated_list(items, add_space=False)
    assert result == expected


def test_separate_empty_list_with_comma():
    """Test separating an empty list with commas and spaces."""
    items = []
    expected = ""
    result = format_comma_separated_list(items)
    assert result == expected


def test_separate_single_item_list_with_comma():
    """Test separating a single-item list with commas and spaces."""
    items = [1]
    expected = "1"
    result = format_comma_separated_list(items)
    assert result == expected


def test_separate_list_of_numbers_with_comma_with_custom_str_func():
    items = [1, 2, 3]
    expected = "1 -,2 -,3 -"
    result = format_comma_separated_list(
        items, str_func=lambda x: f"{x} -", add_space=False
    )
    assert result == expected
