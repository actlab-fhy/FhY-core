"""String utilities."""

from collections.abc import Iterable
from typing import Callable, TypeVar

_T = TypeVar("_T")


def format_comma_separated_list(
    items: Iterable[_T], str_func: Callable[[_T], str] = repr, add_space: bool = True
) -> str:
    """Return a comma-separated list of items.

    Args:
        items: An iterable of items.
        str_func: A function to convert each item to a string.
        add_space: Whether to add a space after each comma.

    Returns:
        A comma-separated list of items.

    """
    join_char = ", " if add_space else ","
    return join_char.join(str_func(item) for item in items)
