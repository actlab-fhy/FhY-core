"""The exceptions of :mod:`fhy_core.search_space`.

Each message is the Rust core's text for the problem, which names an
identifier by its ``repr``, ``name::id``.
"""

__all__ = ["ConfigurationError", "DuplicateNameError", "SearchSpaceError"]

from fhy_core.error import register_error
from fhy_core.utils.override import override


@register_error
class SearchSpaceError(ValueError):
    """A search space, a choice or an alternative cannot be built."""


@register_error
class DuplicateNameError(SearchSpaceError):
    """A name occurs twice among the names a search space holds."""


@register_error
class ConfigurationError(SearchSpaceError):
    """A configuration's entries are not a valid point of its space.

    Attributes:
        problems: The text of every problem found, in the order the core
            checks them.

    """

    problems: tuple[str, ...]

    def __init__(self, message: str, problems: tuple[str, ...] = ()) -> None:
        super().__init__(message, tuple(problems))
        self.problems = tuple(problems)

    @override
    def __str__(self) -> str:
        return str(self.args[0])
