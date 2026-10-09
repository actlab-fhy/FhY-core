"""The exceptions of :mod:`fhy_core.search_space`.

Each message is the Rust core's text for the problem, which names an
identifier by its ``repr``, ``name::id``.
"""

__all__ = [
    "ConfigurationError",
    "DeadEndError",
    "DuplicateNameError",
    "InadmissibleAnswerError",
    "MeasurementError",
    "NotEnumerableError",
    "ReplayMismatchError",
    "SearchSpaceError",
    "StepDomainError",
    "TraceError",
]

from fhy_core.error import register_error
from fhy_core.utils.override import override


@register_error
class SearchSpaceError(ValueError):
    """The base of the search-space errors.

    Raised itself when a search space, a choice or an alternative cannot be
    built.
    """


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


@register_error
class StepDomainError(SearchSpaceError):
    """A step's domain cannot be built: empty, repeating, or out of order."""


@register_error
class TraceError(SearchSpaceError):
    """A run of a search stopped: a step cannot be asked, or its answer is refused.

    None of the search errors derives from anything a search catches as an
    infeasible program: they report a malformed search.
    """


@register_error
class InadmissibleAnswerError(TraceError):
    """An oracle answered outside the step's domain, or with a refused value."""


@register_error
class ReplayMismatchError(TraceError):
    """A recorded trace does not describe the run replaying it."""


@register_error
class NotEnumerableError(TraceError):
    """A step asks a variable whose domain is not finite."""


@register_error
class DeadEndError(TraceError):
    """A step has no admissible value, or its variable's param admits none."""


@register_error
class MeasurementError(SearchSpaceError):
    """An objective or a measurement cannot be built, or two cannot be compared."""
