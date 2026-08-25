"""Errors raised by constraint construction, validation, and solver-backed checks.

``ConstraintError`` is the general-purpose domain error for constructing,
validating, and converting a constraint or a ``ConstraintSystem``.
``MissingSymbolTypeError`` covers the more specific precondition that the
solver-backed entry points on ``ConstraintSystem`` impose on their
``symbol_types`` argument.
"""

__all__ = [
    "ConstraintError",
    "MissingSymbolTypeError",
]

from fhy_core.error import register_error


@register_error
class ConstraintError(ValueError):
    """Domain error for constraint construction, validation, and conversion."""


@register_error
class MissingSymbolTypeError(ValueError):
    """Raised when ``symbol_types`` lacks an entry for a free identifier being lowered.

    ``ConstraintSystem.check_satisfiability`` and
    ``check_satisfiability_with_bindings`` require a Z3 sort for every free
    identifier of the expression they lower. A missing entry is a caller
    precondition violation, not a dictionary lookup miss, so this is a
    ``ValueError`` rather than a ``KeyError``: the missing-identifier
    message must render cleanly in a traceback, and a bare ``except
    KeyError`` elsewhere in a caller's code must not silently swallow it.
    """
