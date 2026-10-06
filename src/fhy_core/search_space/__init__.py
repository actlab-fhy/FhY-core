"""Search spaces: the decisions a compiler search makes and the points it visits.

A :class:`Variable` is a decision over a param's values. An
:class:`Alternative` is one option of a :class:`Choice`, holding the
variables and sub-choices that exist only while it is chosen. A
:class:`Space` holds top-level variables and choices, conditions
(:class:`Condition`) on when a decision is active, and forbidden
combinations of values (:class:`Forbidden`). A :class:`Configuration` is a
point of a space, checked against it when it is built; its
:class:`ConfigurationKey` identifies it within its space, the same for
configurations of spaces that differ only in their names.

The classes run on the Rust core, ``fhy_core::search_space``. Downstream
packages extend :class:`Variable` and :class:`Alternative` with data of
their own, as Python subclasses or as Rust kinds registered with the
extension.
"""

__all__ = [
    "Activity",
    "Alternative",
    "Choice",
    "Condition",
    "Configuration",
    "ConfigurationError",
    "ConfigurationKey",
    "DuplicateNameError",
    "Forbidden",
    "SearchSpaceError",
    "Space",
    "Variable",
]

from .core import (
    Activity,
    Alternative,
    Choice,
    Condition,
    Configuration,
    ConfigurationKey,
    Forbidden,
    Space,
    Variable,
)
from .errors import ConfigurationError, DuplicateNameError, SearchSpaceError
