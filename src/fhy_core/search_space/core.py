"""The search-space classes, backed by the Rust core (``fhy_core::search_space``).

Each class is a thin subclass of its ``fhy_core._rs`` class, which holds the
core value and the objects it was built from, and returns those objects.
Every class but :class:`ConfigurationKey` compares and hashes by identity;
the relations are spelled ``is_structurally_equivalent`` and
``is_alpha_equivalent``.

:class:`Variable` and :class:`Alternative` stay open to subclassing: a
subclass adds data of its own in its ``__init__``, after calling
``super().__init__``, is frozen at the end of its outermost ``__init__``, and
compares that data through the ``extension_*`` hooks, which the core calls
only for two parts of one kind. Its kind is its registered type id, and it
serializes as its type id and its ``serialize_data_to_dict()``.
"""

__all__ = [
    "Activity",
    "Alternative",
    "Choice",
    "Condition",
    "Configuration",
    "ConfigurationKey",
    "Forbidden",
    "Space",
    "Variable",
]

from collections.abc import Iterable
from typing import Any, Generic, TypeVar, final

from fhy_core import _rs
from fhy_core.diagnostic import Note
from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    Serializable,
    WrappedFamilySerializable,
    register_serializable,
)

# Imported for its effect: a configuration is checked under the default
# solver, which importing the solver module sets.
from fhy_core.symbolic import solver as _solver  # noqa: F401
from fhy_core.symbolic.param import Param
from fhy_core.term import AlphaRenaming
from fhy_core.traits import FrozenMixin
from fhy_core.traits.frozen import _FrozenAfterInit
from fhy_core.utils import StrEnum
from fhy_core.utils.override import override

_T = TypeVar("_T")


class Activity(StrEnum):
    """Whether a decision exists in a configuration."""

    ACTIVE = "active"
    """Its parent is active and its condition holds."""
    INACTIVE = "inactive"
    """Its choice chose another alternative or is inactive, or its condition fails."""
    PENDING = "pending"
    """It depends on a decision that is active but unassigned."""


@register_serializable(type_id="search_space.variable")
class Variable(_rs.Variable, _FrozenAfterInit, WrappedFamilySerializable, Generic[_T]):
    """A decision over the values of a param, named in its space.

    The name is how conditions, forbidden clauses and configurations refer
    to the variable; the param's own variable is a different identifier,
    which only the param's constraints name.

    Args:
        param: The param whose values the variable takes.
        name: The variable's name; a fresh ``Identifier("variable")`` when
            omitted.
        notes: Notes attached to the variable.

    Raises:
        TypeError: If an argument has the wrong type.

    """

    def __init__(
        self,
        param: Param[_T],
        name: Identifier | None = None,
        notes: Iterable[Note] = (),
    ) -> None:
        _rs.Variable._initialize(self, param, name, notes)

    def extension_is_structurally_equivalent(self, other: Any) -> bool:
        """Return whether this subclass's own data equals ``other``'s.

        The core compares the name, the param and the notes itself, and
        calls this only for ``other`` of the same kind. The default has no
        data of its own and answers ``True``.
        """
        raise NotImplementedError

    def extension_is_alpha_equivalent_under(
        self, other: Any, renaming: AlphaRenaming
    ) -> bool:
        """Return whether this subclass's own data corresponds to ``other``'s.

        ``renaming`` pairs every name of the space (or of the variable
        compared on its own); compare identifiers in the data only through
        ``renaming.are_identifiers_alpha_equivalent``. The default answers
        ``True``.
        """
        raise NotImplementedError

    @override
    def __setstate__(self, state: Any) -> None:
        """Restore a pickled subclass instance: its base fields, then its data."""
        raise NotImplementedError


@register_serializable(type_id="search_space.alternative")
class Alternative(_rs.Alternative, _FrozenAfterInit, WrappedFamilySerializable):
    """One option of a choice.

    Its variables and sub-choices exist only while it is chosen, and its
    name is the value its choice takes when it is chosen.

    Args:
        variables: The variables that exist while the alternative is chosen.
        choices: The choices that exist while the alternative is chosen.
        name: The alternative's name; a fresh ``Identifier("alternative")``
            when omitted.
        notes: Notes attached to the alternative.

    Raises:
        TypeError: If an argument has the wrong type.
        DuplicateNameError: If its name, its variables' names and the names
            its sub-choices hold repeat.

    """

    def __init__(
        self,
        variables: Iterable[Variable[Any]] = (),
        choices: Iterable["Choice"] = (),
        name: Identifier | None = None,
        notes: Iterable[Note] = (),
    ) -> None:
        _rs.Alternative._initialize(self, variables, choices, name, notes)

    def extension_bound_identifiers(self) -> Iterable[Identifier]:
        """Return the identifiers this subclass's own data binds, in a fixed order.

        They must be distinct, the same on every call, and as many for two
        alternatives that should correspond. The default binds none.
        """
        raise NotImplementedError

    def extension_is_structurally_equivalent(self, other: Any) -> bool:
        """Return whether this subclass's own data equals ``other``'s.

        The core compares the name, the variables, the sub-choices and the
        notes itself, and calls this only for ``other`` of the same kind.
        The default answers ``True``.
        """
        raise NotImplementedError

    def extension_is_alpha_equivalent_under(
        self, other: Any, renaming: AlphaRenaming
    ) -> bool:
        """Return whether this subclass's own data corresponds to ``other``'s.

        ``renaming`` already pairs the names of the space, the bound
        identifiers included. The default answers ``True``.
        """
        raise NotImplementedError

    @override
    def __setstate__(self, state: Any) -> None:
        """Restore a pickled subclass instance: its base fields, then its data."""
        raise NotImplementedError


@final
@register_serializable(type_id="search_space.choice")
class Choice(_rs.Choice, Serializable):
    """A named decision among one or more alternatives.

    Args:
        alternatives: The alternatives, in order.
        name: The choice's name; a fresh ``Identifier("choice")`` when
            omitted.
        notes: Notes attached to the choice.

    Raises:
        TypeError: If an argument has the wrong type.
        SearchSpaceError: If there is no alternative.
        DuplicateNameError: If the names the choice holds repeat.
        RecursionError: If choices nest deeper than the recursion limit.

    """

    __slots__ = ()


@final
class Condition(_rs.Condition):
    """When a decision is active: a constraint system that must hold.

    Args:
        target: The name of the decision the condition is on.
        when: A ``ConstraintSystem``, or the constraints of one.

    Raises:
        TypeError: If an argument has the wrong type.

    """

    __slots__ = ()


@final
class Forbidden(_rs.Forbidden):
    """A combination of values no configuration may take.

    Args:
        when: A ``ConstraintSystem``, or the constraints of one, that no
            configuration may satisfy.

    Raises:
        TypeError: If an argument has the wrong type.

    """

    __slots__ = ()


@final
@register_serializable(type_id="search_space.space")
class Space(_rs.Space, Serializable):
    """The whole search space: decisions, conditions and forbidden clauses.

    Every name a space holds is distinct: its own, every decision's, every
    alternative's and every identifier an alternative binds.

    Args:
        variables: The top-level variables.
        choices: The top-level choices.
        conditions: When decisions are active.
        forbidden: Combinations of values no configuration may take.
        name: The space's name; a fresh ``Identifier("space")`` when
            omitted.
        notes: Notes attached to the space.

    Raises:
        TypeError: If an argument has the wrong type.
        DuplicateNameError: If a name repeats.
        SearchSpaceError: If a condition or clause names something that is
            not a decision, or decisions depend on each other in a cycle.
        RecursionError: If choices nest deeper than the recursion limit.

    """

    __slots__ = ()


@final
@register_serializable(type_id="search_space.configuration")
class Configuration(_rs.Configuration, Serializable):
    """A point of a space: a value for some of its active decisions.

    A choice's value is the chosen alternative's name. A configuration is
    checked against its space when it is built, so every configuration is
    valid; :meth:`is_complete` says whether it assigns every active decision.

    Args:
        space: The space the configuration is a point of.
        entries: A mapping, or ``(name, value)`` pairs, of decisions to
            values.

    Raises:
        TypeError: If an argument has the wrong type.
        ConfigurationError: Carrying every problem of the entries.

    """

    __slots__ = ()


ConfigurationKey = _rs.ConfigurationKey
"""The identity of a configuration within its space; hashable, compared structurally."""


# The classes are registered, not derived: `FrozenMixin` carries an instance
# layout a Rust-backed class cannot share.
for _frozen_class in (
    Variable,
    Alternative,
    Choice,
    Condition,
    Forbidden,
    Space,
    Configuration,
    ConfigurationKey,
):
    FrozenMixin.register(_frozen_class)

Variable._register_public_class()
Alternative._register_public_class()
Choice._register_public_class()
Condition._register_public_class()
Forbidden._register_public_class()
Space._register_public_class()
Configuration._register_public_class()
