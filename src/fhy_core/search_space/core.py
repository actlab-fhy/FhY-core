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
    "Cardinality",
    "CardinalityKind",
    "Choice",
    "ChoiceDomain",
    "Condition",
    "Configuration",
    "ConfigurationKey",
    "Direction",
    "ExhaustiveOracle",
    "Forbidden",
    "GuidedOracle",
    "Measurement",
    "MeasurementStatus",
    "Measurer",
    "Objective",
    "OrderDomain",
    "PendingStep",
    "RandomOracle",
    "Recorder",
    "ReplayOracle",
    "Rng",
    "SearchOracle",
    "Space",
    "StridedDomain",
    "StridedRun",
    "Trace",
    "TraceKey",
    "TraceStep",
    "Variable",
    "non_dominated",
]

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, Protocol, TypeVar, final, runtime_checkable

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
from fhy_core.utils.override import override

_T = TypeVar("_T")
_S_contra = TypeVar("_S_contra", contravariant=True)


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
        del other
        return True

    def extension_is_alpha_equivalent_under(
        self, other: Any, renaming: AlphaRenaming
    ) -> bool:
        """Return whether this subclass's own data corresponds to ``other``'s.

        ``renaming`` pairs every name of the space (or of the variable
        compared on its own); compare identifiers in the data only through
        ``renaming.are_identifiers_alpha_equivalent``. The default answers
        ``True``.
        """
        del other, renaming
        return True

    def extension_search_domain(
        self,
    ) -> "ChoiceDomain | OrderDomain | StridedDomain | None":
        """Return the domain a step over this variable offers, or ``None``.

        ``None``, the default, derives it from the param: a categorical or
        ordinal param's values, a permutation param's orderings, or an
        integer param bounded at both ends. A subclass whose param has a
        custom domain returns one holding exactly the values that domain
        admits, in an order that never changes.
        """
        return None

    @override
    def __setstate__(self, state: Any) -> None:
        """Restore a pickled subclass instance: its base fields, then its data."""
        base, data = state
        _rs.Variable._initialize(self, *base)
        for name, attribute in (data or {}).items():
            object.__setattr__(self, name, attribute)


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
        return ()

    def extension_is_structurally_equivalent(self, other: Any) -> bool:
        """Return whether this subclass's own data equals ``other``'s.

        The core compares the name, the variables, the sub-choices and the
        notes itself, and calls this only for ``other`` of the same kind.
        The default answers ``True``.
        """
        del other
        return True

    def extension_is_alpha_equivalent_under(
        self, other: Any, renaming: AlphaRenaming
    ) -> bool:
        """Return whether this subclass's own data corresponds to ``other``'s.

        ``renaming`` already pairs the names of the space, the bound
        identifiers included. The default answers ``True``.
        """
        del other, renaming
        return True

    @override
    def __setstate__(self, state: Any) -> None:
        """Restore a pickled subclass instance: its base fields, then its data."""
        base, data = state
        _rs.Alternative._initialize(self, *base)
        for name, attribute in (data or {}).items():
            object.__setattr__(self, name, attribute)


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
        SearchSpaceError: If there is no alternative, or the choice nests
            choices more than 16 levels deep, itself included.
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
        SearchSpaceError: If a condition or clause names no decision or
            something that is not a decision, or a set constraint over a
            choice holds a member that is none of its alternatives' names,
            or decisions depend on each other in a cycle.
        RecursionError: If choices nest deeper than the recursion limit.

    """

    __slots__ = ()

    def cardinality(self, *, budget: int = 100_000) -> "Cardinality":
        """Return how many complete configurations the space has.

        Args:
            budget: The most configurations checked in each part of the
                space that has no closed-form count.

        Returns:
            The count: exact, a lower bound when the budget ran out,
            unbounded, or unknown.

        """
        kind, count, decision = self._cardinality(budget=budget)
        return Cardinality(CardinalityKind(kind), count, decision)


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

Rng = _rs.Rng
"""A seeded generator, SplitMix64, whose stream is the same from Python and Rust."""

ChoiceDomain = _rs.ChoiceDomain
"""A step's domain of one or more distinct values, the objects given."""

OrderDomain = _rs.OrderDomain
"""A step's domain of the orderings of one or more distinct elements."""

StridedRun = _rs.StridedRun
"""A run of integers below a stop by a stride."""

StridedDomain = _rs.StridedDomain
"""A step's domain of the integers of disjoint, ascending strided runs."""

PendingStep = _rs.PendingStep
"""A step as an oracle is asked it."""

RandomOracle = _rs.RandomOracle
"""Answers each step uniformly among its admissible coordinates."""

ReplayOracle = _rs.ReplayOracle
"""Answers the steps of a recorded trace back, refusing a run that leaves it."""

GuidedOracle = _rs.GuidedOracle
"""Answers a recorded trace back where it fits, and asks a fallback otherwise."""

ExhaustiveOracle = _rs.ExhaustiveOracle
"""Takes every path of a deterministic stream once, over successive runs."""

Recorder = _rs.Recorder
"""One run of a search: asks each step of an oracle, checks and records it."""

TraceStep = _rs.TraceStep
"""One recorded step: its kind, subject, domain signature and answer."""

TraceKey = _rs.TraceKey
"""The identity of a trace without its subjects; hashable, compared structurally."""


@runtime_checkable
class SearchOracle(Protocol):
    """Answers the steps of a run, one at a time.

    Any object with a ``decide`` method is an oracle; no base class is
    needed. An exception it raises stops the run and propagates as itself.
    """

    def decide(self, step: PendingStep) -> int | tuple[int, ...]:
        """Return the answer to ``step``: a coordinate of its domain."""
        ...


@final
@register_serializable(type_id="search_space.trace")
class Trace(_rs.Trace, Serializable):
    """The steps of one run, in the order asked.

    ``==`` and ``hash`` are structural. Two traces recorded over different
    modules differ in their subjects; compare their ``coordinates`` to ask
    whether they took the same answers.

    Args:
        steps: The ``TraceStep``s, in order.

    Raises:
        TypeError: If a step is no ``TraceStep``.

    """

    __slots__ = ()


class CardinalityKind(StrEnum):
    """What a :class:`Cardinality` says of a space's count."""

    EXACT = "exact"
    """The count is exact."""
    AT_LEAST = "at_least"
    """The count is at least this: the budget ran out."""
    UNBOUNDED = "unbounded"
    """A configuration activates a variable over unbounded integers or reals."""
    UNKNOWN = "unknown"
    """A configuration activates a variable whose domain cannot be enumerated."""


@dataclass(frozen=True)
class Cardinality:
    """How many complete configurations a space has.

    Attributes:
        kind: Whether the count is exact, a lower bound, unbounded or unknown.
        count: The count, or its lower bound; ``None`` when unbounded or
            unknown.
        decision: The variable that makes it unbounded or unknown, else
            ``None``.

    """

    kind: CardinalityKind
    count: int | None
    decision: Identifier | None


class Direction(StrEnum):
    """Which way an objective's values are better."""

    MINIMIZE = "minimize"
    """Lower is better: a cost."""
    MAXIMIZE = "maximize"
    """Higher is better: a benefit."""
    REPORT = "report"
    """Recorded, never compared: a diagnostic."""


class MeasurementStatus(StrEnum):
    """How a measurement of a configuration went."""

    OK = "ok"
    """The configuration was measured: a value per objective."""
    INFEASIBLE = "infeasible"
    """The configuration cannot be realized: data about the space."""
    FAILED = "failed"
    """The measurement was attempted and broke: a fault of the measuring."""
    TIMEOUT = "timeout"
    """The measurement ran out of time."""


@final
@register_serializable(type_id="search_space.objective")
class Objective(_rs.Objective, Serializable):
    """A named quantity a search measures, and which way it is better.

    Its name is a string, stable across processes. ``==`` and ``hash`` are
    structural: two objectives are equal when their names and directions
    are.

    Args:
        name: The objective's name.
        direction: Which way its values are better: a :class:`Direction` or
            its value.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``direction`` names no direction.
        MeasurementError: If ``name`` is empty.

    """

    __slots__ = ()


@final
@register_serializable(type_id="search_space.measurement")
class Measurement(_rs.Measurement, Serializable):
    """The record of one measured configuration or run.

    It holds the :class:`ConfigurationKey` of the configuration, or the
    :class:`TraceKey` of the run, it measured, a
    :class:`MeasurementStatus` and, when the status is ``OK``, a finite
    value per objective, and notes. Build one with :meth:`ok`,
    :meth:`infeasible`, :meth:`failed` or :meth:`timeout`. ``==`` and
    ``hash`` are identity.
    """

    __slots__ = ()


@runtime_checkable
class Measurer(Protocol[_S_contra]):
    """Measures subjects, such as a lowered program, as configurations or runs.

    A subject that cannot be measured is a :class:`Measurement` with a
    failing status; an exception is a fault of the measurer, which stops
    the search. A successful measurement holds a value for each of
    :attr:`objectives`, and only those.
    """

    @property
    def objectives(self) -> Sequence[Objective]:
        """The objectives every successful measurement holds a value for."""
        ...

    def measure(
        self, key: ConfigurationKey | TraceKey, subject: _S_contra
    ) -> Measurement:
        """Return the measurement of ``subject``, which realizes ``key``."""
        ...


def non_dominated(measurements: Iterable[Measurement]) -> list[Measurement]:
    """Return the measurements no other successful one dominates: the Pareto front.

    Measurements that did not succeed are left out; the rest keep their
    order. Two measurements with equal values are both kept unless a third
    dominates them.

    Args:
        measurements: The measurements to filter.

    Returns:
        The successful measurements no other successful one dominates.

    Raises:
        TypeError: If an element is no :class:`Measurement`.
        MeasurementError: If two successful measurements are over different
            objectives.

    """
    return _rs.non_dominated(measurements)


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
    TraceKey,
    Trace,
    Objective,
    Measurement,
):
    FrozenMixin.register(_frozen_class)

Variable._register_public_class()
Alternative._register_public_class()
Choice._register_public_class()
Condition._register_public_class()
Forbidden._register_public_class()
Space._register_public_class()
Configuration._register_public_class()
Trace._register_public_class()
Objective._register_public_class()
Measurement._register_public_class()
