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

A search runs as steps: a :class:`Recorder` asks each one of a
:class:`SearchOracle`, over a domain (:class:`ChoiceDomain`,
:class:`OrderDomain`, :class:`StridedDomain`), checks the answer, a
coordinate in the domain, and records it in a :class:`Trace`.
:class:`RandomOracle` draws from an :class:`Rng`, :class:`ReplayOracle`
answers a recorded trace back, :class:`GuidedOracle` answers it back where
it fits and asks another oracle where it does not, and
:class:`ExhaustiveOracle` takes every path once over successive runs; a
:class:`Space` also samples, completes, replays, enumerates, counts
(:class:`Cardinality`), mutates and crosses over on its own.

A :class:`Measurer` measures a realization of a configuration, or of one
run, and returns a :class:`Measurement`: the configuration's key or the
run's :class:`TraceKey`, a :class:`MeasurementStatus` and, when it
succeeded, a value per :class:`Objective`, whose :class:`Direction` says
which way is better. :func:`non_dominated` keeps the measurements no other
one dominates.
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
    "ConfigurationError",
    "ConfigurationKey",
    "DeadEndError",
    "Direction",
    "DuplicateNameError",
    "ExhaustiveOracle",
    "Forbidden",
    "GuidedOracle",
    "InadmissibleAnswerError",
    "Measurement",
    "MeasurementError",
    "MeasurementStatus",
    "Measurer",
    "NotEnumerableError",
    "Objective",
    "OrderDomain",
    "PendingStep",
    "RandomOracle",
    "Recorder",
    "ReplayMismatchError",
    "ReplayOracle",
    "Rng",
    "SearchOracle",
    "SearchSpaceError",
    "Space",
    "StepDomainError",
    "StridedDomain",
    "StridedRun",
    "Trace",
    "TraceError",
    "TraceKey",
    "TraceStep",
    "Variable",
    "non_dominated",
]

from .core import (
    Activity,
    Alternative,
    Cardinality,
    CardinalityKind,
    Choice,
    ChoiceDomain,
    Condition,
    Configuration,
    ConfigurationKey,
    Direction,
    ExhaustiveOracle,
    Forbidden,
    GuidedOracle,
    Measurement,
    MeasurementStatus,
    Measurer,
    Objective,
    OrderDomain,
    PendingStep,
    RandomOracle,
    Recorder,
    ReplayOracle,
    Rng,
    SearchOracle,
    Space,
    StridedDomain,
    StridedRun,
    Trace,
    TraceKey,
    TraceStep,
    Variable,
    non_dominated,
)
from .errors import (
    ConfigurationError,
    DeadEndError,
    DuplicateNameError,
    InadmissibleAnswerError,
    MeasurementError,
    NotEnumerableError,
    ReplayMismatchError,
    SearchSpaceError,
    StepDomainError,
    TraceError,
)
