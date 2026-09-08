"""Tests for the `evaluate_with_bindings` contract on the set-constraint leaves.

`EquationConstraint`'s own bindings behavior (tri-state evaluation,
chained/swap substitution semantics, DEBUG-vs-WARNING logging) lives in
`test_equation_constraint.py`. This module covers `InSetConstraint` /
`NotInSetConstraint`'s bindings behavior -- which each implement directly
(there is no more base-class default keyed on a designated `variable`) --
plus cross-cutting bindings-API contracts shared by every kind.
"""

import logging
import re
from collections.abc import Iterator, Mapping
from decimal import Decimal
from typing import Any, cast

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import (
    ConstraintError,
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import (
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
    make_binary_expression,
)
from fhy_core.utils.override import override

from .conftest import ALL_KINDS, SET_KINDS, SerializableHashRaises, mock_identifier

_CONSTRAINT_LOGGER = "fhy_core.symbolic.constraint.core"

SET_KINDS_WITH_MEMBER_OUTCOMES = [
    pytest.param(
        InSetConstraint,
        ConstraintOutcome.SATISFIED,
        ConstraintOutcome.VIOLATED,
        id="in_set",
    ),
    pytest.param(
        NotInSetConstraint,
        ConstraintOutcome.VIOLATED,
        ConstraintOutcome.SATISFIED,
        id="not_in_set",
    ),
]
"""Each set-constraint kind with its decided outcome for a member and a non-member.

Membership polarity is inverted between the two kinds: a bound value that is
a member SATISFIES `InSetConstraint` but VIOLATES `NotInSetConstraint`, and
vice versa for a non-member.
"""

SET_KINDS_WITH_MEMBER_OUTCOME = [
    pytest.param(InSetConstraint, ConstraintOutcome.SATISFIED, id="in_set"),
    pytest.param(NotInSetConstraint, ConstraintOutcome.VIOLATED, id="not_in_set"),
]
"""Each set-constraint kind with its decided outcome for a bound member value."""


def _find_records(
    caplog: pytest.LogCaptureFixture, level: int
) -> list[logging.LogRecord]:
    """Return the constraint module's records emitted at exactly ``level``."""
    return [
        record
        for record in caplog.records
        if record.levelno == level and record.name == _CONSTRAINT_LOGGER
    ]


# =============================================================================
# Set-constraint bindings evaluation
# =============================================================================


@pytest.mark.parametrize(
    ("factory", "member_outcome", "non_member_outcome"),
    SET_KINDS_WITH_MEMBER_OUTCOMES,
)
def test_set_constraint_bindings_bound_value_decides_membership(
    factory: Any,
    member_outcome: ConstraintOutcome,
    non_member_outcome: ConstraintOutcome,
) -> None:
    """Test a bound value under the constrained variable decides membership."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2, 3})

    assert constraint.evaluate_with_bindings({x: 2}) is member_outcome
    assert constraint.evaluate_with_bindings({x: 4}) is non_member_outcome


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_bindings_missing_variable_is_undecided(
    factory: Any,
) -> None:
    """Test a bindings mapping missing the constrained variable is UNDECIDED."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = factory(x, {1, 2, 3})

    assert constraint.evaluate_with_bindings({y: 2}) is ConstraintOutcome.UNDECIDED


@pytest.mark.parametrize(("factory", "member_outcome"), SET_KINDS_WITH_MEMBER_OUTCOME)
def test_set_constraint_bindings_unwraps_literal_expression_binding(
    factory: Any,
    member_outcome: ConstraintOutcome,
) -> None:
    """Test a `LiteralExpression`-valued binding unwraps and decides membership."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2, 3})

    outcome = constraint.evaluate_with_bindings({x: LiteralExpression(2)})

    assert outcome is member_outcome


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_bindings_symbolic_expression_binding_is_undecided(
    factory: Any,
) -> None:
    """Test a non-literal `Expression` binding cannot be decided."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = factory(x, {1, 2, 3})

    outcome = constraint.evaluate_with_bindings({x: IdentifierExpression(y)})

    assert outcome is ConstraintOutcome.UNDECIDED


@pytest.mark.parametrize(("factory", "member_outcome"), SET_KINDS_WITH_MEMBER_OUTCOME)
def test_set_constraint_bindings_ignores_extraneous_keys(
    factory: Any,
    member_outcome: ConstraintOutcome,
) -> None:
    """Test identifiers the constraint does not reference are ignored."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = factory(x, {1, 2, 3})

    outcome = constraint.evaluate_with_bindings({x: 2, y: 999})

    assert outcome is member_outcome


def test_in_set_constraint_bindings_are_type_strict_for_bool_vs_int() -> None:
    """Test a bound `True` is distinct from a member `1` under type-strict rules."""
    x = mock_identifier("x", 0)
    constraint = InSetConstraint(x, {1})

    assert constraint.evaluate_with_bindings({x: True}) is ConstraintOutcome.VIOLATED


def test_not_in_set_constraint_bindings_are_type_strict_for_bool_vs_int() -> None:
    """Test a bound `True` is distinct from a forbidden member `1`."""
    x = mock_identifier("x", 0)
    constraint = NotInSetConstraint(x, {1})

    assert constraint.evaluate_with_bindings({x: True}) is ConstraintOutcome.SATISFIED


# =============================================================================
# `evaluate_with_bindings` reads the mapping once (snapshot)
# =============================================================================


class _SingleReadMapping(Mapping[Identifier, Any]):
    """A mapping that raises if any key is read from it more than once.

    Used to prove `evaluate_with_bindings` takes a single snapshot of the
    caller's mapping rather than performing a membership check and a
    separate lookup against the live mapping (which would read the same
    key twice).
    """

    def __init__(self, data: dict[Identifier, Any]) -> None:
        self._data = dict(data)
        self._read_counts: dict[Identifier, int] = {}

    @override
    def __iter__(self) -> Iterator[Identifier]:
        return iter(self._data)

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def __getitem__(self, key: Identifier) -> Any:
        self._read_counts[key] = self._read_counts.get(key, 0) + 1
        if self._read_counts[key] > 1:
            raise AssertionError(f"{key!r} was read more than once from the mapping")
        return self._data[key]


@pytest.mark.parametrize(("factory", "member_outcome"), SET_KINDS_WITH_MEMBER_OUTCOME)
def test_set_constraint_bindings_reads_the_mapping_only_once(
    factory: Any,
    member_outcome: ConstraintOutcome,
) -> None:
    """Test evaluation snapshots the mapping instead of re-reading it."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2, 3})
    bindings = _SingleReadMapping({x: 2})

    outcome = constraint.evaluate_with_bindings(bindings)

    assert outcome is member_outcome


# =============================================================================
# DEBUG-vs-WARNING logging split: reporting the cause of UNDECIDED
# =============================================================================


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_bindings_logs_debug_when_the_variable_is_unbound(
    caplog: pytest.LogCaptureFixture, factory: Any
) -> None:
    """Test a lookup miss on the constrained variable is reported at DEBUG.

    A set constraint's bindings evaluation never reports UNDECIDED for a
    reason other than "variable unbound" or "non-literal binding", so the
    record has to name the variable and the keys that were supplied.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = factory(x, {1, 2})

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        outcome = constraint.evaluate_with_bindings({y: 1})

    assert outcome is ConstraintOutcome.UNDECIDED
    debug_records = _find_records(caplog, logging.DEBUG)
    assert debug_records, "expected a DEBUG record naming the unbound variable"
    message = debug_records[0].getMessage()
    assert repr(x) in message
    assert repr(y) in message
    assert not _find_records(caplog, logging.WARNING), (
        "an unbound variable is an ordinary partial assignment, not an anomaly"
    )


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_bindings_logs_debug_for_a_non_literal_expression_binding(
    caplog: pytest.LogCaptureFixture, factory: Any
) -> None:
    """Test a symbolic binding value the leaf cannot consume is reported at DEBUG."""
    x = mock_identifier("x", 0)
    w = mock_identifier("w", 1)
    constraint = factory(x, {1, 2})
    binding = IdentifierExpression(w)

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        outcome = constraint.evaluate_with_bindings({x: binding})

    assert outcome is ConstraintOutcome.UNDECIDED
    debug_records = _find_records(caplog, logging.DEBUG)
    assert debug_records, "expected a DEBUG record naming the rejected expression"
    message = debug_records[0].getMessage()
    assert repr(x) in message
    assert repr(binding) in message
    assert not _find_records(caplog, logging.WARNING), (
        "a symbolic binding value is a structural limitation, not an anomaly"
    )


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_bindings_logs_nothing_when_decidable(
    caplog: pytest.LogCaptureFixture, factory: Any
) -> None:
    """Test a decided outcome under bindings emits no record at any level."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2})

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        constraint.evaluate_with_bindings({x: 1})

    assert not _find_records(caplog, logging.DEBUG)
    assert not _find_records(caplog, logging.WARNING)


# =============================================================================
# Binding-value construction boundary (shared across every kind)
# =============================================================================

OFF_UNION_BINDING_VALUES = [
    pytest.param(None, id="none"),
    pytest.param(Decimal("1"), id="decimal"),
    pytest.param([1, 2], id="list"),
    pytest.param(object(), id="object"),
]
"""Parametrize list of values outside ``Expression | LiteralType``.

``ConstraintBindings`` admits none of these; each reaches the API only
from code the type checker has not seen or has been silenced on, which is
exactly the case the runtime boundary has to answer for.
"""


@pytest.mark.parametrize("value", OFF_UNION_BINDING_VALUES)
def test_equation_constraint_bindings_rejects_a_value_outside_the_declared_union(
    value: Any,
) -> None:
    """Test a binding value outside `Expression | LiteralType` raises.

    ``ConstraintBindings`` declares ``Expression | LiteralType``. A value
    in neither arm cannot be lifted into the substitution environment, so
    the override rejects it at the boundary with a domain error naming the
    identifier, the value, and its type, rather than letting it reach the
    expression passes as an internal error.
    """
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 10))

    with pytest.raises(ConstraintError, match=re.escape(repr(x))) as exception_info:
        constraint.evaluate_with_bindings({x: value})

    message = str(exception_info.value)
    assert repr(value) in message
    assert type(value).__name__ in message


@pytest.mark.parametrize("factory", SET_KINDS)
@pytest.mark.parametrize("value", OFF_UNION_BINDING_VALUES)
def test_set_constraint_bindings_rejects_a_value_outside_the_declared_union(
    factory: Any, value: Any
) -> None:
    """Test a binding value that could never be a constraint member raises.

    `InSetConstraint`/`NotInSetConstraint` decide membership against the
    wider `ConstraintMember` union rather than `Expression |
    LiteralType`, but each of these four values falls outside both:
    none could ever be a member, so each raises the same domain error
    naming the identifier, the value, and its type, rather than being
    decided against (a hashable value) or crashing inside `hash` with no
    identifier named (an unhashable one).
    """
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2, 3})

    with pytest.raises(ConstraintError, match=re.escape(repr(x))) as exception_info:
        constraint.evaluate_with_bindings({x: value})

    message = str(exception_info.value)
    assert repr(value) in message
    assert type(value).__name__ in message


@pytest.mark.parametrize("factory", SET_KINDS)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param((1, 2), id="tuple"),
        pytest.param(frozenset({1, 2}), id="frozenset"),
    ],
)
def test_set_constraint_bindings_decides_a_container_shaped_off_union_value(
    factory: Any, value: Any
) -> None:
    """Test a tuple/frozenset binding value is decided, not rejected.

    `ConstraintBindings` declares `Expression | LiteralType`, but a set
    constraint decides membership against the wider `ConstraintMember`
    union, which also allows `tuple`/`frozenset`/`Serializable` values --
    exactly the shapes a container member can take. Rejecting these
    would break every set constraint whose members are containers.
    """
    x = mock_identifier("x", 0)
    constraint = factory(x, [value])

    outcome = constraint.evaluate_with_bindings({x: value})

    in_set = factory is InSetConstraint
    assert outcome is (
        ConstraintOutcome.SATISFIED if in_set else ConstraintOutcome.VIOLATED
    )


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_bindings_rejects_a_value_whose_hash_raises(
    factory: Any,
) -> None:
    """Test a value that lies about being hashable raises `ConstraintError`.

    `SerializableHashRaises` passes structural member-shape validation
    (it is `Serializable` and nominally `Hashable`), but calling `hash`
    on it raises `TypeError` -- the same failure mode member declaration
    already guards against after validation, now guarded here too so it
    never surfaces as an unattributed crash.
    """
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2, 3})
    bad = SerializableHashRaises()

    with pytest.raises(ConstraintError, match=re.escape(repr(x))):
        constraint.evaluate_with_bindings({x: bad})


def test_equation_constraint_bindings_names_the_offending_identifier_only() -> None:
    """Test the boundary error names the bad binding, not an acceptable one."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = make_binary_expression(
        BinaryOperation.LESS, make_binary_expression(BinaryOperation.ADD, x, y), 10
    )
    constraint = EquationConstraint(expression)

    with pytest.raises(ConstraintError, match=re.escape(repr(y))) as exception_info:
        constraint.evaluate_with_bindings({x: 3, y: None})  # type: ignore[dict-item]

    assert repr(x) not in str(exception_info.value)


# =============================================================================
# `is_satisfied_with_bindings` cross-cutting contract (every kind)
# =============================================================================

_KIND_SATISFYING_AND_VIOLATING_BINDINGS = [
    pytest.param("equation", True, False, id="equation"),
    pytest.param("in_set", 1, 99, id="in_set"),
    pytest.param("not_in_set", 99, 1, id="not_in_set"),
]
"""(kind id, a value satisfying that kind's default constraint, a violator)."""

_ALL_KINDS_BY_ID: dict[str, Any] = {
    cast(str, param.id): param.values[0] for param in ALL_KINDS
}


@pytest.mark.parametrize(
    "kind_id, satisfying_value, violating_value",
    _KIND_SATISFYING_AND_VIOLATING_BINDINGS,
)
def test_is_satisfied_with_bindings_matches_the_documented_true_false_split(
    kind_id: str, satisfying_value: Any, violating_value: Any
) -> None:
    """Test `is_satisfied_with_bindings` is `True` only for the satisfying value.

    Every kind's default constraint, one concrete value that satisfies it,
    and one that violates it -- the expected booleans are literal, not
    derived from `evaluate_with_bindings` itself.
    """
    x = mock_identifier("x", 0)
    constraint = _ALL_KINDS_BY_ID[kind_id](x)

    assert constraint.is_satisfied_with_bindings({x: satisfying_value}) is True
    assert constraint.is_satisfied_with_bindings({x: violating_value}) is False


@pytest.mark.parametrize("factory", ALL_KINDS)
def test_is_satisfied_with_bindings_folds_undecided_to_false(
    factory: Any,
) -> None:
    """Test an UNDECIDED outcome maps `is_satisfied_with_bindings` to `False`."""
    x = mock_identifier("x", 0)
    constraint = factory(x)

    assert constraint.evaluate_with_bindings({}) is ConstraintOutcome.UNDECIDED
    assert constraint.is_satisfied_with_bindings({}) is False
