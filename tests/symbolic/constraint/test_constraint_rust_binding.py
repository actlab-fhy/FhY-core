"""Interface tests of the Rust-backed constraints (S13a of ``python-switch.md``).

The behavior of the constraints is specified by the Rust tests of
``fhy_core::constraint``; these tests cover what the binding adds over the
core: the class structure, the objects kept, the readers of Python values,
opaque members and their exceptions, the default solver, the log records,
the errors, pickling and payloads, and threads.
"""

import copy
import logging
import pickle
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import IntEnum
from typing import Any

import pytest

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.serialization import Serializable, register_serializable
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintError,
    ConstraintOutcome,
    ConstraintSystem,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
    SymbolicPredicate,
    create_constraint_system,
    does_member_lift_to_expression,
)
from fhy_core.symbolic.expression import (
    BinaryExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    NonBooleanLogicalOperandError,
)
from fhy_core.symbolic.solver import (
    SatResult,
    Simplifier,
    Solver,
    get_default_solver,
    set_default_solver,
)
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.term import excluded_from_equivalence
from fhy_core.traits import FrozenMixin, FrozenMutationError
from fhy_core.utils.override import override

from ..conftest import RecordingSmtSolver
from .conftest import SET_KINDS, mock_identifier

_CORE_LOGGER = "fhy_core.symbolic.constraint.core"


class _Level(IntEnum):
    LOW = 1
    HIGH = 3


class _Measure(float):
    """A float subclass, as a NumPy ``float64`` is one."""


class _Name(str):
    """A str subclass."""

    __slots__ = ()


@register_serializable(type_id="tests.constraint_binding.token")
class _Token(Serializable):
    """A ``Serializable`` member equal and hashed by its value."""

    def __init__(self, value: int) -> None:
        self.value = value

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Token) and self.value == other.value

    @override
    def __hash__(self) -> int:
        return hash(self.value)

    @override
    def __repr__(self) -> str:
        return f"_Token({self.value})"

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "_Token":
        return cls(int(data["value"]))


class _RaisingEquality(Exception):
    """Raised by ``_Touchy.__eq__``."""


@register_serializable(type_id="tests.constraint_binding.touchy")
class _Touchy(Serializable):
    """A ``Serializable`` member whose ``==`` raises what it is given."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    @override
    def __eq__(self, other: object) -> bool:
        raise self.error

    @override
    def __hash__(self) -> int:
        return 7

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "_Touchy":
        return cls(_RaisingEquality())


@register_serializable(type_id="tests.constraint_binding.unhashable")
class _Unhashable(_Token):
    """A ``Serializable``, nominally ``Hashable``, whose hash raises."""

    @override
    def __hash__(self) -> int:
        raise TypeError("no hash")


class _RecordingSimplifier(Simplifier):
    """A Python simplifier returning a fixed result and recording its inputs."""

    def __init__(self, result: Expression) -> None:
        super().__init__()
        self.result = result
        self.inputs: list[Expression] = []

    @override
    def simplify(self, expression: Expression) -> Expression:
        self.inputs.append(expression)
        return self.result


class _SingleReadMapping(Mapping[Identifier, Any]):
    """A mapping that is not a dict and refuses to read a key twice."""

    def __init__(self, data: dict[Identifier, Any]) -> None:
        self._data = data
        self._reads: set[Identifier] = set()

    @override
    def __getitem__(self, key: Identifier) -> Any:
        assert key not in self._reads, f"{key!r} read twice"
        self._reads.add(key)
        return self._data[key]

    @override
    def __iter__(self) -> Iterator[Identifier]:
        return iter(self._data)

    @override
    def __len__(self) -> int:
        return len(self._data)


def _bind(identifier: Identifier, value: Any) -> dict[Identifier, Any]:
    """Return the bindings of `identifier` to a member-shaped `value`.

    A set constraint decides any member-shaped value, which is wider than
    ``ConstraintBindings`` declares.
    """
    return {identifier: value}


@pytest.fixture
def x() -> Identifier:
    """Return the constrained variable."""
    return mock_identifier("x", 0)


@pytest.fixture
def restore_default_solver() -> Iterator[None]:
    """Restore the default solver after a test replaces it."""
    original = get_default_solver()
    yield
    set_default_solver(original)


# =============================================================================
# Class structure
# =============================================================================


@pytest.mark.parametrize(
    ("public", "native"),
    [
        (EquationConstraint, _rs.EquationConstraint),
        (InSetConstraint, _rs.InSetConstraint),
        (NotInSetConstraint, _rs.NotInSetConstraint),
    ],
)
def test_each_leaf_is_a_thin_subclass_registered_as_a_constraint(
    public: type[Any], native: type[Any]
) -> None:
    """Test each public leaf extends its Rust class and is a virtual Constraint."""
    assert issubclass(public, native)
    assert issubclass(public, Constraint)
    assert issubclass(public, FrozenMixin)
    assert Constraint not in public.__mro__


def test_leaves_are_constraints_symbolic_predicates_and_frozen(x: Identifier) -> None:
    """Test instances pass the checks param and the system make."""
    for constraint in (
        EquationConstraint(IdentifierExpression(x) > 0),
        InSetConstraint(x, [1]),
        NotInSetConstraint(x, [1]),
    ):
        assert isinstance(constraint, Constraint)
        assert isinstance(constraint, SymbolicPredicate)
        assert isinstance(constraint, FrozenMixin)
        assert constraint.is_frozen
        constraint.freeze()
        constraint.assert_frozen()


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_refuses_every_mutation_naming_the_attribute(
    factory: Any, x: Identifier
) -> None:
    """Test setting or deleting any attribute raises ``FrozenMutationError``."""
    constraint = factory(x, [1])

    with pytest.raises(FrozenMutationError, match='"variable"'):
        constraint.variable = x
    with pytest.raises(FrozenMutationError, match='"values"'):
        del constraint.values
    with pytest.raises(FrozenMutationError, match='"anything"'):
        constraint.anything = 1


def test_equality_and_hash_are_identity(x: Identifier) -> None:
    """Test two equal constraints are distinct keys, as ``eq=False`` made them."""
    left = InSetConstraint(x, [1, 2])
    right = InSetConstraint(x, [1, 2])

    assert left == left  # noqa: PLR0124
    assert left != right
    assert len({left, right}) == 2
    assert left.is_structurally_equivalent(right)


def test_equivalence_needs_the_same_class(x: Identifier) -> None:
    """Test constraints of another class, or other values, are never equivalent."""

    class _Subclass(InSetConstraint):
        __slots__ = ()

    constraint = InSetConstraint(x, [1])

    assert not constraint.is_structurally_equivalent(_Subclass(x, [1]))
    assert not constraint.is_structurally_equivalent(NotInSetConstraint(x, [1]))
    assert not constraint.is_structurally_equivalent("x in {1}")
    assert not EquationConstraint(IdentifierExpression(x) > 0).is_alpha_equivalent(None)


def test_alpha_equivalence_refuses_a_value_that_is_no_renaming(x: Identifier) -> None:
    """Test ``is_alpha_equivalent_under`` raises ``TypeError`` for a non-renaming."""
    constraint = InSetConstraint(x, [1])

    with pytest.raises(TypeError, match="AlphaRenaming"):
        constraint.is_alpha_equivalent_under(constraint, {})  # type: ignore[arg-type]


# =============================================================================
# The objects kept
# =============================================================================


def test_constraints_return_the_objects_they_were_given(x: Identifier) -> None:
    """Test the expression, the variable and opaque members come back as themselves."""
    expression = IdentifierExpression(x) > 0
    token = _Token(4)
    equation = EquationConstraint(expression)
    in_set = InSetConstraint(x, [token, 1])

    assert equation.expression is expression
    assert equation.convert_to_expression() is expression
    assert in_set.variable is x
    assert in_set.values[-1] is token
    assert in_set.members is in_set.values
    assert in_set.get_free_identifiers() == frozenset({x})


def test_set_conversion_references_the_variable_object(x: Identifier) -> None:
    """Test the comparisons of ``convert_to_expression`` hold the variable given."""
    expression = InSetConstraint(x, [2]).convert_to_expression()

    assert isinstance(expression, BinaryExpression)
    assert isinstance(expression.left, IdentifierExpression)
    assert expression.left.identifier is x
    assert expression.right == LiteralExpression(2)


# =============================================================================
# Reading members and bound values
# =============================================================================


@pytest.mark.parametrize(
    ("given", "stored", "stored_type"),
    [
        pytest.param(_Level.HIGH, 3, int, id="int_enum"),
        pytest.param(_Measure(1.5), 1.5, float, id="float_subclass"),
        pytest.param(_Name("a"), "a", str, id="str_subclass"),
        pytest.param(-0.0, 0.0, float, id="negative_zero"),
        pytest.param(10**30, 10**30, int, id="big_int"),
    ],
)
def test_member_is_stored_as_the_exact_value_it_denotes(
    x: Identifier, given: Any, stored: Any, stored_type: type[Any]
) -> None:
    """Test a subclass instance is stored as the exact builtin value."""
    (member,) = InSetConstraint(x, [given]).values

    assert type(member) is stored_type
    assert member == stored
    assert str(member) != "-0.0"


def test_container_members_are_plain_tuples_and_frozensets(x: Identifier) -> None:
    """Test nested containers come back as plain tuples and frozensets."""
    (member,) = InSetConstraint(x, [(1, frozenset({_Level.LOW}), "a")]).values

    assert member == (1, frozenset({1}), "a")
    assert type(member) is tuple
    assert type(member[1]) is frozenset


def test_members_are_decided_type_strictly_against_subclass_bindings(
    x: Identifier,
) -> None:
    """Test bound subclass instances are the exact values they denote."""
    constraint = InSetConstraint(x, [3, 1.5])

    assert constraint.evaluate_with_bindings({x: _Level.HIGH}) is (
        ConstraintOutcome.SATISFIED
    )
    assert constraint.evaluate_with_bindings({x: _Measure(1.5)}) is (
        ConstraintOutcome.SATISFIED
    )
    assert constraint.evaluate_with_bindings({x: True}) is ConstraintOutcome.VIOLATED


@pytest.mark.usefixtures("restore_default_solver")
def test_bindings_may_be_any_mapping_read_once(x: Identifier) -> None:
    """Test a mapping that is not a dict is read, each key once."""
    y = mock_identifier("y", 1)
    simplifier = _RecordingSimplifier(LiteralExpression(True))
    set_default_solver(Solver(simplifier=simplifier))
    equation = EquationConstraint(IdentifierExpression(x) < 5)
    in_set = InSetConstraint(x, [1])

    assert in_set.evaluate_with_bindings(_SingleReadMapping({x: 1})) is (
        ConstraintOutcome.SATISFIED
    )
    assert equation.evaluate_with_bindings(_SingleReadMapping({y: object(), x: 1})) is (
        ConstraintOutcome.SATISFIED
    )
    assert simplifier.inputs == [LiteralExpression(1) < 5]


def test_keys_that_are_no_identifiers_are_ignored(x: Identifier) -> None:
    """Test a binding under a key of another type is never read."""
    constraint = InSetConstraint(x, [1])

    assert (
        constraint.evaluate_with_bindings(
            {"x": 1, x: 1}  # type: ignore[dict-item]
        )
        is ConstraintOutcome.SATISFIED
    )


def test_bindings_that_are_no_mapping_are_refused(x: Identifier) -> None:
    """Test an equation refuses bindings that are no mapping with ``TypeError``."""
    constraint = EquationConstraint(IdentifierExpression(x) > 0)

    with pytest.raises(TypeError, match="mapping"):
        constraint.evaluate_with_bindings([(x, 1)])  # type: ignore[arg-type]


# =============================================================================
# Opaque members
# =============================================================================


def test_opaque_members_compare_with_python_equality(x: Identifier) -> None:
    """Test a ``Serializable`` member is found by ``==`` and its type."""
    constraint = InSetConstraint(x, [_Token(1), _Token(2), _Token(1)])

    assert len(constraint.values) == 2
    assert constraint.evaluate_with_bindings(_bind(x, _Token(2))) is (
        ConstraintOutcome.SATISFIED
    )
    assert constraint.evaluate_with_bindings(_bind(x, _Token(3))) is (
        ConstraintOutcome.VIOLATED
    )
    assert InSetConstraint(x, [_Token(1)]).is_structurally_equivalent(
        InSetConstraint(x, [_Token(1)])
    )


def test_an_exception_from_an_opaque_equality_propagates_as_itself(
    x: Identifier,
) -> None:
    """Test the exception a member's ``==`` raises reaches the caller unchanged."""
    error = _RaisingEquality("compared")
    constraint = InSetConstraint(x, [_Touchy(error)])

    with pytest.raises(_RaisingEquality) as raised:
        constraint.evaluate_with_bindings(_bind(x, _Touchy(_RaisingEquality())))
    assert raised.value is error
    with pytest.raises(_RaisingEquality):
        constraint.is_structurally_equivalent(
            InSetConstraint(x, [_Touchy(_RaisingEquality())])
        )


def test_a_keyboard_interrupt_from_an_opaque_equality_passes_through(
    x: Identifier,
) -> None:
    """Test a ``KeyboardInterrupt`` raised by ``==`` is not wrapped."""
    constraint = InSetConstraint(x, [_Touchy(KeyboardInterrupt())])

    with pytest.raises(KeyboardInterrupt):
        constraint.evaluate_with_bindings(_bind(x, _Touchy(KeyboardInterrupt())))


_TAGGED_CALLS: list[str] = []
"""The tags of the ``_Tagged`` members whose ``==`` ran, in order."""


@register_serializable(type_id="tests.constraint_binding.tagged")
class _Tagged(Serializable):
    """A ``Serializable`` member that records its ``==`` and raises its error."""

    def __init__(self, tag: str, error: BaseException | None) -> None:
        self.tag = tag
        self.error = error

    @override
    def __eq__(self, other: object) -> bool:
        _TAGGED_CALLS.append(self.tag)
        if self.error is not None:
            raise self.error
        return NotImplemented

    @override
    def __hash__(self) -> int:
        return hash(self.tag)

    @override
    def __repr__(self) -> str:
        return f"_Tagged({self.tag})"

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {"tag": self.tag}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "_Tagged":
        return cls(str(data["tag"]), None)


def test_no_member_equality_runs_after_the_first_exception(x: Identifier) -> None:
    """Test the first exception of a call stops the comparisons that follow it."""
    first = ValueError("first")
    constraint = InSetConstraint(
        x, [_Tagged("a", first), _Tagged("b", KeyboardInterrupt())]
    )
    _TAGGED_CALLS.clear()

    with pytest.raises(ValueError, match="first") as raised:
        constraint.evaluate_with_bindings(_bind(x, _Tagged("v", None)))

    assert raised.value is first
    assert _TAGGED_CALLS == ["a"]


def test_a_keyboard_interrupt_from_the_first_member_is_raised_alone(
    x: Identifier,
) -> None:
    """Test a first ``KeyboardInterrupt`` is raised, and nothing runs after it."""
    constraint = InSetConstraint(
        x, [_Tagged("a", KeyboardInterrupt()), _Tagged("b", ValueError("later"))]
    )
    _TAGGED_CALLS.clear()

    with pytest.raises(KeyboardInterrupt):
        constraint.evaluate_with_bindings(_bind(x, _Tagged("v", None)))

    assert _TAGGED_CALLS == ["a"]


# =============================================================================
# Evaluation through the default solver
# =============================================================================


@pytest.mark.usefixtures("restore_default_solver")
def test_equation_asks_the_default_solver_with_the_bindings_substituted(
    x: Identifier,
) -> None:
    """Test a plugged simplifier decides an equation and sees the substitution."""
    simplifier = _RecordingSimplifier(LiteralExpression(False))
    set_default_solver(Solver(simplifier=simplifier))
    constraint = EquationConstraint(IdentifierExpression(x) >= 0)

    outcome = constraint.evaluate_with_bindings({x: 3})

    assert outcome is ConstraintOutcome.VIOLATED
    assert simplifier.inputs == [LiteralExpression(3) >= 0]


@pytest.mark.usefixtures("restore_default_solver")
def test_a_non_boolean_result_raises_with_the_core_text(x: Identifier) -> None:
    """Test a literal result that is not a ``bool`` raises."""
    set_default_solver(Solver(simplifier=_RecordingSimplifier(LiteralExpression(1))))
    constraint = EquationConstraint(IdentifierExpression(x) >= 0)

    with pytest.raises(NonBooleanLogicalOperandError, match="not a boolean"):
        constraint.evaluate_with_bindings({x: 3})


@pytest.mark.usefixtures("restore_default_solver")
def test_a_set_constraint_never_asks_the_solver(x: Identifier) -> None:
    """Test membership is decided without a solver."""
    set_default_solver(Solver())

    assert InSetConstraint(x, [1]).is_satisfied_with_bindings({x: 1}) is True


# =============================================================================
# Log records
# =============================================================================


def test_unbound_variable_debug_record_lists_the_supplied_keys(
    x: Identifier, caplog: pytest.LogCaptureFixture
) -> None:
    """Test the DEBUG record of an unbound variable names the bound keys."""
    y = mock_identifier("y", 1)
    with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
        InSetConstraint(x, [1]).evaluate_with_bindings({y: 1})

    (record,) = [r for r in caplog.records if r.name == _CORE_LOGGER]
    assert record.levelno == logging.DEBUG
    assert record.getMessage() == (
        f"InSetConstraint.evaluate_with_bindings: no binding for variable {x!r}; "
        f"the bindings supplied {y!r}; reporting UNDECIDED"
    )


def test_no_record_is_built_when_the_level_is_disabled(
    x: Identifier, caplog: pytest.LogCaptureFixture
) -> None:
    """Test an undecided outcome logs nothing above its level."""
    with caplog.at_level(logging.WARNING, logger=_CORE_LOGGER):
        InSetConstraint(x, [1]).evaluate_with_bindings({})

    assert not [r for r in caplog.records if r.name == _CORE_LOGGER]


# =============================================================================
# Errors
# =============================================================================


def test_a_member_whose_hash_raises_is_refused_with_the_cause(x: Identifier) -> None:
    """Test the ``ConstraintError`` of an unhashable member chains the hash's error."""
    with pytest.raises(ConstraintError, match="unhashable") as raised:
        InSetConstraint(x, [_Unhashable(1)])
    assert isinstance(raised.value.__cause__, TypeError)


def test_a_nan_member_is_refused(x: Identifier) -> None:
    """Test a NaN member raises with the core's text."""
    with pytest.raises(ConstraintError, match="NaN"):
        InSetConstraint(x, [(1, float("nan"))])


def test_a_string_member_does_not_convert(x: Identifier) -> None:
    """Test ``convert_to_expression`` refuses a string with the core's text."""
    with pytest.raises(ConstraintError, match="type-strict"):
        InSetConstraint(x, ["5"]).convert_to_expression()


def test_an_unliftable_decimal_binding_chains_the_constructor_error(
    x: Identifier,
) -> None:
    """Test a negative ``Decimal`` binding raises, caused by the literal's refusal."""
    constraint = EquationConstraint(IdentifierExpression(x) > 0)

    with pytest.raises(ConstraintError, match="cannot be lifted") as raised:
        constraint.evaluate_with_bindings({x: Decimal("-1")})
    assert isinstance(raised.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(True, True, id="bool"),
        pytest.param(1, True, id="int"),
        pytest.param(1.5, True, id="float"),
        pytest.param(float("nan"), True, id="nan"),
        pytest.param(Decimal("0.5"), True, id="decimal"),
        pytest.param(Decimal("-0.5"), False, id="negative_decimal"),
        pytest.param("1", False, id="str"),
        pytest.param((1,), False, id="tuple"),
        pytest.param(_Token(1), False, id="serializable"),
    ],
)
def test_does_member_lift_to_expression(value: Any, expected: bool) -> None:
    """Test which values lift to a literal expression."""
    assert does_member_lift_to_expression(value) is expected


# =============================================================================
# Pickling, copying and payloads
# =============================================================================


@pytest.mark.parametrize(
    "build",
    [
        lambda x: EquationConstraint(IdentifierExpression(x) > 0),
        lambda x: InSetConstraint(x, [3, "a", (1, 2)]),
        lambda x: NotInSetConstraint(x, [_Token(1)]),
    ],
)
def test_constraints_pickle_and_copy_as_equivalent_constraints(build: Any) -> None:
    """Test pickle, copy and deepcopy give structurally equivalent constraints."""
    constraint = build(Identifier("x"))

    for restored in (
        pickle.loads(pickle.dumps(constraint)),
        copy.copy(constraint),
        copy.deepcopy(constraint),
    ):
        assert type(restored) is type(constraint)
        assert restored.is_structurally_equivalent(constraint)


@pytest.mark.usefixtures("v1_wire")
def test_a_payload_in_another_order_decodes_to_the_canonical_order() -> None:
    """Test a payload written before S13, in ``repr`` order, still decodes."""
    x = Identifier("x")
    payload = InSetConstraint(x, [2, 10]).serialize_to_dict()
    data: Any = payload["__data__"]
    data["values"] = list(reversed(data["values"]))

    restored = Constraint.deserialize_from_dict(payload)

    assert isinstance(restored, InSetConstraint)
    assert restored.values == (2, 10)


# =============================================================================
# Threads
# =============================================================================


def test_concurrent_evaluations_agree(x: Identifier) -> None:
    """Test eight threads deciding one constraint get the same answers."""
    constraint = InSetConstraint(x, list(range(100)))
    results: list[list[ConstraintOutcome]] = []
    lock = threading.Lock()

    def evaluate() -> None:
        answers = [
            constraint.evaluate_with_bindings({x: value}) for value in range(95, 105)
        ]
        with lock:
            results.append(answers)

    threads = [threading.Thread(target=evaluate) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    expected = [ConstraintOutcome.SATISFIED] * 5 + [ConstraintOutcome.VIOLATED] * 5
    assert results == [expected] * 8


# =============================================================================
# The system (S13b)
# =============================================================================


_PROBE_CALLS: dict[str, list[tuple[str, Any]]] = {}
"""The calls made on the probes, by label; each test uses its own labels."""


@dataclass(frozen=True, eq=False)
class _Probe(Constraint):
    """A Python-defined constraint that records the calls the system makes."""

    label: str
    outcome: Any = field(default=None, metadata=excluded_from_equivalence())

    @property
    def calls(self) -> list[tuple[str, Any]]:
        """Return the calls made on probes of this label, in order."""
        return _PROBE_CALLS.setdefault(self.label, [])

    @override
    def get_free_identifiers(self) -> frozenset[Identifier]:
        return frozenset()

    @override
    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        self.calls.append(("evaluate", bindings))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return ConstraintOutcome.SATISFIED if self.outcome is None else self.outcome

    @override
    def convert_to_expression(self) -> Expression:
        self.calls.append(("convert", None))
        return LiteralExpression(True)

    @override
    def build_ordering_key(self) -> str:
        self.calls.append(("key", None))
        return f"_Probe|{self.label}"

    @override
    def serialize_data_to_dict(self) -> dict[str, Any]:
        return {"label": self.label}

    @classmethod
    @override
    def deserialize_data_from_dict(cls, data: Any) -> "_Probe":
        return cls(str(data["label"]))

    @override
    def __repr__(self) -> str:
        return f"_Probe({self.label})"

    @override
    def __str__(self) -> str:
        return self.label


def test_system_is_a_thin_frozen_subclass_holding_the_objects_given(
    x: Identifier,
) -> None:
    """Test the system's class structure and the members it returns."""
    member = InSetConstraint(x, [1])
    probe = _Probe("probe")
    system = ConstraintSystem((member, probe))

    assert issubclass(ConstraintSystem, _rs.ConstraintSystem)
    assert isinstance(system, SymbolicPredicate)
    assert isinstance(system, FrozenMixin)
    assert system.constraints == (probe, member)
    assert system.constraints[0] is probe
    assert system != ConstraintSystem((member, probe))
    with pytest.raises(FrozenMutationError, match='"constraints"'):
        system.constraints = ()  # type: ignore[misc]


def test_a_python_member_key_is_read_once_when_the_system_is_built() -> None:
    """Test the ordering key of a Python-defined member is read on construction."""
    probe = _Probe("probe_key")

    system = ConstraintSystem((probe,))
    system.evaluate_with_bindings({})

    assert [call for call, _ in probe.calls].count("key") == 1


def test_a_python_member_receives_a_snapshot_of_the_bindings(x: Identifier) -> None:
    """Test a Python-defined member is handed a dict of the bindings given."""
    probe = _Probe("probe_snapshot")
    token = _Token(1)
    system = create_constraint_system(probe)

    system.evaluate_with_bindings(_bind(x, token))

    ((_, bindings),) = [call for call in probe.calls if call[0] == "evaluate"]
    assert isinstance(bindings, dict)
    assert bindings[x] is token


def test_a_python_member_error_propagates_as_itself() -> None:
    """Test an exception a Python-defined member raises reaches the caller."""
    error = RuntimeError("probe failed")
    system = create_constraint_system(_Probe("probe", error))

    with pytest.raises(RuntimeError) as raised:
        system.evaluate_with_bindings({})
    assert raised.value is error


def test_a_python_member_returning_no_outcome_is_refused() -> None:
    """Test a member whose evaluation returns another value raises ``TypeError``."""
    system = create_constraint_system(_Probe("probe", "yes"))

    with pytest.raises(TypeError, match="ConstraintOutcome"):
        system.evaluate_with_bindings({})


def test_a_member_that_is_no_constraint_is_refused() -> None:
    """Test a system refuses a member that is not a ``Constraint``."""
    with pytest.raises(ConstraintError, match="must be Constraint instances"):
        ConstraintSystem((object(),))  # type: ignore[arg-type]


def test_system_logs_each_undecided_member_on_its_module_logger(
    x: Identifier, caplog: pytest.LogCaptureFixture
) -> None:
    """Test an undecided member is logged at DEBUG, and its own record kept."""
    member = InSetConstraint(x, [1])
    system = create_constraint_system(member)

    with caplog.at_level(logging.DEBUG):
        outcome = system.evaluate_with_bindings({})

    assert outcome is ConstraintOutcome.UNDECIDED
    system_records = [
        r for r in caplog.records if r.name == "fhy_core.symbolic.constraint.system"
    ]
    core_records = [r for r in caplog.records if r.name == _CORE_LOGGER]
    assert [r.getMessage() for r in system_records] == [
        f"ConstraintSystem.evaluate_with_bindings: member {member!r} is undecided "
        "under the given bindings; the conjunction reports UNDECIDED unless a later "
        "member is violated"
    ]
    assert (
        core_records[0]
        .getMessage()
        .startswith("InSetConstraint.evaluate_with_bindings: no binding for variable")
    )


@pytest.mark.usefixtures("restore_default_solver")
def test_system_questions_convert_python_members_through_their_methods(
    x: Identifier,
    plug_smt_solver: Callable[[SatResult], RecordingSmtSolver],
) -> None:
    """Test a question asks a Python-defined member for its expression."""
    backend = plug_smt_solver(SatResult.SAT)
    probe = _Probe("probe_convert")
    system = create_constraint_system(probe, InSetConstraint(x, [1]))

    outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED
    assert ("convert", None) in probe.calls
    assert len(backend.checks) == 1


def test_check_implication_refuses_a_value_that_is_no_system(x: Identifier) -> None:
    """Test ``check_implication`` raises ``TypeError`` for another value."""
    system = create_constraint_system(InSetConstraint(x, [1]))

    with pytest.raises(TypeError, match="ConstraintSystem"):
        system.check_implication(object(), {})  # type: ignore[arg-type]


def test_systems_with_python_members_compare_through_their_methods(
    x: Identifier,
) -> None:
    """Test a pair of Python-defined members compares by the member's method."""
    left = create_constraint_system(_Probe("probe"), InSetConstraint(x, [1]))
    right = create_constraint_system(_Probe("probe"), InSetConstraint(x, [1]))
    other = create_constraint_system(_Probe("other"), InSetConstraint(x, [1]))

    assert left.is_structurally_equivalent(right)
    assert left.is_alpha_equivalent(right)
    assert not left.is_structurally_equivalent(other)
    assert not left.is_structurally_equivalent(create_constraint_system())


def test_system_pickles_and_round_trips_its_payload() -> None:
    """Test a system pickles and deserializes to an equivalent system."""
    x = Identifier("x")
    system = create_constraint_system(
        InSetConstraint(x, [2, 1]), EquationConstraint(IdentifierExpression(x) > 0)
    )

    restored = ConstraintSystem.deserialize_from_dict(system.serialize_to_dict())

    assert restored.is_structurally_equivalent(system)
    assert pickle.loads(pickle.dumps(system)).is_structurally_equivalent(system)


@pytest.mark.usefixtures("restore_default_solver")
def test_concurrent_system_questions_agree(
    plug_smt_solver: Callable[[SatResult], RecordingSmtSolver],
) -> None:
    """Test eight threads asking one system with a Python backend agree."""
    plug_smt_solver(SatResult.UNSAT)
    x = Identifier("x")
    system = create_constraint_system(EquationConstraint(IdentifierExpression(x) > 0))
    results: list[ConstraintOutcome] = []
    lock = threading.Lock()

    def ask() -> None:
        outcome = system.check_satisfiability({x: SymbolType.INT})
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [ConstraintOutcome.VIOLATED] * 8
