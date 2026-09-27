"""Tests of the Python API over the Rust-backed params (S16b).

``Param`` and ``ParamAssignment`` of ``fhy_core.symbolic.param.core`` are
thin subclasses of ``fhy_core._rs`` classes backed by ``fhy_core::param``.
These tests pin what the binding adds over the core's semantics, which
``rust/fhy-core/tests/it/param/`` specifies: the class structure, the
objects kept, the messages that name Python values, the operators, a
Python-defined domain and constraint driven by the core, assignments,
pickling, payloads, and threads.
"""

import copy
import pickle
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.serialization import DeserializationDictStructureError, Serializable
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintOutcome,
    EquationConstraint,
    create_constraint_system,
)
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
    get_native_constant_identifier,
)
from fhy_core.symbolic.param import (
    IntegerDomain,
    Param,
    ParamAssignment,
    ParamDomain,
    ParamError,
    create_categorical_param,
    create_integer_param,
    create_interval_integer_param_between,
    create_natural_param,
    create_ordinal_param,
    create_permutation_param,
)
from fhy_core.symbolic.solver import (
    SatResult,
    SmtScript,
    SmtSolver,
    Solver,
    get_default_solver,
    set_default_solver,
)
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.traits import FrozenMixin, FrozenMutationError
from fhy_core.utils.override import override


class _EvenDomain(ParamDomain):
    """A Python-defined domain of the even integers, recording its hooks."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    @override
    def symbol_type(self) -> SymbolType | None:
        return SymbolType.INT

    @override
    def is_value_admissible(self, value: Any) -> bool:
        self.calls.append("is_value_admissible")
        return isinstance(value, int) and value % 2 == 0

    @override
    def normalize_value(self, value: Any) -> Any:
        self.calls.append("normalize_value")
        return value

    @override
    def validate_constraint(self, constraint: Constraint, variable: Identifier) -> None:
        self.calls.append("validate_constraint")

    @override
    def get_implied_constraints(self, variable: Identifier) -> tuple[Constraint, ...]:
        self.calls.append("get_implied_constraints")
        return (EquationConstraint(IdentifierExpression(variable) >= 0),)

    @override
    def is_value_set_subset(self, other: ParamDomain) -> bool:
        return False

    @override
    def compute_feasibility_subset(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
    ) -> ConstraintOutcome:
        return ConstraintOutcome.UNDECIDED

    @override
    def has_feasible_value(
        self, constraints: Sequence[Constraint], variable: Identifier
    ) -> ConstraintOutcome:
        self.calls.append("has_feasible_value")
        return ConstraintOutcome.SATISFIED

    @override
    def compute_intersection(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
        variable: Identifier,
    ) -> tuple[ParamDomain, tuple[Constraint, ...]]:
        return self, ()

    @override
    def is_structurally_equivalent(self, other: object) -> bool:
        return other is self

    @override
    def render_set_string(self) -> str:
        return "2Z"

    @override
    def render_set_repr(self) -> str:
        return ""

    @override
    def serialize_data_to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    @override
    def deserialize_data_from_dict(cls, data: dict[str, Any]) -> "_EvenDomain":
        return cls()


_SEEN: list[Mapping[Identifier, Any]] = []


@dataclass(frozen=True)
class _RecordingConstraint(Constraint):
    """A Python-defined constraint recording the bindings it is evaluated under."""

    scope: frozenset[Identifier]

    @override
    def get_free_identifiers(self) -> frozenset[Identifier]:
        return self.scope

    @override
    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        _SEEN.append(bindings)
        return ConstraintOutcome.SATISFIED

    @override
    def convert_to_expression(self) -> Expression:
        return LiteralExpression(True)

    @override
    def build_ordering_key(self) -> str:
        return "_RecordingConstraint"

    @override
    def __repr__(self) -> str:
        return "_RecordingConstraint()"

    @override
    def __str__(self) -> str:
        return "recording"


# =============================================================================
# Class structure
# =============================================================================


def test_param_extends_its_rust_class_and_mixes_in_the_protocols() -> None:
    """Test `Param` subclasses `_rs.Param`, `Serializable` and `FrozenMixin`."""
    param = create_integer_param()

    assert isinstance(param, _rs.Param)
    assert isinstance(param, Serializable)
    assert isinstance(param, FrozenMixin)
    assert Param[int] is not Param
    assert "constraints" in Param.__slots__


def test_param_is_frozen_and_equal_by_identity() -> None:
    """Test a param refuses mutation, and `==` and `hash` are identity."""
    x = Identifier("x")
    param = create_integer_param(name=x)
    twin = create_integer_param(name=x)

    with pytest.raises(FrozenMutationError, match="frozen"):
        param.domain = IntegerDomain()
    assert param != twin
    assert param.is_structurally_equivalent(twin)
    assert hash(param) == object.__hash__(param)


# =============================================================================
# Objects kept
# =============================================================================


def test_param_keeps_its_domain_variable_and_constraint_objects() -> None:
    """Test a param returns the objects given, and new implied constraints."""
    x = Identifier("x")
    domain = IntegerDomain(non_negative=True)
    bound = EquationConstraint(IdentifierExpression(x) <= 10)

    param: Param[int] = Param(
        domain, variable=x, constraint_system=create_constraint_system(bound)
    )

    assert param.domain is domain
    assert param.variable is x
    assert bound in param.constraints
    implied = [
        constraint for constraint in param.constraints if constraint is not bound
    ]
    assert len(implied) == 1
    assert isinstance(implied[0], EquationConstraint)


def test_adding_an_equivalent_constraint_returns_the_same_param() -> None:
    """Test `add_constraint` returns `self` for an equivalent constraint."""
    x = Identifier("x")
    param = create_integer_param(name=x).add_lower_bound_constraint(1)

    assert param.add_lower_bound_constraint(1) is param
    assert (
        param.add_constraint(EquationConstraint(IdentifierExpression(x) >= 1)) is param
    )


# =============================================================================
# Messages that name Python values
# =============================================================================


@pytest.mark.sympy
def test_value_messages_name_the_value_the_constraint_and_the_param() -> None:
    """Test `validate_value` names the value, constraint and param by `repr`."""
    x = Identifier("x")
    bound = EquationConstraint(IdentifierExpression(x) <= 10)
    param = create_integer_param(name=x, constraints=[bound])

    with pytest.raises(ParamError) as violated:
        param.validate_value(11)
    with pytest.raises(ParamError, match="is not admissible"):
        param.validate_value("a")

    message = str(violated.value)
    assert message.startswith("Value 11 violates constraint")
    assert repr(bound) in message
    assert repr(param) in message


def test_bindings_of_the_own_variable_are_refused_first() -> None:
    """Test bindings of the param's own variable raise before anything else."""
    x = Identifier("x")
    param = create_integer_param(name=x)

    with pytest.raises(ParamError, match="own variable"):
        param.is_value_valid("not even admissible", bindings={x: 1})


def test_native_constant_variable_names_the_constant() -> None:
    """Test a native constant's identifier as the variable names the constant."""
    pi = get_native_constant_identifier("pi")

    with pytest.raises(ParamError, match="native constant 'pi'"):
        create_integer_param(name=pi)


def test_constraint_outside_the_scope_names_it_and_its_scope() -> None:
    """Test a constraint whose scope lacks the variable names it and its scope."""
    x, y = Identifier("x"), Identifier("y")
    foreign = EquationConstraint(IdentifierExpression(y) >= 0)

    with pytest.raises(ParamError, match="scope must include") as caught:
        create_integer_param(name=x, constraints=[foreign])

    assert repr(foreign) in str(caught.value)


@pytest.mark.parametrize(
    ("bound", "error", "text"),
    [
        pytest.param(True, ValueError, "bare Python bool", id="bool"),
        pytest.param("1e5", ValueError, "invalid literal text", id="text"),
    ],
)
def test_bounds_are_lifted_as_expression_operands(
    bound: Any, error: type[Exception], text: str
) -> None:
    """Test a bound is lifted as an expression operand is, with its errors."""
    with pytest.raises(error, match=text):
        create_integer_param().add_lower_bound_constraint(bound)


# =============================================================================
# Operators
# =============================================================================


def test_integers_combine_with_interval_params_from_either_side() -> None:
    """Test `+`, `-` and `*` take an `int` on either side of an interval param."""
    param = create_interval_integer_param_between(0, 2)

    for result in (param + 3, 3 + param, param - 1, 10 - param, 2 * param, -param):
        assert isinstance(result, Param)


def test_interval_param_refuses_an_operand_of_another_type() -> None:
    """Test an interval param refuses a `str` or `bool` operand with `TypeError`."""
    param = create_interval_integer_param_between(0, 2)

    with pytest.raises(TypeError, match=r"Unsupported operand type.*str"):
        param + "a"
    with pytest.raises(TypeError, match=r"Unsupported operand type.*bool"):
        param * True


def test_non_interval_params_decline_arithmetic() -> None:
    """Test two params neither of which is an interval operand do not add."""
    left = create_ordinal_param([1, 2])

    with pytest.raises(TypeError):
        left + left
    with pytest.raises(TypeError, match="arithmetic is only supported"):
        -create_integer_param()


def test_non_interval_params_return_not_implemented_from_each_operator() -> None:
    """Test each binary operator of two non-interval params is `NotImplemented`."""
    left, right = create_integer_param(), create_integer_param()

    for operator in (
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
    ):
        assert getattr(left, operator)(right) is NotImplemented
    with pytest.raises(TypeError):
        left + right
    with pytest.raises(TypeError):
        left - right


def test_union_names_the_unsupported_domain_class() -> None:
    """Test a union of numeric params names the domain class, and `|` needs a param."""
    param = create_integer_param()

    with pytest.raises(TypeError, match="domain kind IntegerDomain"):
        param | param
    with pytest.raises(TypeError):
        param | 3  # type: ignore[operator]  # test: invalid operand


# =============================================================================
# Python-defined domains and constraints
# =============================================================================


@pytest.mark.sympy
def test_python_defined_domain_is_driven_by_the_core() -> None:
    """Test a param over a Python domain calls its hooks with Python objects."""
    domain = _EvenDomain()

    param: Param[int] = Param(domain)

    assert domain.calls == ["get_implied_constraints"]
    assert param.is_value_valid(4)
    assert not param.is_value_valid(3)
    assert param.check_feasibility() is ConstraintOutcome.SATISFIED
    assert domain.calls[1:] == [
        "is_value_admissible",
        "normalize_value",
        "is_value_admissible",
        "has_feasible_value",
    ]


class _ScopeError(Exception):
    """Raised by ``_UnscopedConstraint.get_free_identifiers``."""


class _UnscopedConstraint(_RecordingConstraint):
    """A Python-defined constraint whose scope raises."""

    @override
    def get_free_identifiers(self) -> frozenset[Identifier]:
        raise _ScopeError("no scope")


def test_a_python_constraint_whose_scope_raises_is_refused_with_its_error() -> None:
    """Test a scope that raises is the param's error, not an empty scope."""
    x = Identifier("x")

    with pytest.raises(_ScopeError, match="no scope"):
        create_integer_param(name=x, constraints=[_UnscopedConstraint(frozenset({x}))])


def test_python_defined_constraint_receives_the_bindings_snapshot() -> None:
    """Test a Python constraint in a param sees the value and bindings given."""
    x, y = Identifier("x"), Identifier("y")
    token = object()
    _SEEN.clear()
    constraint = _RecordingConstraint(frozenset({x, y}))
    param = create_integer_param(name=x, constraints=[constraint])

    bindings: Any = {y: token}
    assert param.is_constraints_satisfied(3, bindings=bindings)

    (seen,) = _SEEN
    assert seen[x] == 3
    assert seen[y] is token


# =============================================================================
# Assignments
# =============================================================================


def test_assignment_normalizes_its_value() -> None:
    """Test a permutation assignment stores its value as a tuple."""
    value: Any = [2, 1]
    assignment = create_permutation_param([1, 2]).assign(value)

    assert assignment.value == (2, 1)
    assert isinstance(assignment, ParamAssignment)
    assert assignment.is_value_set()


@pytest.mark.sympy
def test_dependent_assignment_pickles_without_its_bindings() -> None:
    """Test an assignment proved with bindings duplicates without them."""
    x, y = Identifier("x"), Identifier("y")
    dependent = EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))
    param = create_integer_param(name=x, constraints=[dependent])
    assignment = param.assign(3, bindings={y: 5})

    for duplicate in (
        pickle.loads(pickle.dumps(assignment)),
        copy.deepcopy(assignment),
    ):
        assert duplicate.value == 3
        assert duplicate.param.is_alpha_equivalent(param)


def test_assignment_values_compare_type_strictly() -> None:
    """Test assignments of `1` and `True` are not equivalent (P-9)."""
    param = create_categorical_param([1, True])

    assert not param.assign(1).is_structurally_equivalent(param.assign(True))
    assert param.assign(1).is_structurally_equivalent(param.assign(1))


@pytest.mark.sympy
def test_assignment_payload_rejects_only_a_provable_violation() -> None:
    """Test decoding an assignment refuses a violation and keeps an undecided one."""
    x, y = Identifier("x"), Identifier("y")
    param = create_natural_param(
        name=x,
        constraints=[
            EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y)),
            EquationConstraint(IdentifierExpression(x) <= 5),
        ],
    )

    undecided = ParamAssignment.construct_from_fields({"param": param, "value": 3})
    with pytest.raises(ParamError, match="violates"):
        ParamAssignment.construct_from_fields({"param": param, "value": 7})
    with pytest.raises(ParamError, match="not admissible"):
        ParamAssignment.construct_from_fields({"param": param, "value": -1})

    assert undecided.value == 3


# =============================================================================
# Pickling, payloads, threads
# =============================================================================


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param(lambda value: pickle.loads(pickle.dumps(value)), id="pickle"),
        pytest.param(copy.deepcopy, id="deepcopy"),
    ],
)
def test_param_survives_duplication(duplicate: Any) -> None:
    """Test a param duplicates to an equivalent param of its class."""
    param = create_interval_integer_param_between(0, 5)

    duplicated = duplicate(param)

    assert type(duplicated) is Param
    assert duplicated.is_alpha_equivalent(param)


@pytest.mark.usefixtures("v1_wire")
def test_param_payload_keeps_its_shape() -> None:
    """Test a param's payload is its three fields, and a malformed one is refused."""
    param = create_ordinal_param([2, 1])

    payload = param.serialize_to_dict()
    rebuilt: Param[Any] = Param.deserialize_from_dict(payload)

    assert set(payload) == {"domain", "variable", "constraint_system"}
    assert rebuilt.is_structurally_equivalent(param)
    with pytest.raises(DeserializationDictStructureError, match='"Param"'):
        Param.deserialize_from_dict({"domain": {}})


def test_threads_agree() -> None:
    """Test eight threads building params and assigning values agree."""
    results: list[Any] = []
    lock = threading.Lock()

    def work() -> None:
        param = create_ordinal_param(list(range(10, 0, -1)))
        assignment = param.assign(3)
        with lock:
            results.append((param.constraints, param.domain, assignment.value))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 8
    assert all(len(constraints) == 0 for constraints, _, _ in results)
    assert all(value == 3 for _, _, value in results)


# =============================================================================
# Members the interface suites did not name (R2-030)
# =============================================================================


class _InterruptingSmtSolver(SmtSolver):
    """A backend whose every check raises one `KeyboardInterrupt`."""

    def __init__(self) -> None:
        super().__init__()
        self.interrupt = KeyboardInterrupt()
        self.checks = 0

    @override
    def check(
        self, script: SmtScript, *, timeout_milliseconds: int | None
    ) -> SatResult:
        self.checks += 1
        raise self.interrupt


@pytest.fixture
def interrupting_default_solver() -> Iterator[_InterruptingSmtSolver]:
    """Make the default solver's backend interrupt, restoring it afterwards."""
    original = get_default_solver()
    backend = _InterruptingSmtSolver()
    set_default_solver(Solver(smt_solver=backend))
    try:
        yield backend
    finally:
        set_default_solver(original)


def _bounded_param() -> Param[int]:
    param = create_integer_param()
    variable = param.variable_expression
    return param.add_constraints(
        [EquationConstraint(variable >= 3), EquationConstraint(variable <= 9)]
    )


def test_variable_expression_is_a_fresh_reference_to_the_param_s_variable() -> None:
    """Test `variable_expression` holds the param's own identifier object."""
    param = create_integer_param()

    first, second = param.variable_expression, param.variable_expression

    assert isinstance(first, IdentifierExpression)
    assert first.identifier is param.variable
    assert first is not second
    assert first.is_structurally_equivalent(second)


def test_add_constraints_and_replace_constraints_return_new_params() -> None:
    """Test both build a new param over the same variable and domain.

    `add_constraints` keeps the param's constraints and appends the new
    ones; `replace_constraints` keeps only the ones it is given; the
    original is unchanged.
    """
    param = create_integer_param()
    variable = param.variable_expression

    added = param.add_constraints(
        [EquationConstraint(variable >= 3), EquationConstraint(variable <= 9)]
    )
    replaced = added.replace_constraints([EquationConstraint(variable >= 5)])

    assert added is not param and replaced is not added
    assert (
        len(param.constraints),
        len(added.constraints),
        len(replaced.constraints),
    ) == (0, 2, 1)
    assert added.variable is param.variable is replaced.variable
    assert added.domain is param.domain
    assert [added.is_value_valid(value) for value in (2, 3, 9, 10)] == [
        False,
        True,
        True,
        False,
    ]
    assert [replaced.is_value_valid(value) for value in (3, 5, 10)] == [
        False,
        True,
        True,
    ]


def test_add_upper_bound_constraint_is_inclusive_unless_told_otherwise() -> None:
    """Test the upper bound admits itself by default, and not when exclusive."""
    param = create_integer_param()

    inclusive = param.add_upper_bound_constraint(4)
    exclusive = param.add_upper_bound_constraint(4, is_inclusive=False)

    assert inclusive is not param
    assert (inclusive.is_value_valid(4), inclusive.is_value_valid(5)) == (True, False)
    assert (exclusive.is_value_valid(3), exclusive.is_value_valid(4)) == (True, False)


def test_is_feasible_is_subset_and_check_subset_answer_by_the_solver() -> None:
    """Test the three questions answer through the default solver."""
    bounded = _bounded_param()
    unbounded = create_integer_param(name=bounded.variable)
    variable = bounded.variable_expression
    empty = bounded.add_constraints([EquationConstraint(variable <= 1)])

    assert bounded.check_subset(unbounded) is ConstraintOutcome.SATISFIED
    assert bounded.is_subset(unbounded) is True
    assert unbounded.is_subset(bounded) is False
    assert bounded.is_feasible() is True
    assert empty.is_feasible() is False


@pytest.mark.parametrize("question", ["check_subset", "is_subset", "is_feasible"])
def test_a_solver_question_raises_the_backend_s_keyboard_interrupt(
    interrupting_default_solver: _InterruptingSmtSolver, question: str
) -> None:
    """Test the backend's `KeyboardInterrupt` is raised as itself, after one check."""
    bounded = _bounded_param()
    arguments = () if question == "is_feasible" else (create_integer_param(),)

    with pytest.raises(KeyboardInterrupt) as exception_info:
        getattr(bounded, question)(*arguments)

    assert exception_info.value is interrupting_default_solver.interrupt
    assert interrupting_default_solver.checks == 1


def test_add_constraints_raises_a_python_constraint_s_keyboard_interrupt() -> None:
    """Test a Python constraint's `KeyboardInterrupt` from its scope is raised as is."""
    interrupt = KeyboardInterrupt()

    @dataclass(frozen=True)
    class InterruptingConstraint(_RecordingConstraint):
        @override
        def get_free_identifiers(self) -> frozenset[Identifier]:
            raise interrupt

    param = create_integer_param()

    with pytest.raises(KeyboardInterrupt) as exception_info:
        param.add_constraints([InterruptingConstraint(frozenset({param.variable}))])

    assert exception_info.value is interrupt
