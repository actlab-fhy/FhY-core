"""Tests of the Python API over the Rust-backed param domains (S16a).

The six kinds of ``fhy_core.symbolic.param.domains`` are thin subclasses of
``fhy_core._rs`` classes backed by ``fhy_core::param``, and the module
functions call into the core. These tests pin what the binding adds over
the core's semantics, which ``rust/fhy-core/tests/it/param/`` specifies:
the class structure, the objects kept, the value readers, the exceptions
user code raises, a Python-defined domain driven by the core, the log
records, the error texts, pickling, payloads, and threads.
"""

import copy
import logging
import pickle
import threading
from collections.abc import Iterator, Sequence
from enum import IntEnum
from typing import Any

import pytest

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import PassExecutionError
from fhy_core.serialization import Serializable, register_serializable
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintError,
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
    create_constraint_system,
)
from fhy_core.symbolic.expression import Expression, IdentifierExpression
from fhy_core.symbolic.param import (
    CategoricalDomain,
    IntegerDomain,
    IntervalIntegerDomain,
    IntervalProfile,
    OrdinalDomain,
    Param,
    ParamDomain,
    ParamError,
    PermutationDomain,
    RealDomain,
)
from fhy_core.symbolic.param.domains import (
    evaluate_system_outcome,
    is_bound_expression,
)
from fhy_core.symbolic.solver import (
    Simplifier,
    Solver,
    get_default_solver,
    set_default_solver,
)
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.traits import FrozenMixin, FrozenMutationError
from fhy_core.utils.override import override

_DOMAINS_LOGGER = "fhy_core.symbolic.param.domains"

_KINDS: list[tuple[type[Any], type[Any], tuple[Any, ...]]] = [
    (IntegerDomain, _rs.IntegerDomain, ()),
    (IntervalIntegerDomain, _rs.IntervalIntegerDomain, ()),
    (RealDomain, _rs.RealDomain, ()),
    (OrdinalDomain, _rs.OrdinalDomain, ((1, 2),)),
    (CategoricalDomain, _rs.CategoricalDomain, (("a", "b"),)),
    (PermutationDomain, _rs.PermutationDomain, ((1, 2),)),
]


class _Level(IntEnum):
    """An ``int`` subclass."""

    HIGH = 3


class _Raising(Exception):
    """An exception a value's comparison raises."""


@register_serializable(type_id="tests.param.rank")
class _Rank(Serializable):
    """A ``Serializable`` value, equal, hashed and ordered by its value."""

    def __init__(self, value: int) -> None:
        self.value = value

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Rank) and self.value == other.value

    @override
    def __hash__(self) -> int:
        return hash(self.value)

    def __lt__(self, other: "_Rank") -> bool:
        return self.value < other.value

    @override
    def __repr__(self) -> str:
        return f"_Rank({self.value})"

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "_Rank":
        return cls(int(data["value"]))


class _TouchyRank(_Rank):
    """A rank whose ``<`` raises the exception it was built with."""

    def __init__(self, value: int, error: BaseException) -> None:
        super().__init__(value)
        self.error = error

    @override
    def __lt__(self, other: "_Rank") -> bool:
        raise self.error

    @override
    def __hash__(self) -> int:
        return hash(self.value)


class _EvenDomain(ParamDomain):
    """A Python-defined domain of the even integers, recording its calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    @property
    @override
    def symbol_type(self) -> SymbolType | None:
        self.calls.append(("symbol_type", ()))
        return SymbolType.INT

    @override
    def is_value_admissible(self, value: Any) -> bool:
        self.calls.append(("is_value_admissible", (value,)))
        return isinstance(value, int) and value % 2 == 0

    @override
    def normalize_value(self, value: Any) -> Any:
        return value

    @override
    def validate_constraint(self, constraint: Constraint, variable: Identifier) -> None:
        self.calls.append(("validate_constraint", (constraint, variable)))

    @override
    def get_implied_constraints(self, variable: Identifier) -> tuple[Constraint, ...]:
        return ()

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


class _RaisingSimplifier(Simplifier):
    """A Python simplifier raising the exception it was built with."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    @override
    def simplify(self, expression: Expression) -> Expression:
        raise self.error


def _ordinal(values: tuple[Any, ...]) -> OrdinalDomain:
    """Return the ordinal domain of `values`, of any type."""
    return OrdinalDomain(values)


@pytest.fixture
def restore_default_solver() -> Iterator[None]:
    """Restore the default solver after a test replaces it."""
    original = get_default_solver()
    yield
    set_default_solver(original)


# =============================================================================
# Class structure
# =============================================================================


@pytest.mark.parametrize(("public", "native", "arguments"), _KINDS)
def test_kind_extends_its_rust_class_and_is_a_virtual_param_domain(
    public: type[Any], native: type[Any], arguments: tuple[Any, ...]
) -> None:
    """Test each kind subclasses its `_rs` class and registers with the ABCs."""
    domain = public(*arguments)

    assert isinstance(domain, native)
    assert isinstance(domain, ParamDomain)
    assert isinstance(domain, FrozenMixin)
    assert ParamDomain not in type(domain).__mro__


@pytest.mark.parametrize(("public", "native", "arguments"), _KINDS)
def test_kind_is_frozen_and_equal_by_identity(
    public: type[Any], native: type[Any], arguments: tuple[Any, ...]
) -> None:
    """Test a domain refuses mutation, and `==` and `hash` are identity."""
    domain = public(*arguments)
    twin = public(*arguments)

    with pytest.raises(FrozenMutationError, match="frozen"):
        domain.anything = 1
    assert domain.is_frozen
    assert domain == domain  # noqa: PLR0124
    assert domain != twin
    assert domain.is_structurally_equivalent(twin)
    assert hash(domain) == object.__hash__(domain)


def test_attributes_are_slot_reads() -> None:
    """Test a kind's attributes live in slots of the public class."""
    domain = IntervalIntegerDomain(prefer_inclusive=False, non_negative=True)

    assert domain.prefer_inclusive is False
    assert domain.non_negative is True
    assert "non_negative" in IntervalIntegerDomain.__slots__


def test_zero_included_is_canonical_without_non_negative() -> None:
    """Test `zero_included` reads `True` unless the domain is non-negative."""
    assert IntegerDomain(zero_included=False).zero_included is True
    assert IntegerDomain(non_negative=True, zero_included=False).zero_included is False


# =============================================================================
# Objects kept and values read
# =============================================================================


def test_finite_domain_keeps_serializable_value_objects() -> None:
    """Test an ordinal domain returns the very `Serializable` objects given."""
    low, high = _Rank(1), _Rank(2)

    domain = _ordinal((high, low))

    assert domain.sorted_values[0] is low
    assert domain.sorted_values[1] is high


def test_number_subclasses_are_stored_as_exact_numbers() -> None:
    """Test an `IntEnum` value is stored as the `int` it denotes (P-2)."""
    domain = OrdinalDomain((_Level.HIGH, 1))

    assert domain.sorted_values == (1, 3)
    assert type(domain.sorted_values[1]) is int
    assert domain.is_value_admissible(_Level.HIGH)


def test_numpy_float_is_read_as_a_float() -> None:
    """Test a NumPy `float64` is admitted as the float it denotes."""
    numpy = pytest.importorskip("numpy")

    assert OrdinalDomain((1.5,)).is_value_admissible(numpy.float64(1.5))
    assert RealDomain().is_value_admissible(numpy.float64(1.5))


def test_ordinal_ties_order_by_kind_and_categories_canonically() -> None:
    """Test equal numbers of three kinds order `bool`, `float`, `int` (P-1, P-2)."""
    assert OrdinalDomain((1, True, 1.0)).sorted_values == (True, 1.0, 1)
    assert CategoricalDomain(("b", 10, 2, True)).categories == (True, 2, 10, "b")


def test_permutation_domain_admits_any_sequence_and_normalizes_to_a_tuple() -> None:
    """Test a permutation domain admits a list and normalizes it to a tuple."""
    domain = PermutationDomain((1, 2))

    assert domain.is_value_admissible([2, 1])
    assert not domain.is_value_admissible("ab")
    assert domain.normalize_value([2, 1]) == (2, 1)


def test_value_of_the_wrong_kind_raises_the_python_type_error() -> None:
    """Test a value that is no ordinal value raises `TypeError` with its text."""
    with pytest.raises(TypeError, match="orderable semantics"):
        _ordinal(([1],))


# =============================================================================
# Exceptions user code raises
# =============================================================================


def test_raising_type_error_in_less_than_is_chained_under_the_order_error() -> None:
    """Test a `<` raising `TypeError` makes the values incomparable, chained."""
    with pytest.raises(TypeError, match="mutually comparable") as caught:
        _ordinal((_Rank(1), _TouchyRank(2, TypeError("no order"))))

    assert isinstance(caught.value.__cause__, TypeError)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(_Raising("boom"), id="exception"),
        pytest.param(KeyboardInterrupt(), id="keyboard_interrupt"),
    ],
)
def test_raising_less_than_propagates_its_exception(error: BaseException) -> None:
    """Test another exception `<` raises reaches the caller as itself."""
    with pytest.raises(type(error)) as caught:
        _ordinal((_Rank(1), _TouchyRank(2, error)))

    assert caught.value is error


def test_serializable_and_primitive_values_do_not_order() -> None:
    """Test an ordinal `Serializable` does not order against an `int` (P-3)."""
    with pytest.raises(TypeError, match="mutually comparable"):
        _ordinal((_Rank(1), 2))


# =============================================================================
# A Python-defined domain
# =============================================================================


def test_native_procedure_drives_a_python_defined_domain() -> None:
    """Test a numeric subset asks a Python domain its sort and its values."""
    x, y = Identifier("x"), Identifier("y")
    other = _EvenDomain()

    outcome = IntegerDomain().compute_feasibility_subset(
        (InSetConstraint(x, {2, 3}),), x, other, (), y
    )

    assert outcome is ConstraintOutcome.VIOLATED
    assert other.calls == [
        ("symbol_type", ()),
        ("is_value_admissible", (2,)),
        ("is_value_admissible", (3,)),
    ]


@pytest.mark.parametrize(("value", "admissible"), [(-5, False), (0, True), (7, True)])
def test_a_natural_domain_admits_only_its_own_values(
    value: int, admissible: bool
) -> None:
    """Test the sign restriction holds at the domain level (F2-021)."""
    assert IntegerDomain(non_negative=True).is_value_admissible(value) is admissible
    assert IntegerDomain().is_value_admissible(value) is True


@pytest.mark.z3
def test_domain_questions_fold_in_the_domain_s_restriction() -> None:
    """Test the TYP probe's rows answer as a param over the domain does."""
    x, y = Identifier("x"), Identifier("y")
    natural, integer = IntegerDomain(non_negative=True), IntegerDomain()
    below_zero = EquationConstraint(IdentifierExpression(x) <= -1)

    assert natural.has_feasible_value((below_zero,), x) is ConstraintOutcome.VIOLATED
    assert integer.has_feasible_value((below_zero,), x) is ConstraintOutcome.SATISFIED
    assert (
        integer.compute_feasibility_subset((), x, natural, (), y)
        is ConstraintOutcome.VIOLATED
    )
    assert (
        natural.compute_feasibility_subset((), x, integer, (), y)
        is ConstraintOutcome.SATISFIED
    )
    assert integer.is_value_set_subset(natural) is False
    assert natural.is_value_set_subset(integer) is True


def test_python_defined_domain_exception_propagates() -> None:
    """Test an exception a Python domain's hook raises reaches the caller."""
    x = Identifier("x")

    class _Failing(_EvenDomain):
        @property
        @override
        def symbol_type(self) -> SymbolType | None:
            raise _Raising("no sort")

    with pytest.raises(_Raising, match="no sort"):
        IntegerDomain().compute_feasibility_subset((), x, _Failing(), (), x)


def test_structural_equivalence_is_false_against_a_python_defined_domain() -> None:
    """Test a native domain is not equivalent to a Python-defined one."""
    assert not IntegerDomain().is_structurally_equivalent(_EvenDomain())


class _CountingDomain(_EvenDomain):
    """A Python-defined domain that counts each set hook and what it received."""

    def __init__(self) -> None:
        super().__init__()
        # The domain is frozen once built, so a test arms it through this list.
        self.failure: list[BaseException] = []

    def _record(self, hook: str, *received: Any) -> None:
        self.calls.append((hook, received))
        if self.failure:
            raise self.failure[0]

    @override
    def is_value_set_subset(self, other: ParamDomain) -> bool:
        self._record("is_value_set_subset", other)
        return True

    @override
    def compute_feasibility_subset(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
    ) -> ConstraintOutcome:
        self._record(
            "compute_feasibility_subset",
            len(own_constraints),
            own_variable,
            other,
            len(other_constraints),
            other_variable,
        )
        return ConstraintOutcome.SATISFIED

    @override
    def compute_union(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
        variable: Identifier,
    ) -> tuple[ParamDomain, tuple[Constraint, ...]] | None:
        self._record("compute_union", own_variable, other, other_variable, variable)
        return other, ()

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
        self._record(
            "compute_intersection", own_variable, other, other_variable, variable
        )
        return other, ()

    @override
    def is_structurally_equivalent(self, other: object) -> bool:
        self._record("is_structurally_equivalent", other)
        return other is self


def _hooks(domain: _CountingDomain) -> list[str]:
    """Return the set hooks `domain` was asked, in order, and forget its calls."""
    hooks = [
        hook
        for hook, _ in domain.calls
        if hook not in {"symbol_type", "is_value_admissible", "validate_constraint"}
    ]
    domain.calls.clear()
    return hooks


def test_each_set_procedure_asks_a_python_defined_domain_once() -> None:
    """Test the set procedures reach a Python-defined domain's hooks, once each."""
    x, y, u, i = (Identifier(name) for name in "xyui")
    domain = _CountingDomain()
    own: Param[int] = Param(domain, x)
    other: Param[int] = Param(_EvenDomain(), y)
    domain.calls.clear()

    assert own.is_value_set_subset(other)
    assert _hooks(domain) == ["is_value_set_subset"]
    assert own.check_subset(other) is ConstraintOutcome.SATISFIED
    assert domain.calls == [("compute_feasibility_subset", (0, x, other.domain, 0, y))]
    domain.calls.clear()
    assert own.union(other, u).domain is other.domain
    assert domain.calls == [("compute_union", (x, other.domain, y, u))]
    domain.calls.clear()
    assert own.intersection(other, i).domain is other.domain
    assert _hooks(domain) == ["compute_intersection"]
    assert own.is_structurally_equivalent(Param(domain, x))
    assert _hooks(domain) == ["is_structurally_equivalent"]


def test_a_python_defined_domain_on_the_right_is_asked_only_its_sort() -> None:
    """Test a native left operand asks a Python-defined right one only its sort."""
    domain = _CountingDomain()
    native: Param[int] = Param(IntegerDomain(), Identifier("x"))
    custom: Param[int] = Param(domain, Identifier("y"))
    domain.calls.clear()

    assert native.is_value_set_subset(custom)
    assert not native.is_structurally_equivalent(custom)
    assert domain.calls == [("symbol_type", ())]


@pytest.mark.parametrize("error", [_Raising("hook failed"), KeyboardInterrupt()])
def test_a_python_defined_domain_s_set_hook_raises_through_each_procedure(
    error: BaseException,
) -> None:
    """Test the exception a set hook raises reaches the caller as itself."""
    x, y = Identifier("x"), Identifier("y")
    domain = _CountingDomain()
    own: Param[int] = Param(domain, x)
    other: Param[int] = Param(_EvenDomain(), y)
    domain.failure.append(error)

    for procedure in (
        lambda: own.is_value_set_subset(other),
        lambda: own.check_subset(other),
        lambda: own.union(other),
        lambda: own.intersection(other),
    ):
        with pytest.raises(type(error)) as raised:
            procedure()
        assert raised.value is error


# =============================================================================
# Records and errors
# =============================================================================


@pytest.mark.sympy
def test_enumeration_logs_the_undecided_candidates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test an undecided enumeration logs a WARNING naming its candidates."""
    x, y = Identifier("x"), Identifier("y")
    dependent = EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))

    with caplog.at_level(logging.WARNING, logger=_DOMAINS_LOGGER):
        outcome = IntegerDomain().has_feasible_value(
            (InSetConstraint(x, {2, 1}), dependent), x
        )

    assert outcome is ConstraintOutcome.UNDECIDED
    records = [record for record in caplog.records if record.name == _DOMAINS_LOGGER]
    assert len(records) == 1
    assert "candidate(s) 1, 2" in records[0].getMessage()
    assert repr(x) in records[0].getMessage()


@pytest.mark.parametrize(
    ("build", "error", "text"),
    [
        pytest.param(lambda: OrdinalDomain(()), ParamError, "non-empty", id="empty"),
        pytest.param(
            lambda: PermutationDomain((1.0, float("nan"))), ParamError, "NaN", id="nan"
        ),
        pytest.param(lambda: CategoricalDomain((1, 1)), ParamError, "unique", id="dup"),
        pytest.param(
            lambda: OrdinalDomain((1, "a")),
            TypeError,
            "mutually comparable",
            id="order",
        ),
    ],
)
def test_construction_errors_take_the_core_text(
    build: Any, error: type[Exception], text: str
) -> None:
    """Test construction errors raise the Python classes with the core's text."""
    with pytest.raises(error, match=text):
        build()


def test_kind_mismatch_names_the_other_class() -> None:
    """Test a set operation over another kind raises `TypeError` naming it."""
    x = Identifier("x")

    with pytest.raises(
        TypeError, match="Cannot union an OrdinalDomain with a domain of type"
    ):
        OrdinalDomain((1,)).compute_union((), x, CategoricalDomain((1,)), (), x, x)


def test_interval_domain_refuses_a_set_constraint_with_type_error() -> None:
    """Test an interval domain refuses a set constraint with `TypeError`."""
    x = Identifier("x")

    with pytest.raises(TypeError, match="interval integer parameters"):
        IntervalIntegerDomain().validate_constraint(InSetConstraint(x, {1}), x)


def test_rescoping_a_foreign_set_constraint_raises_constraint_error() -> None:
    """Test an intersection refuses a set constraint on another variable."""
    x, y, z = Identifier("x"), Identifier("y"), Identifier("z")

    with pytest.raises(ConstraintError, match="scoped"):
        IntegerDomain().compute_intersection(
            (NotInSetConstraint(y, {1}),), x, IntegerDomain(), (), x, z
        )


def test_interval_profile_is_the_python_dataclass() -> None:
    """Test `get_interval_profile` returns an `IntervalProfile` value."""
    profile = IntervalIntegerDomain(prefer_inclusive=False).get_interval_profile()

    assert profile == IntervalProfile(
        admits_only_bounds=True,
        non_negative=False,
        zero_included=True,
        prefer_inclusive=False,
    )
    assert RealDomain().get_interval_profile() is None


def test_is_bound_expression_refuses_a_non_expression() -> None:
    """Test `is_bound_expression` answers `False` for a value of another type."""
    x = Identifier("x")

    assert is_bound_expression(IdentifierExpression(x) >= 1)
    assert not is_bound_expression("x >= 1")  # type: ignore[arg-type]


# =============================================================================
# Undecidable failures of evaluation
# =============================================================================


@pytest.mark.usefixtures("restore_default_solver")
def test_evaluation_reads_a_pass_execution_error_as_undecided(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a simplifier raising `PassExecutionError` makes its member undecided."""
    x = Identifier("x")
    set_default_solver(
        Solver(simplifier=_RaisingSimplifier(PassExecutionError("bridge failed")))
    )
    system = create_constraint_system(
        EquationConstraint(IdentifierExpression(x) >= 0), InSetConstraint(x, {5})
    )

    with caplog.at_level(logging.WARNING, logger=_DOMAINS_LOGGER):
        outcome = evaluate_system_outcome(system, {x: 3})

    assert outcome is ConstraintOutcome.VIOLATED
    assert any("expression bridge" in record.getMessage() for record in caplog.records)


@pytest.mark.usefixtures("restore_default_solver")
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(_Raising("boom"), id="exception"),
        pytest.param(KeyboardInterrupt(), id="keyboard_interrupt"),
    ],
)
def test_evaluation_propagates_another_simplifier_failure(error: BaseException) -> None:
    """Test any other exception a Python simplifier raises reaches the caller."""
    x = Identifier("x")
    set_default_solver(Solver(simplifier=_RaisingSimplifier(error)))
    system = create_constraint_system(EquationConstraint(IdentifierExpression(x) >= 0))

    with pytest.raises(type(error)):
        evaluate_system_outcome(system, {x: 3})


# =============================================================================
# Pickling, payloads, threads
# =============================================================================


@pytest.mark.parametrize(("public", "native", "arguments"), _KINDS)
@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param(lambda value: pickle.loads(pickle.dumps(value)), id="pickle"),
        pytest.param(copy.deepcopy, id="deepcopy"),
    ],
)
def test_kind_survives_duplication(
    public: type[Any], native: type[Any], arguments: tuple[Any, ...], duplicate: Any
) -> None:
    """Test a domain duplicates to an equivalent instance of its class."""
    domain = public(*arguments)

    duplicated = duplicate(domain)

    assert type(duplicated) is public
    assert duplicated.is_structurally_equivalent(domain)


@pytest.mark.usefixtures("v1_wire")
def test_payload_in_another_order_decodes_to_the_canonical_order() -> None:
    """Test a categorical payload listing its values in any order decodes."""
    payload = CategoricalDomain(("a", "b")).serialize_to_dict()
    data = payload["__data__"]
    assert isinstance(data, dict)
    categories = data["categories"]
    assert isinstance(categories, list)
    data["categories"] = list(reversed(categories))

    rebuilt = ParamDomain.deserialize_from_dict(payload)

    assert isinstance(rebuilt, CategoricalDomain)
    assert rebuilt.categories == ("a", "b")


def test_threads_agree() -> None:
    """Test eight threads building and asking domains agree."""
    x = Identifier("x")
    results: list[Any] = []
    lock = threading.Lock()

    def work() -> None:
        domain = OrdinalDomain(tuple(range(20, 0, -1)))
        outcome = domain.has_feasible_value((InSetConstraint(x, {3, 30}),), x)
        with lock:
            results.append((domain.sorted_values, outcome))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 8
    assert all(result == results[0] for result in results)
    assert results[0][1] is ConstraintOutcome.SATISFIED
