"""Tests that cycles through the extension's objects are collected.

Each test builds an ordinary cycle that runs through a Rust-backed object,
drops every name of it, and asserts a weak reference to a Python object in
the cycle dies after `gc.collect()`: a class that held its Python objects
where the collector cannot see them would keep the whole cycle alive for
the life of the process.
"""

import dataclasses
import gc
import weakref
from collections.abc import Callable
from typing import Any

import pytest

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.lattice import Lattice
from fhy_core.pass_infrastructure import CompilerPass, PassManager
from fhy_core.symbol_table import (
    SymbolTable,
    SymbolTableFrame,
    VariableSymbolTableFrame,
)
from fhy_core.symbolic.expression import Expression, FunctionSort, NativeFunction
from fhy_core.symbolic.expression.pattern import RewriteRule, WildcardPattern
from fhy_core.symbolic.param import CategoricalDomain, OrdinalDomain, Param
from fhy_core.symbolic.solver import GroundSimplifier, Simplifier, Solver
from fhy_core.types import (
    CoreDataType,
    NumericalType,
    PrimitiveDataType,
    TypeQualifier,
    TypeUnificationEnvironment,
)
from fhy_core.utils.override import override
from fhy_core.utils.poset import PartiallyOrderedSet

from .symbolic.param.test_domain_rust_binding import _Rank
from .symbolic.param.test_param_rust_binding import _EvenDomain
from .types.test_types_rust_binding import _Opaque, _Tagged

_Py_TPFLAGS_HAVE_GC = 1 << 14


def _collects(build: Callable[[], object]) -> bool:
    """Return whether the cycle `build` makes is freed by `gc.collect()`.

    `build` returns the object a weak reference watches; the cycle is
    otherwise unreachable once `build` returns.
    """
    watched = weakref.ref(build())
    gc.collect()
    return watched() is None


class _Node:
    """A plain Python object, for the control cycle."""

    other: "_Node | None" = None


def test_a_plain_python_cycle_is_collected() -> None:
    """Test the control: a cycle of Python objects alone is collected."""

    def build() -> object:
        first, second = _Node(), _Node()
        first.other, second.other = second, first
        return first

    assert _collects(build)


class _ManagedPass(CompilerPass[Any, Any]):
    """A pass that keeps the manager it runs in."""

    manager: Any = None

    @override
    def run_pass(self, ir: Any) -> Any:
        return ir


def test_a_pass_keeping_its_manager_is_collected() -> None:
    """Test a pass manager and a pass that points back at it."""

    def build() -> object:
        compiler_pass = _ManagedPass()
        manager: PassManager[Any] = PassManager()
        manager.add_pass(compiler_pass)
        compiler_pass.manager = manager
        return compiler_pass

    assert _collects(build)


def test_a_rewrite_callback_closing_over_its_rule_list_is_collected() -> None:
    """Test a rule whose rewrite callable closes over a list holding the rule."""

    def build() -> object:
        rules: list[RewriteRule] = []

        def rewrite(bindings: object) -> Expression:
            raise AssertionError(len(rules))

        rules.append(RewriteRule(WildcardPattern(), rewrite))
        return rewrite

    assert _collects(build)


def test_a_rewrite_guard_closing_over_its_rule_is_collected() -> None:
    """Test a rule whose guard closes over the rule."""

    def build() -> object:
        holder: list[RewriteRule] = []

        def guard(bindings: object) -> bool:
            return bool(holder)

        def rewrite(bindings: object) -> Expression:
            raise AssertionError

        holder.append(RewriteRule(WildcardPattern(), rewrite, guard=guard))
        return guard

    assert _collects(build)


class _SolverHoldingSimplifier(Simplifier):
    """A simplifier that keeps the solver it serves."""

    solver: Any = None

    @override
    def simplify(self, expression: Expression) -> Expression:
        return expression


def test_a_simplifier_holding_its_solver_is_collected() -> None:
    """Test a solver and a Python simplifier that points back at it."""

    def build() -> object:
        simplifier = _SolverHoldingSimplifier()
        simplifier.solver = Solver(simplifier=simplifier)
        return simplifier

    assert _collects(build)


def test_a_fallback_holding_its_ground_simplifier_is_collected() -> None:
    """Test a ground simplifier and the Python fallback that points back at it."""

    def build() -> object:
        fallback = _SolverHoldingSimplifier()
        ground = GroundSimplifier(fallback)
        fallback.solver = Solver(simplifier=ground)
        return fallback

    assert _collects(build)


class _Element:
    """A hashable element that keeps the order it belongs to."""

    container: Any = None


@pytest.mark.parametrize("order_class", [PartiallyOrderedSet, Lattice])
def test_an_element_pointing_at_its_order_is_collected(order_class: type) -> None:
    """Test a poset or lattice and an element that points back at it."""

    def build() -> object:
        element = _Element()
        order = order_class()
        order.add_element(element)
        element.container = order
        return element

    assert _collects(build)


@dataclasses.dataclass(frozen=True)
class _TableFrame(SymbolTableFrame):
    """A frame Python defines."""


def test_a_frame_pointing_at_its_symbol_table_is_collected() -> None:
    """Test a symbol table and a frame of it that points back at it."""

    def build() -> object:
        namespace, symbol = Identifier("ns"), Identifier("x")
        frame = _TableFrame(symbol)
        table = SymbolTable()
        table.add_namespace(namespace)
        table.add_symbol(namespace, symbol, frame)
        # A frozen frame still has a `__dict__`; the attribute is the cycle.
        object.__setattr__(frame, "table", table)
        return frame

    assert _collects(build)


def test_a_domain_pointing_at_its_param_is_collected() -> None:
    """Test a param over a Python-defined domain that points back at it.

    The param holds the domain twice, as its `domain` object and in the
    core param's custom part; both references are visited.
    """

    def build() -> object:
        domain = _EvenDomain()
        domain.calls.append(Param(domain))  # type: ignore[arg-type]
        return domain

    assert _collects(build)


def test_a_native_function_whose_implementation_holds_it_is_collected() -> None:
    """Test a registry entry whose implementation points back at the entry."""

    def build() -> object:
        box: list[object] = []

        def implementation(value: float) -> float:
            return value + len(box)

        box.append(
            NativeFunction(
                "gc_cycle_entry", [FunctionSort.REAL], FunctionSort.REAL, implementation
            )
        )
        return implementation

    assert _collects(build)


@pytest.mark.parametrize("domain_class", [CategoricalDomain, OrdinalDomain])
def test_an_opaque_member_pointing_at_its_domain_is_collected(
    domain_class: type,
) -> None:
    """Test a finite domain and a `Serializable` member that points back at it.

    The domain holds the member twice, in its values and in the core
    domain's opaque value; both references are visited.
    """

    def build() -> object:
        member = _Rank(1)
        domain = domain_class([member, _Rank(2)])
        member.owner = domain  # type: ignore[attr-defined]
        return member

    assert _collects(build)


def test_a_bound_data_type_pointing_at_its_environment_is_collected() -> None:
    """Test an environment and a Python-defined data type it binds, which
    points back at it."""

    def build() -> object:
        data_type = _Opaque("cycle")
        environment = TypeUnificationEnvironment.empty().with_data_type_binding(
            Identifier("T"), data_type
        )
        object.__setattr__(data_type, "owner", environment)
        return data_type

    assert _collects(build)


def test_a_frame_s_type_pointing_at_the_frame_is_collected() -> None:
    """Test a variable frame and its Python-defined type, which points back."""

    def build() -> object:
        ty = _Tagged("cycle", NumericalType(PrimitiveDataType(CoreDataType.INT32)))
        frame = VariableSymbolTableFrame(Identifier("x"), ty, TypeQualifier.INPUT)
        object.__setattr__(ty, "owner", frame)
        return ty

    assert _collects(build)


def test_a_data_type_pointing_at_its_numerical_type_is_collected() -> None:
    """Test a numerical type over a Python-defined data type that points back."""

    def build() -> object:
        data_type = _Opaque("cycle")
        numerical = NumericalType(data_type)
        object.__setattr__(data_type, "owner", numerical)
        return data_type

    assert _collects(build)


_NOT_TRACKED = {
    **dict.fromkeys(
        (
            "AnalysisBase",
            "BuiltinNativeImplementation",
            "CompilerPassBase",
            "DataType",
            "EquivalenceRole",
            "Provenance",
            "RuleBase",
            "SatResult",
            "SimplifierBase",
            "SimplifyContext",
            "SmtSolverBase",
            "Type",
            "UnknownProvenance",
            "ValidatorBase",
            "WildcardPattern",
        ),
        "holds no Python object; a Python subclass gains the flag from its "
        "own `__dict__`",
    ),
    "SmtScript": "caches only its declarations' identifiers and names, read "
    "through a `PyOnceLock`, which cannot be read without the interpreter",
    "PreservedAnalyses": "caches only the preserved analyses' names",
    "ConfigurationKey": "holds only the core key, no Python object",
    "Rng": "holds only its seed and the generator's state, no Python object",
    "StridedRun": "holds only the run's integers, no Python object",
    "ExhaustiveOracle": "holds only its path's coordinates and domain "
    "signatures, which keep no value of a domain, so no Python object",
    "Objective": "holds only its name and direction, no Python object",
    "Measurement": "holds only core values: its key's opaque values are the "
    "configuration's, as a ConfigurationKey's are, and its notes are core notes",
    "SympySimplifier": "holds only the SymPy module and its classes, which "
    "`sys.modules` keeps reachable, so no garbage cycle runs through them",
}
"""The exported classes that do not take part in GC, each with the reason."""


def test_every_exported_class_that_holds_objects_takes_part_in_gc() -> None:
    """Test each class that holds Python objects has `Py_TPFLAGS_HAVE_GC`."""
    classes = {
        name: value
        for name, value in vars(_rs).items()
        if isinstance(value, type)
        and value.__module__ == "fhy_core._rs"
        and not issubclass(value, BaseException)
    }

    missing = sorted(
        name
        for name, cls in classes.items()
        if name not in _NOT_TRACKED and not cls.__flags__ & _Py_TPFLAGS_HAVE_GC
    )

    assert _NOT_TRACKED.keys() <= classes.keys()
    assert not missing, missing
