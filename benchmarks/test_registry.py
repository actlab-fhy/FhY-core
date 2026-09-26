"""Benchmarks of the function registry, its lookups and the inliner.

They measure the registry API before and after it switches to the Rust core
(S7 of ``docs/design/python-switch.md``). Registration mutates the
process-wide registry, so the registration rows restore a snapshot before
each round through :func:`_restore`, and the fixtures that register user
entries restore one after their benchmark.

The trees are the expression benchmarks' deep tree and doubling DAG. The
nested row inlines ``relu`` nested `_NESTED_DEPTH` deep; before the switch
the Python inliner walks the substituted body per occurrence, so the cost
doubles with each level, and depth 100 did not finish in five minutes.
"""

import math
from collections.abc import Iterator, Mapping

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    Expression,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    RegisteredEntry,
    RegisteredFunction,
    call,
    evaluate_expression,
    get_native_constant_identifier,
    get_registered_entries,
    get_registered_entry,
    inline_functions,
    is_entry_registered,
    logical_and,
    register_function,
    register_native_constant,
    register_native_function,
    try_get_native_constant_for_identifier,
    try_get_registered_result_sort,
    validate_predicate,
)
from fhy_core.symbolic.expression.errors import EntryLookupError
from fhy_core.symbolic.expression.registry import set_registry_state_for_tests
from fhy_core.term import AlphaRenaming
from fhy_core.types.checking import check_all_registered_function_bodies

from .conftest import Benchmark
from .test_expression import (
    _DAG_DEPTH,
    _DEEP_TREE_DEPTH,
    _DEEP_TREE_IDENTIFIER_COUNT,
    _build_deep_tree,
    _build_identifiers,
)

pytestmark = pytest.mark.benchmark(group="registry")

# How many rounds a registration row times, each after a restore.
_REGISTRATION_ROUNDS = 2_000
# How many user entries the snapshot row lists beside the built-ins.
_LISTED_USER_ENTRY_COUNT = 50
# How many user functions the sweep row checks beside the built-ins.
_SWEPT_USER_FUNCTION_COUNT = 20
# How many calls the screened conjunction joins, and how many functions
# they cycle through.
_SCREENED_CALL_COUNT = 100
_SCREENED_FUNCTION_COUNT = 5
# How deep the nested row nests `relu`.
_NESTED_DEPTH = 10
# How many user functions the chain row links.
_CHAIN_LENGTH = 10


def _restore(snapshot: Mapping[str, RegisteredEntry]) -> None:
    """Restore the registry to `snapshot`, dropping every later entry."""
    set_registry_state_for_tests(snapshot)


@pytest.fixture()
def snapshot() -> Iterator[Mapping[str, RegisteredEntry]]:
    """Return the registry's entries, and restore them after the benchmark."""
    entries = dict(get_registered_entries())
    try:
        yield entries
    finally:
        _restore(entries)


def _register_increment(name: str) -> RegisteredFunction:
    """Register `name(x) = x + 1` over the reals."""
    x = Identifier("x")
    return register_function(
        name, [x], [FunctionSort.REAL], FunctionSort.REAL, IdentifierExpression(x) + 1
    )


def _register_chain(length: int) -> str:
    """Register `length` user functions, each calling the next.

    The last is `x + 1`; each other one adds one to a call of the next.
    Return the name of the first.
    """
    names = [f"chain_{index}" for index in range(length)]
    _register_increment(names[-1])
    for index in range(length - 2, -1, -1):
        x = Identifier("x")
        register_function(
            names[index],
            [x],
            [FunctionSort.REAL],
            FunctionSort.REAL,
            call(names[index + 1], x) + 1,
        )
    return names[0]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body_kind", ["small", "deep_body"])
def test_register_function(
    benchmark: Benchmark, snapshot: Mapping[str, RegisteredEntry], body_kind: str
) -> None:
    """Time registering a function over a small body, and over a deep one."""
    identifiers = _build_identifiers(_DEEP_TREE_IDENTIFIER_COUNT, "p")
    body = (
        IdentifierExpression(identifiers[0]) + 1
        if body_kind == "small"
        else _build_deep_tree(identifiers, _DEEP_TREE_DEPTH)
    )
    sorts = [FunctionSort.REAL] * len(identifiers)

    def register() -> RegisteredFunction:
        return register_function("f", identifiers, sorts, FunctionSort.REAL, body)

    benchmark.pedantic(
        register, setup=lambda: _restore(snapshot), rounds=_REGISTRATION_ROUNDS
    )


def test_register_native_function(
    benchmark: Benchmark, snapshot: Mapping[str, RegisteredEntry]
) -> None:
    """Time registering a native function, with its arity check."""

    def register() -> object:
        return register_native_function(
            "softplus", [FunctionSort.REAL], FunctionSort.REAL, math.log1p
        )

    benchmark.pedantic(
        register, setup=lambda: _restore(snapshot), rounds=_REGISTRATION_ROUNDS
    )


def test_register_native_constant(
    benchmark: Benchmark, snapshot: Mapping[str, RegisteredEntry]
) -> None:
    """Time registering a constant, which mints its identifier."""

    def register() -> object:
        return register_native_constant("tau", FunctionSort.REAL, math.tau)

    benchmark.pedantic(
        register, setup=lambda: _restore(snapshot), rounds=_REGISTRATION_ROUNDS
    )


# ---------------------------------------------------------------------------
# Entry values
# ---------------------------------------------------------------------------


def _build_function(name: str, parameter: Identifier) -> RegisteredFunction:
    """Return `name(parameter) = max(parameter, 0) * parameter`, unregistered."""
    reference = IdentifierExpression(parameter)
    return RegisteredFunction(
        name,
        (parameter,),
        (FunctionSort.REAL,),
        FunctionSort.REAL,
        call("max", parameter, 0) * reference,
    )


def test_registered_function_construction(benchmark: Benchmark) -> None:
    """Time building an unregistered function entry."""
    x = Identifier("x")
    body = call("max", x, 0) * IdentifierExpression(x)
    benchmark(
        RegisteredFunction, "f", (x,), (FunctionSort.REAL,), FunctionSort.REAL, body
    )


def test_registered_function_eq(benchmark: Benchmark) -> None:
    """Time ``==`` of two equal function entries."""
    x = Identifier("x")
    left = _build_function("f", x)
    right = _build_function("f", x)
    benchmark(left.__eq__, right)


def test_registered_function_hash(benchmark: Benchmark) -> None:
    """Time ``hash`` of a function entry."""
    benchmark(hash, _build_function("f", Identifier("x")))


def test_registered_function_alpha_equivalence(benchmark: Benchmark) -> None:
    """Time the binder equivalence of two parameter-renamed functions."""
    left = _build_function("f", Identifier("x"))
    right = _build_function("g", Identifier("y"))
    assert benchmark(left.is_alpha_equivalent_under, right, AlphaRenaming.empty())


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


@pytest.fixture()
def user_function(snapshot: Mapping[str, RegisteredEntry]) -> str:
    """Register a user function and return its name."""
    _register_increment("increment")
    return "increment"


@pytest.mark.parametrize("kind", ["user", "builtin", "miss"])
def test_get_registered_entry(
    benchmark: Benchmark, user_function: str, kind: str
) -> None:
    """Time the per-call lookup of a user entry, a built-in, and a miss."""
    if kind == "miss":

        def look_up_missing() -> None:
            try:
                get_registered_entry("missing")
            except EntryLookupError:
                pass

        benchmark(look_up_missing)
        return
    benchmark(get_registered_entry, user_function if kind == "user" else "max")


def test_is_entry_registered(benchmark: Benchmark, user_function: str) -> None:
    """Time the presence check of a user entry."""
    assert benchmark(is_entry_registered, user_function)


@pytest.mark.parametrize("kind", ["user", "miss"])
def test_try_get_registered_result_sort(
    benchmark: Benchmark, user_function: str, kind: str
) -> None:
    """Time the result-sort lookup of a user function, and of a miss."""
    benchmark(try_get_registered_result_sort, user_function if kind == "user" else "no")


@pytest.mark.parametrize("kind", ["hit", "miss"])
def test_try_get_native_constant_for_identifier(
    benchmark: Benchmark, kind: str
) -> None:
    """Time the constant lookup by identifier, a hit and a look-alike miss."""
    identifier = (
        get_native_constant_identifier("pi") if kind == "hit" else Identifier("pi")
    )
    benchmark(try_get_native_constant_for_identifier, identifier)


def test_get_native_constant_identifier(benchmark: Benchmark) -> None:
    """Time the lookup of a constant's identifier by name."""
    benchmark(get_native_constant_identifier, "pi")


def test_get_registered_entries(
    benchmark: Benchmark, snapshot: Mapping[str, RegisteredEntry]
) -> None:
    """Time the registry snapshot, with user entries beside the built-ins."""
    for index in range(_LISTED_USER_ENTRY_COUNT):
        _register_increment(f"listed_{index}")
    benchmark(get_registered_entries)


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------


def test_validate_predicate_of_user_calls(
    benchmark: Benchmark, snapshot: Mapping[str, RegisteredEntry]
) -> None:
    """Time the screen of a conjunction of user calls and a constant."""
    for index in range(_SCREENED_FUNCTION_COUNT):
        x = Identifier("x")
        register_function(
            f"is_positive_{index}",
            [x],
            [FunctionSort.REAL],
            FunctionSort.BOOL,
            IdentifierExpression(x) > index,
        )
    register_native_constant("enabled", FunctionSort.BOOL, value=True)
    y = Identifier("y")
    operands: list[Expression] = [
        call(f"is_positive_{index % _SCREENED_FUNCTION_COUNT}", y)
        for index in range(_SCREENED_CALL_COUNT)
    ]
    operands.append(IdentifierExpression(get_native_constant_identifier("enabled")))
    benchmark(validate_predicate, logical_and(*operands))


# ---------------------------------------------------------------------------
# The inliner
# ---------------------------------------------------------------------------


def _nest_relu(depth: int) -> Expression:
    """Return `relu` applied `depth` times to an identifier."""
    tree: Expression = IdentifierExpression(Identifier("x"))
    for _ in range(depth):
        tree = call("relu", tree)
    return tree


@pytest.mark.parametrize(
    "kind", ["no_calls", "nested_builtins", "user_chain", "shared_dag"]
)
def test_inline_functions(
    benchmark: Benchmark, snapshot: Mapping[str, RegisteredEntry], kind: str
) -> None:
    """Time the inliner over four trees.

    They are a tree with no call, nested built-ins, a chain of user calls,
    and a shared DAG with a call at its leaf.
    """
    if kind == "no_calls":
        tree = _build_deep_tree(
            _build_identifiers(_DEEP_TREE_IDENTIFIER_COUNT, "v"), _DEEP_TREE_DEPTH
        )
    elif kind == "nested_builtins":
        tree = _nest_relu(_NESTED_DEPTH)
    elif kind == "user_chain":
        tree = call(_register_chain(_CHAIN_LENGTH), Identifier("y"))
    else:
        tree = _build_dag_over(call("sigmoid", Identifier("z")))
    benchmark(inline_functions, tree)


def _build_dag_over(leaf: Expression) -> Expression:
    """Return `_DAG_DEPTH` additions over `leaf`, each of one node to itself."""
    dag = leaf
    for _ in range(_DAG_DEPTH):
        dag = dag + dag
    return dag


def _register_floor_chain(length: int) -> str:
    """Register `length` user functions, each flooring a call of the next.

    The last floors its argument. Return the name of the first.
    """
    names = [f"floor_chain_{index}" for index in range(length)]
    for index in range(length - 1, -1, -1):
        x = Identifier("x")
        inner = call(names[index + 1], x) if index + 1 < length else x
        register_function(
            names[index],
            [x],
            [FunctionSort.REAL],
            FunctionSort.INT,
            call("floor", inner),
        )
    return names[0]


def test_evaluate_after_inline(
    benchmark: Benchmark, snapshot: Mapping[str, RegisteredEntry]
) -> None:
    """Time evaluating an inlined chain, which looks up a native per call."""
    tree = call(_register_floor_chain(_CHAIN_LENGTH), LiteralExpression(3.5))

    def evaluate() -> Expression:
        return evaluate_expression(inline_functions(tree))

    result = benchmark(evaluate)
    assert isinstance(result, LiteralExpression)


def test_check_all_registered_function_bodies(
    benchmark: Benchmark, snapshot: Mapping[str, RegisteredEntry]
) -> None:
    """Time the body sweep over the built-ins and user functions."""
    for index in range(_SWEPT_USER_FUNCTION_COUNT):
        _register_increment(f"swept_{index}")
    report = benchmark(check_all_registered_function_bodies)
    assert not report.has_errors()
