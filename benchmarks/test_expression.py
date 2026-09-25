"""Benchmarks of the expression tree's hot paths.

They measure the expression API before and after it switched to the Rust
core (S4). The switch gave expressions the Rust semantics on the Rust
backend (decision D-S4-1 of ``docs/design/python-switch.md``): ``==`` and
``hash`` are structural, literals are normalized, the logical connectives
are one n-ary node, and the printed text is the core's. The pure-Python
backend keeps the old semantics until it is retired, so no benchmark
asserts a result whose value differs between the backends, and the
helpers the switch affects are marked ``D-S4-1``:

- :func:`_build_conjunction` builds a conjunction, one n-ary
  ``LogicalExpression`` on the Rust backend and a chain of binary nodes on
  the pure-Python one;
- :func:`_build_distinct_equal_trees` returns trees whose ``==`` is true
  (structural) on the Rust backend and false (identity) on the pure-Python
  one; the benchmarks time it without checking.
"""

import operator
import pickle
from collections.abc import Callable, Sequence

import pytest

from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import VisitablePass
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    call,
    logical_and,
    logical_not,
    make_binary_expression,
    pformat_expression,
    piecewise,
    validate_logical_operands,
    validate_predicate,
)
from fhy_core.term import AlphaRenaming
from fhy_core.utils.override import override

from .conftest import Benchmark

pytestmark = pytest.mark.benchmark(group="expression")

# How many operations the benchmarked deep trees stack on their first leaf.
_DEEP_TREE_DEPTH = 100
# How many identifiers the deep trees cycle through.
_DEEP_TREE_IDENTIFIER_COUNT = 4
# How many additions the shared DAG stacks: each adds a node to itself, so
# the DAG has `_DAG_DEPTH + 1` distinct nodes and `2**(_DAG_DEPTH + 1) - 1`
# occurrences.
_DAG_DEPTH = 10
# How many comparisons the benchmarked conjunction joins.
_CONJUNCTION_SIZE = 100
# How many piecewise levels the benchmarked predicate nests.
_PIECEWISE_NESTING = 20

_NODE_KINDS = (
    UnaryExpression,
    BinaryExpression,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    CallExpression,
)


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------


def _build_identifiers(count: int, prefix: str) -> tuple[Identifier, ...]:
    return tuple(Identifier(f"{prefix}{index}") for index in range(count))


def _build_deep_tree(identifiers: Sequence[Identifier], depth: int) -> Expression:
    """Return a chain of `depth` operations over `identifiers` and literals.

    The chain alternates additions of an identifier with multiplications by
    a literal and wraps every tenth level in a negation, so it holds unary,
    binary, identifier and literal nodes.
    """
    tree: Expression = IdentifierExpression(identifiers[0])
    for level in range(1, depth + 1):
        if level % 10 == 0:
            tree = -tree
        elif level % 2:
            tree = tree + identifiers[(level // 2) % len(identifiers)]
        else:
            tree = tree * level
    return tree


def _build_doubling_dag(identifier: Identifier, depth: int) -> Expression:
    """Return `depth` additions, each of one shared node to itself."""
    dag: Expression = IdentifierExpression(identifier)
    for _ in range(depth):
        dag = dag + dag
    return dag


def _build_distinct_equal_trees() -> tuple[Expression, Expression]:
    """Return two separately built, structurally equal small trees.

    D-S4-1: on the Rust backend ``==`` of the two is structural and true;
    on the pure-Python backend it is identity and false. The benchmarks
    time the call and do not assert its result.
    """
    x = Identifier("x")
    return IdentifierExpression(x) + 1, IdentifierExpression(x) + 1


def _build_conjunction(operands: Sequence[Expression]) -> Expression:
    """Return the conjunction of `operands`.

    D-S4-1: on the Rust backend one n-ary ``LogicalExpression`` of all
    the operands; on the pure-Python backend a right-folded chain of
    binary ``LOGICAL_AND`` nodes.
    """
    return logical_and(*operands)


def _build_deep_conjunction(size: int) -> Expression:
    """Return a conjunction of `size` comparisons of distinct identifiers."""
    references = [
        IdentifierExpression(identifier) for identifier in _build_identifiers(size, "c")
    ]
    return _build_conjunction([reference > 0 for reference in references])


def _build_nested_piecewise(nesting: int) -> Expression:
    """Return a predicate of `nesting` nested piecewise levels.

    Each level sits in the otherwise branch of the one above.
    """
    x = IdentifierExpression(Identifier("x"))
    predicate: Expression = x > 0
    for level in range(nesting):
        predicate = piecewise((x > level, x < level), otherwise=predicate)
    return predicate


@pytest.fixture()
def identifiers() -> tuple[Identifier, ...]:
    """Return the identifiers the deep trees are built over."""
    return _build_identifiers(_DEEP_TREE_IDENTIFIER_COUNT, "v")


@pytest.fixture()
def deep_tree(identifiers: tuple[Identifier, ...]) -> Expression:
    """Return a tree `_DEEP_TREE_DEPTH` operations deep."""
    return _build_deep_tree(identifiers, _DEEP_TREE_DEPTH)


@pytest.fixture()
def deep_tree_copy(identifiers: tuple[Identifier, ...]) -> Expression:
    """Return a separately built tree structurally equal to ``deep_tree``."""
    return _build_deep_tree(identifiers, _DEEP_TREE_DEPTH)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.fixture()
def x() -> IdentifierExpression:
    """Return a reference to a fresh identifier ``x``."""
    return IdentifierExpression(Identifier("x"))


@pytest.fixture()
def y() -> IdentifierExpression:
    """Return a reference to a fresh identifier ``y``."""
    return IdentifierExpression(Identifier("y"))


def test_identifier_expression_construction(benchmark: Benchmark) -> None:
    """Benchmark wrapping an identifier in a reference."""
    benchmark(IdentifierExpression, Identifier("x"))


@pytest.mark.parametrize(
    "value",
    [7, 10**30, 1.5, "5", "1.50", True],
    ids=["int", "big_int", "float", "integer_text", "decimal_text", "bool"],
)
def test_literal_expression_construction(
    benchmark: Benchmark, value: int | float | str | bool
) -> None:
    """Benchmark constructing a literal from each kind of value."""
    benchmark(LiteralExpression, value)


def test_unary_expression_construction(
    benchmark: Benchmark, x: IdentifierExpression
) -> None:
    """Benchmark constructing a negation directly."""
    benchmark(UnaryExpression, UnaryOperation.NEGATE, x)


def test_binary_expression_construction(
    benchmark: Benchmark, x: IdentifierExpression
) -> None:
    """Benchmark constructing an addition directly."""
    one = LiteralExpression(1)
    benchmark(BinaryExpression, BinaryOperation.ADD, x, one)


def test_make_binary_expression(benchmark: Benchmark, x: IdentifierExpression) -> None:
    """Benchmark the binary builder, which coerces an `int` operand."""
    benchmark(make_binary_expression, BinaryOperation.ADD, x, 1)


def test_logical_not_construction(
    benchmark: Benchmark, x: IdentifierExpression, y: IdentifierExpression
) -> None:
    """Benchmark negating a comparison."""
    benchmark(logical_not, x < y)


def test_conjunction_construction(
    benchmark: Benchmark, x: IdentifierExpression, y: IdentifierExpression
) -> None:
    """Benchmark conjoining three comparisons."""
    operands = (x < y, y < LiteralExpression(10), x > 0)
    benchmark(_build_conjunction, operands)


def test_piecewise_construction(benchmark: Benchmark, x: IdentifierExpression) -> None:
    """Benchmark building a one-case piecewise, which coerces its values."""
    condition = x > 0
    benchmark(piecewise, (condition, 1), otherwise=0)


def test_call_construction_of_a_builtin(
    benchmark: Benchmark, x: IdentifierExpression
) -> None:
    """Benchmark building a call of a built-in function."""
    benchmark(call, "max", x, 2)


def test_call_construction_of_a_user_function(
    benchmark: Benchmark, x: IdentifierExpression
) -> None:
    """Benchmark building a call of a name that is not a built-in."""
    benchmark(call, "user_function", x, 2)


def test_deep_tree_construction(
    benchmark: Benchmark, identifiers: tuple[Identifier, ...]
) -> None:
    """Benchmark building a tree `_DEEP_TREE_DEPTH` operations deep."""
    benchmark(_build_deep_tree, identifiers, _DEEP_TREE_DEPTH)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


_BINARY_OPERATORS: dict[str, Callable[[Expression, Expression], Expression]] = {
    "add": operator.add,
    "multiply": operator.mul,
    "true_divide": operator.truediv,
    "floor_divide": operator.floordiv,
    "modulo": operator.mod,
    "power": operator.pow,
    "less": operator.lt,
    "greater_equal": operator.ge,
    "equals": Expression.equals,
    "not_equals": Expression.not_equals,
}


@pytest.mark.parametrize("name", list(_BINARY_OPERATORS))
def test_binary_operator_of_two_expressions(
    benchmark: Benchmark, x: IdentifierExpression, y: IdentifierExpression, name: str
) -> None:
    """Benchmark a binary operator or builder method on two expressions."""
    benchmark(_BINARY_OPERATORS[name], x, y)


def test_add_operator_with_int(benchmark: Benchmark, x: IdentifierExpression) -> None:
    """Benchmark ``x + 1``, which coerces the `int`."""
    benchmark(operator.add, x, 1)


def test_reflected_subtract_operator_with_int(
    benchmark: Benchmark, x: IdentifierExpression
) -> None:
    """Benchmark ``1 - x``, the reflected operator."""
    benchmark(operator.sub, 1, x)


@pytest.mark.parametrize("name", ["neg", "pos"])
def test_unary_operator(
    benchmark: Benchmark, x: IdentifierExpression, name: str
) -> None:
    """Benchmark unary ``-`` and ``+``."""
    benchmark(getattr(operator, name), x)


# ---------------------------------------------------------------------------
# Field reads and isinstance dispatch
# ---------------------------------------------------------------------------


def _build_node_of_each_kind() -> dict[str, Expression]:
    x = IdentifierExpression(Identifier("x"))
    return {
        "unary": -x,
        "binary": x + 1,
        "identifier": x,
        "literal": LiteralExpression(1),
        "piecewise": piecewise((x > 0, 1), otherwise=0),
        "call": call("max", x, 2),
    }


_NODE_FIELDS = {
    "unary": ("operation", "operand"),
    "binary": ("operation", "left", "right"),
    "identifier": ("identifier",),
    "literal": ("value",),
    "piecewise": ("conditions", "values", "otherwise"),
    "call": ("function_name", "arguments"),
}


@pytest.mark.parametrize("kind", list(_NODE_FIELDS))
def test_node_attribute_access(benchmark: Benchmark, kind: str) -> None:
    """Benchmark reading every field of a node of one kind."""
    node = _build_node_of_each_kind()[kind]
    benchmark(operator.attrgetter(*_NODE_FIELDS[kind]), node)


def test_isinstance_of_node_kind(benchmark: Benchmark) -> None:
    """Benchmark one `isinstance` check that holds."""
    node = _build_node_of_each_kind()["binary"]
    assert benchmark(isinstance, node, BinaryExpression)


def _find_node_kind(node: Expression) -> type[Expression]:
    """Return the node class of `node` by an `isinstance` cascade."""
    for kind in _NODE_KINDS:
        if isinstance(node, kind):
            return kind
    raise TypeError(type(node).__name__)


def test_isinstance_dispatch_over_node_kinds(benchmark: Benchmark) -> None:
    """Benchmark an `isinstance` cascade, as the passes dispatch, to its last kind."""
    node = _build_node_of_each_kind()["call"]
    assert benchmark(_find_node_kind, node) is CallExpression


# ---------------------------------------------------------------------------
# Equality, hashing and equivalence
# ---------------------------------------------------------------------------


def test_eq_of_one_node(benchmark: Benchmark, x: IdentifierExpression) -> None:
    """Benchmark ``==`` of a node with itself."""
    benchmark(operator.eq, x, x)


def test_eq_of_distinct_equal_trees(benchmark: Benchmark) -> None:
    """Benchmark ``==`` of two separately built, structurally equal trees."""
    left, right = _build_distinct_equal_trees()
    benchmark(operator.eq, left, right)


def test_eq_of_distinct_equal_deep_trees(
    benchmark: Benchmark, deep_tree: Expression, deep_tree_copy: Expression
) -> None:
    """Benchmark ``==`` of two separately built, structurally equal deep trees."""
    benchmark(operator.eq, deep_tree, deep_tree_copy)


def test_hash_of_small_tree(benchmark: Benchmark) -> None:
    """Benchmark hashing a two-level tree."""
    tree, _ = _build_distinct_equal_trees()
    benchmark(hash, tree)


def test_hash_of_deep_tree(benchmark: Benchmark, deep_tree: Expression) -> None:
    """Benchmark hashing a deep tree."""
    benchmark(hash, deep_tree)


def test_dict_lookup_by_deep_tree(benchmark: Benchmark, deep_tree: Expression) -> None:
    """Benchmark looking a deep tree up in a dict keyed by it."""
    table = {deep_tree: 1}
    benchmark(table.get, deep_tree)


def test_structural_equivalence_of_deep_trees(
    benchmark: Benchmark, deep_tree: Expression, deep_tree_copy: Expression
) -> None:
    """Benchmark structural equivalence of two separately built deep trees."""
    assert benchmark(deep_tree.is_structurally_equivalent, deep_tree_copy)


def test_structural_equivalence_of_shared_dags(benchmark: Benchmark) -> None:
    """Benchmark structural equivalence of two separately built doubling DAGs."""
    x = Identifier("x")
    left = _build_doubling_dag(x, _DAG_DEPTH)
    right = _build_doubling_dag(x, _DAG_DEPTH)
    assert benchmark(left.is_structurally_equivalent, right)


def test_alpha_equivalence_under_free_renaming_of_deep_trees(
    benchmark: Benchmark, identifiers: tuple[Identifier, ...], deep_tree: Expression
) -> None:
    """Benchmark alpha equivalence of a deep tree and a renamed copy."""
    fresh = _build_identifiers(len(identifiers), "w")
    renamed = _build_deep_tree(fresh, _DEEP_TREE_DEPTH)
    renaming = AlphaRenaming.with_free_renaming(
        dict(zip(identifiers, fresh, strict=True))
    )
    assert benchmark(deep_tree.is_alpha_equivalent_under, renamed, renaming)


# ---------------------------------------------------------------------------
# Term operations
# ---------------------------------------------------------------------------


def test_substitute_in_deep_tree(
    benchmark: Benchmark, identifiers: tuple[Identifier, ...], deep_tree: Expression
) -> None:
    """Benchmark substituting one identifier of a deep tree."""
    replacements = {identifiers[1]: IdentifierExpression(Identifier("r"))}
    benchmark(deep_tree.substitute, replacements)


def test_substitute_in_shared_dag(benchmark: Benchmark) -> None:
    """Benchmark substituting the leaf of a doubling DAG."""
    x = Identifier("x")
    dag = _build_doubling_dag(x, _DAG_DEPTH)
    benchmark(dag.substitute, {x: IdentifierExpression(Identifier("r"))})


def test_free_identifiers_of_deep_tree(
    benchmark: Benchmark, identifiers: tuple[Identifier, ...], deep_tree: Expression
) -> None:
    """Benchmark collecting the free identifiers of a deep tree."""
    assert benchmark(deep_tree.get_free_identifiers) == frozenset(identifiers)


# ---------------------------------------------------------------------------
# Printing and visitor walks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("show_id", "functional"),
    [(False, False), (False, True), (True, False)],
    ids=["symbolic", "functional", "show_id"],
)
def test_pformat_expression_of_deep_tree(
    benchmark: Benchmark, deep_tree: Expression, show_id: bool, functional: bool
) -> None:
    """Benchmark pretty-formatting a deep tree."""
    benchmark(pformat_expression, deep_tree, show_id=show_id, functional=functional)


class _NodeCounter(VisitablePass[Expression, int]):
    """Count the nodes of a tree, visiting each kind through its own method."""

    def _count_children(self, node: Expression) -> int:
        return 1 + sum(self.visit(child) for child in node.get_visit_children())

    def visit_unary_expression(self, node: UnaryExpression) -> int:
        return self._count_children(node)

    def visit_binary_expression(self, node: BinaryExpression) -> int:
        return self._count_children(node)

    def visit_identifier_expression(self, node: IdentifierExpression) -> int:
        _ = node
        return 1

    def visit_literal_expression(self, node: LiteralExpression) -> int:
        _ = node
        return 1

    def visit_piecewise_expression(self, node: PiecewiseExpression) -> int:
        return self._count_children(node)

    def visit_call_expression(self, node: CallExpression) -> int:
        return self._count_children(node)

    @override
    def get_noop_output(self, ir: Expression) -> int:
        _ = ir
        return 0


def test_visitable_pass_walk_of_deep_tree(
    benchmark: Benchmark, deep_tree: Expression
) -> None:
    """Benchmark a visitor pass counting the nodes of a deep tree."""
    counter = _NodeCounter()
    assert benchmark(counter, deep_tree) > _DEEP_TREE_DEPTH


# ---------------------------------------------------------------------------
# The Boolean-position screen
# ---------------------------------------------------------------------------


def test_validate_logical_operands_of_deep_conjunction(benchmark: Benchmark) -> None:
    """Benchmark screening a conjunction of `_CONJUNCTION_SIZE` comparisons."""
    conjunction = _build_deep_conjunction(_CONJUNCTION_SIZE)
    benchmark(validate_logical_operands, conjunction)


def test_validate_predicate_of_nested_piecewise(benchmark: Benchmark) -> None:
    """Benchmark screening a predicate of nested piecewise expressions."""
    predicate = _build_nested_piecewise(_PIECEWISE_NESTING)
    benchmark(validate_predicate, predicate)


def test_validate_predicate_of_comparison(
    benchmark: Benchmark, x: IdentifierExpression, y: IdentifierExpression
) -> None:
    """Benchmark screening one comparison."""
    benchmark(validate_predicate, x < y)


# ---------------------------------------------------------------------------
# Serialization and pickling
# ---------------------------------------------------------------------------


def test_serialize_to_dict_of_deep_tree(
    benchmark: Benchmark, deep_tree: Expression
) -> None:
    """Benchmark serializing a deep tree to a dict."""
    benchmark(deep_tree.serialize_to_dict)


def test_deserialize_from_dict_of_deep_tree(
    benchmark: Benchmark, deep_tree: Expression
) -> None:
    """Benchmark deserializing a deep tree from a dict."""
    payload = deep_tree.serialize_to_dict()
    rebuilt = benchmark(Expression.deserialize_from_dict, payload)
    assert rebuilt.is_structurally_equivalent(deep_tree)


def test_json_round_trip_of_deep_tree(
    benchmark: Benchmark, deep_tree: Expression
) -> None:
    """Benchmark serializing a deep tree to JSON and back."""

    def round_trip() -> Expression:
        return Expression.from_json(deep_tree.to_json())

    assert benchmark(round_trip).is_structurally_equivalent(deep_tree)


def test_pickle_round_trip_of_deep_tree(
    benchmark: Benchmark, deep_tree: Expression
) -> None:
    """Benchmark pickling a deep tree and loading it back."""

    def round_trip() -> Expression:
        loaded: Expression = pickle.loads(pickle.dumps(deep_tree))
        return loaded

    assert benchmark(round_trip).is_structurally_equivalent(deep_tree)
