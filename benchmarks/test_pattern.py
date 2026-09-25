"""Benchmarks of pattern matching and rule-driven rewriting.

They measure the pattern API before and after it switches to the Rust core
(S5 of ``docs/design/python-switch.md``). The trees are the expression
benchmarks' deep tree (`_DEEP_TREE_DEPTH` operations over four identifiers)
and doubling DAG, and "the rules" are four: ``x + 0 -> x``,
``x * 1 -> x``, ``-(-x) -> x`` and ``x - x -> 0``.

The benchmarks call only API whose meaning S5 keeps, and assert no result
that S5 changes. The calls whose spelling S5 changes sit in helpers marked
with their decisions: :func:`_capture` returns what a ``CapturePattern``
binds (D-S5-2), and :func:`_read` reads it from a ``MatchBindings``
(D-S5-3).
"""

from collections.abc import Callable, Sequence

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    AlternativesPattern,
    BinaryExpression,
    BinaryExpressionPattern,
    BinaryOperation,
    Capture,
    CapturePattern,
    Expression,
    IdentifierExpression,
    IdentifierPattern,
    LiteralExpression,
    LiteralPattern,
    MatchBindings,
    Pattern,
    PredicatePattern,
    RewriteRule,
    RewriteRuleApplier,
    UnaryExpression,
    UnaryExpressionPattern,
    UnaryOperation,
    WildcardPattern,
    apply_rewrite_rule,
    apply_rewrite_rules,
)

from .conftest import Benchmark
from .test_expression import (
    _DAG_DEPTH,
    _DEEP_TREE_DEPTH,
    _DEEP_TREE_IDENTIFIER_COUNT,
    _build_deep_tree,
    _build_doubling_dag,
    _build_identifiers,
)

pytestmark = pytest.mark.benchmark(group="pattern")

# How many alternatives the benchmarked alternatives pattern tries.
_ALTERNATIVE_COUNT = 4
# Every how many levels the firing variant of the deep tree adds a zero.
_FIRING_PERIOD = 4


def _capture(name: str) -> Capture:
    """Return a new capture named `name`.

    D-S5-2: a ``Capture`` handle, which two positions share by passing the
    same object; before the switch, the name itself, which two positions
    shared by spelling it the same.
    """
    return Capture(name)


def _read(bindings: MatchBindings, capture: Capture) -> Expression:
    """Return the expression `bindings` binds to `capture`.

    D-S5-3: ``bindings[capture]``; before the switch, ``bindings.get(name)``,
    which raised ``KeyError`` for an unbound name as ``[]`` does now.
    """
    return bindings[capture]


# ---------------------------------------------------------------------------
# Trees and rules
# ---------------------------------------------------------------------------


def _build_firing_deep_tree(
    identifiers: Sequence[Identifier], depth: int
) -> Expression:
    """Return a deep tree in which the rules fire about once every four levels.

    It is the deep tree with every `_FIRING_PERIOD`-th multiplication
    replaced by an addition of zero, which ``x + 0 -> x`` rewrites.
    """
    tree: Expression = IdentifierExpression(identifiers[0])
    for level in range(1, depth + 1):
        if level % 10 == 0:
            tree = -tree
        elif level % 2:
            tree = tree + identifiers[(level // 2) % len(identifiers)]
        elif level % _FIRING_PERIOD == 0:
            tree = tree + 0
        else:
            tree = tree * level
    return tree


def _build_rules(*, named: bool = False) -> tuple[RewriteRule, ...]:
    """Return the four rules, named when `named` is set."""
    x = _capture("x")
    captured_x = CapturePattern(x, WildcardPattern())

    def keep_x(bindings: MatchBindings) -> Expression:
        return _read(bindings, x)

    def zero(bindings: MatchBindings) -> Expression:
        _ = bindings
        return LiteralExpression(0)

    shapes: tuple[tuple[str, Pattern, Callable[[MatchBindings], Expression]], ...] = (
        (
            "x + 0 -> x",
            BinaryExpressionPattern(BinaryOperation.ADD, captured_x, LiteralPattern(0)),
            keep_x,
        ),
        (
            "x * 1 -> x",
            BinaryExpressionPattern(
                BinaryOperation.MULTIPLY, captured_x, LiteralPattern(1)
            ),
            keep_x,
        ),
        (
            "-(-x) -> x",
            UnaryExpressionPattern(
                UnaryOperation.NEGATE,
                UnaryExpressionPattern(UnaryOperation.NEGATE, captured_x),
            ),
            keep_x,
        ),
        (
            "x - x -> 0",
            BinaryExpressionPattern(BinaryOperation.SUBTRACT, captured_x, captured_x),
            zero,
        ),
    )
    return tuple(
        RewriteRule(pattern, rewrite, name=name if named else None)
        for name, pattern, rewrite in shapes
    )


def _build_mirroring_pattern(tree: Expression) -> Pattern:
    """Return a pattern of the shape of `tree` capturing every leaf.

    `tree` holds unary, binary and leaf nodes only; each leaf gets its own
    capture.
    """
    leaf_count = 0

    def mirror(node: Expression) -> Pattern:
        nonlocal leaf_count
        if isinstance(node, UnaryExpression):
            return UnaryExpressionPattern(node.operation, mirror(node.operand))
        if isinstance(node, BinaryExpression):
            return BinaryExpressionPattern(
                node.operation, mirror(node.left), mirror(node.right)
            )
        leaf_count += 1
        return CapturePattern(
            _capture(f"leaf{leaf_count}"),
            WildcardPattern(),
        )

    return mirror(tree)


@pytest.fixture()
def identifiers() -> tuple[Identifier, ...]:
    """Return the identifiers the deep trees are built over."""
    return _build_identifiers(_DEEP_TREE_IDENTIFIER_COUNT, "v")


@pytest.fixture()
def deep_tree(identifiers: tuple[Identifier, ...]) -> Expression:
    """Return a tree `_DEEP_TREE_DEPTH` operations deep."""
    return _build_deep_tree(identifiers, _DEEP_TREE_DEPTH)


@pytest.fixture()
def a() -> IdentifierExpression:
    """Return a reference to a fresh identifier ``a``."""
    return IdentifierExpression(Identifier("a"))


@pytest.fixture()
def add_zero_pattern() -> tuple[Capture, Pattern]:
    """Return a capture ``x`` and the pattern ``x + 0`` capturing it."""
    x = _capture("x")
    pattern = BinaryExpressionPattern(
        BinaryOperation.ADD,
        CapturePattern(x, WildcardPattern()),
        LiteralPattern(0),
    )
    return x, pattern


# ---------------------------------------------------------------------------
# Construction and field reads
# ---------------------------------------------------------------------------


def test_capture_pattern_construction(benchmark: Benchmark) -> None:
    """Benchmark building a capture pattern over a wildcard."""
    x = _capture("x")
    wildcard = WildcardPattern()
    benchmark(CapturePattern, x, wildcard)


def test_binary_expression_pattern_construction(benchmark: Benchmark) -> None:
    """Benchmark building a binary pattern over two sub-patterns."""
    left = CapturePattern(_capture("x"), WildcardPattern())
    right = LiteralPattern(0)
    benchmark(BinaryExpressionPattern, BinaryOperation.ADD, left, right)


def test_rewrite_rule_construction(
    benchmark: Benchmark, add_zero_pattern: tuple[Capture, Pattern]
) -> None:
    """Benchmark building a named, guarded rule."""
    x, pattern = add_zero_pattern

    def rewrite(bindings: MatchBindings) -> Expression:
        return _read(bindings, x)

    def guard(bindings: MatchBindings) -> bool:
        _ = bindings
        return True

    benchmark(RewriteRule, pattern, rewrite, guard=guard, name="x + 0 -> x")


def test_pattern_attribute_access(
    benchmark: Benchmark, add_zero_pattern: tuple[Capture, Pattern]
) -> None:
    """Benchmark reading every field of a binary pattern."""
    _, pattern = add_zero_pattern
    assert isinstance(pattern, BinaryExpressionPattern)

    def read_fields() -> tuple[object, ...]:
        return (pattern.operation, pattern.left, pattern.right)

    benchmark(read_fields)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["hit", "miss"])
def test_match_of_a_small_pattern(
    benchmark: Benchmark,
    add_zero_pattern: tuple[Capture, Pattern],
    a: IdentifierExpression,
    outcome: str,
) -> None:
    """Benchmark matching ``x + 0`` against a match and a root mismatch."""
    _, pattern = add_zero_pattern
    candidate = a + 0 if outcome == "hit" else a * 0
    result = benchmark(pattern.match, candidate)
    assert (result is not None) == (outcome == "hit")


def test_match_of_a_mirroring_pattern_of_a_deep_tree(
    benchmark: Benchmark, deep_tree: Expression
) -> None:
    """Benchmark matching a pattern of a deep tree's shape, capturing each leaf."""
    pattern = _build_mirroring_pattern(deep_tree)
    assert benchmark(pattern.match, deep_tree) is not None


def test_match_of_a_repeated_capture_over_deep_operands(
    benchmark: Benchmark, identifiers: tuple[Identifier, ...]
) -> None:
    """Benchmark ``x - x`` over two distinct, equal deep trees."""
    x = _capture("x")
    captured_x = CapturePattern(x, WildcardPattern())
    pattern = BinaryExpressionPattern(BinaryOperation.SUBTRACT, captured_x, captured_x)
    candidate = _build_deep_tree(identifiers, _DEEP_TREE_DEPTH) - _build_deep_tree(
        identifiers, _DEEP_TREE_DEPTH
    )
    assert benchmark(pattern.match, candidate) is not None


def test_match_of_alternatives(benchmark: Benchmark, a: IdentifierExpression) -> None:
    """Benchmark four alternatives, of which only the last matches."""
    pattern = AlternativesPattern(
        tuple(
            BinaryExpressionPattern(BinaryOperation.ADD, WildcardPattern(), right)
            for right in (
                *(LiteralPattern(value) for value in range(1, _ALTERNATIVE_COUNT)),
                LiteralPattern(0),
            )
        )
    )
    assert benchmark(pattern.match, a + 0) is not None


def test_match_bindings_get(
    benchmark: Benchmark,
    add_zero_pattern: tuple[Capture, Pattern],
    a: IdentifierExpression,
) -> None:
    """Benchmark reading one capture of a match."""
    x, pattern = add_zero_pattern
    bindings = pattern.match(a + 0)
    assert bindings is not None
    benchmark(_read, bindings, x)


# ---------------------------------------------------------------------------
# Rewriting
# ---------------------------------------------------------------------------


def test_apply_rewrite_rule_at_the_root(
    benchmark: Benchmark, a: IdentifierExpression
) -> None:
    """Benchmark trying one rule once, at the root: the per-call overhead."""
    rule = _build_rules()[0]
    assert benchmark(apply_rewrite_rule, rule, a + 0) is not None


@pytest.mark.parametrize("variant", ["no_firing", "firing"])
def test_apply_rewrite_rules_to_a_deep_tree(
    benchmark: Benchmark, identifiers: tuple[Identifier, ...], variant: str
) -> None:
    """Benchmark the four rules over a deep tree, firing nowhere or often."""
    if variant == "no_firing":
        tree = _build_deep_tree(identifiers, _DEEP_TREE_DEPTH)
    else:
        tree = _build_firing_deep_tree(identifiers, _DEEP_TREE_DEPTH)
    rules = _build_rules()
    output = benchmark(apply_rewrite_rules, tree, rules)
    assert (output is tree) == (variant == "no_firing")


def test_apply_rewrite_rules_to_a_shared_dag(benchmark: Benchmark) -> None:
    """Benchmark the four rules over the doubling DAG."""
    dag = _build_doubling_dag(Identifier("d"), _DAG_DEPTH)
    rules = _build_rules()
    assert benchmark(apply_rewrite_rules, dag, rules) is dag


def test_rewrite_rule_applier_execute_of_a_deep_tree(
    benchmark: Benchmark, identifiers: tuple[Identifier, ...]
) -> None:
    """Benchmark the named rules through the pass, reporting each firing."""
    tree = _build_firing_deep_tree(identifiers, _DEEP_TREE_DEPTH)
    applier = RewriteRuleApplier(_build_rules(named=True))
    result = benchmark(applier.execute, tree)
    assert result.changed
    assert result.diagnostics


def test_apply_rewrite_rules_with_a_predicate_at_every_node(
    benchmark: Benchmark, deep_tree: Expression
) -> None:
    """Benchmark one rule whose predicate runs at every node and never holds."""

    def never(expression: Expression) -> bool:
        _ = expression
        return False

    rule = RewriteRule(PredicatePattern(never), lambda bindings: LiteralExpression(0))
    assert benchmark(apply_rewrite_rules, deep_tree, (rule,)) is deep_tree


def test_apply_rewrite_rules_with_a_guard_and_rewrite_at_every_leaf(
    benchmark: Benchmark, deep_tree: Expression
) -> None:
    """Benchmark a rule whose guard and rewrite run at every identifier leaf.

    The rule negates each captured reference, so every call reads a
    binding and builds a node.
    """
    x = _capture("x")

    def guard(bindings: MatchBindings) -> bool:
        return _read(bindings, x) is not None

    def negate(bindings: MatchBindings) -> Expression:
        return -_read(bindings, x)

    rule = RewriteRule(
        CapturePattern(x, IdentifierPattern()),
        negate,
        guard=guard,
    )
    assert benchmark(apply_rewrite_rules, deep_tree, (rule,)) is not deep_tree
