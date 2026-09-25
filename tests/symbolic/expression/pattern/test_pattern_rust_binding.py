"""Tests for the Python interface over the Rust-backed patterns and rules.

The pattern classes, ``Capture``, ``MatchBindings``, ``RewriteRule`` and
``FiredRule`` are thin Python subclasses of ``fhy_core._rs`` classes
(pattern P2 of ``docs/design/python-switch.md``), and ``Rule`` is a
Python ABC over ``_rs.RuleBase`` whose subclasses the Rust walk drives
(pattern P3). These tests cover what the binding adds around the core:
the class structure and registration, argument checks, the node objects
callbacks receive and results keep, the callbacks' errors, deep trees and
patterns, the pass, and pickles.
"""

import pickle
import re
import sys
from collections.abc import Callable

import pytest

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import CompilerPass, PassExecutionError
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.symbolic.expression.pattern import (
    AlternativesPattern,
    BinaryExpressionPattern,
    CallExpressionPattern,
    Capture,
    CapturePattern,
    FiredRule,
    IdentifierPattern,
    LiteralPattern,
    LogicalExpressionPattern,
    MatchBindings,
    Pattern,
    PiecewiseExpressionPattern,
    PredicatePattern,
    RewriteCallbackError,
    RewriteError,
    RewriteRebuildError,
    RewriteRule,
    RewriteRuleApplier,
    Rule,
    UnaryExpressionPattern,
    WildcardPattern,
    apply_rewrite_rule,
    apply_rewrite_rules,
)
from fhy_core.traits import FrozenMixin, FrozenMutationError

_PATTERN_CLASSES = [
    (WildcardPattern, _rs.WildcardPattern),
    (CapturePattern, _rs.CapturePattern),
    (LiteralPattern, _rs.LiteralPattern),
    (IdentifierPattern, _rs.IdentifierPattern),
    (UnaryExpressionPattern, _rs.UnaryExpressionPattern),
    (BinaryExpressionPattern, _rs.BinaryExpressionPattern),
    (LogicalExpressionPattern, _rs.LogicalExpressionPattern),
    (PiecewiseExpressionPattern, _rs.PiecewiseExpressionPattern),
    (CallExpressionPattern, _rs.CallExpressionPattern),
    (PredicatePattern, _rs.PredicatePattern),
    (AlternativesPattern, _rs.AlternativesPattern),
]


def _reference(name: str) -> IdentifierExpression:
    return IdentifierExpression(Identifier(name))


def _keep(capture: Capture) -> Callable[[MatchBindings], Expression]:
    """Return a rewrite returning what ``capture`` bound."""

    def rewrite(bindings: MatchBindings) -> Expression:
        return bindings[capture]

    return rewrite


def _build_add_zero_rule(name: str | None = "x + 0 -> x") -> RewriteRule:
    x = Capture("x")
    pattern = BinaryExpressionPattern(
        BinaryOperation.ADD, CapturePattern(x), LiteralPattern(0)
    )
    return RewriteRule(pattern, _keep(x), name=name)


def is_literal(expression: Expression) -> bool:
    """Return whether ``expression`` is a literal; a module-level predicate."""
    return isinstance(expression, LiteralExpression)


def to_zero(bindings: MatchBindings) -> Expression:
    """Return the literal zero; a module-level rewrite."""
    _ = bindings
    return LiteralExpression(0)


def always(bindings: MatchBindings) -> bool:
    """Return ``True``; a module-level guard."""
    _ = bindings
    return True


# =============================================================================
# Class structure and registration
# =============================================================================


@pytest.mark.parametrize(("public_class", "rust_class"), _PATTERN_CLASSES)
def test_public_pattern_class_subclasses_its_rust_class_and_pattern(
    public_class: type[Pattern], rust_class: type
) -> None:
    """Test each public pattern class extends its ``_rs`` class and ``Pattern``."""
    assert issubclass(public_class, rust_class)
    assert issubclass(public_class, Pattern)
    assert issubclass(public_class, _rs.Pattern)


@pytest.mark.parametrize(
    ("public_class", "rust_class"),
    [
        *_PATTERN_CLASSES[:1],
        (Capture, _rs.Capture),
        (MatchBindings, _rs.MatchBindings),
        (RewriteRule, _rs.RewriteRule),
        (FiredRule, _rs.FiredRule),
    ],
)
def test_registering_another_public_class_is_refused(
    public_class: type, rust_class: type
) -> None:
    """Test the registered public class never changes once registered."""
    assert issubclass(public_class, rust_class)

    class _Impostor(public_class):  # type: ignore[misc]
        pass

    public_class._register_public_class()  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="registered already"):
        _Impostor._register_public_class()


def test_a_new_pattern_kind_cannot_be_built() -> None:
    """Test a Python subclass of ``Pattern`` that is no kind cannot be built."""

    class _NewKind(Pattern):
        pass

    with pytest.raises(TypeError):
        _NewKind()
    with pytest.raises(TypeError):
        Pattern()


def test_a_subclass_of_a_kind_is_that_kind() -> None:
    """Test a Python subclass of a pattern kind matches as that kind."""

    class _Literal(LiteralPattern):  # type: ignore[misc]
        pass

    assert _Literal(1).match(LiteralExpression(1)) is not None
    assert _Literal(1).match(LiteralExpression(2)) is None


def test_the_classes_are_virtual_frozen_mixins_and_rules() -> None:
    """Test the classes stand in for ``FrozenMixin``, and rules for ``Rule``."""
    rule = _build_add_zero_rule()
    x = Capture("x")
    bindings = CapturePattern(x).match(LiteralExpression(1))

    for value in (WildcardPattern(), x, bindings, rule, FiredRule(0, None)):
        assert isinstance(value, FrozenMixin)
        assert value.is_frozen
    assert isinstance(rule, Rule)
    assert not issubclass(RewriteRule, _rs.RuleBase)


@pytest.mark.parametrize(
    "build",
    [
        WildcardPattern,
        lambda: Capture("x"),
        MatchBindings,
        _build_add_zero_rule,
        lambda: FiredRule(0, "r"),
    ],
    ids=["pattern", "capture", "bindings", "rule", "fired_rule"],
)
def test_mutation_raises_the_frozen_error(build: Callable[[], object]) -> None:
    """Test setting or deleting an attribute raises ``FrozenMutationError``."""
    value = build()

    with pytest.raises(FrozenMutationError, match="Cannot modify"):
        value.anything = 1  # type: ignore[attr-defined]
    with pytest.raises(FrozenMutationError, match="Cannot delete"):
        del value.anything  # type: ignore[attr-defined]


# =============================================================================
# Argument checks
# =============================================================================


@pytest.mark.parametrize(
    ("build", "error", "message"),
    [
        (lambda: Capture(1), TypeError, "Capture name must be a str, got int."),  # type: ignore[arg-type]
        (
            lambda: CapturePattern("x"),  # type: ignore[arg-type]
            TypeError,
            "CapturePattern capture must be a Capture, got str.",
        ),
        (
            lambda: CapturePattern(Capture("x"), None),  # type: ignore[arg-type]
            TypeError,
            "CapturePattern sub_pattern must be a Pattern, got NoneType.",
        ),
        (
            lambda: IdentifierPattern("x"),  # type: ignore[arg-type]
            TypeError,
            "IdentifierPattern identifier must be an Identifier, got str.",
        ),
        (
            lambda: UnaryExpressionPattern("nope", WildcardPattern()),  # type: ignore[arg-type]
            ValueError,
            "'nope' is not a valid UnaryOperation",
        ),
        (
            lambda: BinaryExpressionPattern(None, 1, WildcardPattern()),  # type: ignore[arg-type]
            TypeError,
            "BinaryExpressionPattern left must be a Pattern, got int.",
        ),
        (
            lambda: PiecewiseExpressionPattern([1], WildcardPattern()),  # type: ignore[list-item]
            TypeError,
            "PiecewiseExpressionPattern cases must be (condition, value) pairs of "
            "Patterns, got int.",
        ),
        (
            lambda: CallExpressionPattern(1, None),  # type: ignore[arg-type]
            TypeError,
            "CallExpressionPattern function_name must be a str, got int.",
        ),
        (lambda: CallExpressionPattern("", None), ValueError, "function name"),
        (
            lambda: PredicatePattern(1),  # type: ignore[arg-type]
            TypeError,
            "PredicatePattern predicate must be callable, got int.",
        ),
        (
            lambda: LiteralPattern(object()),  # type: ignore[arg-type]
            TypeError,
            "Unsupported type for literal expression value",
        ),
        (lambda: LiteralPattern("abc"), ValueError, "invalid literal text"),
        (
            lambda: RewriteRule(1, to_zero),  # type: ignore[arg-type]
            TypeError,
            "RewriteRule pattern must be a Pattern, got int.",
        ),
        (
            lambda: RewriteRule(WildcardPattern(), 1),  # type: ignore[arg-type]
            TypeError,
            "RewriteRule rewrite must be callable, got int.",
        ),
        (
            lambda: RewriteRule(WildcardPattern(), to_zero, guard=1),  # type: ignore[arg-type]
            TypeError,
            "RewriteRule guard must be callable or None, got int.",
        ),
        (
            lambda: RewriteRule(WildcardPattern(), to_zero, name=1),  # type: ignore[arg-type]
            TypeError,
            "RewriteRule name must be a str or None, got int.",
        ),
        (
            lambda: FiredRule(0, 1),  # type: ignore[arg-type]
            TypeError,
            "FiredRule name must be a str or None, got int.",
        ),
    ],
)
def test_an_argument_of_the_wrong_type_is_refused(
    build: Callable[[], object], error: type[Exception], message: str
) -> None:
    """Test the binding's argument checks and their messages."""
    with pytest.raises(error, match=_escape(message)):
        build()


def _escape(message: str) -> str:
    return re.escape(message)


@pytest.mark.parametrize(
    "pattern",
    [
        LogicalExpressionPattern(None, ()),
        LogicalExpressionPattern(None, (WildcardPattern(),)),
        PiecewiseExpressionPattern((), WildcardPattern()),
        AlternativesPattern(()),
    ],
    ids=["no_operand", "one_operand", "no_case", "no_alternative"],
)
def test_a_degenerate_pattern_builds_and_matches_nothing(pattern: Pattern) -> None:
    """Test the degenerate patterns build, and match no expression (D-S5-5)."""
    a, b = _reference("a"), _reference("b")
    candidates = [
        a,
        LiteralExpression(True),
        a.logical_and(b),
        PiecewiseExpression((LiteralExpression(True),), (a,), b),
    ]

    assert all(pattern.match(candidate) is None for candidate in candidates)


def test_a_call_pattern_compares_callees() -> None:
    """Test a call pattern of a built-in's name matches the built-in call."""
    a = _reference("a")

    assert CallExpressionPattern("sin", None).match(Expression.call("sin", a)) == (
        MatchBindings()
    )
    assert CallExpressionPattern("f", None).match(Expression.call("g", a)) is None


def test_match_refuses_a_value_that_is_not_an_expression() -> None:
    """Test ``match`` of a non-expression raises ``TypeError``."""
    with pytest.raises(TypeError, match=r"Pattern\.match expression must be an"):
        WildcardPattern().match(1)  # type: ignore[arg-type]


# =============================================================================
# Capture
# =============================================================================


def test_a_capture_compares_and_hashes_by_identity() -> None:
    """Test a capture equals only itself, whatever its name."""
    x, other_x = Capture("x"), Capture("x")

    assert x == x  # noqa: PLR0124
    assert x != other_x
    assert len({x, other_x, x}) == 2


@pytest.mark.parametrize("name", ["x", "", "a b"])
def test_a_capture_keeps_its_name(name: str) -> None:
    """Test ``name``, ``str`` and ``repr`` of a capture."""
    capture = Capture(name)

    assert capture.name == name
    assert str(capture) == name
    assert repr(capture) == f"Capture({name!r})"


# =============================================================================
# MatchBindings
# =============================================================================


def test_bindings_read_by_capture() -> None:
    """Test ``get``, ``[]``, ``has``, ``in``, ``len``, iteration and ``items``."""
    x, y, unbound = Capture("x"), Capture("y"), Capture("x")
    left, right = _reference("a"), LiteralExpression(2)
    pattern = BinaryExpressionPattern(None, CapturePattern(x), CapturePattern(y))

    bindings = pattern.match(BinaryExpression(BinaryOperation.ADD, left, right))

    assert bindings is not None
    assert bindings.get(x) is left
    assert bindings[y] is right
    assert bindings.get(unbound) is None
    assert bindings.get("x") is None
    assert bindings.has(x) and x in bindings
    assert not bindings.has(unbound) and unbound not in bindings
    assert len(bindings) == 2
    assert list(bindings) == [x, y]
    assert bindings.items() == ((x, left), (y, right))
    with pytest.raises(KeyError, match="capture `x` is not bound"):
        bindings[unbound]


def test_bindings_order_follows_the_matching_order() -> None:
    """Test inner captures bind before the capture around them."""
    outer, inner = Capture("outer"), Capture("inner")
    pattern = CapturePattern(outer, UnaryExpressionPattern(None, CapturePattern(inner)))

    bindings = pattern.match(-_reference("a"))

    assert bindings is not None
    assert list(bindings) == [inner, outer]


def test_bindings_compare_structurally_and_ignore_order() -> None:
    """Test equality and hashing of bindings from separate matches."""
    x = Capture("x")
    a = Identifier("a")

    first = CapturePattern(x).match(IdentifierExpression(a) + 1)
    second = CapturePattern(x).match(IdentifierExpression(a) + 1)

    assert first is not None and second is not None
    assert first is not second
    assert first == second
    assert hash(first) == hash(second)
    assert first != CapturePattern(x).match(LiteralExpression(1))


def test_no_public_constructor_binds_a_capture() -> None:
    """Test ``MatchBindings()`` and ``empty()`` bind nothing, and arguments fail."""
    assert MatchBindings().is_empty()
    assert MatchBindings.empty() == MatchBindings()
    with pytest.raises(TypeError, match="only a match produces bindings"):
        MatchBindings(((Capture("x"), LiteralExpression(1)),))  # type: ignore[call-arg]


def test_bindings_are_falsy_when_empty() -> None:
    """Test bindings of no capture are falsy, as an empty container is.

    A match is tested with ``is not None``: a pattern without captures
    matches with empty bindings.
    """
    bindings = WildcardPattern().match(LiteralExpression(1))

    assert bindings is not None
    assert not bindings
    assert CapturePattern(Capture("x")).match(LiteralExpression(1))


def test_bindings_repr_lists_the_pairs() -> None:
    """Test the ``repr`` of bindings."""
    x = Capture("x")

    bindings = CapturePattern(x).match(LiteralExpression(1))

    assert repr(bindings) == "MatchBindings({Capture('x'): LiteralExpression(1)})"
    assert repr(MatchBindings()) == "MatchBindings({})"


def test_bindings_do_not_pickle() -> None:
    """Test ``MatchBindings`` refuses to pickle (D-S5-13)."""
    with pytest.raises(TypeError, match="only a match produces bindings"):
        pickle.dumps(MatchBindings())


def test_bound_objects_are_the_objects_of_the_matched_tree() -> None:
    """Test a match binds the node objects of the tree it matched."""
    x = Capture("x")
    operand = _reference("a") * 2
    expression = UnaryExpression(UnaryOperation.NEGATE, -operand)
    pattern = UnaryExpressionPattern(
        None, UnaryExpressionPattern(None, CapturePattern(x))
    )

    bindings = pattern.match(expression)

    assert bindings is not None
    assert bindings[x] is operand


def test_a_rebuilt_node_gets_one_object() -> None:
    """Test callbacks at a node the walk rebuilt see one object, built once."""
    seen: list[Expression] = []

    def record(expression: Expression) -> bool:
        seen.append(expression)
        return False

    a = _reference("a")
    expression = (a + 0) * 2
    rules = [_build_add_zero_rule(), RewriteRule(PredicatePattern(record), to_zero)]
    rules.append(RewriteRule(PredicatePattern(record), to_zero))

    output = apply_rewrite_rules(expression, rules)

    rebuilt = [node for node in seen if isinstance(node, BinaryExpression)]
    assert len(rebuilt) == 2
    assert rebuilt[0] is rebuilt[1]
    assert rebuilt[0] is output
    assert isinstance(output, BinaryExpression)
    assert output.left is a


# =============================================================================
# Callbacks
# =============================================================================


def test_callbacks_receive_the_right_objects_and_are_read_by_truthiness() -> None:
    """Test predicates, guards and rewrites, and truthy results (D-S5-6)."""
    x = Capture("x")
    calls: list[object] = []

    def predicate(expression: Expression) -> object:
        calls.append(expression)
        return [1]

    def guard(bindings: MatchBindings) -> object:
        calls.append(bindings)
        return "yes"

    def rewrite(bindings: MatchBindings) -> Expression:
        calls.append(bindings)
        return LiteralExpression(9)

    expression = _reference("a")
    rule = RewriteRule(CapturePattern(x, PredicatePattern(predicate)), rewrite, guard)

    result = rule.apply(expression)

    assert result == LiteralExpression(9)
    assert calls[0] is expression
    assert isinstance(calls[1], MatchBindings)
    assert calls[1] is calls[2]
    assert calls[1][x] is expression
    assert (
        RewriteRule(WildcardPattern(), to_zero, guard=lambda _: 0).apply(expression)
        is None
    )


def test_a_truth_value_error_is_the_callbacks_error() -> None:
    """Test an exception from ``__bool__`` of a result is the callback's error."""

    class _Undecided:
        def __bool__(self) -> bool:
            raise ArithmeticError("undecided")

    with pytest.raises(ArithmeticError, match="undecided"):
        PredicatePattern(lambda _: _Undecided()).match(LiteralExpression(1))


@pytest.mark.parametrize(
    "run",
    [
        lambda rule, expression: rule.pattern.match(expression),
        lambda rule, expression: rule.apply(expression),
        apply_rewrite_rule,
    ],
    ids=["match", "apply", "apply_rewrite_rule"],
)
def test_the_same_exception_object_propagates(
    run: Callable[[RewriteRule, Expression], object],
) -> None:
    """Test a callback's exception propagates unchanged (D-S5-7)."""
    failure = LookupError("no")

    def predicate(expression: Expression) -> bool:
        raise failure

    rule = RewriteRule(PredicatePattern(predicate), to_zero)

    with pytest.raises(LookupError) as caught:
        run(rule, LiteralExpression(1))

    assert caught.value is failure


def test_an_exception_that_is_not_an_exception_passes_through_a_walk() -> None:
    """Test ``KeyboardInterrupt`` from a callback is never wrapped (D-S5-7)."""

    def interrupt(bindings: MatchBindings) -> Expression:
        raise KeyboardInterrupt

    rule = RewriteRule(WildcardPattern(), interrupt)

    with pytest.raises(KeyboardInterrupt):
        apply_rewrite_rules(LiteralExpression(1), [rule])
    with pytest.raises(KeyboardInterrupt):
        RewriteRuleApplier([rule]).execute(LiteralExpression(1))


def test_nested_matches_and_walks_inside_a_callback_work() -> None:
    """Test a callback may match and rewrite other trees."""
    x = Capture("x")
    inner_rule = _build_add_zero_rule()

    def rewrite(bindings: MatchBindings) -> Expression:
        operand = bindings[x]
        assert CapturePattern(Capture("y")).match(operand) is not None
        return apply_rewrite_rules(operand + 0, [inner_rule])

    rule = RewriteRule(UnaryExpressionPattern(None, CapturePattern(x)), rewrite)
    a = _reference("a")

    assert apply_rewrite_rules(-a, [rule]) is a


def test_a_rewrite_result_that_is_no_expression_raises_type_error() -> None:
    """Test the rewrite-result ``TypeError``, naming the rule (D-S5-8)."""
    rule = RewriteRule(WildcardPattern(), lambda _: None, name="n")  # type: ignore[arg-type,return-value]
    unnamed = RewriteRule(WildcardPattern(), lambda _: 1)  # type: ignore[arg-type,return-value]
    partial = RewriteRule.new_partial(WildcardPattern(), lambda _: 1, name="p")  # type: ignore[arg-type,return-value]

    with pytest.raises(
        TypeError,
        match=r"RewriteRule 'n' rewrite must return an Expression, got NoneType\.",
    ):
        rule.apply(LiteralExpression(1))
    with pytest.raises(TypeError, match="RewriteRule '<unnamed>' rewrite must return"):
        unnamed.apply(LiteralExpression(1))
    with pytest.raises(TypeError, match="an Expression or None, got int"):
        partial.apply(LiteralExpression(1))


def test_a_partial_rule_declines_with_none() -> None:
    """Test ``new_partial``'s rewrite may return ``None`` to decline."""
    declining = RewriteRule.new_partial(WildcardPattern(), lambda _: None)
    firing = RewriteRule(WildcardPattern(), to_zero)

    assert declining.apply(LiteralExpression(1)) is None
    assert apply_rewrite_rules(LiteralExpression(1), [declining, firing]) == (
        LiteralExpression(0)
    )


def test_a_rewrite_returning_the_matched_node_declines() -> None:
    """Test a result that is the matched node itself declines (D-S5-8)."""
    x = Capture("x")
    identity = RewriteRule(CapturePattern(x), _keep(x))
    expression = _reference("a") + 1

    assert identity.apply(expression) is None
    assert apply_rewrite_rules(expression, [identity]) is expression
    assert apply_rewrite_rules(
        expression, [identity, RewriteRule(x_is_sum(), to_zero)]
    ) == LiteralExpression(0)


def x_is_sum() -> Pattern:
    """Return the pattern of any sum."""
    return BinaryExpressionPattern(
        BinaryOperation.ADD, WildcardPattern(), WildcardPattern()
    )


def test_guards_run_in_order_and_the_first_refusal_stops_the_rest() -> None:
    """Test ``with_guard`` appends guards, which run in order."""
    calls: list[str] = []

    def guard(name: str, verdict: bool) -> Callable[[MatchBindings], bool]:
        def check(bindings: MatchBindings) -> bool:
            calls.append(name)
            return verdict

        return check

    base = RewriteRule(WildcardPattern(), to_zero, guard=guard("first", True))
    rule = base.with_guard(guard("second", False)).with_guard(guard("third", True))

    assert rule.apply(LiteralExpression(1)) is None
    assert calls == ["first", "second"]
    assert len(rule.guards) == 3
    assert len(base.guards) == 1
    assert rule.with_name("renamed").name == "renamed"
    assert rule.name is None


# =============================================================================
# The walk
# =============================================================================


def test_a_shared_subtree_is_rewritten_once() -> None:
    """Test a node that occurs twice has its rule tried once (D-S5-10)."""
    calls: list[Expression] = []

    def count(expression: Expression) -> bool:
        calls.append(expression)
        return True

    shared = _reference("s") * 3
    rule = RewriteRule(
        BinaryExpressionPattern(
            BinaryOperation.MULTIPLY, PredicatePattern(count), LiteralPattern(3)
        ),
        to_zero,
    )

    output = apply_rewrite_rules(shared + shared, [rule])

    assert len(calls) == 1
    assert output == LiteralExpression(0) + LiteralExpression(0)


def test_the_input_comes_back_itself_when_nothing_fires() -> None:
    """Test the walk returns its input object when no rule fires."""
    expression = _reference("a") * 2 + 1

    assert apply_rewrite_rules(expression, [_build_add_zero_rule()]) is expression


def test_a_deep_tree_rewrites() -> None:
    """Test a 20,000-level tree rewrites on the walk's own stack."""
    a = _reference("a")
    expression: Expression = a
    for _ in range(20_000):
        expression = expression + 0

    assert apply_rewrite_rules(expression, [_build_add_zero_rule()]) is a


def test_a_pattern_deeper_than_the_recursion_limit_raises() -> None:
    """Test a pattern deeper than the limit raises ``RecursionError`` (D-S5-15)."""
    pattern: Pattern = WildcardPattern()
    for _ in range(sys.getrecursionlimit() + 1):
        pattern = UnaryExpressionPattern(None, pattern)
    rule = RewriteRule(pattern, to_zero)

    with pytest.raises(RecursionError, match="levels deep"):
        pattern.match(LiteralExpression(1))
    with pytest.raises(RecursionError, match="levels deep"):
        rule.apply(LiteralExpression(1))
    with pytest.raises(RecursionError, match="levels deep"):
        apply_rewrite_rules(LiteralExpression(1), [rule])


def test_a_walk_refuses_a_rule_that_is_not_a_rule() -> None:
    """Test a rule list item that is not a ``Rule`` raises ``TypeError``."""
    with pytest.raises(TypeError, match="apply_rewrite_rules rules must be Rules"):
        apply_rewrite_rules(LiteralExpression(1), [1])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="apply_rewrite_rules expression must be"):
        apply_rewrite_rules(1, [])  # type: ignore[arg-type]


# =============================================================================
# Walk errors
# =============================================================================


def test_a_failing_callback_raises_rewrite_callback_error() -> None:
    """Test the callback error's class, rule, text and cause (D-S5-11)."""
    failure = ValueError("boom")

    def rewrite(bindings: MatchBindings) -> Expression:
        raise failure

    rules = [_build_add_zero_rule(), RewriteRule(WildcardPattern(), rewrite, name="n")]

    with pytest.raises(RewriteCallbackError) as caught:
        apply_rewrite_rules(LiteralExpression(1), rules)

    error = caught.value
    assert isinstance(error, RewriteError)
    assert isinstance(error, RuntimeError)
    assert str(error) == "rewrite rule 1 (n) failed"
    assert (error.rule_index, error.rule_name) == (1, "n")
    assert error.__cause__ is failure


def test_a_refused_rebuild_blames_the_rule_that_rewrote_the_child() -> None:
    """Test the rebuild error names the rule behind the refused child."""
    condition = _reference("c")
    expression = PiecewiseExpression(
        (condition.equals(1),), (LiteralExpression(1),), LiteralExpression(2)
    )
    to_number = RewriteRule(
        BinaryExpressionPattern(
            BinaryOperation.EQUAL, WildcardPattern(), WildcardPattern()
        ),
        lambda _: LiteralExpression(3),
        name="to a number",
    )

    with pytest.raises(RewriteRebuildError) as caught:
        apply_rewrite_rules(expression, [_build_add_zero_rule(), to_number])

    error = caught.value
    assert str(error) == "rebuilding a node after rewrite rule 1 (to a number) failed"
    assert (error.rule_index, error.rule_name) == (1, "to a number")
    assert isinstance(error.__cause__, ValueError)


def test_a_rewrite_error_pickles() -> None:
    """Test the walk's errors keep their fields through a pickle."""
    error = RewriteCallbackError("rewrite rule 0 failed", 0, None)

    loaded = pickle.loads(pickle.dumps(error))

    assert type(loaded) is RewriteCallbackError
    assert (str(loaded), loaded.rule_index, loaded.rule_name) == (
        "rewrite rule 0 failed",
        0,
        None,
    )


# =============================================================================
# Rule
# =============================================================================


class _DoubleLiteral(Rule):
    """Replace each non-zero integer literal with its double."""

    def __init__(self, label: str | None = None) -> None:
        super().__init__()
        self.label = label
        self.calls = 0

    def apply(self, expression: Expression) -> Expression | None:  # type: ignore[explicit-override]
        self.calls += 1
        if isinstance(expression, LiteralExpression) and expression.value:
            return LiteralExpression(2 * expression.value)
        return None

    @property
    def name(self) -> str | None:  # type: ignore[explicit-override]
        return self.label


def test_rule_abstract_methods_are_enforced() -> None:
    """Test a ``Rule`` subclass without ``apply`` cannot be built."""

    class _Incomplete(Rule):
        pass

    with pytest.raises(TypeError, match="abstract"):
        _Incomplete()  # type: ignore[abstract]


def test_a_python_rule_is_driven_from_rust_mixed_with_rewrite_rules() -> None:
    """Test a Python rule and a ``RewriteRule`` in one walk and one pass."""
    python_rule = _DoubleLiteral("double")
    a = _reference("a")
    applier = RewriteRuleApplier([_build_add_zero_rule(), python_rule])

    result = applier.execute((a + 0) * 3)

    assert result.output == a * 6
    assert isinstance(result.output, BinaryExpression)
    assert result.output.left is a
    assert applier.fired == (FiredRule(0, "x + 0 -> x"), FiredRule(1, "double"))
    assert [diagnostic.message_text for diagnostic in result.diagnostics] == [
        'applied rewrite rule "x + 0 -> x"',
        'applied rewrite rule "double"',
    ]


def test_a_python_rule_name_appears_in_errors() -> None:
    """Test a failing Python rule is named in the walk's error."""

    class _Failing(_DoubleLiteral):
        def apply(self, expression: Expression) -> Expression | None:  # type: ignore[explicit-override]
            raise OSError("disk")

    with pytest.raises(RewriteCallbackError, match=r"rewrite rule 0 \(failing\)"):
        apply_rewrite_rules(LiteralExpression(1), [_Failing("failing")])


def test_a_python_rule_result_of_another_type_raises_type_error() -> None:
    """Test a Python rule returning neither an expression nor ``None``."""

    class _Wrong(_DoubleLiteral):
        def apply(self, expression: Expression) -> Expression | None:  # type: ignore[explicit-override]
            return 1  # type: ignore[return-value]

    with pytest.raises(RewriteCallbackError) as caught:
        apply_rewrite_rules(LiteralExpression(1), [_Wrong()])

    assert isinstance(caught.value.__cause__, TypeError)
    assert "must return an Expression or None, got int" in str(caught.value.__cause__)


def test_a_python_rule_name_is_read_once_per_walk() -> None:
    """Test the walk reads ``name`` once, however often the rule fires."""
    reads = 0

    class _Counted(_DoubleLiteral):
        @property
        def name(self) -> str | None:  # type: ignore[explicit-override]
            nonlocal reads
            reads += 1
            return "counted"

    rule = _Counted()
    expression = LiteralExpression(1) + LiteralExpression(2) + LiteralExpression(3)

    apply_rewrite_rules(expression, [rule])

    assert reads == 1
    assert rule.calls >= 5


# =============================================================================
# The applier
# =============================================================================


def test_the_applier_is_registered_and_builds_with_no_rules() -> None:
    """Test ``CompilerPass.create`` builds an applier of no rules (D-S5-12)."""
    applier = CompilerPass.create("fhy_core.symbolic.expression.apply_rewrite_rules")

    assert isinstance(applier, RewriteRuleApplier)
    assert applier.rules == ()
    assert applier.fired == ()
    assert (
        RewriteRuleApplier.get_pass_description()
        == "Apply a sequence of rewrite rules bottom-up over an expression tree."
    )


def test_the_applier_records_firings_after_a_success_and_a_failure() -> None:
    """Test ``fired`` holds the last run's firings, also before a failure."""
    failure = RuntimeError("late")

    def fail(expression: Expression) -> bool:
        if isinstance(expression, BinaryExpression) and expression.operation is (
            BinaryOperation.MULTIPLY
        ):
            raise failure
        return False

    applier = RewriteRuleApplier(
        [_build_add_zero_rule(), RewriteRule(PredicatePattern(fail), to_zero)]
    )
    a = _reference("a")

    applier.execute(a + 0)
    assert applier.fired == (FiredRule(0, "x + 0 -> x"),)

    with pytest.raises(PassExecutionError) as caught:
        applier.execute((a + 0) * 2)

    assert applier.fired == (FiredRule(0, "x + 0 -> x"),)
    assert isinstance(caught.value.__cause__, RewriteCallbackError)
    assert caught.value.__cause__.__cause__ is failure
    assert applier.diagnostics[0].message_text == 'applied rewrite rule "x + 0 -> x"'


def test_the_applier_escapes_names_as_the_core_does() -> None:
    """Test the diagnostic writes the name as Rust's ``Debug`` writes a string."""
    applier = RewriteRuleApplier([_build_add_zero_rule('say "hi"\n')])

    result = applier.execute(_reference("a") + 0)

    assert result.diagnostics[0].message_text == (
        'applied rewrite rule "say \\"hi\\"\\n"'
    )


def test_the_applier_reports_a_change_by_identity() -> None:
    """Test ``changed`` is whether the output is another object."""
    x = Capture("x")
    fresh_equal = RewriteRule(
        CapturePattern(x, LiteralPattern(1)), lambda _: LiteralExpression(1)
    )
    expression = LiteralExpression(1)

    result = RewriteRuleApplier([fresh_equal]).execute(expression)

    assert result.output == expression
    assert result.output is not expression
    assert result.changed


# =============================================================================
# Pickles and reprs
# =============================================================================


def test_patterns_pickle_as_calls_of_their_class() -> None:
    """Test patterns round-trip, and a shared capture stays shared (D-S5-13)."""
    x = Capture("x")
    pattern = BinaryExpressionPattern(
        BinaryOperation.SUBTRACT,
        CapturePattern(x),
        AlternativesPattern((CapturePattern(x), PredicatePattern(is_literal))),
    )

    loaded = pickle.loads(pickle.dumps(pattern))

    assert type(loaded) is BinaryExpressionPattern
    assert isinstance(loaded.left, CapturePattern)
    assert isinstance(loaded.right, AlternativesPattern)
    first = loaded.right.alternatives[0]
    assert isinstance(first, CapturePattern)
    assert loaded.left.capture is first.capture
    assert loaded.left.capture is not x
    a = _reference("a")
    assert loaded.match(a - a) is not None
    assert pickle.loads(pickle.dumps(LiteralPattern("05"))) == LiteralPattern("05")


def test_rules_pickle_with_module_level_callables() -> None:
    """Test a rule of module-level callables round-trips."""
    rule = RewriteRule(PredicatePattern(is_literal), to_zero, guard=always, name="z")
    partial = RewriteRule.new_partial(WildcardPattern(), to_zero)

    loaded = pickle.loads(pickle.dumps(rule))
    loaded_partial = pickle.loads(pickle.dumps(partial))

    assert type(loaded) is RewriteRule
    assert loaded.name == "z"
    assert loaded.guards == (always,)
    assert loaded.rewrite is to_zero
    assert loaded.apply(LiteralExpression(4)) == LiteralExpression(0)
    assert loaded_partial.apply(LiteralExpression(4)) == LiteralExpression(0)
    assert pickle.loads(pickle.dumps(FiredRule(2, "n"))) == FiredRule(2, "n")


def test_a_lambda_predicate_fails_to_pickle_as_pickle_fails() -> None:
    """Test a pattern pickles only if its callable does."""
    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(PredicatePattern(lambda _: True))


def test_reprs() -> None:
    """Test the dataclass-style reprs (D-S5-14)."""
    x = Capture("x")
    pattern = BinaryExpressionPattern(None, CapturePattern(x), LiteralPattern(1))

    assert repr(pattern) == (
        "BinaryExpressionPattern(operation=None, left=CapturePattern("
        "capture=Capture('x'), sub_pattern=WildcardPattern()), "
        "right=LiteralPattern(value=1))"
    )
    assert repr(RewriteRule(pattern, to_zero, name="n")) == (
        f"RewriteRule(pattern={pattern!r}, name='n')"
    )
    assert repr(FiredRule(0, None)) == "FiredRule(rule_index=0, name=None)"


def test_patterns_compare_and_hash_by_their_fields() -> None:
    """Test pattern equality and hashing follow their field objects."""
    x = Capture("x")

    assert CapturePattern(x) == CapturePattern(x)
    assert hash(CapturePattern(x)) == hash(CapturePattern(x))
    assert CapturePattern(x) != CapturePattern(Capture("x"))
    assert LiteralPattern(1) != LiteralPattern("1")
    assert WildcardPattern() != IdentifierPattern()
    assert BinaryExpressionPattern(
        BinaryOperation.ADD, WildcardPattern(), WildcardPattern()
    ) == BinaryExpressionPattern("add", WildcardPattern(), WildcardPattern())  # type: ignore[arg-type]


def test_unary_operation_fields_are_members() -> None:
    """Test an operation given by value is stored as its member."""
    pattern = UnaryExpressionPattern("negate", WildcardPattern())  # type: ignore[arg-type]

    assert pattern.operation is UnaryOperation.NEGATE
    assert (
        pattern.match(UnaryExpression(UnaryOperation.NEGATE, _reference("a")))
        is not None
    )
