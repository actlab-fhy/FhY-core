"""Generate golden pattern-matching and rewrite cases from the Python oracle.

Trees are given by their wire dict (``serialize_to_dict``) with every
identifier written as ``{"slot": name}``; the Rust replay mints one
identifier per slot and case, as ``generate_expression_cases.py`` does.

Patterns are declarative specs, one JSON object per pattern with a
``"kind"``:

* ``wildcard``;
* ``capture`` with ``name`` and ``sub_pattern``;
* ``literal`` with ``value``: ``null`` for any literal, or the raw literal
  value as the wire form writes it (a JSON Boolean, integer, float, or a
  literal text);
* ``identifier`` with ``slot``: ``null`` for any identifier, or a slot;
* ``unary`` with ``operation`` (wire name or ``null``) and ``operand``;
* ``binary`` with ``operation``, ``left`` and ``right``;
* ``piecewise`` with ``cases`` (``null`` or a list of ``[condition, value]``
  pattern pairs) and ``otherwise``;
* ``call`` with ``function_name`` and ``arguments`` (each ``null`` or given);
* ``alternatives`` with ``alternatives``;
* ``predicate`` with ``predicate``, one of a fixed set both sides implement
  identically: ``is_literal``, ``is_identifier``, ``has_children``, and
  ``fail`` (the predicate fails).

A rule is ``{"pattern", "guard", "rewrite", "name"}``. A guard is ``null``
or ``{"kind": ...}`` with kind ``always``, ``never``, ``fail``, or
``bound_is_literal`` with ``capture`` (whether that capture is bound to a
literal; an unbound capture fails). A rewrite is ``{"kind": ...}`` with kind
``capture`` (the expression bound to ``capture``; an unbound capture
fails), ``literal`` with ``value``, ``negate_capture`` (the negation of the
expression bound to ``capture``), or ``fail``.

``match_cases`` record ``match_pattern`` on a tree: ``null`` for no match,
``{"bindings": [[name, wire], ...]}`` in binding order, or
``{"error": "callback"}`` when a predicate fails. ``rewrite_cases`` record
``apply_rewrite_rules``: ``{"output": wire, "changed": bool, "fired":
[{"rule_index", "name"}, ...]}`` with the firings in walk order and
``changed`` meaning the output is not the input object, or
``{"error": {"kind": "callback", "rule_index": i}}`` naming the rule whose
callback failed, or ``{"error": {"kind": "non_boolean_condition_literal"}}``
when a rebuilt piecewise gets a non-Boolean literal condition. A firing is a
rule's rewrite returning, which in the oracle is the moment the applier
takes that rule's result.

No literal pattern holds a NaN or a text outside the literal grammar (the
Rust literal type cannot hold such text, and JSON cannot carry a NaN).

The Rust replay is ``rust/fhy-core/tests/pattern_equivalence.rs``.

Run from the repository root:

    uv run --no-sync python rust/fhy-core/tests/golden/generate_pattern_cases.py

This overwrites ``rust/fhy-core/tests/golden/pattern_cases.json``.
``--random-count`` sets how many random trees are drawn, ``--max-ops`` sets
the largest leaf count of a random tree, and ``--seed`` seeds the draws. The
ignored expanded-corpus test reads a corpus written with other options from
the file named in ``FHY_PATTERN_CORPUS``.
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from _golden_support import add_corpus_arguments, build_provenance, write_document

from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import PassExecutionError
from fhy_core.symbolic.expression import (
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.symbolic.expression.pattern import (
    AlternativesPattern,
    BinaryExpressionPattern,
    CallExpressionPattern,
    CapturePattern,
    IdentifierPattern,
    LiteralPattern,
    MatchBindings,
    Pattern,
    PiecewiseExpressionPattern,
    PredicatePattern,
    RewriteRule,
    UnaryExpressionPattern,
    WildcardPattern,
    apply_rewrite_rules,
    match_pattern,
)

GENERATOR_COMMAND = (
    "uv run --no-sync python rust/fhy-core/tests/golden/generate_pattern_cases.py"
)

_RANDOM_SEED = 20260922
_RANDOM_COUNT = 30
_RANDOM_MAX_LEAVES = 8

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_OUTPUT = Path(__file__).resolve().with_name("pattern_cases.json")

_SLOTS = ("x", "y", "z", "p", "q")
_FUNCTION_NAMES = ("f", "g", "h")
_CAPTURE_NAMES = ("a", "b", "x")
_MAX_CASES = 2
_MAX_ARGUMENTS = 3

JsonDict = dict[str, Any]


class _ScriptError(Exception):
    """A scripted callback failure, carrying the index of its rule."""

    def __init__(self, rule_index: int) -> None:
        super().__init__(f"scripted failure in rule {rule_index}")
        self.rule_index = rule_index


# =============================================================================
# Slots and wire dicts
# =============================================================================


class _SlotTable:
    """Identifiers minted for the slots of one case, in both directions."""

    def __init__(self) -> None:
        self._by_slot = {slot: Identifier(slot) for slot in _SLOTS}
        self._by_id = {
            identifier.id: slot for slot, identifier in self._by_slot.items()
        }

    def resolve(self, slot: str) -> Identifier:
        """Return the identifier of ``slot``."""
        return self._by_slot[slot]

    def bind(self, wire: Any) -> Any:
        """Return ``wire`` with every slot placeholder replaced by its identifier."""
        if isinstance(wire, dict):
            if set(wire) == {"slot"}:
                identifier = self.resolve(wire["slot"])
                return {"id": identifier.id, "name_hint": identifier.name_hint}
            return {key: self.bind(value) for key, value in wire.items()}
        if isinstance(wire, list):
            return [self.bind(item) for item in wire]
        return wire

    def unbind(self, wire: Any) -> Any:
        """Return ``wire`` with every identifier replaced by its slot placeholder."""
        if isinstance(wire, dict):
            if set(wire) == {"id", "name_hint"}:
                return {"slot": self._by_id[wire["id"]]}
            return {key: self.unbind(value) for key, value in wire.items()}
        if isinstance(wire, list):
            return [self.unbind(item) for item in wire]
        return wire

    def decode(self, wire: JsonDict) -> Expression:
        """Deserialize a slot-form wire dict."""
        return Expression.deserialize_from_dict(self.bind(wire))

    def encode(self, expression: Expression) -> JsonDict:
        """Serialize ``expression`` to a slot-form wire dict."""
        wire: JsonDict = self.unbind(expression.serialize_to_dict())
        return wire


def _identifier(slot: str) -> JsonDict:
    """Return the wire dict of a reference to ``slot``."""
    return {
        "__type__": "identifier_expression",
        "__data__": {"identifier": {"slot": slot}},
    }


def _literal(value: bool | int | float | str) -> JsonDict:
    """Return the wire dict of a literal."""
    return {"__type__": "literal_expression", "__data__": {"value": value}}


def _unary(operation: str, operand: JsonDict) -> JsonDict:
    """Return the wire dict of a unary node."""
    return {
        "__type__": "unary_expression",
        "__data__": {"operation": operation, "operand": operand},
    }


def _binary(operation: str, left: JsonDict, right: JsonDict) -> JsonDict:
    """Return the wire dict of a binary node."""
    return {
        "__type__": "binary_expression",
        "__data__": {"operation": operation, "left": left, "right": right},
    }


def _piecewise(cases: list[tuple[JsonDict, JsonDict]], otherwise: JsonDict) -> JsonDict:
    """Return the wire dict of a piecewise node."""
    return {
        "__type__": "piecewise_expression",
        "__data__": {
            "conditions": [condition for condition, _ in cases],
            "values": [value for _, value in cases],
            "otherwise": otherwise,
        },
    }


def _call(function_name: str, arguments: list[JsonDict]) -> JsonDict:
    """Return the wire dict of a call."""
    return {
        "__type__": "call_expression",
        "__data__": {"function_name": function_name, "arguments": arguments},
    }


def _children(wire: JsonDict) -> list[JsonDict]:
    """Return the children of a wire tree in visiting order."""
    data = wire["__data__"]
    kind = wire["__type__"]
    if kind == "unary_expression":
        return [data["operand"]]
    if kind == "binary_expression":
        return [data["left"], data["right"]]
    if kind == "piecewise_expression":
        interleaved = []
        for condition, value in zip(data["conditions"], data["values"], strict=True):
            interleaved += [condition, value]
        return [*interleaved, data["otherwise"]]
    if kind == "call_expression":
        return list(data["arguments"])
    return []


def _subtrees(wire: JsonDict) -> list[JsonDict]:
    """Return every subtree of a wire tree, pre-order."""
    found = [wire]
    for child in _children(wire):
        found += _subtrees(child)
    return found


# =============================================================================
# Pattern and rule specs
# =============================================================================


def _p_wildcard() -> JsonDict:
    return {"kind": "wildcard"}


def _p_capture(name: str, sub_pattern: JsonDict | None = None) -> JsonDict:
    return {
        "kind": "capture",
        "name": name,
        "sub_pattern": _p_wildcard() if sub_pattern is None else sub_pattern,
    }


def _p_literal(value: Any = None) -> JsonDict:
    return {"kind": "literal", "value": value}


def _p_identifier(slot: str | None = None) -> JsonDict:
    return {"kind": "identifier", "slot": slot}


def _p_unary(operation: str | None, operand: JsonDict) -> JsonDict:
    return {"kind": "unary", "operation": operation, "operand": operand}


def _p_binary(operation: str | None, left: JsonDict, right: JsonDict) -> JsonDict:
    return {"kind": "binary", "operation": operation, "left": left, "right": right}


def _p_piecewise(cases: list[list[JsonDict]] | None, otherwise: JsonDict) -> JsonDict:
    return {"kind": "piecewise", "cases": cases, "otherwise": otherwise}


def _p_call(function_name: str | None, arguments: list[JsonDict] | None) -> JsonDict:
    return {"kind": "call", "function_name": function_name, "arguments": arguments}


def _p_alternatives(*alternatives: JsonDict) -> JsonDict:
    return {"kind": "alternatives", "alternatives": list(alternatives)}


def _p_predicate(name: str) -> JsonDict:
    return {"kind": "predicate", "predicate": name}


def _rule(
    pattern: JsonDict,
    rewrite: JsonDict,
    guard: JsonDict | None = None,
    name: str | None = None,
) -> JsonDict:
    return {"pattern": pattern, "guard": guard, "rewrite": rewrite, "name": name}


def _r_capture(capture: str) -> JsonDict:
    return {"kind": "capture", "capture": capture}


def _r_literal(value: bool | int | float | str) -> JsonDict:
    return {"kind": "literal", "value": value}


def _r_negate(capture: str) -> JsonDict:
    return {"kind": "negate_capture", "capture": capture}


def _r_fail() -> JsonDict:
    return {"kind": "fail"}


def _g(kind: str, capture: str | None = None) -> JsonDict:
    guard: JsonDict = {"kind": kind}
    if capture is not None:
        guard["capture"] = capture
    return guard


# =============================================================================
# Building oracle objects from specs
# =============================================================================


def _build_predicate(name: str, rule_index: int) -> Callable[[Expression], bool]:
    """Return the named predicate."""
    if name == "is_literal":
        return lambda expression: isinstance(expression, LiteralExpression)
    if name == "is_identifier":
        return lambda expression: isinstance(expression, IdentifierExpression)
    if name == "has_children":
        return lambda expression: len(expression.get_visit_children()) > 0
    if name == "fail":

        def fail(_: Expression) -> bool:
            raise _ScriptError(rule_index)

        return fail
    raise AssertionError(f"unknown predicate {name}")


def _build_optional(value: Any, convert: Callable[[Any], Any]) -> Any:
    """Return ``None`` for ``None``, else ``convert(value)``."""
    return None if value is None else convert(value)


def _build_pattern(spec: JsonDict, table: _SlotTable, rule_index: int = -1) -> Pattern:
    """Return the oracle pattern of a spec; its predicates fail with ``rule_index``."""

    def build(child: JsonDict) -> Pattern:
        return _build_pattern(child, table, rule_index)

    def build_all(children: list[JsonDict]) -> tuple[Pattern, ...]:
        return tuple(build(child) for child in children)

    builders: dict[str, Callable[[], Pattern]] = {
        "wildcard": WildcardPattern,
        "capture": lambda: CapturePattern(spec["name"], build(spec["sub_pattern"])),
        "literal": lambda: LiteralPattern(spec["value"]),
        "identifier": lambda: IdentifierPattern(
            _build_optional(spec["slot"], table.resolve)
        ),
        "unary": lambda: UnaryExpressionPattern(
            _build_optional(spec["operation"], UnaryOperation), build(spec["operand"])
        ),
        "binary": lambda: BinaryExpressionPattern(
            _build_optional(spec["operation"], BinaryOperation),
            build(spec["left"]),
            build(spec["right"]),
        ),
        "piecewise": lambda: PiecewiseExpressionPattern(
            _build_optional(
                spec["cases"],
                lambda cases: tuple((build(c), build(v)) for c, v in cases),
            ),
            build(spec["otherwise"]),
        ),
        "call": lambda: CallExpressionPattern(
            spec["function_name"], _build_optional(spec["arguments"], build_all)
        ),
        "alternatives": lambda: AlternativesPattern(build_all(spec["alternatives"])),
        "predicate": lambda: PredicatePattern(
            _build_predicate(spec["predicate"], rule_index)
        ),
    }
    return builders[spec["kind"]]()


def _bound(bindings: MatchBindings, capture: str, rule_index: int) -> Expression:
    """Return the expression bound to ``capture``, failing when it is unbound."""
    if not bindings.has(capture):
        raise _ScriptError(rule_index)
    return bindings.get(capture)


def _build_guard(
    spec: JsonDict | None, rule_index: int
) -> Callable[[MatchBindings], bool] | None:
    """Return the oracle guard of a spec."""
    if spec is None:
        return None
    kind = spec["kind"]
    if kind == "always":
        return lambda _: True
    if kind == "never":
        return lambda _: False
    if kind == "bound_is_literal":
        capture = spec["capture"]
        return lambda bindings: isinstance(
            _bound(bindings, capture, rule_index), LiteralExpression
        )
    if kind == "fail":

        def fail(_: MatchBindings) -> bool:
            raise _ScriptError(rule_index)

        return fail
    raise AssertionError(f"unknown guard kind {kind}")


def _build_rewrite(
    spec: JsonDict, rule_index: int, table: _SlotTable
) -> Callable[[MatchBindings], Expression]:
    """Return the oracle rewrite of a spec."""
    kind = spec["kind"]
    if kind == "capture":
        return lambda bindings: _bound(bindings, spec["capture"], rule_index)
    if kind == "literal":
        return lambda _: table.decode(_literal(spec["value"]))
    if kind == "negate_capture":
        return lambda bindings: UnaryExpression(
            UnaryOperation.NEGATE, _bound(bindings, spec["capture"], rule_index)
        )
    if kind == "fail":

        def fail(_: MatchBindings) -> Expression:
            raise _ScriptError(rule_index)

        return fail
    raise AssertionError(f"unknown rewrite kind {kind}")


def _build_rules(
    specs: list[JsonDict], table: _SlotTable, fired: list[JsonDict]
) -> list[RewriteRule]:
    """Return the oracle rules of specs, logging each firing in ``fired``."""
    rules = []
    for index, spec in enumerate(specs):
        rewrite = _build_rewrite(spec["rewrite"], index, table)

        def logged(
            bindings: MatchBindings,
            rewrite: Callable[[MatchBindings], Expression] = rewrite,
            index: int = index,
            name: str | None = spec["name"],
        ) -> Expression:
            result = rewrite(bindings)
            fired.append({"rule_index": index, "name": name})
            return result

        rules.append(
            RewriteRule(
                pattern=_build_pattern(spec["pattern"], table, index),
                rewrite=logged,
                guard=_build_guard(spec["guard"], index),
                name=spec["name"],
            )
        )
    return rules


# =============================================================================
# Recording
# =============================================================================


def _record_match(name: str, tree: JsonDict, pattern: JsonDict) -> JsonDict:
    """Return one match case."""
    table = _SlotTable()
    expression = table.decode(tree)
    oracle_pattern = _build_pattern(pattern, table)
    result: JsonDict | None
    try:
        bindings = match_pattern(oracle_pattern, expression)
    except _ScriptError:
        result = {"error": "callback"}
    else:
        result = (
            None
            if bindings is None
            else {
                "bindings": [
                    [capture, table.encode(bound)]
                    for capture, bound in bindings.bindings.items()
                ]
            }
        )
    return {"name": name, "tree": tree, "pattern": pattern, "result": result}


def _record_rewrite(name: str, tree: JsonDict, rules: list[JsonDict]) -> JsonDict:
    """Return one rewrite case."""
    table = _SlotTable()
    expression = table.decode(tree)
    fired: list[JsonDict] = []
    oracle_rules = _build_rules(rules, table, fired)
    result: JsonDict
    try:
        output = apply_rewrite_rules(expression, oracle_rules)
    except PassExecutionError as error:
        cause = error.__cause__
        if isinstance(cause, _ScriptError):
            result = {"error": {"kind": "callback", "rule_index": cause.rule_index}}
        elif isinstance(
            cause, ValueError
        ) and "condition literal must be a boolean" in str(cause):
            result = {"error": {"kind": "non_boolean_condition_literal"}}
        else:
            raise
    else:
        result = {
            "output": table.encode(output),
            "changed": output is not expression,
            "fired": fired,
        }
    return {"name": name, "tree": tree, "rules": rules, "result": result}


# =============================================================================
# Hand-picked cases
# =============================================================================


def _hand_picked_match_cases() -> list[tuple[str, JsonDict, JsonDict]]:
    """Return match cases covering every pattern kind and its decisive shapes."""
    one, two = _literal(1), _literal(2)
    true, false = _literal(True), _literal(False)
    x, y = _identifier("x"), _identifier("y")
    sum_1_2 = _binary("add", one, two)
    repeated = _p_binary(None, _p_capture("x"), _p_capture("x"))
    two_case = _piecewise([(true, _literal(10)), (false, _literal(11))], _literal(12))
    cases: list[tuple[str, JsonDict, JsonDict]] = [
        ("wildcard_literal", one, _p_wildcard()),
        ("wildcard_compound", sum_1_2, _p_wildcard()),
        ("capture_compound", sum_1_2, _p_capture("x")),
        ("literal_any", _literal("1.50"), _p_literal()),
        ("literal_any_rejects_identifier", x, _p_literal()),
        ("literal_int_int", _literal(5), _p_literal(5)),
        ("literal_int_other_int", _literal(6), _p_literal(5)),
        ("literal_int_integer_text", _literal("5"), _p_literal(5)),
        ("literal_integer_text_int", _literal(5), _p_literal("5")),
        ("literal_integer_texts_differ", _literal("05"), _p_literal("5")),
        ("literal_integer_texts_same", _literal("05"), _p_literal("05")),
        ("literal_int_float", _literal(5.0), _p_literal(5)),
        ("literal_int_bool", true, _p_literal(1)),
        ("literal_bool_int", one, _p_literal(True)),
        ("literal_bool_bool", false, _p_literal(False)),
        ("literal_decimal_texts_differ", _literal("1.50"), _p_literal("1.5")),
        ("literal_float_decimal_text", _literal("1.5"), _p_literal(1.5)),
        ("literal_zero_negative_zero", _literal(-0.0), _p_literal(0.0)),
        ("literal_negative_zero_zero", _literal(0.0), _p_literal(-0.0)),
        ("literal_big_int", _literal(10**30), _p_literal(10**30)),
        ("literal_big_int_off_by_one", _literal(10**30 + 1), _p_literal(10**30)),
        ("identifier_any", x, _p_identifier()),
        ("identifier_same_slot", x, _p_identifier("x")),
        ("identifier_other_slot", y, _p_identifier("x")),
        ("identifier_rejects_literal", one, _p_identifier()),
        (
            "unary_any_operation",
            _unary("logical_not", true),
            _p_unary(None, _p_wildcard()),
        ),
        (
            "unary_wrong_operation",
            _unary("positive", one),
            _p_unary("negate", _p_wildcard()),
        ),
        (
            "unary_capture_operand",
            _unary("negate", x),
            _p_unary("negate", _p_capture("a")),
        ),
        (
            "binary_wrong_operation",
            sum_1_2,
            _p_binary("subtract", _p_wildcard(), _p_wildcard()),
        ),
        (
            "binary_left_then_right",
            sum_1_2,
            _p_binary("add", _p_capture("b"), _p_capture("a")),
        ),
        (
            "binding_order_is_completion_order",
            sum_1_2,
            _p_capture("outer", _p_binary(None, _p_capture("b"), _p_capture("a"))),
        ),
        ("nested_captures", one, _p_capture("outer", _p_capture("inner"))),
        ("repeated_capture_equal", _binary("subtract", one, one), repeated),
        ("repeated_capture_different", sum_1_2, repeated),
        (
            "repeated_capture_int_and_text",
            _binary("subtract", _literal(5), _literal("5")),
            repeated,
        ),
        (
            "repeated_capture_int_and_padded_text",
            _binary("add", _literal(5), _literal("05")),
            repeated,
        ),
        (
            "repeated_capture_int_and_float",
            _binary("add", one, _literal(1.0)),
            repeated,
        ),
        (
            "repeated_capture_decimal_texts",
            _binary("add", _literal("1.5"), _literal("1.50")),
            repeated,
        ),
        (
            "repeated_capture_zeros",
            _binary("add", _literal(0.0), _literal(-0.0)),
            repeated,
        ),
        ("repeated_capture_bool_and_int", _binary("add", true, one), repeated),
        ("repeated_capture_identifiers_same", _binary("add", x, x), repeated),
        ("repeated_capture_identifiers_different", _binary("add", x, y), repeated),
        (
            "piecewise_capture_order",
            two_case,
            _p_piecewise(
                [
                    [_p_capture("c1"), _p_capture("v1")],
                    [_p_capture("c2"), _p_capture("v2")],
                ],
                _p_capture("o"),
            ),
        ),
        ("piecewise_any_cases", two_case, _p_piecewise(None, _p_capture("o"))),
        (
            "piecewise_case_count_mismatch",
            two_case,
            _p_piecewise([[_p_wildcard(), _p_wildcard()]], _p_wildcard()),
        ),
        (
            "piecewise_repeated_capture_value_and_otherwise",
            _piecewise([(true, one)], _literal("01")),
            _p_piecewise([[_p_wildcard(), _p_capture("v")]], _p_capture("v")),
        ),
        ("piecewise_rejects_call", _call("f", []), _p_piecewise(None, _p_wildcard())),
        ("call_any_arguments", _call("f", [one, two]), _p_call("f", None)),
        ("call_empty_arguments_zero_ary", _call("f", []), _p_call("f", [])),
        ("call_empty_arguments_one_ary", _call("f", [one]), _p_call("f", [])),
        ("call_other_name", _call("g", [one]), _p_call("f", None)),
        ("call_any_name", _call("h", [one]), _p_call(None, [_p_capture("a")])),
        (
            "call_arity_mismatch",
            _call("f", [one]),
            _p_call("f", [_p_wildcard(), _p_wildcard()]),
        ),
        (
            "call_repeated_capture",
            _call("f", [x, x, one]),
            _p_call(None, [_p_capture("x"), _p_capture("x"), _p_literal(1)]),
        ),
        (
            "alternatives_first_wins",
            one,
            _p_alternatives(_p_capture("first", _p_literal()), _p_capture("second")),
        ),
        (
            "alternatives_fall_through",
            x,
            _p_alternatives(_p_capture("first", _p_literal()), _p_capture("second")),
        ),
        (
            "alternatives_discard_failed_captures",
            sum_1_2,
            _p_alternatives(
                _p_binary("add", _p_capture("failed", _p_literal(1)), _p_literal(99)),
                _p_capture("chosen"),
            ),
        ),
        (
            "alternatives_commit_without_backtracking",
            sum_1_2,
            _p_binary(
                None,
                _p_alternatives(_p_capture("x", _p_literal()), _p_wildcard()),
                _p_capture("x"),
            ),
        ),
        (
            "alternatives_skip_failing_predicate_after_a_match",
            one,
            _p_alternatives(_p_wildcard(), _p_predicate("fail")),
        ),
        ("predicate_is_literal", one, _p_predicate("is_literal")),
        ("predicate_is_literal_rejects", x, _p_predicate("is_literal")),
        (
            "predicate_has_children",
            sum_1_2,
            _p_capture("n", _p_predicate("has_children")),
        ),
        ("predicate_fails", one, _p_predicate("fail")),
        (
            "predicate_fails_under_a_binary",
            sum_1_2,
            _p_binary(None, _p_capture("a"), _p_predicate("fail")),
        ),
        (
            "predicate_after_a_failed_operand_is_not_run",
            sum_1_2,
            _p_binary(None, _p_literal(99), _p_predicate("fail")),
        ),
    ]
    return cases


def _hand_picked_rewrite_cases() -> list[tuple[str, JsonDict, list[JsonDict]]]:
    """Return rewrite cases covering the walk's order, identity, and failures."""
    zero, one, two = _literal(0), _literal(1), _literal(2)
    true, false = _literal(True), _literal(False)
    x, y = _identifier("x"), _identifier("y")
    x_plus_zero = _rule(
        _p_binary("add", _p_capture("x"), _p_literal(0)),
        _r_capture("x"),
        name="x + 0 -> x",
    )
    zero_plus_x = _rule(
        _p_binary("add", _p_literal(0), _p_capture("x")),
        _r_capture("x"),
        name="0 + x -> x",
    )
    x_times_one = _rule(
        _p_binary("multiply", _p_capture("x"), _p_literal(1)),
        _r_capture("x"),
        name="x * 1 -> x",
    )
    x_minus_x = _rule(
        _p_binary("subtract", _p_capture("x"), _p_capture("x")),
        _r_literal(0),
        name="x - x -> 0",
    )
    algebraic = [x_plus_zero, zero_plus_x, x_times_one, x_minus_x]
    double_not = _rule(
        _p_unary("logical_not", _p_unary("logical_not", _p_capture("x"))),
        _r_capture("x"),
        name="!!x -> x",
    )
    identity = _rule(_p_capture("x"), _r_capture("x"), name="x -> x")
    literal_identity = _rule(_p_capture("x", _p_literal()), _r_capture("x"))
    plus_zero_chain = x
    for _ in range(8):
        plus_zero_chain = _binary("add", plus_zero_chain, zero)
    four_nots = x
    for _ in range(4):
        four_nots = _unary("logical_not", four_nots)
    literal_markers = [
        _rule(_p_literal(value), _r_literal(value + 100), name=f"lit {value}")
        for value in (10, 11, 12, 13)
    ]
    return [
        ("no_rules", _binary("add", x, zero), []),
        ("x_plus_zero_at_root", _binary("add", x, zero), [x_plus_zero]),
        ("x_plus_zero_misses", _binary("add", x, one), [x_plus_zero]),
        (
            "x_plus_zero_in_subtree",
            _binary("multiply", _binary("add", x, zero), two),
            [x_plus_zero],
        ),
        (
            "bottom_up_single_pass",
            _binary("multiply", _binary("add", x, zero), one),
            algebraic,
        ),
        (
            "parent_sees_rewritten_children",
            _binary("add", _binary("add", x, zero), zero),
            algebraic,
        ),
        ("long_chain", plus_zero_chain, [x_plus_zero]),
        ("left_zero_addend", _binary("add", zero, y), algebraic),
        (
            "nested_neutral_operations",
            _binary(
                "subtract",
                _binary("multiply", _binary("add", x, zero), one),
                _binary("multiply", _binary("add", x, zero), one),
            ),
            algebraic,
        ),
        ("x_minus_x_distinct", _binary("subtract", x, y), [x_minus_x]),
        (
            "x_minus_x_equal_literals",
            _binary("subtract", _literal(5), _literal("05")),
            [x_minus_x],
        ),
        ("four_negations_one_walk", four_nots, [double_not]),
        ("identity_rule_at_leaf_root", one, [identity]),
        ("identity_rule_everywhere", _unary("negate", x), [identity]),
        ("literal_identity_below_root", _binary("add", one, two), [literal_identity]),
        ("literal_identity_without_literals", _binary("add", x, y), [literal_identity]),
        (
            "first_rule_wins",
            zero,
            [
                _rule(_p_wildcard(), _r_literal(101)),
                _rule(_p_wildcard(), _r_literal(202)),
            ],
        ),
        (
            "guard_refuses_then_next_rule",
            zero,
            [
                _rule(_p_wildcard(), _r_literal(101), guard=_g("never")),
                _rule(_p_wildcard(), _r_literal(202), guard=_g("always")),
            ],
        ),
        (
            "guard_sees_bindings",
            _binary("add", _unary("negate", one), _unary("negate", x)),
            [
                _rule(
                    _p_unary("negate", _p_capture("v")),
                    _r_literal(0),
                    guard=_g("bound_is_literal", "v"),
                )
            ],
        ),
        (
            "replacement_not_rewritten_again",
            zero,
            [_rule(_p_capture("x", _p_literal(0)), _r_negate("x"))],
        ),
        (
            "piecewise_interleaved_walk_order",
            _call(
                "f",
                [
                    _piecewise(
                        [(_binary("less", x, _literal(10)), _literal(11))], _literal(12)
                    ),
                    _literal(13),
                ],
            ),
            literal_markers,
        ),
        (
            "piecewise_branch_rewrite",
            _piecewise([(_identifier("p"), _binary("add", x, zero))], y),
            [x_plus_zero],
        ),
        (
            "piecewise_condition_becomes_number",
            _piecewise([(true, _literal(5))], _literal(6)),
            [_rule(_p_literal(True), _r_literal(1))],
        ),
        (
            "piecewise_condition_stays_boolean",
            _piecewise([(true, _literal(5))], _literal(6)),
            [_rule(_p_literal(True), _r_literal(False))],
        ),
        (
            "call_argument_rewrite",
            _call("f", [_binary("add", x, zero), _literal(3)]),
            [x_plus_zero],
        ),
        (
            "guard_fails",
            one,
            [x_plus_zero, _rule(_p_wildcard(), _r_literal(0), guard=_g("fail"))],
        ),
        (
            "rewrite_fails",
            _unary("negate", one),
            [_rule(_p_literal(), _r_fail(), name="failing")],
        ),
        (
            "predicate_fails",
            _unary("negate", one),
            [_rule(_p_predicate("fail"), _r_literal(0))],
        ),
        ("rewrite_capture_unbound", one, [_rule(_p_wildcard(), _r_capture("x"))]),
        (
            "failure_after_a_firing",
            _binary("add", _binary("add", x, zero), false),
            [x_plus_zero, _rule(_p_literal(False), _r_fail())],
        ),
    ]


# =============================================================================
# Random cases
# =============================================================================

_LEAF_PROBABILITY = 0.25
"""Chance that an inner draw stops at a leaf although its budget allows more."""

_IDENTIFIER_LEAF_PROBABILITY = 0.4
"""Chance that a leaf is an identifier reference rather than a literal."""

_MIN_PIECEWISE_LEAVES = 3
"""Leaves the smallest piecewise needs: a condition, a value, an otherwise."""

_MAX_LOOSE_DEPTH = 4
"""Depth below which a loose pattern is only a wildcard."""

_RULE_NAME_PROBABILITY = 0.6
"""Chance that a random rule is named."""

_NODE_KIND_WEIGHTS = {"unary": 2, "binary": 5, "piecewise": 2, "call": 1}
_BINARY_OPERATIONS = ("add", "subtract", "multiply", "less", "logical_and")
_SMALL_LITERALS: tuple[bool | int | float | str, ...] = (
    True,
    False,
    0,
    1,
    2,
    -1,
    10**20,
    0.0,
    -0.0,
    1.0,
    2.5,
    "0",
    "1",
    "01",
    "2",
    "1.5",
    "1.50",
)

# Relative weights of the choices a random pattern or rule makes. A field
# of a node pattern is mostly the tree's own ("same"), else "any" (`None`)
# or another value ("other").
_FIELD_WEIGHTS = {"same": 14, "any": 3, "other": 3}
_LOOSE_PATTERN_WEIGHTS = {
    "wildcard": 8,
    "capture": 14,
    "predicate": 5,
    "failing_predicate": 1,
    "alternatives": 7,
    "unrelated": 3,
    "structural": 62,
}
_CASE_LIST_WEIGHTS = {"same": 8, "any": 2, "extra_case": 1}
_ARGUMENT_LIST_WEIGHTS = {"same": 7, "any": 2, "other_arity": 1}
_GUARD_WEIGHTS = {
    "none": 59,
    "always": 15,
    "never": 10,
    "bound_is_literal": 15,
    "fail": 1,
}
_REWRITE_WEIGHTS = {"capture": 45, "literal": 30, "negate_capture": 23, "fail": 2}
_REWRITE_LITERALS: tuple[bool | int | str, ...] = (0, 1, 2, True, "1.5")


class _Drawer:
    """Seeded random trees, patterns and rules."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def _pick(self, weights: dict[str, int]) -> str:
        """Return one key of ``weights``, drawn with its weight."""
        return self._rng.choices(list(weights), list(weights.values()))[0]

    def draw_literal(self) -> bool | int | float | str:
        """Return a literal value from a small space, so equal values recur."""
        return self._rng.choice(_SMALL_LITERALS)

    def draw_leaf(self) -> JsonDict:
        """Return a random identifier reference or literal."""
        if self._rng.random() < _IDENTIFIER_LEAF_PROBABILITY:
            return _identifier(self._rng.choice(_SLOTS[:3]))
        return _literal(self.draw_literal())

    def draw_tree(self, max_leaves: int) -> JsonDict:
        """Return a random tree with at most ``max_leaves`` leaves."""
        rng = self._rng
        if max_leaves <= 1 or rng.random() < _LEAF_PROBABILITY:
            return self.draw_leaf()
        weights = dict(_NODE_KIND_WEIGHTS)
        if max_leaves < _MIN_PIECEWISE_LEAVES:
            del weights["piecewise"]
        draw = {
            "unary": self._draw_unary,
            "binary": self._draw_binary,
            "piecewise": self._draw_piecewise,
            "call": self._draw_call,
        }[self._pick(weights)]
        return draw(max_leaves)

    def _draw_unary(self, max_leaves: int) -> JsonDict:
        """Return a random unary node."""
        operation = self._rng.choice([str(operation) for operation in UnaryOperation])
        return _unary(operation, self.draw_tree(max_leaves - 1))

    def _draw_binary(self, max_leaves: int) -> JsonDict:
        """Return a random binary node splitting the budget between its operands."""
        left_budget = self._rng.randint(1, max_leaves - 1)
        return _binary(
            self._rng.choice(_BINARY_OPERATIONS),
            self.draw_tree(left_budget),
            self.draw_tree(max_leaves - left_budget),
        )

    def _draw_call(self, max_leaves: int) -> JsonDict:
        """Return a random call."""
        count = self._rng.randint(0, min(_MAX_ARGUMENTS, max_leaves))
        budget = max(1, max_leaves // max(1, count))
        arguments = [self.draw_tree(budget) for _ in range(count)]
        return _call(self._rng.choice(_FUNCTION_NAMES), arguments)

    def _draw_piecewise(self, max_leaves: int) -> JsonDict:
        """Return a random piecewise whose literal conditions are Booleans."""
        rng = self._rng
        case_count = rng.randint(1, min(_MAX_CASES, (max_leaves - 1) // 2))
        budget = max(1, max_leaves // (2 * case_count + 1))
        cases = []
        for _ in range(case_count):
            condition = self.draw_tree(budget)
            if condition["__type__"] == "literal_expression" and not isinstance(
                condition["__data__"]["value"], bool
            ):
                condition = _literal(rng.choice((True, False)))
            cases.append((condition, self.draw_tree(budget)))
        return _piecewise(cases, self.draw_tree(budget))

    def _draw_field(self, same: Any, others: list[Any]) -> Any:
        """Return ``same`` mostly, else ``None`` or one of ``others``."""
        choice = self._pick(_FIELD_WEIGHTS)
        if choice == "same":
            return same
        return None if choice == "any" else self._rng.choice(others)

    def mirror(
        self, tree: JsonDict, leaf_capture: Callable[[JsonDict], JsonDict]
    ) -> JsonDict:
        """Return a pattern of ``tree``'s exact shape with ``leaf_capture`` leaves."""
        kind = tree["__type__"]
        data = tree["__data__"]

        def sub(child: JsonDict) -> JsonDict:
            return self.mirror(child, leaf_capture)

        if kind == "unary_expression":
            return _p_unary(data["operation"], sub(data["operand"]))
        if kind == "binary_expression":
            return _p_binary(data["operation"], sub(data["left"]), sub(data["right"]))
        if kind == "piecewise_expression":
            pairs = zip(data["conditions"], data["values"], strict=True)
            cases = [[sub(condition), sub(value)] for condition, value in pairs]
            return _p_piecewise(cases, sub(data["otherwise"]))
        if kind == "call_expression":
            arguments = [sub(argument) for argument in data["arguments"]]
            return _p_call(data["function_name"], arguments)
        return leaf_capture(tree)

    def draw_loose_pattern(
        self, tree: JsonDict, depth: int = 0, *, may_fail: bool = True
    ) -> JsonDict:
        """Return a pattern roughly shaped like ``tree``, often but not always matching.

        With ``may_fail`` set, a predicate that fails may appear in it.
        """
        rng = self._rng

        def loose(subtree: JsonDict) -> JsonDict:
            return self.draw_loose_pattern(subtree, depth + 1, may_fail=may_fail)

        weights = dict(_LOOSE_PATTERN_WEIGHTS)
        if not may_fail:
            del weights["failing_predicate"]
        choice = "wildcard" if depth > _MAX_LOOSE_DEPTH else self._pick(weights)
        draws: dict[str, Callable[[], JsonDict]] = {
            "wildcard": _p_wildcard,
            "capture": lambda: _p_capture(rng.choice(_CAPTURE_NAMES), loose(tree)),
            "predicate": lambda: _p_predicate(
                rng.choice(("is_literal", "is_identifier", "has_children"))
            ),
            "failing_predicate": lambda: _p_predicate("fail"),
            "alternatives": lambda: _p_alternatives(loose(tree), loose(tree)),
            "unrelated": lambda: loose(self.draw_tree(3)),
            "structural": lambda: self._draw_structural_pattern(tree, loose),
        }
        return draws[choice]()

    def _draw_structural_pattern(
        self, tree: JsonDict, loose: Callable[[JsonDict], JsonDict]
    ) -> JsonDict:
        """Return a pattern of ``tree``'s node kind with perturbed fields."""
        data = tree["__data__"]
        draws: dict[str, Callable[[], JsonDict]] = {
            "literal_expression": lambda: _p_literal(
                self._draw_field(data["value"], list(_SMALL_LITERALS))
            ),
            "identifier_expression": lambda: _p_identifier(
                self._draw_field(data["identifier"]["slot"], list(_SLOTS))
            ),
            "unary_expression": lambda: _p_unary(
                self._draw_field(data["operation"], [str(op) for op in UnaryOperation]),
                loose(data["operand"]),
            ),
            "binary_expression": lambda: _p_binary(
                self._draw_field(data["operation"], list(_BINARY_OPERATIONS)),
                loose(data["left"]),
                loose(data["right"]),
            ),
            "piecewise_expression": lambda: self._draw_piecewise_pattern(data, loose),
            "call_expression": lambda: self._draw_call_pattern(data, loose),
        }
        return draws[tree["__type__"]]()

    def _draw_piecewise_pattern(
        self, data: JsonDict, loose: Callable[[JsonDict], JsonDict]
    ) -> JsonDict:
        """Return a piecewise pattern over a piecewise node's fields."""
        pairs = zip(data["conditions"], data["values"], strict=True)
        cases: list[list[JsonDict]] | None = [
            [loose(condition), loose(value)] for condition, value in pairs
        ]
        choice = self._pick(_CASE_LIST_WEIGHTS)
        if choice == "any":
            cases = None
        elif choice == "extra_case" and cases is not None:
            cases = [*cases, [_p_wildcard(), _p_wildcard()]]
        return _p_piecewise(cases, loose(data["otherwise"]))

    def _draw_call_pattern(
        self, data: JsonDict, loose: Callable[[JsonDict], JsonDict]
    ) -> JsonDict:
        """Return a call pattern over a call node's fields."""
        name = self._draw_field(data["function_name"], list(_FUNCTION_NAMES))
        arguments: list[JsonDict] | None = [
            loose(argument) for argument in data["arguments"]
        ]
        choice = self._pick(_ARGUMENT_LIST_WEIGHTS)
        if choice == "any":
            arguments = None
        elif choice == "other_arity" and arguments is not None:
            arguments = arguments[:-1] if arguments else [_p_wildcard()]
        return _p_call(name, arguments)

    def draw_rule(self, tree: JsonDict, index: int) -> JsonDict:
        """Return a random rule whose pattern is drawn from a subtree of ``tree``.

        The pattern is a capture ``x`` of a loose pattern without failing
        predicates; guards and rewrites name captures every match binds.
        """
        rng = self._rng
        target = rng.choice(_subtrees(tree))
        pattern = _p_capture("x", self.draw_loose_pattern(target, may_fail=False))
        captures = _collect_capture_names(pattern)
        guard_choice = self._pick(_GUARD_WEIGHTS)
        guard = {
            "none": None,
            "always": _g("always"),
            "never": _g("never"),
            "bound_is_literal": _g("bound_is_literal", rng.choice(captures)),
            "fail": _g("fail"),
        }[guard_choice]
        rewrite = {
            "capture": lambda: _r_capture(rng.choice(captures)),
            "literal": lambda: _r_literal(rng.choice(_REWRITE_LITERALS)),
            "negate_capture": lambda: _r_negate(rng.choice(captures)),
            "fail": _r_fail,
        }[self._pick(_REWRITE_WEIGHTS)]()
        name = f"rule {index}" if rng.random() < _RULE_NAME_PROBABILITY else None
        return _rule(pattern, rewrite, guard=guard, name=name)


def _collect_capture_names(pattern: JsonDict) -> list[str]:
    """Return the capture names a match of a pattern spec always binds.

    Captures inside alternatives are left out: a match binds only those of
    the chosen alternative.
    """
    if pattern["kind"] == "alternatives":
        return []
    names = [pattern["name"]] if pattern["kind"] == "capture" else []
    for value in pattern.values():
        if isinstance(value, dict) and "kind" in value:
            names += _collect_capture_names(value)
        elif isinstance(value, list):
            for item in value:
                for element in item if isinstance(item, list) else [item]:
                    names += _collect_capture_names(element)
    return names


def _build_unique_leaf_capture() -> Callable[[JsonDict], JsonDict]:
    """Return a leaf pattern builder capturing each leaf under a fresh name."""
    count = 0

    def capture(_: JsonDict) -> JsonDict:
        nonlocal count
        name = f"leaf_{count}"
        count += 1
        return _p_capture(name)

    return capture


def _capture_leaf_by_kind(leaf: JsonDict) -> JsonDict:
    """Return a capture of a leaf named by its slot, or ``literal`` for literals."""
    if leaf["__type__"] == "identifier_expression":
        return _p_capture(leaf["__data__"]["identifier"]["slot"])
    return _p_capture("literal")


def _draw_random_cases(
    drawer: _Drawer, rng: random.Random, count: int, max_leaves: int
) -> tuple[
    list[tuple[str, JsonDict, JsonDict]], list[tuple[str, JsonDict, list[JsonDict]]]
]:
    """Return random match and rewrite cases over ``count`` random trees.

    Each tree is matched against a mirror capturing every leaf under its own
    name, a mirror sharing captures (one per slot, one for all literals, so
    repeated captures meet literals of every bucket), and two loose
    patterns; it is rewritten with two random rule lists.
    """
    match_cases = []
    rewrite_cases = []
    for index in range(count):
        tree = drawer.draw_tree(rng.randint(1, max_leaves))
        match_cases += [
            (
                f"random_{index}_mirror",
                tree,
                drawer.mirror(tree, _build_unique_leaf_capture()),
            ),
            (
                f"random_{index}_mirror_shared_captures",
                tree,
                drawer.mirror(tree, _capture_leaf_by_kind),
            ),
            (f"random_{index}_loose_0", tree, drawer.draw_loose_pattern(tree)),
            (f"random_{index}_loose_1", tree, drawer.draw_loose_pattern(tree)),
        ]
        for variant in range(2):
            rule_count = rng.randint(1, 3)
            rules = [drawer.draw_rule(tree, position) for position in range(rule_count)]
            rewrite_cases.append((f"random_{index}_rules_{variant}", tree, rules))
    return match_cases, rewrite_cases


def main() -> None:
    """Parse the options, run the oracle, and write the corpus."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_corpus_arguments(
        parser,
        seed=_RANDOM_SEED,
        random_count=_RANDOM_COUNT,
        max_ops=_RANDOM_MAX_LEAVES,
        default_output=_DEFAULT_OUTPUT,
    )
    arguments = parser.parse_args()
    # Failing rewrites are recorded as data; the applier's error log is noise.
    logging.disable(logging.CRITICAL)
    rng = random.Random(arguments.seed)
    drawer = _Drawer(rng)
    match_inputs = _hand_picked_match_cases()
    rewrite_inputs = _hand_picked_rewrite_cases()
    random_matches, random_rewrites = _draw_random_cases(
        drawer, rng, arguments.random_count, arguments.max_ops
    )
    match_inputs += random_matches
    rewrite_inputs += random_rewrites
    document = {
        "provenance": build_provenance(_REPOSITORY_ROOT, GENERATOR_COMMAND),
        "match_cases": [
            _record_match(name, copy.deepcopy(tree), pattern)
            for name, tree, pattern in match_inputs
        ],
        "rewrite_cases": [
            _record_rewrite(name, copy.deepcopy(tree), rules)
            for name, tree, rules in rewrite_inputs
        ],
    }
    write_document(arguments.output, document)


if __name__ == "__main__":
    main()
