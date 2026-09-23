"""Generate golden expression-tree cases from the Python expression oracle.

Each case is one expression tree, given by its wire dict
(``serialize_to_dict``) with every identifier written as ``{"slot": name}``
instead of ``{"id": ..., "name_hint": ...}``. The Rust replay mints one
identifier per slot and name. Every wire dict (a tree, a replacement, a bound
value, a refused payload) is stored as compact JSON text, and a node inside
a tree is named by its path: the ``/``-joined ``__data__`` field names from
the root, with a list index after a list field (``conditions/0/left``; the
root is ``""``). For each tree the oracle records:

* ``children``: ``get_visit_children()`` in order, as the paths of the
  children;
* ``free_identifiers``: ``get_free_identifiers()`` as sorted slot names;
* the wire dict of ``Expression.deserialize_from_dict`` of the tree,
  serialized again, is the tree itself (checked here, so nothing is stored);
* ``substitutions``: ``substitute`` under slot-to-tree mappings, each the
  paths of the identifier leaves the result replaces by their mapped trees
  (checked here to rebuild the oracle's result exactly), or the refusal of a
  piecewise case condition the mapping turns into a non-Boolean literal;
* ``screens``: ``validate_logical_operands`` and ``validate_predicate``
  under an environment (slot to tree) and declared symbol types, each
  ``null`` when the screen passes, or the offending operand and the node that
  puts it in a Boolean position (``null`` for a predicate root) as paths in
  the tree or in the bound value of ``source``, and that position. The pair
  comes from the oracle's own search (``_find_non_boolean_logical_operand``
  and ``_is_provably_non_boolean``, cross-checked against the public screens);
  the paths and the position are found by identity, which is by position here
  because a deserialized tree shares no nodes;
* ``comparisons``: ``is_structurally_equivalent`` against other trees, each
  a case's tree, optionally with the leaf at ``path`` replaced by ``leaf``: a
  second decode of the same tree, a copy with one literal respelled, a copy
  with one leaf changed, and the next case's tree.

Only unregistered function names (``f``, ``g``, ``h``) and freshly minted
identifiers appear, so no screen verdict depends on the function registry:
every verdict equals the one an empty sort lookup gives. No literal is NaN or
infinite, since JSON cannot carry one.

A second section, ``decode_errors``, records payloads the oracle refuses to
deserialize, with the exception type. A literal payload with an extra key is
left out: the oracle accepts it, while every other node refuses extra keys.

The Rust replay is ``rust/fhy-core/tests/expression_equivalence.rs``.

Run from the repository root:

    uv run --no-sync python rust/fhy-core/tests/golden/generate_expression_cases.py

This overwrites ``rust/fhy-core/tests/golden/expression_cases.json``.
``--random-count`` sets how many random trees are drawn, ``--max-ops`` sets
the largest leaf count of a random tree, and ``--seed`` seeds the draws. The
ignored expanded-corpus test reads a corpus written with other options from
the file named in ``FHY_EXPRESSION_CORPUS``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from _golden_support import add_corpus_arguments, build_provenance, write_document

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    NonBooleanLogicalOperandError,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    validate_logical_operands,
    validate_predicate,
)
from fhy_core.symbolic.expression.core import (
    _find_non_boolean_logical_operand,
    _is_provably_non_boolean,
)
from fhy_core.symbolic.expression.registry import try_get_registered_result_sort
from fhy_core.symbolic.symbol_type import SymbolType

GENERATOR_COMMAND = (
    "uv run --no-sync python rust/fhy-core/tests/golden/generate_expression_cases.py"
)

_RANDOM_SEED = 20260922
_RANDOM_COUNT = 60
_RANDOM_MAX_LEAVES = 10

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_OUTPUT = Path(__file__).resolve().with_name("expression_cases.json")

_SLOTS = ("x", "y", "z", "p", "q")
_FUNCTION_NAMES = ("f", "g", "h")
_SYMBOL_TYPES = (SymbolType.INT, SymbolType.REAL, SymbolType.BOOL)
_MAX_CASES = 3
_MAX_ARGUMENTS = 3


def _slot(name: str) -> dict[str, str]:
    """Return the identifier placeholder for slot ``name``."""
    return {"slot": name}


def _identifier(name: str) -> dict[str, Any]:
    """Return the wire dict of an identifier reference to slot ``name``."""
    return {
        "__type__": "identifier_expression",
        "__data__": {"identifier": _slot(name)},
    }


def _literal(value: bool | int | float | str) -> dict[str, Any]:
    """Return the wire dict of a literal."""
    return {"__type__": "literal_expression", "__data__": {"value": value}}


def _unary(operation: str, operand: dict[str, Any]) -> dict[str, Any]:
    """Return the wire dict of a unary node."""
    return {
        "__type__": "unary_expression",
        "__data__": {"operation": operation, "operand": operand},
    }


def _binary(
    operation: str, left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    """Return the wire dict of a binary node."""
    return {
        "__type__": "binary_expression",
        "__data__": {"operation": operation, "left": left, "right": right},
    }


def _piecewise(
    cases: list[tuple[dict[str, Any], dict[str, Any]]], otherwise: dict[str, Any]
) -> dict[str, Any]:
    """Return the wire dict of a piecewise node."""
    return {
        "__type__": "piecewise_expression",
        "__data__": {
            "conditions": [condition for condition, _ in cases],
            "values": [value for _, value in cases],
            "otherwise": otherwise,
        },
    }


def _call(function_name: str, arguments: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the wire dict of a call."""
    return {
        "__type__": "call_expression",
        "__data__": {"function_name": function_name, "arguments": arguments},
    }


class _SlotTable:
    """Identifiers minted for the slots of one case, in both directions."""

    def __init__(self) -> None:
        self._by_slot: dict[str, Identifier] = {}
        self._by_id: dict[int, str] = {}

    def resolve(self, slot: str) -> Identifier:
        """Return the identifier of ``slot``, minting it on first use."""
        if slot not in self._by_slot:
            identifier = Identifier(slot)
            self._by_slot[slot] = identifier
            self._by_id[identifier.id] = slot
        return self._by_slot[slot]

    def slot_of(self, identifier: Identifier) -> str:
        """Return the slot ``identifier`` was minted for."""
        return self.slot_of_id(identifier.id)

    def slot_of_id(self, identifier_id: int) -> str:
        """Return the slot the identifier with id ``identifier_id`` was minted for."""
        return self._by_id[identifier_id]


def _bind_slots(wire: Any, table: _SlotTable) -> Any:
    """Return ``wire`` with every slot placeholder replaced by its identifier dict."""
    if isinstance(wire, dict):
        if set(wire) == {"slot"}:
            identifier = table.resolve(wire["slot"])
            return {"id": identifier.id, "name_hint": identifier.name_hint}
        return {key: _bind_slots(value, table) for key, value in wire.items()}
    if isinstance(wire, list):
        return [_bind_slots(item, table) for item in wire]
    return wire


def _unbind_slots(wire: Any, table: _SlotTable) -> Any:
    """Return ``wire`` with every identifier dict replaced by its slot placeholder."""
    if isinstance(wire, dict):
        if set(wire) == {"id", "name_hint"}:
            return _slot(table.slot_of_id(wire["id"]))
        return {key: _unbind_slots(value, table) for key, value in wire.items()}
    if isinstance(wire, list):
        return [_unbind_slots(item, table) for item in wire]
    return wire


def _decode(wire: dict[str, Any], table: _SlotTable) -> Expression:
    """Deserialize a slot-form wire dict."""
    return Expression.deserialize_from_dict(_bind_slots(wire, table))


def _encode(expression: Expression, table: _SlotTable) -> dict[str, Any]:
    """Serialize ``expression`` to a slot-form wire dict."""
    wire: dict[str, Any] = _unbind_slots(expression.serialize_to_dict(), table)
    return wire


def _describe_position(parent: Expression, operand: Expression) -> dict[str, Any]:
    """Return where ``operand`` sits in ``parent``'s Boolean positions."""
    if isinstance(parent, UnaryExpression):
        return {"kind": "negated_operand"}
    if isinstance(parent, BinaryExpression):
        return {"kind": "logical_operand", "operation": str(parent.operation)}
    if isinstance(parent, PiecewiseExpression):
        for index, condition in enumerate(parent.conditions):
            if condition is operand:
                return {"kind": "case_condition", "case_index": index}
        for index, value in enumerate(parent.values):
            if value is operand:
                return {"kind": "case_value", "case_index": index}
        if parent.otherwise is operand:
            return {"kind": "otherwise"}
    raise AssertionError(f"no Boolean position of {parent!r} holds {operand!r}")


def _write_compact(wire: Any) -> str:
    """Return ``wire`` as compact JSON text."""
    return json.dumps(wire, ensure_ascii=False, separators=(",", ":"))


def _join_path(path: str, *steps: str) -> str:
    """Return ``path`` extended by ``steps``."""
    return "/".join(step for step in (path, *steps) if step)


def _map_node_paths(root: Expression) -> dict[int, str]:
    """Return the path of every node object under ``root``, keyed by identity."""
    paths: dict[int, str] = {}
    pending = [(root, "")]
    while pending:
        node, path = pending.pop()
        paths[id(node)] = path
        if not dataclasses.is_dataclass(node):
            raise AssertionError(f"{node!r} is not a dataclass node")
        for field in dataclasses.fields(node):
            value = getattr(node, field.name)
            if isinstance(value, Expression):
                pending.append((value, _join_path(path, field.name)))
            elif isinstance(value, tuple):
                pending.extend(
                    (item, _join_path(path, field.name, str(index)))
                    for index, item in enumerate(value)
                    if isinstance(item, Expression)
                )
    return paths


def _iter_wire_nodes(
    wire: dict[str, Any], path: str = ""
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield the path and wire dict of every node of ``wire``, in wire order."""
    yield path, wire
    for key, value in wire["__data__"].items():
        if isinstance(value, dict) and "__type__" in value:
            yield from _iter_wire_nodes(value, _join_path(path, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from _iter_wire_nodes(item, _join_path(path, key, str(index)))


def _replace_at_paths(
    wire: dict[str, Any], replacements: dict[str, dict[str, Any]], path: str = ""
) -> dict[str, Any]:
    """Return ``wire`` with the node at each path of ``replacements`` replaced."""
    if path in replacements:
        return replacements[path]
    data: dict[str, Any] = {}
    for key, value in wire["__data__"].items():
        if isinstance(value, dict) and "__type__" in value:
            data[key] = _replace_at_paths(value, replacements, _join_path(path, key))
        elif isinstance(value, list):
            data[key] = [
                _replace_at_paths(item, replacements, _join_path(path, key, str(index)))
                for index, item in enumerate(value)
            ]
        else:
            data[key] = value
    return {"__type__": wire["__type__"], "__data__": data}


def _record_screen(
    expression: Expression,
    environment: dict[Identifier, Expression],
    symbol_types: dict[Identifier, SymbolType],
    table: _SlotTable,
    *,
    is_predicate: bool,
) -> dict[str, Any] | None:
    """Return the oracle's verdict for one screen, cross-checked with the public one."""
    verdict: dict[str, Any] | None = None
    if is_predicate and _is_provably_non_boolean(expression, environment, symbol_types):
        verdict = {
            "position": {"kind": "predicate_root"},
            "source": None,
            "operand": "",
            "parent": None,
        }
    else:
        found = _find_non_boolean_logical_operand(
            expression, environment, symbol_types, is_in_boolean_position=is_predicate
        )
        if found is not None:
            parent, operand = found
            sources = [(None, expression)] + [
                (table.slot_of(identifier), value)
                for identifier, value in environment.items()
            ]
            for source, root in sources:
                paths = _map_node_paths(root)
                if id(parent) in paths:
                    verdict = {
                        "position": _describe_position(parent, operand),
                        "source": source,
                        "operand": paths[id(operand)],
                        "parent": paths[id(parent)],
                    }
                    break
            else:
                raise AssertionError(f"{parent!r} is in no screened tree")
    validate = validate_predicate if is_predicate else validate_logical_operands
    try:
        validate(expression, environment, symbol_types=symbol_types)
    except NonBooleanLogicalOperandError:
        refused = True
    else:
        refused = False
    if refused != (verdict is not None):
        raise AssertionError(f"private and public screens disagree on {expression!r}")
    return verdict


_LEAF_PROBABILITY = 0.25
"""Chance that an inner draw stops at a leaf although its budget allows more."""

_SMALL_INTEGER_BOUND = 20
_LARGE_INTEGER_BOUND = 10**30
_FLOAT_BOUND = 100.0
_MAX_FLOAT_DIGITS = 3
_MAX_TEXT_INTEGER = 999
_MAX_TEXT_PADDING = 4
_SPECIAL_FLOATS = (0.0, -0.0, 1e16, 1e-05, 2.5)

# Relative weights of the inner node kinds a random tree draws.
_NODE_KIND_WEIGHTS = {"unary": 2, "binary": 5, "piecewise": 2, "call": 1}

_MIN_PIECEWISE_LEAVES = 3
"""Leaves the smallest piecewise needs: a condition, a value, an otherwise."""


class _TreeDrawer:
    """Seeded random expression trees in slot-form wire dicts."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def _draw_integer_text(self) -> str:
        """Return a random integer text, possibly zero-padded."""
        rng = self._rng
        return str(rng.randint(0, _MAX_TEXT_INTEGER)).zfill(
            rng.randint(1, _MAX_TEXT_PADDING)
        )

    def _draw_decimal_text(self) -> str:
        """Return a random decimal text, possibly with a trailing zero."""
        rng = self._rng
        whole = str(rng.randint(0, _MAX_TEXT_INTEGER))
        fraction = str(rng.randint(0, _MAX_TEXT_INTEGER)).rstrip("0")
        return f"{whole}.{fraction}{'0' * rng.randrange(2)}"

    def draw_literal(self) -> dict[str, Any]:
        """Return a random finite literal of any kind."""
        rng = self._rng
        draws = (
            lambda: rng.choice((True, False)),
            lambda: rng.randint(-_SMALL_INTEGER_BOUND, _SMALL_INTEGER_BOUND),
            lambda: rng.randint(-_LARGE_INTEGER_BOUND, _LARGE_INTEGER_BOUND),
            lambda: round(
                rng.uniform(-_FLOAT_BOUND, _FLOAT_BOUND),
                rng.randint(0, _MAX_FLOAT_DIGITS),
            ),
            self._draw_integer_text,
            self._draw_decimal_text,
            lambda: rng.choice(_SPECIAL_FLOATS),
        )
        return _literal(rng.choice(draws)())

    def draw_leaf(self) -> dict[str, Any]:
        """Return a random identifier reference or literal."""
        rng = self._rng
        draws = (lambda: _identifier(rng.choice(_SLOTS)), self.draw_literal)
        return rng.choice(draws)()

    def draw_tree(self, max_leaves: int) -> dict[str, Any]:
        """Return a random tree with at most ``max_leaves`` leaves."""
        rng = self._rng
        if max_leaves <= 1 or rng.random() < _LEAF_PROBABILITY:
            return self.draw_leaf()
        kinds = list(_NODE_KIND_WEIGHTS)
        if max_leaves < _MIN_PIECEWISE_LEAVES:
            kinds.remove("piecewise")
        kind = rng.choices(kinds, [_NODE_KIND_WEIGHTS[kind] for kind in kinds])[0]
        draw = {
            "unary": self._draw_unary,
            "binary": self._draw_binary,
            "piecewise": self._draw_piecewise,
            "call": self._draw_call,
        }[kind]
        return draw(max_leaves)

    def _draw_unary(self, max_leaves: int) -> dict[str, Any]:
        """Return a random unary node."""
        operation = self._rng.choice([str(operation) for operation in UnaryOperation])
        return _unary(operation, self.draw_tree(max_leaves - 1))

    def _draw_binary(self, max_leaves: int) -> dict[str, Any]:
        """Return a random binary node splitting the budget between its operands."""
        rng = self._rng
        operation = rng.choice([str(operation) for operation in BinaryOperation])
        left_budget = rng.randint(1, max_leaves - 1)
        return _binary(
            operation,
            self.draw_tree(left_budget),
            self.draw_tree(max_leaves - left_budget),
        )

    def _draw_call(self, max_leaves: int) -> dict[str, Any]:
        """Return a random call to an unregistered function."""
        rng = self._rng
        argument_count = rng.randint(0, min(_MAX_ARGUMENTS, max_leaves))
        budget = max(1, max_leaves // max(1, argument_count))
        return _call(
            rng.choice(_FUNCTION_NAMES),
            [self.draw_tree(budget) for _ in range(argument_count)],
        )

    def _draw_piecewise(self, max_leaves: int) -> dict[str, Any]:
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


def _find_first_leaf_edit(
    wire: dict[str, Any], replace: Callable[[dict[str, Any]], dict[str, Any] | None]
) -> tuple[str, dict[str, Any]] | None:
    """Return the path and replacement of the first leaf ``replace`` maps.

    Leaves are visited in wire order; ``replace`` returns ``None`` for a leaf
    it leaves alone.
    """
    for path, node in _iter_wire_nodes(wire):
        if node["__type__"] in _LEAF_TYPE_IDS:
            replacement = replace(node)
            if replacement is not None:
                return path, replacement
    return None


_LEAF_TYPE_IDS = ("identifier_expression", "literal_expression")


def _respell_literal(leaf: dict[str, Any]) -> dict[str, Any] | None:
    """Return a literal leaf in another spelling of the same value, if it has one."""
    if leaf["__type__"] != "literal_expression":
        return None
    value = leaf["__data__"]["value"]
    if isinstance(value, str):
        return _literal(value + "0" if "." in value else "0" + value)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return _literal("0" + str(value))
    return None


def _change_leaf(leaf: dict[str, Any]) -> dict[str, Any]:
    """Return a leaf of the same kind with a different identifier or value."""
    if leaf["__type__"] == "identifier_expression":
        slot = leaf["__data__"]["identifier"]["slot"]
        return _identifier(_SLOTS[(_SLOTS.index(slot) + 1) % len(_SLOTS)])
    value = leaf["__data__"]["value"]
    if isinstance(value, bool):
        changed: bool | int | float | str = not value
    elif isinstance(value, int):
        changed = value + 1
    elif isinstance(value, float):
        changed = value + 1.0 if value + 1.0 != value else value * 2.0
    else:
        changed = value + ("1" if "." in value else "7")
    return _literal(changed)


def _hand_picked_trees() -> list[tuple[str, dict[str, Any]]]:
    """Return trees covering every node kind and the screens' decisive shapes."""
    true, false = _literal(True), _literal(False)
    x, y, p, q = _identifier("x"), _identifier("y"), _identifier("p"), _identifier("q")
    return [
        ("identifier", x),
        ("int_literal", _literal(5)),
        ("big_int_literal", _literal(10**30)),
        ("negative_big_int_literal", _literal(-(2**64))),
        ("float_literal", _literal(1.5)),
        ("integral_float_literal", _literal(5.0)),
        ("negative_zero_literal", _literal(-0.0)),
        ("large_float_literal", _literal(1e16)),
        ("integer_text_literal", _literal("05")),
        ("decimal_text_literal", _literal("1.50")),
        ("bool_literal", true),
        ("negate", _unary("negate", x)),
        ("logical_not_of_bool", _unary("logical_not", true)),
        ("logical_not_of_number", _unary("logical_not", _literal(2))),
        (
            "logical_not_of_negation",
            _unary("logical_not", _unary("negate", _literal(1))),
        ),
        ("sum", _binary("add", _literal(1), _literal(2))),
        ("and_of_numbers", _binary("logical_and", _literal(2), _literal(4))),
        ("or_of_numbers", _binary("logical_or", _literal(2), _literal(4))),
        ("and_bool_then_number", _binary("logical_and", true, _literal(4))),
        ("and_of_floats", _binary("logical_and", _literal(1.5), _literal(2.5))),
        ("and_of_texts", _binary("logical_and", _literal("2"), _literal("4"))),
        (
            "and_arithmetic_operand",
            _binary("logical_and", _binary("add", _literal(1), _literal(2)), true),
        ),
        ("and_of_bools", _binary("logical_and", true, false)),
        ("and_of_identifiers", _binary("logical_and", p, q)),
        (
            "and_of_comparisons",
            _binary(
                "logical_and",
                _binary("greater", x, _literal(0)),
                _binary("less", x, _literal(5)),
            ),
        ),
        ("and_nested_not", _binary("logical_and", _unary("logical_not", p), true)),
        (
            "numeric_operand_below_well_typed_root",
            _binary(
                "logical_and",
                _binary("greater", x, _literal(0)),
                _binary("logical_and", _literal(2), _literal(4)),
            ),
        ),
        (
            "all_numeric_piecewise_under_and",
            _binary(
                "logical_and",
                _piecewise(
                    [(_binary("greater", x, _literal(0)), _literal(1))], _literal(2)
                ),
                true,
            ),
        ),
        (
            "boolean_piecewise_under_and",
            _binary(
                "logical_and",
                _piecewise([(_binary("greater", x, _literal(0)), true)], false),
                true,
            ),
        ),
        (
            "not_of_piecewise_with_numeric_value",
            _unary(
                "logical_not",
                _piecewise([(_binary("greater", x, _literal(0)), _literal(2))], true),
            ),
        ),
        (
            "not_of_piecewise_with_numeric_otherwise",
            _unary(
                "logical_not",
                _piecewise([(_binary("greater", x, _literal(0)), true)], _literal(2)),
            ),
        ),
        (
            "numeric_piecewise_compared",
            _binary(
                "logical_and",
                _binary(
                    "greater",
                    _piecewise(
                        [(_binary("greater", x, _literal(0)), _literal(2))], _literal(3)
                    ),
                    _literal(1),
                ),
                true,
            ),
        ),
        (
            "piecewise_with_arithmetic_condition",
            _piecewise([(_binary("add", x, _literal(1)), _literal(5))], _literal(0)),
        ),
        (
            "piecewise_with_product_condition",
            _piecewise(
                [(_binary("multiply", x, _literal(2)), _literal(5))], _literal(0)
            ),
        ),
        (
            "piecewise_with_identifier_condition",
            _piecewise([(_identifier("z"), _literal(1))], _literal(0)),
        ),
        (
            "piecewise_with_numeric_branch",
            _piecewise([(_identifier("z"), true)], _literal(2)),
        ),
        (
            "two_case_piecewise",
            _piecewise([(true, _literal(1)), (false, _literal(2))], _literal(0)),
        ),
        (
            "nested_piecewise",
            _piecewise(
                [(false, _piecewise([(true, _literal(1))], _literal(0)))], _literal(9)
            ),
        ),
        ("zero_argument_call", _call("f", [])),
        ("call_of_identifiers", _call("f", [x, y])),
        ("call_under_and", _binary("logical_and", _call("g", [true, true]), true)),
        (
            "deep_mixed",
            _binary(
                "power",
                _unary("negate", _binary("floor_divide", x, _literal(3))),
                _call("h", [y, _literal("0.25")]),
            ),
        ),
    ]


def _draw_substitutions(
    drawer: _TreeDrawer, rng: random.Random, wire: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return two or three random slot-to-tree mappings for ``wire``."""
    mappings = []
    for _ in range(rng.randint(2, 3)):
        domain = rng.sample(_SLOTS, rng.randint(1, 2))
        mappings.append({slot: drawer.draw_tree(3) for slot in domain})
    mappings.append({"z": _literal(1), "x": _identifier("y")})
    return mappings


def _record_substitution(
    wire: dict[str, Any],
    expression: Expression,
    mapping_wire: dict[str, Any],
    table: _SlotTable,
) -> dict[str, Any]:
    """Return one substitution's outcome."""
    mapping = {
        table.resolve(slot): _decode(value, table)
        for slot, value in mapping_wire.items()
    }
    record: dict[str, Any] = {
        "mapping": {slot: _write_compact(value) for slot, value in mapping_wire.items()}
    }
    try:
        result = expression.substitute(mapping)
    except ValueError as error:
        if "condition literal must be a boolean" not in str(error):
            raise
        record["error"] = "non_boolean_condition_literal"
        return record
    replaced = {
        path: mapping_wire[node["__data__"]["identifier"]["slot"]]
        for path, node in _iter_wire_nodes(wire)
        if node["__type__"] == "identifier_expression"
        and node["__data__"]["identifier"]["slot"] in mapping_wire
    }
    if _encode(result, table) != _replace_at_paths(wire, replaced):
        raise AssertionError(f"the substitution result of {wire} is not a leaf rewrite")
    record["replaced"] = list(replaced)
    return record


def _draw_screen_settings(
    drawer: _TreeDrawer, rng: random.Random
) -> list[tuple[dict[str, Any], dict[str, str]]]:
    """Return the environments and symbol types each tree is screened under."""
    declared = {slot: str(rng.choice(_SYMBOL_TYPES)) for slot in rng.sample(_SLOTS, 3)}
    bound_slot = rng.choice(_SLOTS)
    return [
        ({}, {}),
        ({}, declared),
        ({bound_slot: drawer.draw_tree(4)}, declared),
        ({"z": _literal(1), "p": _literal(False)}, {"q": "int"}),
    ]


def _record_case(
    name: str,
    index: int,
    wires: list[dict[str, Any]],
    drawer: _TreeDrawer,
    rng: random.Random,
) -> dict[str, Any]:
    """Return one golden case: the tree ``wires[index]`` and its observations."""
    wire = wires[index]
    table = _SlotTable()
    for slot in _SLOTS:
        table.resolve(slot)
    expression = _decode(wire, table)
    if _encode(_decode(_encode(expression, table), table), table) != wire:
        raise AssertionError(f"the tree of {name} does not round-trip")
    paths = _map_node_paths(expression)
    case: dict[str, Any] = {
        "name": name,
        "tree": _write_compact(wire),
        "children": [paths[id(child)] for child in expression.get_visit_children()],
        "free_identifiers": sorted(
            table.slot_of(identifier)
            for identifier in expression.get_free_identifiers()
        ),
    }
    case["substitutions"] = [
        _record_substitution(wire, expression, mapping, table)
        for mapping in _draw_substitutions(drawer, rng, wire)
    ]
    screens = []
    for environment_wire, symbol_types_wire in _draw_screen_settings(drawer, rng):
        environment = {
            table.resolve(slot): _decode(value, table)
            for slot, value in environment_wire.items()
        }
        symbol_types = {
            table.resolve(slot): SymbolType(value)
            for slot, value in symbol_types_wire.items()
        }
        screens.append(
            {
                "environment": {
                    slot: _write_compact(value)
                    for slot, value in environment_wire.items()
                },
                "symbol_types": symbol_types_wire,
                "logical_operands": _record_screen(
                    expression, environment, symbol_types, table, is_predicate=False
                ),
                "predicate": _record_screen(
                    expression, environment, symbol_types, table, is_predicate=True
                ),
            }
        )
    case["screens"] = screens
    others: list[tuple[dict[str, Any], dict[str, Any]]] = [({"case": index}, wire)]
    for replace in (_respell_literal, _change_leaf):
        edit = _find_first_leaf_edit(wire, replace)
        if edit is None:
            others.append(({"case": index}, wire))
        else:
            path, leaf = edit
            reference = {"case": index, "path": path, "leaf": _write_compact(leaf)}
            others.append((reference, _replace_at_paths(wire, {path: leaf})))
    next_index = (index + 1) % len(wires)
    others.append(({"case": next_index}, wires[next_index]))
    comparisons = [
        {
            "other": reference,
            "equivalent": expression.is_structurally_equivalent(_decode(other, table)),
        }
        for reference, other in others
    ]
    case["comparisons"] = comparisons
    return case


def _decode_error_payloads() -> list[tuple[str, Any]]:
    """Return payloads the oracle refuses, each named."""
    one = _literal(1)
    return [
        (
            "unknown_type_id",
            {
                "__type__": "ternary_expression",
                "__data__": {"condition": _literal(True)},
            },
        ),
        ("missing_data", {"__type__": "literal_expression"}),
        (
            "extra_top_level_key",
            {"__type__": "literal_expression", "__data__": {"value": 1}, "extra": 1},
        ),
        ("literal_missing_value", {"__type__": "literal_expression", "__data__": {}}),
        (
            "literal_list_value",
            {"__type__": "literal_expression", "__data__": {"value": [1, 2, 3]}},
        ),
        (
            "literal_null_value",
            {"__type__": "literal_expression", "__data__": {"value": None}},
        ),
        (
            "literal_bad_text",
            {"__type__": "literal_expression", "__data__": {"value": "abc"}},
        ),
        (
            "literal_signed_text",
            {"__type__": "literal_expression", "__data__": {"value": "-5"}},
        ),
        ("unary_unknown_operation", _unary("not", one)),
        ("unary_symbol_operation", _unary("-", one)),
        ("binary_unknown_operation", _binary("plus", one, one)),
        (
            "binary_missing_right",
            {
                "__type__": "binary_expression",
                "__data__": {"operation": "add", "left": one},
            },
        ),
        (
            "binary_extra_field",
            {
                "__type__": "binary_expression",
                "__data__": {"operation": "add", "left": one, "right": one, "extra": 1},
            },
        ),
        (
            "identifier_missing_name_hint",
            {
                "__type__": "identifier_expression",
                "__data__": {"identifier": {"id": 1}},
            },
        ),
        (
            "piecewise_length_mismatch",
            {
                "__type__": "piecewise_expression",
                "__data__": {
                    "conditions": [_literal(True), _literal(False)],
                    "values": [one],
                    "otherwise": _literal(0),
                },
            },
        ),
        ("piecewise_no_cases", _piecewise([], _literal(0))),
        ("piecewise_numeric_condition", _piecewise([(one, one)], _literal(0))),
        (
            "piecewise_nested_numeric_condition",
            _binary("add", one, _piecewise([(_literal("1.5"), one)], _literal(0))),
        ),
        ("call_empty_name", _call("", [])),
        (
            "call_arguments_not_a_list",
            {
                "__type__": "call_expression",
                "__data__": {"function_name": "f", "arguments": one},
            },
        ),
        (
            "child_not_a_map",
            {
                "__type__": "unary_expression",
                "__data__": {"operation": "negate", "operand": 5},
            },
        ),
    ]


def _record_decode_error(name: str, payload: Any) -> dict[str, Any]:
    """Return one refused payload with the oracle's exception type."""
    table = _SlotTable()
    try:
        Expression.deserialize_from_dict(_bind_slots(payload, table))
    except Exception as error:  # recorded as data, whatever its type
        return {
            "name": name,
            "payload": _write_compact(payload),
            "oracle_error": type(error).__name__,
        }
    raise AssertionError(f"the oracle accepted the payload {name}")


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
    for function_name in _FUNCTION_NAMES:
        if try_get_registered_result_sort(function_name) is not None:
            raise AssertionError(f"{function_name} is registered; pick another name")
    rng = random.Random(arguments.seed)
    drawer = _TreeDrawer(rng)
    trees = _hand_picked_trees()
    trees += [
        (f"random_{index}", drawer.draw_tree(rng.randint(1, arguments.max_ops)))
        for index in range(arguments.random_count)
    ]
    wires = [wire for _, wire in trees]
    cases = [
        _record_case(name, index, wires, drawer, rng)
        for index, (name, _) in enumerate(trees)
    ]
    document = {
        "provenance": build_provenance(_REPOSITORY_ROOT, GENERATOR_COMMAND),
        "cases": cases,
        "decode_errors": [
            _record_decode_error(name, payload)
            for name, payload in _decode_error_payloads()
        ],
    }
    write_document(arguments.output, document)


if __name__ == "__main__":
    main()
