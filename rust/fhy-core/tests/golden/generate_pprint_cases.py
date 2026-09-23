"""Generate golden printed-expression cases from the Python printer oracle.

Each case is one expression tree, given by its wire dict
(``serialize_to_dict``) with every identifier written as ``{"slot": name}``
instead of ``{"id": ..., "name_hint": ...}``. The document's ``slots`` block
maps every slot to the name hint its identifier carries; a slot's name hint
may differ from the slot name, and two slots may share a name hint.

For each tree the oracle records ``pformat_expression`` in all four option
combinations:

* ``symbolic`` and ``functional``: the text with ``show_id=False``, exactly;
* ``symbolic_with_ids`` and ``functional_with_ids``: the text with
  ``show_id=True`` as a list of segments, each a string written verbatim or
  ``{"slot": name}`` standing for that slot's identifier reference.

Why segments: with ``show_id=True`` an identifier prints as
``name_hint::id``, and the id is whatever the identifier counter handed out,
which differs between the oracle's process and the Rust replay. So the
generator mints one identifier per slot and case, binds the tree to those
identifiers, prints it, and then cuts every ``name_hint::id`` of a minted
identifier out of the text as a slot segment. The Rust replay mints its own
identifier per slot and case, deserializes the same tree with them (a
deserialized identifier keeps its payload's id), and expands each slot
segment to ``name_hint::id`` with the id of the identifier it minted for that
slot. The expected text is therefore exactly the text the oracle would print
had its counter handed out the replay's ids.

The cut is checked twice before a case is written: the slot segments, in
order, must be exactly the tree's identifier references in the order the
printer visits them (counted per slot), and replacing each slot segment by
the slot's bare name hint must give back the ``show_id=False`` text. An id is
matched only when no digit follows it, and ids are unique, so a
``name_hint::id`` inside a longer name hint cannot be mistaken for another
slot's.

No literal is NaN or infinite, since JSON cannot carry one, and no tree is
deep enough to exceed the JSON nesting limit of the Rust replay's parser.

The Rust replay is ``rust/fhy-core/tests/pprint_equivalence.rs``.

Run from the repository root:

    uv run --no-sync python rust/fhy-core/tests/golden/generate_pprint_cases.py

This overwrites ``rust/fhy-core/tests/golden/pprint_cases.json``.
``--random-count`` sets how many random trees are drawn, ``--max-ops`` sets
the largest leaf count of a random tree, and ``--seed`` seeds the draws. The
ignored expanded-corpus test reads a corpus written with other options from
the file named in ``FHY_PPRINT_CORPUS``.
"""

from __future__ import annotations

import argparse
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from _golden_support import add_corpus_arguments, build_provenance, write_document

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryOperation,
    Expression,
    UnaryOperation,
    pformat_expression,
)

GENERATOR_COMMAND = (
    "uv run --no-sync python rust/fhy-core/tests/golden/generate_pprint_cases.py"
)

_RANDOM_SEED = 20260922
_RANDOM_COUNT = 80
_RANDOM_MAX_LEAVES = 12

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_OUTPUT = Path(__file__).resolve().with_name("pprint_cases.json")

# Slot name -> name hint of the identifier minted for it. ``x_twin`` shares
# ``x``'s name hint; the others spell name hints the printer writes raw.
_SLOT_NAME_HINTS = {
    "x": "x",
    "y": "y",
    "z": "z",
    "p": "p",
    "q": "q",
    "x_twin": "x",
    "spaced": "a b",
    "colons": "m::n",
    "accented": "\u00e9t\u00e9",
    "digits": "42",
    "true_word": "True",
    "parenthesized": "(p)",
}
_PLAIN_SLOTS = ("x", "y", "z", "p", "q")
_FUNCTION_NAMES = ("f", "g", "h")
_MAX_CASES = 3
_MAX_ARGUMENTS = 3

_DEEP_CHAIN_DEPTH = 30
"""Depth of the hand-picked chains, well inside the replay parser's nesting limit."""

_WIDE_PIECEWISE_CASES = 30


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
    """Identifiers minted for the slots of one case."""

    def __init__(self) -> None:
        self._by_slot = {
            slot: Identifier(name_hint) for slot, name_hint in _SLOT_NAME_HINTS.items()
        }

    def bind(self, wire: Any) -> Any:
        """Return ``wire`` with every slot placeholder replaced by an identifier."""
        if isinstance(wire, dict):
            if set(wire) == {"slot"}:
                identifier = self._by_slot[wire["slot"]]
                return {"id": identifier.id, "name_hint": identifier.name_hint}
            return {key: self.bind(value) for key, value in wire.items()}
        if isinstance(wire, list):
            return [self.bind(item) for item in wire]
        return wire

    def cut_identifiers(self, text: str) -> list[str | dict[str, str]]:
        """Return ``text`` split into verbatim strings and slot segments.

        Every ``name_hint::id`` of a minted identifier not followed by a digit
        becomes ``{"slot": name}``; the text between them is kept verbatim.
        """
        slot_by_repr = {
            repr(identifier): slot for slot, identifier in self._by_slot.items()
        }
        alternatives = sorted(slot_by_repr, key=len, reverse=True)
        escaped = "|".join(re.escape(alternative) for alternative in alternatives)
        pattern = re.compile(f"({escaped})" + r"(?!\d)")
        segments: list[str | dict[str, str]] = []
        for index, piece in enumerate(pattern.split(text)):
            if index % 2 == 1:
                segments.append(_slot(slot_by_repr[piece]))
            elif piece:
                segments.append(piece)
        return segments


def _read_piecewise_children(data: dict[str, Any], functional: bool) -> list[Any]:
    """Return a piecewise's children in printing order.

    Symbolic notation writes a case's value before its condition; functional
    notation writes the condition first.
    """
    children: list[Any] = []
    for condition, value in zip(data["conditions"], data["values"], strict=True):
        children += [condition, value] if functional else [value, condition]
    return [*children, data["otherwise"]]


# Type id -> reader of a node's children in printing order, given its data
# and whether the notation is functional.
_CHILD_READERS: dict[str, Callable[[dict[str, Any], bool], list[Any]]] = {
    "literal_expression": lambda data, functional: [],
    "unary_expression": lambda data, functional: [data["operand"]],
    "binary_expression": lambda data, functional: [data["left"], data["right"]],
    "piecewise_expression": _read_piecewise_children,
    "call_expression": lambda data, functional: list(data["arguments"]),
}


def _collect_references(wire: dict[str, Any], functional: bool) -> list[str]:
    """Return the slots of ``wire``'s identifier references in printing order."""
    data = wire["__data__"]
    if wire["__type__"] == "identifier_expression":
        return [data["identifier"]["slot"]]
    slots: list[str] = []
    for child in _CHILD_READERS[wire["__type__"]](data, functional):
        slots += _collect_references(child, functional)
    return slots


def _check_cut(
    segments: list[str | dict[str, str]],
    plain_text: str,
    wire: dict[str, Any],
    *,
    functional: bool,
) -> None:
    """Check the cut against the plain text and the tree's references."""
    cut_slots = [segment["slot"] for segment in segments if isinstance(segment, dict)]
    if cut_slots != _collect_references(wire, functional):
        raise AssertionError(f"cut {segments!r} disagrees with the tree {wire!r}")
    rejoined = "".join(
        segment if isinstance(segment, str) else _SLOT_NAME_HINTS[segment["slot"]]
        for segment in segments
    )
    if rejoined != plain_text:
        raise AssertionError(f"cut {segments!r} does not rejoin to {plain_text!r}")


def _record_case(name: str, wire: dict[str, Any]) -> dict[str, Any]:
    """Return one golden case: the tree and its four printed forms."""
    table = _SlotTable()
    expression = Expression.deserialize_from_dict(table.bind(wire))
    case: dict[str, Any] = {"name": name, "tree": wire}
    for key, functional in (("symbolic", False), ("functional", True)):
        plain_text = pformat_expression(expression, functional=functional)
        segments = table.cut_identifiers(
            pformat_expression(expression, show_id=True, functional=functional)
        )
        _check_cut(segments, plain_text, wire, functional=functional)
        case[key] = plain_text
        case[f"{key}_with_ids"] = segments
    return case


_SMALL_INTEGER_BOUND = 20
_LARGE_INTEGER_BOUND = 10**30
_FLOAT_BOUND = 100.0
_MAX_FLOAT_DIGITS = 3
_MAX_TEXT_INTEGER = 999
_MAX_TEXT_PADDING = 4
_SPECIAL_FLOATS = (
    0.0,
    -0.0,
    1e16,
    1e-05,
    1e-07,
    2.5,
    1 / 3,
    5e-324,
    1.7976931348623157e308,
    1e22,
)

_LEAF_PROBABILITY = 0.25
"""Chance that an inner draw stops at a leaf although its budget allows more."""

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
        """Return a random decimal text, possibly without a whole or fraction part."""
        rng = self._rng
        whole = str(rng.randint(0, _MAX_TEXT_INTEGER))
        fraction = str(rng.randint(0, _MAX_TEXT_INTEGER))
        shape = rng.randrange(4)
        if shape == 0:
            return f".{fraction}"
        if shape == 1:
            return f"{whole}."
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
            lambda: rng.choice((-1.0, 1.0)) * 10.0 ** rng.randint(-30, 30),
            self._draw_integer_text,
            self._draw_decimal_text,
            lambda: rng.choice(_SPECIAL_FLOATS),
        )
        return _literal(rng.choice(draws)())

    def draw_leaf(self) -> dict[str, Any]:
        """Return a random identifier reference or literal."""
        rng = self._rng
        draws = (
            lambda: _identifier(rng.choice(tuple(_SLOT_NAME_HINTS))),
            self.draw_literal,
        )
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
        """Return a random call."""
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


def _build_chain(depth: int, wrap: Any, leaf: dict[str, Any]) -> dict[str, Any]:
    """Return ``leaf`` wrapped ``depth`` times by ``wrap``."""
    tree = leaf
    for _ in range(depth):
        tree = wrap(tree)
    return tree


def _hand_picked_literal_trees() -> list[tuple[str, dict[str, Any]]]:
    """Return one literal per printed form the printer must reproduce."""
    values: list[tuple[str, bool | int | float | str]] = [
        ("true", True),
        ("false", False),
        ("zero", 0),
        ("negative_one", -1),
        ("big_int", 10**30),
        ("negative_big_int", -(2**64)),
        ("float_one", 1.0),
        ("float_tenth", 0.1),
        ("float_third", 1 / 3),
        ("float_positional_limit", 1e15),
        ("float_1e16", 1e16),
        ("float_1e22", 1e22),
        ("float_long_mantissa", 1.2345678901234568e17),
        ("float_1e_minus_4", 0.0001),
        ("float_1e_minus_5", 1e-05),
        ("float_1e_minus_7", 1e-07),
        ("float_negative_zero", -0.0),
        ("float_subnormal", 5e-324),
        ("float_max", 1.7976931348623157e308),
        ("float_integral", 123456789.0),
        ("float_negative", -2.5),
        ("integer_text", "05"),
        ("integer_text_zero", "0"),
        ("decimal_text_trailing_zero", "1.50"),
        ("decimal_text_no_whole", ".5"),
        ("decimal_text_no_fraction", "1."),
        ("decimal_text_zeros", "000.000"),
        ("decimal_text_long", "3.1415926535897932384626433832795028841971"),
    ]
    return [(f"literal_{name}", _literal(value)) for name, value in values]


def _hand_picked_operation_trees() -> list[tuple[str, dict[str, Any]]]:
    """Return every operation over simple operands, both unary operand signs."""
    x, two = _identifier("x"), _literal(2)
    trees = [
        (f"unary_{operation}", _unary(str(operation), x))
        for operation in UnaryOperation
    ]
    trees += [
        (f"unary_{operation}_of_negative_literal", _unary(str(operation), _literal(-1)))
        for operation in UnaryOperation
    ]
    trees += [
        (f"binary_{operation}", _binary(str(operation), x, two))
        for operation in BinaryOperation
    ]
    return trees


def _hand_picked_shape_trees() -> list[tuple[str, dict[str, Any]]]:
    """Return trees whose shape the parentheses must show."""
    x, y, z = _identifier("x"), _identifier("y"), _identifier("z")
    p, q = _identifier("p"), _identifier("q")
    one, two = _literal(1), _literal(2)
    true, false = _literal(True), _literal(False)
    return [
        ("sum_of_product", _binary("add", x, _binary("multiply", y, two))),
        ("product_of_sum", _binary("multiply", _binary("add", x, y), two)),
        (
            "right_nested_difference",
            _binary("subtract", x, _binary("subtract", y, one)),
        ),
        ("left_nested_difference", _binary("subtract", _binary("subtract", x, y), one)),
        ("right_nested_power", _binary("power", x, _binary("power", y, two))),
        (
            "right_folded_conjunction",
            _binary("logical_and", p, _binary("logical_and", q, _identifier("z"))),
        ),
        ("negated_sum", _unary("negate", _binary("add", x, one))),
        ("double_negation", _unary("negate", _unary("negate", x))),
        ("not_of_comparison", _unary("logical_not", _binary("less", x, y))),
        ("one_case_piecewise", _piecewise([(true, one)], two)),
        ("two_case_piecewise", _piecewise([(true, one), (false, two)], _literal(3))),
        (
            "three_case_piecewise",
            _piecewise(
                [
                    (_binary("greater", x, _literal(0.0)), one),
                    (_binary("less", x, _literal(0.0)), _literal(-1)),
                    (p, _literal(0)),
                ],
                z,
            ),
        ),
        (
            "piecewise_in_value",
            _piecewise([(true, _piecewise([(false, one)], two))], _literal(3)),
        ),
        (
            "piecewise_in_condition",
            _piecewise([(_piecewise([(p, q)], false), x)], y),
        ),
        (
            "piecewise_in_otherwise",
            _piecewise([(p, x)], _piecewise([(q, y)], z)),
        ),
        (
            "wide_piecewise",
            _piecewise(
                [
                    (_binary("equal", x, _literal(index)), _literal(index))
                    for index in range(_WIDE_PIECEWISE_CASES)
                ],
                _literal(-1),
            ),
        ),
        ("zero_argument_call", _call("f", [])),
        ("one_argument_call", _call("f", [x])),
        ("three_argument_call", _call("g", [x, one, _literal("1.50")])),
        ("call_in_first_argument", _call("f", [_call("g", [x, one]), two])),
        ("call_in_last_argument", _call("f", [one, _call("g", [])])),
        ("call_of_piecewise", _call("h", [_piecewise([(p, x)], y)])),
        (
            "piecewise_of_calls",
            _piecewise([(_call("f", [p]), _call("g", []))], _call("h", [q])),
        ),
        (
            "gelu_body",
            _binary(
                "multiply",
                _binary("multiply", _literal(0.5), x),
                _binary(
                    "add",
                    _literal(1.0),
                    _call(
                        "erf", [_binary("divide", x, _call("sqrt", [_literal(2.0)]))]
                    ),
                ),
            ),
        ),
        (
            "sign_body",
            _piecewise(
                [
                    (_binary("greater", x, _literal(0.0)), one),
                    (_binary("less", x, _literal(0.0)), _literal(-1)),
                ],
                _literal(0),
            ),
        ),
        (
            "deep_left_sum",
            _build_chain(_DEEP_CHAIN_DEPTH, lambda tree: _binary("add", tree, one), x),
        ),
        (
            "deep_right_conjunction",
            _build_chain(
                _DEEP_CHAIN_DEPTH, lambda tree: _binary("logical_and", true, tree), p
            ),
        ),
        (
            "deep_negation",
            _build_chain(_DEEP_CHAIN_DEPTH, lambda tree: _unary("negate", tree), x),
        ),
        (
            "deep_piecewise_in_condition",
            _build_chain(
                _DEEP_CHAIN_DEPTH // 2,
                lambda tree: _piecewise([(tree, one)], _literal(0)),
                p,
            ),
        ),
        (
            "deep_call_in_first_argument",
            _build_chain(_DEEP_CHAIN_DEPTH, lambda tree: _call("f", [tree, two]), x),
        ),
    ]


def _hand_picked_name_hint_trees() -> list[tuple[str, dict[str, Any]]]:
    """Return trees over identifiers whose name hints the printer writes raw."""
    x, twin = _identifier("x"), _identifier("x_twin")
    trees = [
        (f"name_hint_{slot}", _identifier(slot))
        for slot in _SLOT_NAME_HINTS
        if slot not in _PLAIN_SLOTS
    ]
    trees += [
        ("same_name_hint_twins", _binary("subtract", x, twin)),
        ("same_name_hint_twins_in_call", _call("f", [twin, x, twin])),
        (
            "name_hints_in_piecewise",
            _piecewise(
                [(_identifier("true_word"), _identifier("digits"))],
                _identifier("colons"),
            ),
        ),
        (
            "digit_name_hint_beside_literal",
            _binary("add", _identifier("digits"), _literal(42)),
        ),
        (
            "true_name_hint_beside_literal",
            _binary("logical_and", _identifier("true_word"), _literal(True)),
        ),
    ]
    return trees


def _hand_picked_trees() -> list[tuple[str, dict[str, Any]]]:
    """Return every hand-picked tree, each named."""
    return (
        _hand_picked_literal_trees()
        + _hand_picked_operation_trees()
        + _hand_picked_shape_trees()
        + _hand_picked_name_hint_trees()
    )


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
    rng = random.Random(arguments.seed)
    drawer = _TreeDrawer(rng)
    trees = _hand_picked_trees()
    trees += [
        (f"random_{index}", drawer.draw_tree(rng.randint(1, arguments.max_ops)))
        for index in range(arguments.random_count)
    ]
    document = {
        "provenance": build_provenance(_REPOSITORY_ROOT, GENERATOR_COMMAND),
        "slots": _SLOT_NAME_HINTS,
        "cases": [_record_case(name, wire) for name, wire in trees],
    }
    write_document(arguments.output, document)


if __name__ == "__main__":
    main()
