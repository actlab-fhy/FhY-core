"""Generate the serialization corpus: the V2 text the Python package writes.

Writes, for one object of each serializable class and variant, and for
values holding Python-defined parts, the class, the Rust type of its value,
the canonical V2 text ``to_json()`` writes, and the type ids of the foreign
parts it holds. The Rust replay
(``rust/fhy-core/tests/it/serialization_golden.rs``) reads each text into its
Rust type and writes it back byte-identically, so Python's and Rust's
serialization cannot drift apart (slice S17 of
``docs/design/python-switch.md``, D-S17-20).

Every case is checked here first: the text decodes to an object that writes
the same text again. Every identifier holds a fixed id, so the corpus is the
same in every process.

Run from the repository root:

    uv run --no-sync python rust/fhy-core/tests/golden/generate_serialization_cases.py

This overwrites ``rust/fhy-core/tests/golden/serialization_cases.json``.
Options select a larger random corpus written elsewhere, which the ignored
expanded-corpus replay reads from the file named in
``FHY_SERIALIZATION_CORPUS``; ``uv run nox -s golden_expanded`` generates and
replays an expanded corpus for every generator.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from _golden_support import add_corpus_arguments, build_provenance, write_document

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPOSITORY_ROOT))

from fhy_core.diagnostic import Note, NoteKind  # noqa: E402
from fhy_core.identifier import Identifier  # noqa: E402
from fhy_core.op_attribute import OpAttribute  # noqa: E402
from fhy_core.provenance import Position, Provenance, Span  # noqa: E402
from fhy_core.serialization import Serializable  # noqa: E402
from fhy_core.symbol_table import (  # noqa: E402
    SymbolTable,
    SymbolTableFrame,
    VariableSymbolTableFrame,
)
from fhy_core.symbolic.constraint import (  # noqa: E402
    Constraint,
    ConstraintSystem,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import (  # noqa: E402
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.symbolic.param import (  # noqa: E402
    CategoricalDomain,
    Param,
    ParamAssignment,
    ParamDomain,
)
from fhy_core.types import (  # noqa: E402
    CoreDataType,
    DataType,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
)
from fhy_core.value_domain import ValueDomain  # noqa: E402
from tests.serialization.foreign_parts import (  # noqa: E402  # after the path it is found on
    GoldenDomain,
    GoldenEven,
    GoldenFrame,
    GoldenToken,
    GoldenType,
)
from tests.serialization.frozen_fixtures import build_fixtures  # noqa: E402

GENERATOR_COMMAND = (
    "uv run --no-sync python rust/fhy-core/tests/golden/generate_serialization_cases.py"
)
_DEFAULT_OUTPUT = Path(__file__).with_name("serialization_cases.json")
# The random cases of the committed corpus, and the largest node count of
# one random expression.
_RANDOM_CASE_COUNT = 20
_MAX_NODES = 12
# The first fixed id of the random cases' identifiers.
_RANDOM_ID_BASE = 62_000

# The Rust type each class's value serializes as, by the family base or
# class it belongs to, most specific first.
_RUST_TYPES: tuple[tuple[type, str], ...] = (
    (Identifier, "Identifier"),
    (Position, "Position"),
    (Span, "Span"),
    (Note, "Note"),
    (NoteKind, "NoteKind"),
    (OpAttribute, "OpAttribute"),
    (ValueDomain, "ValueDomain"),
    (Provenance, "Provenance"),
    (Expression, "Expression"),
    (DataType, "DataType"),
    (Type, "Type"),
    (SymbolTableFrame, "SymbolFrame"),
    (SymbolTable, "SymbolTable"),
    (ConstraintSystem, "ConstraintSystem"),
    (Constraint, "Constraint"),
    (ParamDomain, "ParamDomain"),
    (ParamAssignment, "ParamAssignment"),
    (Param, "Param"),
)


def _family(instance: Serializable) -> tuple[type, str]:
    """Return the class or family base of `instance` and its value's Rust type."""
    for cls, name in _RUST_TYPES:
        if isinstance(instance, cls):
            return cls, name
    raise TypeError(f"no Rust type for {type(instance).__name__}")


def _class_path(instance: Serializable) -> str:
    """Return the class a reader decodes `instance`'s text with.

    A Python-defined part of this corpus decodes through its family base, as
    a reader of a foreign part's family would.
    """
    cls: type = type(instance)
    if cls.__module__.endswith("foreign_parts"):
        cls = _family(instance)[0]
    return f"{cls.__module__}.{cls.__qualname__}"


def _identifier(identifier_id: int, name_hint: str) -> Identifier:
    return Identifier.deserialize_from_dict(
        {"id": identifier_id, "name_hint": name_hint}
    )


def _foreign_type_ids(document: Any) -> list[str]:
    """Return the type id of every foreign part in `document`, in order."""
    found: list[str] = []
    pending = [document]
    while pending:
        value = pending.pop(0)
        if isinstance(value, dict):
            if set(value) == {"type_id", "data"} and isinstance(value["data"], str):
                found.append(value["type_id"])
                continue
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return found


def _foreign_cases() -> dict[str, Serializable]:
    """Return values holding each kind of Python-defined part."""
    x = _identifier(61_900, "x")
    y = _identifier(61_901, "y")
    ns = _identifier(61_902, "ns")
    frame_name = _identifier(61_903, "f")
    scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    table = SymbolTable()
    table.add_namespace(ns)
    table.add_symbol(ns, frame_name, GoldenFrame(frame_name, "noted"))
    table.add_symbol(
        ns, x, VariableSymbolTableFrame(x, GoldenType("tile"), TypeQualifier.STATE)
    )
    return {
        "foreign_constraint": GoldenEven(x),
        "foreign_constraint_system": ConstraintSystem(
            (
                GoldenEven(x),
                InSetConstraint(y, [GoldenToken(2), GoldenToken(1), 3]),
                EquationConstraint(IdentifierExpression(y) > 0),
            )
        ),
        "foreign_member": NotInSetConstraint(x, [GoldenToken(5)]),
        "foreign_domain": GoldenDomain(),
        "foreign_categorical_domain": CategoricalDomain(
            (GoldenToken(2), GoldenToken(1))  # type: ignore[arg-type]  # compared by ==
        ),
        "foreign_param": Param(GoldenDomain(), _identifier(61_904, "p")),
        "foreign_assignment": ParamAssignment(
            Param(
                CategoricalDomain(
                    (GoldenToken(1), GoldenToken(2))  # type: ignore[arg-type]  # by ==
                ),
                _identifier(61_905, "q"),
            ),
            GoldenToken(2),
        ),
        "foreign_type": GoldenType("tile"),
        "foreign_type_in_frame": VariableSymbolTableFrame(
            y, GoldenType("tile"), TypeQualifier.INPUT
        ),
        "foreign_frame": GoldenFrame(frame_name, "noted"),
        "foreign_frames_in_table": table,
        "shape_of_a_scalar": scalar,
    }


_LEAF_FLOATS = (0.5, -2.25, 1e300, 5e-324, float("inf"), float("nan"))


def _random_leaf(rng: random.Random, identifiers: list[Identifier]) -> Expression:
    """Return a random reference or literal of any literal kind."""
    makers = (
        lambda: IdentifierExpression(rng.choice(identifiers)),
        lambda: LiteralExpression(rng.randrange(-(2**70), 2**70)),
        lambda: LiteralExpression(rng.choice(_LEAF_FLOATS)),
        lambda: LiteralExpression(
            Decimal(f"{rng.randrange(1000)}.{rng.randrange(1000):03d}")
        ),
        lambda: LiteralExpression(rng.random() < 0.5),  # noqa: PLR2004
    )
    return rng.choice(makers)()


def _random_expression(
    rng: random.Random, identifiers: list[Identifier], size: int
) -> Expression:
    """Return a random expression of about `size` nodes over `identifiers`."""
    if size <= 1:
        return _random_leaf(rng, identifiers)
    left = _random_expression(rng, identifiers, size // 2)
    right = _random_expression(rng, identifiers, size - size // 2 - 1)
    combine = rng.choice(
        (
            lambda: left + right,
            lambda: left * right,
            lambda: -left,
            lambda: left < right,
            lambda: left - right,
        )
    )
    return combine()


def _random_cases(seed: int, count: int, max_nodes: int) -> dict[str, Serializable]:
    """Return `count` random expressions and set constraints, seeded."""
    rng = random.Random(seed)
    identifiers = [
        _identifier(_RANDOM_ID_BASE + index, f"v{index}") for index in range(8)
    ]
    cases: dict[str, Serializable] = {}
    for index in range(count):
        size = rng.randrange(1, max_nodes + 1)
        if index % 4 == 3:  # noqa: PLR2004  # every fourth case is a set
            members: list[Any] = [
                rng.choice(
                    [
                        rng.randrange(-50, 50),
                        f"s{rng.randrange(9)}",
                        rng.random() * 8,
                        True,
                    ]
                )
                for _ in range(rng.randrange(1, 6))
            ]
            cases[f"random_{index:04d}"] = InSetConstraint(
                rng.choice(identifiers), members
            )
        else:
            cases[f"random_{index:04d}"] = _random_expression(rng, identifiers, size)
    return cases


def _build_case(name: str, instance: Serializable) -> dict[str, Any]:
    """Return the corpus case of `instance`, checked to round-trip."""
    text = instance.to_json()
    decoded = type(instance).from_json(text)
    if decoded.to_json() != text:
        raise AssertionError(f"case {name} does not round-trip: {text}")
    if json.loads(text) != instance.serialize_to_dict():
        raise AssertionError(f"case {name}: the dict is not the text's")
    return {
        "name": name,
        "class": _class_path(instance),
        "rust_type": _family(instance)[1],
        "v2": text,
        "foreign": _foreign_type_ids(json.loads(text)),
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    add_corpus_arguments(
        parser,
        seed=17,
        random_count=_RANDOM_CASE_COUNT,
        max_ops=_MAX_NODES,
        default_output=_DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def main() -> None:
    """Write the corpus the options select."""
    arguments = _parse_arguments()
    cases = {
        **build_fixtures(),
        **_foreign_cases(),
        **_random_cases(arguments.seed, arguments.random_count, arguments.max_ops),
    }
    document = {
        "provenance": build_provenance(_REPOSITORY_ROOT, GENERATOR_COMMAND),
        "cases": [_build_case(name, instance) for name, instance in cases.items()],
    }
    write_document(arguments.output, document)


if __name__ == "__main__":
    main()
