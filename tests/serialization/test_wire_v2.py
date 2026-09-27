"""Tests of the V2 wire format: the Rust core's serde shapes, from Python.

V2 is the default (slice S17 of ``docs/design/python-switch.md``). A class
the Rust core backs writes exactly what the core writes, so each case of
the golden serialization corpus, which the Rust replay rewrites
byte-identically, is pinned here from the Python side; a Python-defined part
of a Rust value is a foreign part its registered class decodes.
"""

import importlib
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from fhy_core import serialization
from fhy_core.identifier import Identifier
from fhy_core.provenance import Provenance
from fhy_core.serialization import (
    DeserializationValueError,
    MalformedPayloadError,
    Serializable,
    SerializationFormat,
    SerializedDict,
    UnknownTypeIdError,
    WireVersion,
    WrappedFamilySerializable,
    current_wire_version,
    deserialize_value,
    register_serializable,
    serialize_value,
    wire_version,
)
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintSystem,
    EquationConstraint,
    InSetConstraint,
)
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.symbolic.param import ParamDomain
from fhy_core.types import Type
from fhy_core.utils.override import override

from .foreign_parts import GoldenDomain, GoldenEven, GoldenToken, GoldenType

_CORPUS = (
    Path(__file__).parents[2] / "rust/fhy-core/tests/golden/serialization_cases.json"
)
_ALL_CASES: list[dict[str, Any]] = json.loads(_CORPUS.read_text(encoding="utf-8"))[
    "cases"
]
# A member value is no ``Serializable``: ``serialize_value`` writes it.
_CASES = [case for case in _ALL_CASES if case["rust_type"] != "Value"]
_VALUE_CASES = [case for case in _ALL_CASES if case["rust_type"] == "Value"]


def _class_of(case: dict[str, Any]) -> type[Serializable]:
    module, _, name = case["class"].rpartition(".")
    cls: type[Serializable] = getattr(importlib.import_module(module), name)
    return cls


@pytest.mark.parametrize("case", _CASES, ids=[case["name"] for case in _CASES])
def test_a_corpus_text_reads_and_writes_back_byte_identically(
    case: dict[str, Any],
) -> None:
    """Test each golden V2 text decodes and writes back as itself, in every form."""
    cls = _class_of(case)
    text = case["v2"]

    from_json = cls.from_json(text)
    from_dict = cls.deserialize_from_dict(json.loads(text))

    assert from_json.to_json() == text
    assert from_dict.to_json() == text
    assert from_json.serialize_to_dict() == json.loads(text)
    assert (
        cls.deserialize(
            from_json.serialize(SerializationFormat.BINARY), SerializationFormat.BINARY
        ).to_json()
        == text
    )


@pytest.mark.parametrize(
    "case", _VALUE_CASES, ids=[case["name"] for case in _VALUE_CASES]
)
def test_a_corpus_value_reads_and_writes_back_byte_identically(
    case: dict[str, Any],
) -> None:
    """Test each golden member value decodes and writes back as its V2 text."""
    text = case["v2"]

    value = deserialize_value(json.loads(text))

    assert type(value).__name__ == case["class"].rpartition(".")[2]
    assert serialize_value(value) == json.loads(text)
    assert (
        json.dumps(serialize_value(value), separators=(",", ":"), ensure_ascii=False)
        == text
    )


def test_the_canonical_text_escapes_as_serde_json_does() -> None:
    """Test the text of control, non-ASCII and quote characters is serde_json's."""
    name = 'a\x00\x1f\x7f\u2028"\\/é😀\n'
    identifier = Identifier(name)

    text = IdentifierExpression(identifier).to_json()

    escaped = 'a\\u0000\\u001f\x7f\u2028\\"\\\\/é😀\\n'
    node = f'{{"identifier":{{"id":{identifier.id},"name_hint":"{escaped}"}}}}'
    assert text == f'{{"nodes":[{node}]}}'


def test_indent_and_sort_keys_re_format_the_canonical_text() -> None:
    """Test `indent` and `sort_keys` give the same value in another layout."""
    expression = IdentifierExpression(Identifier("x")) + LiteralExpression(1.5)

    canonical = expression.to_json()

    assert "\n" in expression.to_json(indent=2)
    assert json.loads(expression.to_json(indent=2)) == json.loads(canonical)
    assert json.loads(expression.to_json(sort_keys=True)) == json.loads(canonical)
    assert expression.to_json(sort_keys=True) != canonical


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), -0.0, 5e-324, 1e300]
)
def test_a_float_literal_of_any_value_round_trips(value: float) -> None:
    """Test V2 writes every float literal, NaN and the infinities included."""
    literal = LiteralExpression(value)

    rebuilt = Expression.from_json(literal.to_json())

    assert isinstance(rebuilt, LiteralExpression)
    if math.isnan(value):
        assert math.isnan(rebuilt.value)
    else:
        assert repr(rebuilt.value) == repr(float(value))


def test_a_shared_subtree_is_written_once() -> None:
    """Test a node reached twice appears once in the node table."""
    shared = IdentifierExpression(Identifier("x")) + LiteralExpression(1)

    nodes = json.loads((shared * shared).to_json())["nodes"]

    assert len(nodes) == 4


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(True, {"bool": True}, id="bool"),
        pytest.param(2**80, {"int": str(2**80)}, id="big_int"),
        pytest.param(-3, {"int": "-3"}, id="negative_int"),
        pytest.param(2.5, {"float": "2.5"}, id="float"),
        pytest.param("é", {"str": "é"}, id="str"),
        pytest.param((1, "a"), {"tuple": [{"int": "1"}, {"str": "a"}]}, id="tuple"),
        pytest.param(
            frozenset({"b", 2, 0.5}),
            {"frozen_set": [{"float": "0.5"}, {"int": "2"}, {"str": "b"}]},
            id="frozenset_in_canonical_order",
        ),
        pytest.param(
            GoldenToken(3),
            {"opaque": {"type_id": "golden.token", "data": '{"value":3}'}},
            id="serializable",
        ),
    ],
)
def test_a_value_has_the_cores_form(value: Any, expected: dict[str, Any]) -> None:
    """Test `serialize_value` writes the core's form; `deserialize_value` reads it."""
    assert serialize_value(value) == expected
    assert deserialize_value(expected) == value


def test_a_python_defined_constraint_writes_its_foreign_part() -> None:
    """Test a Python-defined constraint is a foreign part, alone or in a system."""
    x = Identifier("x")
    even = GoldenEven(x)
    system = ConstraintSystem((even, InSetConstraint(x, [GoldenToken(1)])))

    payload = even.serialize_to_dict()
    rebuilt = ConstraintSystem.from_json(system.to_json())

    assert payload == {
        "custom": {
            "type_id": "golden.even",
            "data": json.dumps(
                {"variable": {"id": x.id, "name_hint": "x"}}, separators=(",", ":")
            ),
        }
    }
    assert Constraint.deserialize_from_dict(payload).is_structurally_equivalent(even)
    assert GoldenEven.from_json(even.to_json()).is_structurally_equivalent(even)
    assert rebuilt.is_structurally_equivalent(system)
    assert isinstance(rebuilt.constraints[0], GoldenEven)


def test_python_defined_domains_and_types_write_their_foreign_parts() -> None:
    """Test a Python-defined domain and type are foreign parts."""
    domain = GoldenDomain()
    ty = GoldenType("tile")

    assert domain.serialize_to_dict() == {
        "custom": {"type_id": "golden.domain", "data": '{"modulus":2}'}
    }
    assert isinstance(ParamDomain.from_json(domain.to_json()), GoldenDomain)
    assert ty.serialize_to_dict() == {
        "extension": {"type_id": "golden.type", "data": '{"tag":"tile"}'}
    }
    rebuilt = Type.deserialize_from_dict(ty.serialize_to_dict())
    assert isinstance(rebuilt, GoldenType)
    assert rebuilt.tag == "tile"


def test_a_foreign_part_of_an_unknown_type_id_is_refused() -> None:
    """Test a foreign part is resolved through the registry only."""
    payload: SerializedDict = {"custom": {"type_id": "nowhere.Nothing", "data": "{}"}}

    with pytest.raises(UnknownTypeIdError, match=r"nowhere\.Nothing"):
        Constraint.deserialize_from_dict(payload)


class _Raising(Serializable):
    """A member whose payload hook raises."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    @override
    def __eq__(self, other: object) -> bool:
        return other is self

    @override
    def __hash__(self) -> int:
        return id(self)

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        if getattr(self, "armed", False):
            raise self.error
        return {}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "_Raising":
        raise NotImplementedError


@pytest.mark.parametrize(
    "error",
    [ValueError("hook failed"), KeyboardInterrupt()],
    ids=["value_error", "interrupt"],
)
def test_a_raising_part_hook_propagates_as_itself(error: BaseException) -> None:
    """Test the exception a part's payload hook raises reaches the caller unchanged."""
    member = _Raising(error)
    constraint = InSetConstraint(Identifier("x"), [member])
    member.armed = True  # type: ignore[attr-defined]  # test: arm the hook after keying

    with pytest.raises(type(error)) as raised:
        constraint.to_json()

    assert raised.value is error


@pytest.mark.parametrize(
    ("cls", "payload"),
    [
        pytest.param(Expression, {"nodes": []}, id="empty_table"),
        pytest.param(
            Expression, {"nodes": [{"literal": {"int": "007"}}]}, id="bad_int"
        ),
        pytest.param(Constraint, {"in_between": {}}, id="unknown_variant"),
        pytest.param(Provenance, {"named": {"name": "n"}}, id="missing_field"),
    ],
)
def test_a_malformed_v2_payload_raises_the_deserialization_value_error(
    cls: type[Serializable], payload: dict[str, Any]
) -> None:
    """Test a V2 payload of another shape fails with the core's text."""
    with pytest.raises(DeserializationValueError, match="Invalid V2 payload"):
        cls.deserialize_from_dict(payload)


def test_text_that_is_not_json_is_a_malformed_payload() -> None:
    """Test `from_json` of text that is not JSON raises `MalformedPayloadError`."""
    with pytest.raises(MalformedPayloadError):
        Expression.from_json("{not json")


@register_serializable(type_id="test_wire_v2.shape")
class _Shape(WrappedFamilySerializable):
    """A pure-Python family base."""


@register_serializable(type_id="test_wire_v2.circle")
@dataclass(frozen=True)
class _Circle(_Shape):
    radius: int
    center: Expression


def test_a_python_family_writes_its_type_id_as_the_tag() -> None:
    """Test a pure-Python family member is `{type_id: data}` under V2."""
    circle = _Circle(2, LiteralExpression(0))

    payload = circle.serialize_to_dict()

    assert payload == {
        "test_wire_v2.circle": {
            "radius": 2,
            "center": {"nodes": [{"literal": {"int": "0"}}]},
        }
    }
    assert _Shape.deserialize_from_dict(payload) == _Circle(2, LiteralExpression(0))


def test_the_version_is_per_thread() -> None:
    """Test a V1 block in one thread leaves another thread writing V2."""
    seen: list[WireVersion] = []
    inside = threading.Event()
    done = threading.Event()

    def other_thread() -> None:
        inside.wait()
        seen.append(current_wire_version())
        done.set()

    thread = threading.Thread(target=other_thread)
    thread.start()
    with pytest.warns(DeprecationWarning), wire_version(WireVersion.V1):
        inside.set()
        done.wait()
    thread.join()

    assert seen == [WireVersion.V2]
    assert current_wire_version() is WireVersion.V2


def test_a_member_key_does_not_depend_on_the_version() -> None:
    """Test a `Serializable` member is keyed by its V2 payload in either version."""
    x = Identifier("x")
    members = [EquationConstraint(IdentifierExpression(x) > 0)]
    v2 = InSetConstraint(x, [GoldenToken(2), GoldenToken(10)])
    with pytest.warns(DeprecationWarning), wire_version(WireVersion.V1):
        v1 = InSetConstraint(x, [GoldenToken(2), GoldenToken(10)])

    assert v1.values == v2.values
    assert members


def test_the_module_names_the_versions() -> None:
    """Test the framework exports the versions and the value functions."""
    assert {
        "WireVersion",
        "wire_version",
        "serialize_value",
        "upgrade_v1_payload",
    } <= set(serialization.__all__)
