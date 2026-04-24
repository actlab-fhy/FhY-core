"""Tests the identifier."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    SerializationFormat,
    SerializedDict,
)
from fhy_core.trait import Equal, PartialEqual

# =============================================================================
# Construction & ID generation
# =============================================================================


def test_identifier_initialization() -> None:
    """Test the constructor stores `name_hint` and assigns an `int` `id`."""
    identifier = Identifier("test_name")
    assert identifier.name_hint == "test_name"
    assert isinstance(identifier.id, int)


def test_consecutive_identifiers_have_consecutive_ids() -> None:
    """Test successive constructions yield `id`s that increase by exactly one."""
    base = Identifier("anchor").id
    next_a = Identifier("a").id
    next_b = Identifier("b").id
    assert next_a == base + 1
    assert next_b == base + 2


def test_concurrent_construction_yields_unique_ids() -> None:
    """Test constructing `Identifier`s across many threads yields unique `id`s."""
    num_identifiers = 2000
    with ThreadPoolExecutor(max_workers=32) as executor:
        identifiers = list(
            executor.map(lambda _: Identifier(""), range(num_identifiers))
        )
    ids = [identifier.id for identifier in identifiers]
    assert len(set(ids)) == num_identifiers


def test_concurrent_construction_and_deserialization_yield_unique_ids() -> None:
    """Test concurrent construction and deserialization yield unique `id`s."""
    num_constructed = 1000
    num_deserialized = 1000
    base = Identifier("anchor").id
    deserialize_payloads: list[SerializedDict] = [
        {"id": base + 10_000 + i, "name_hint": f"d{i}"} for i in range(num_deserialized)
    ]

    def construct(_: int) -> Identifier:
        return Identifier("c")

    def deserialize(payload: SerializedDict) -> Identifier:
        return Identifier.deserialize_from_dict(payload)

    with ThreadPoolExecutor(max_workers=32) as executor:
        constructed = list(executor.map(construct, range(num_constructed)))
        deserialized = list(executor.map(deserialize, deserialize_payloads))

    all_ids = [i.id for i in constructed] + [i.id for i in deserialized]
    assert len(set(all_ids)) == len(all_ids)


# =============================================================================
# Equality & hashing
# =============================================================================


def test_distinct_identifiers_are_not_equal() -> None:
    """Test two separately constructed `Identifier`s are unequal symmetrically."""
    a = Identifier("name")
    b = Identifier("name")
    assert a != b
    assert b != a


def test_identifier_equal_to_self() -> None:
    """Test an `Identifier` compares equal to itself."""
    a = Identifier("name")
    assert a == a  # noqa: PLR0124


def test_inequality_with_non_identifier_types() -> None:
    """Test an `Identifier` is not equal to non-`Identifier` values."""
    a = Identifier("x")
    assert a != "x"
    assert a != 0
    assert a != None  # noqa: E711
    assert a != object()


def test_equality_uses_value_not_identity_for_large_ids() -> None:
    """Test equality holds for two distinct objects sharing a large `id`."""
    payload: SerializedDict = {"id": 1_000_000, "name_hint": "x"}
    a = Identifier.deserialize_from_dict(payload)
    b = Identifier.deserialize_from_dict(payload)
    assert a is not b
    assert a == b


def test_hash_consistent_with_equality() -> None:
    """Test two equal `Identifier`s also share the same `hash`."""
    payload: SerializedDict = {"id": 1_000_001, "name_hint": "x"}
    a = Identifier.deserialize_from_dict(payload)
    b = Identifier.deserialize_from_dict(payload)
    assert a == b
    assert hash(a) == hash(b)


def test_identifier_usable_as_dict_key() -> None:
    """Test an `Identifier` as a `dict` key is retrievable via an equal one."""
    payload: SerializedDict = {"id": 1_000_002, "name_hint": "x"}
    a = Identifier.deserialize_from_dict(payload)
    b = Identifier.deserialize_from_dict(payload)
    mapping = {a: "value"}
    assert mapping[b] == "value"


def test_identifier_collapses_in_set() -> None:
    """Test two equal `Identifier`s collapse to a single `set` member."""
    payload: SerializedDict = {"id": 1_000_003, "name_hint": "x"}
    a = Identifier.deserialize_from_dict(payload)
    b = Identifier.deserialize_from_dict(payload)
    assert len({a, b}) == 1


def test_identifier_supports_equal_traits() -> None:
    """Test `Identifier` satisfies the equality trait protocols."""
    identifier = Identifier("name")
    assert isinstance(identifier, PartialEqual)
    assert isinstance(identifier, Equal)
    assert identifier.supports_partial_equality is True
    assert identifier.supports_equality is True


# =============================================================================
# String representations
# =============================================================================


def test_str_returns_name_hint() -> None:
    """Test `str()` of an `Identifier` returns its `name_hint` verbatim."""
    identifier = Identifier("test_repr")
    assert str(identifier) == "test_repr"


def test_repr_includes_name_hint_and_id() -> None:
    """Test `repr()` of an `Identifier` is `"<name_hint>::<id>"`."""
    identifier = Identifier("test_repr")
    assert repr(identifier) == f"test_repr::{identifier.id}"


# =============================================================================
# Serialization (happy paths)
# =============================================================================


def test_serialize_to_dict_round_trip_preserves_equality() -> None:
    """Test a `dict` round trip yields an `Identifier` equal to the original."""
    identifier = Identifier("serialization_test")
    serialized = identifier.serialize_to_dict()
    deserialized = Identifier.deserialize_from_dict(serialized)
    assert deserialized == identifier


def test_serialize_to_dict_round_trip_preserves_name_hint() -> None:
    """Test a `dict` round trip preserves `name_hint` as well as `id`."""
    identifier = Identifier("serialization_test")
    serialized = identifier.serialize_to_dict()
    deserialized = Identifier.deserialize_from_dict(serialized)
    assert repr(deserialized) == repr(identifier)


def test_serialize_via_top_level_format_dict() -> None:
    """Test serialize/deserialize via `SerializationFormat.DICT` round-trip."""
    identifier = Identifier("via_format")
    payload = identifier.serialize(SerializationFormat.DICT)
    deserialized = Identifier.deserialize(payload, SerializationFormat.DICT)
    assert repr(deserialized) == repr(identifier)


def test_serialize_via_top_level_format_json() -> None:
    """Test serialize/deserialize via `SerializationFormat.JSON` round-trip."""
    identifier = Identifier("via_json")
    payload = identifier.serialize(SerializationFormat.JSON)
    deserialized = Identifier.deserialize(payload, SerializationFormat.JSON)
    assert repr(deserialized) == repr(identifier)


def test_serialize_via_registry_binary_round_trip() -> None:
    """Test `to_bytes`/`from_bytes` resolves `Identifier` via the type-id registry."""
    identifier = Identifier("registry_test")
    blob = identifier.to_bytes()
    deserialized = Identifier.from_bytes(blob)
    assert isinstance(deserialized, Identifier)
    assert repr(deserialized) == repr(identifier)


def test_name_hint_empty_string_round_trips() -> None:
    """Test an empty-string name hint survives a `dict` round trip."""
    identifier = Identifier("")
    deserialized = Identifier.deserialize_from_dict(identifier.serialize_to_dict())
    assert deserialized.name_hint == ""


def test_name_hint_with_colons_round_trips() -> None:
    """Test a name hint containing "::" survives a `dict` round trip."""
    identifier = Identifier("foo::bar")
    deserialized = Identifier.deserialize_from_dict(identifier.serialize_to_dict())
    assert deserialized.name_hint == "foo::bar"
    assert str(deserialized) == "foo::bar"


# =============================================================================
# Deserialization (error cases)
# =============================================================================


def test_deserialize_missing_id_raises() -> None:
    """Test deserializing a `dict` missing `"id"` raises a structure error."""
    with pytest.raises(DeserializationDictStructureError):
        Identifier.deserialize_from_dict({"name_hint": "x"})


def test_deserialize_missing_name_hint_raises() -> None:
    """Test deserializing a `dict` missing `"name_hint"` raises a structure error."""
    with pytest.raises(DeserializationDictStructureError):
        Identifier.deserialize_from_dict({"id": 0})


def test_deserialize_wrong_id_type_raises() -> None:
    """Test deserializing a non-`int` `"id"` raises a structure error."""
    with pytest.raises(DeserializationDictStructureError):
        Identifier.deserialize_from_dict({"id": "not_an_int", "name_hint": "x"})


def test_deserialize_wrong_name_hint_type_raises() -> None:
    """Test deserializing a non-`str` `"name_hint"` raises a structure error."""
    with pytest.raises(DeserializationDictStructureError):
        Identifier.deserialize_from_dict({"id": 0, "name_hint": 123})


def test_deserialize_negative_id_raises() -> None:
    """Test deserializing a negative `"id"` raises a value error."""
    with pytest.raises(DeserializationValueError):
        Identifier.deserialize_from_dict({"id": -1, "name_hint": "x"})


def test_deserialize_zero_id_is_valid() -> None:
    """Test deserializing the boundary `id` `0` succeeds."""
    deserialized = Identifier.deserialize_from_dict({"id": 0, "name_hint": "x"})
    assert deserialized.id == 0


# =============================================================================
# Deserialization & generator-state interaction
#
# These tests pin the deserialization path's effect on the id generator
# (next-id bookkeeping) using only the public surface: they observe the id of
# the *next* identifier created via Identifier(...). They use relative offsets
# anchored on a freshly-constructed identifier so the suite needs no patching
# of class-level state.
# =============================================================================


def test_deserialize_higher_id_advances_generator() -> None:
    """Test deserializing an `id` above the generator advances it past that `id`."""
    base = Identifier("anchor").id
    Identifier.deserialize_from_dict({"id": base + 100, "name_hint": "x"})
    assert Identifier("y").id == base + 101


def test_deserialize_equal_boundary_advances_generator() -> None:
    """Test deserializing the next-to-be-issued `id` still advances the generator."""
    Identifier("a")
    b = Identifier("b")
    Identifier.deserialize_from_dict({"id": b.id + 1, "name_hint": "x"})
    assert Identifier("y").id == b.id + 2


def test_deserialize_lower_id_does_not_rewind_generator() -> None:
    """Test deserializing an `id` below the generator does not rewind it."""
    a = Identifier("a")
    Identifier("b")
    c = Identifier("c")
    Identifier.deserialize_from_dict({"id": a.id, "name_hint": "x"})
    assert Identifier("y").id == c.id + 1


def test_deserialize_lower_id_preserves_uniqueness() -> None:
    """Test a low-`id` deserialize plus a new construct produces no collision."""
    a = Identifier("a")
    b = Identifier("b")
    c = Identifier("c")
    Identifier.deserialize_from_dict({"id": a.id, "name_hint": "x"})
    new_id = Identifier("y").id
    assert new_id not in {a.id, b.id, c.id}


def test_deserialize_sets_generator_to_exact_successor() -> None:
    """Test the next constructed `id` is exactly one greater than a deserialized."""
    # Offset 43 chosen deliberately: it separates `+1` from `+0`, `+2`, `*1`,
    # `/1`, `//1`, `%1`, `**1`, `<<1`, `>>1`, `&1`, `|1`, `^1`. Each of those
    # mutants would produce a value other than `base + 44` for the next id.
    base = Identifier("anchor").id
    Identifier.deserialize_from_dict({"id": base + 43, "name_hint": "x"})
    assert Identifier("y").id == base + 44


def test_deserialize_then_construct_avoids_collision() -> None:
    """Test an `Identifier` constructed after deserialize avoids collision."""
    base = Identifier("anchor").id
    Identifier.deserialize_from_dict({"id": base + 43, "name_hint": "x"})
    assert Identifier("y").id != base + 43


# =============================================================================
# Class-level defaults (subprocess-isolated)
#
# The class-level default `_next_id = 0` is only observable in a fresh
# process: any prior Identifier(...) in this test session has already mutated
# it. We spawn a subprocess and observe the id of the very first identifier
# created from a clean import.
# =============================================================================


@pytest.mark.slow
@pytest.mark.subprocess
def test_first_identifier_in_fresh_process_has_id_zero() -> None:
    """Test the very first `Identifier` created in a fresh process has `id` `0`."""
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from fhy_core.identifier import Identifier; print(Identifier('x').id)",
        ],
        text=True,
    ).strip()
    assert output == "0"
