"""Property tests for the generic ``Serializable`` round-trip contract.

Covers every serializable family drawn from ``tests/strategies/serializables.py``:
round-tripping through DICT, JSON, and BINARY under each family's own
equivalence check, and JSON output stability after one round trip (P26).
Also covers the adversarial contract that a malformed dict -- one missing a
top-level key, or carrying an unresolvable type id -- always surfaces
through the ``SerializationError`` hierarchy rather than a raw
``KeyError``/``TypeError``/``ValueError``/``AttributeError`` (P27),
generalizing the hand-picked cases in ``test_malformed_input.py``.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given

from fhy_core.serialization import SerializationError, SerializationFormat

from ..strategies.serializables import SerializableCase, draw_serializable_case

pytestmark = pytest.mark.property

# The top-level key a ``WrappedFamilySerializable`` envelope carries its
# concrete class's type id under (``fhy_core.serialization._WRAPPED_TYPE_KEY``,
# private to that module; see the ``WrappedFamilySerializable`` docstring for
# the ``{"__type__": ..., "__data__": ...}`` envelope shape). Not every
# family's top-level dict has this key: a plain (non-family) ``Serializable``
# such as ``Param`` derives its dict straight from its dataclass fields, and
# ``Position``/``Span`` (unlike their ``Provenance`` siblings) are plain
# ``Serializable`` too.
_WRAPPED_TYPE_KEY = "__type__"


@given(case=draw_serializable_case())
def test_serializable_round_trips_in_every_format_under_its_equivalence(
    case: SerializableCase,
) -> None:
    """Test every ``SerializationFormat`` round-trips a case under its equivalence.

    Oracle: the serialize/deserialize inverse pair for DICT, JSON, and
    BINARY, checked with the case's own ``are_equivalent`` predicate rather
    than ``==`` (expressions, constraints, and several other families do not
    define value equality).
    """
    cls = type(case.instance)
    for fmt in SerializationFormat:
        restored = cls.deserialize(case.instance.serialize(fmt), fmt)
        assert case.are_equivalent(restored, case.instance), (
            f'{case.label}: round trip through "{fmt}" lost equivalence'
        )


@given(case=draw_serializable_case())
def test_json_round_trip_reproduces_the_same_json_text(
    case: SerializableCase,
) -> None:
    """Test re-serializing a JSON round trip reproduces the same JSON text.

    Oracle: idempotence of ``to_json`` after one ``from_json`` round trip.
    Both calls use the default ``sort_keys=True``, so the comparison is exact
    text equality, not merely semantic equivalence.
    """
    cls = type(case.instance)
    first_json = case.instance.to_json()

    restored = cls.from_json(first_json)

    assert restored.to_json() == first_json


@given(case=draw_serializable_case())
def test_deserialize_from_dict_never_escapes_hierarchy_on_a_missing_key(
    case: SerializableCase,
) -> None:
    """Test dropping any one top-level dict key stays inside SerializationError.

    Oracle: the exception-hierarchy contract the module docstring and
    ``test_malformed_input.py`` document -- ``deserialize_from_dict`` on
    malformed input either succeeds or raises a ``SerializationError``
    subclass. A bare ``KeyError``/``TypeError``/``ValueError`` (not already a
    ``SerializationError``)/``AttributeError`` escaping fails the property,
    naming the offending family and key.

    Empirically (see the module's welding commit), every family raises for
    every key: the derived engine requires an exact key set, and every
    hand-written ``deserialize_data_from_dict`` checks each key with an
    explicit membership test. No family tolerates a missing key.
    """
    cls = type(case.instance)
    data = case.instance.serialize_to_dict()
    for key in data:
        malformed = {k: v for k, v in data.items() if k != key}
        try:
            cls.deserialize_from_dict(malformed)
        except SerializationError:
            pass
        except Exception as exc:  # re-raised as a property failure below
            pytest.fail(
                f'{case.label}: removing key "{key}" raised '
                f"{type(exc).__name__}, not a SerializationError subclass: {exc}"
            )


@given(case=draw_serializable_case())
def test_deserialize_from_dict_rejects_an_unresolvable_type_id(
    case: SerializableCase,
) -> None:
    """Test an unresolvable top-level type id raises a SerializationError subclass.

    Only families serialized through the ``WrappedFamilySerializable`` envelope
    carry a top-level type id (see ``_WRAPPED_TYPE_KEY`` above); a case without
    one -- ``Param``, or a ``Position``/``Span`` leaf of the "provenance"
    family -- is not exercised by this property.

    Oracle: ``_resolve_type_id``'s ``UnknownTypeIdError``, a
    ``SerializationError`` subclass reached through
    ``WrappedFamilySerializable.deserialize_from_dict``.
    """
    cls = type(case.instance)
    data = case.instance.serialize_to_dict()
    if _WRAPPED_TYPE_KEY not in data:
        return
    tampered = dict(data)
    tampered[_WRAPPED_TYPE_KEY] = "no.such.type"

    with pytest.raises(SerializationError):
        cls.deserialize_from_dict(tampered)
