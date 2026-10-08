"""Tests of Python-defined provenance subclasses.

A downstream compiler defines provenances of its own, such as MOGA-VM's
`EdgePropagationProvenance`, a frozen dataclass subclassing `Provenance`.
The tests use a class of that shape: it constructs through its own
`__init__`, compares and hashes as a dataclass does, nests in each variant
as the very object it was given, survives `fuse`, the V1 and V2 payloads,
JSON, pickle and `copy.deepcopy`, surfaces an exception its `__eq__` or
`__hash__` raises, and takes part in garbage collection.
"""

import copy
import gc
import pickle
import weakref
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.provenance import (
    CallSiteProvenance,
    FileProvenance,
    FusedProvenance,
    NamedProvenance,
    Provenance,
    UnknownProvenance,
)
from fhy_core.serialization import (
    SerializedDict,
    register_serializable,
)
from fhy_core.utils.override import override

from .v1 import writing_v1

_TYPE_ID = "tests.provenance.edge_propagation"


@register_serializable(type_id=_TYPE_ID)
@dataclass(frozen=True)
class EdgePropagationProvenance(Provenance):
    """A provenance naming the edge a value propagated along, as MOGA-VM's."""

    edge_id: Identifier

    @override
    def serialize_data_to_dict(self) -> SerializedDict:
        return {"edge_id": self.edge_id.serialize_to_dict()}

    @classmethod
    @override
    def deserialize_data_from_dict(
        cls, data: SerializedDict
    ) -> "EdgePropagationProvenance":
        edge_id = data["edge_id"]
        assert isinstance(edge_id, dict)
        return cls(edge_id=Identifier.deserialize_from_dict(edge_id))

    @override
    def __str__(self) -> str:
        return f"edge<{self.edge_id!r}>"


class _Boom(Exception):
    """The exception the exploding provenances raise."""


@dataclass(frozen=True, eq=False)
class _ExplodingEqualityProvenance(Provenance):
    """A provenance whose `==` raises."""

    label: str

    @override
    def __eq__(self, other: object) -> bool:
        raise _Boom("eq")

    @override
    def __hash__(self) -> int:
        return 0

    @override
    def __str__(self) -> str:
        return f"explodes<{self.label}>"


@dataclass(frozen=True, eq=False)
class _ExplodingHashProvenance(Provenance):
    """A provenance whose `hash` raises."""

    label: str

    @override
    def __hash__(self) -> int:
        raise _Boom("hash")

    @override
    def __str__(self) -> str:
        return f"explodes<{self.label}>"


def _edge(label: str = "e3") -> EdgePropagationProvenance:
    """Return an edge provenance over a fresh identifier named `label`."""
    return EdgePropagationProvenance(edge_id=Identifier(label))


def _collects(build: Callable[[], object]) -> bool:
    """Return whether the cycle `build` makes is freed by `gc.collect()`."""
    watched = weakref.ref(build())
    gc.collect()
    return watched() is None


# ===========================================================================
# Construction, text, equality
# ===========================================================================


def test_a_python_subclass_of_provenance_constructs() -> None:
    """Test a frozen dataclass subclass builds through its own `__init__`."""
    edge_id = Identifier("e3")

    edge = EdgePropagationProvenance(edge_id=edge_id)

    assert isinstance(edge, Provenance)
    assert type(edge) is EdgePropagationProvenance
    assert edge.edge_id is edge_id


def test_the_subclass_is_frozen() -> None:
    """Test the dataclass refuses an attribute assignment."""
    edge = _edge()

    with pytest.raises(FrozenInstanceError):
        edge.edge_id = Identifier("other")  # type: ignore[misc]  # test: frozen


def test_the_subclass_text_is_its_own() -> None:
    """Test `str` is the subclass's `__str__`."""
    edge = _edge("e3")

    assert str(edge) == f"edge<{edge.edge_id!r}>"
    assert str(edge).startswith("edge<")


def test_the_subclass_compares_and_hashes_as_a_dataclass() -> None:
    """Test equal fields are equal with one hash, other fields are not."""
    edge_id = Identifier("e3")
    first, second = (
        EdgePropagationProvenance(edge_id),
        EdgePropagationProvenance(edge_id),
    )

    assert first is not second
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1
    assert first != _edge("e3")
    assert first != UnknownProvenance()
    assert first != "edge"


# ===========================================================================
# As a child
# ===========================================================================


def test_a_named_provenance_returns_the_very_child() -> None:
    """Test `NamedProvenance("n", edge).child` is `edge` itself."""
    edge = _edge()

    named = NamedProvenance("n", edge)

    assert named.child is edge
    assert named.name == "n"


def test_a_call_site_provenance_returns_the_very_callee_and_caller() -> None:
    """Test `CallSiteProvenance` returns the subclass objects it was given."""
    callee, caller = _edge("callee"), _edge("caller")

    call_site = CallSiteProvenance(callee, caller)

    assert call_site.callee is callee
    assert call_site.caller is caller


def test_a_fused_provenance_returns_the_very_sources() -> None:
    """Test `FusedProvenance` keeps the subclass objects among its sources."""
    edge, source = _edge(), FileProvenance(Path("a.fhy"))

    fused = FusedProvenance((source, edge), "cse")

    assert fused.sources[1] is edge
    assert fused.sources[0] is source


def test_a_variant_holding_a_subclass_compares_through_its_equality() -> None:
    """Test two variants over equal subclass values are equal with one hash."""
    edge_id = Identifier("e3")
    left = NamedProvenance("n", EdgePropagationProvenance(edge_id))
    right = NamedProvenance("n", EdgePropagationProvenance(edge_id))

    assert left == right
    assert hash(left) == hash(right)
    assert left != NamedProvenance("n", _edge("e3"))
    assert left != NamedProvenance("n", UnknownProvenance())
    assert left != NamedProvenance("m", EdgePropagationProvenance(edge_id))


def test_a_variant_holding_a_subclass_has_its_text() -> None:
    """Test the text of a variant shows the subclass's `str`."""
    edge = _edge()

    assert str(edge) in str(NamedProvenance("n", edge))
    assert str(edge) in str(CallSiteProvenance(edge, UnknownProvenance()))


def test_fuse_keeps_a_subclass_whole() -> None:
    """Test `Provenance.fuse` returns the subclass object a lone survivor is."""
    edge = _edge()

    assert Provenance.fuse(UnknownProvenance(), edge) is edge


def test_fuse_keeps_a_subclass_among_several_sources() -> None:
    """Test `Provenance.fuse` of a subclass and a file keeps the subclass object."""
    edge, source = _edge(), FileProvenance(Path("a.fhy"))

    fused = Provenance.fuse(source, UnknownProvenance(), edge, metadata="cse")

    assert isinstance(fused, FusedProvenance)
    assert fused.metadata == "cse"
    assert len(fused.sources) == 2
    assert fused.sources[0] is source
    assert fused.sources[1] is edge


# ===========================================================================
# Payloads
# ===========================================================================


def test_a_subclass_round_trips_through_its_v2_payload() -> None:
    """Test `Provenance.deserialize_from_dict` returns an equal subclass instance."""
    edge = _edge()

    restored = Provenance.deserialize_from_dict(edge.serialize_to_dict())

    assert type(restored) is EdgePropagationProvenance
    assert restored == edge


def test_a_subclass_round_trips_through_json() -> None:
    """Test `to_json` and `from_json` give an equal subclass instance."""
    edge = _edge()

    restored = Provenance.from_json(edge.to_json())

    assert type(restored) is EdgePropagationProvenance
    assert restored == edge


def test_a_subclass_child_is_written_as_a_custom_part() -> None:
    """Test a variant writes its subclass child under `custom` with its type id."""
    edge = _edge()

    payload: Any = NamedProvenance("n", edge).serialize_to_dict()

    assert set(payload) == {"named"}
    assert payload["named"]["name"] == "n"
    child = payload["named"]["child"]
    assert set(child) == {"custom"}
    assert child["custom"]["type_id"] == _TYPE_ID
    assert "data" in child["custom"]


@pytest.mark.parametrize(
    "build",
    [
        lambda edge: NamedProvenance("n", edge),
        lambda edge: CallSiteProvenance(edge, UnknownProvenance()),
        lambda edge: CallSiteProvenance(UnknownProvenance(), edge),
        lambda edge: FusedProvenance((FileProvenance(Path("a.fhy")), edge), "cse"),
        lambda edge: NamedProvenance("outer", NamedProvenance("inner", edge)),
    ],
    ids=["named", "callee", "caller", "fused", "nested"],
)
def test_a_variant_decoding_a_subclass_child_returns_the_subclass(
    build: Callable[[EdgePropagationProvenance], Provenance],
) -> None:
    """Test a variant over a subclass child round-trips with an equal subclass child."""
    edge = _edge()
    original = build(edge)

    for restored in (
        Provenance.deserialize_from_dict(original.serialize_to_dict()),
        Provenance.from_json(original.to_json()),
    ):
        assert type(restored) is type(original)
        assert restored == original
        assert restored.serialize_to_dict() == original.serialize_to_dict()


def test_a_decoded_subclass_child_is_the_subclass() -> None:
    """Test the child a decoded variant holds is an `EdgePropagationProvenance`."""
    edge = _edge()

    restored = NamedProvenance.from_json(NamedProvenance("n", edge).to_json())

    assert type(restored.child) is EdgePropagationProvenance
    assert restored.child == edge


def test_a_subclass_round_trips_through_the_v1_payload() -> None:
    """Test the V1 envelope of a subclass decodes to an equal subclass instance."""
    edge = _edge()

    with writing_v1():
        payload = edge.serialize_to_dict()
        restored = Provenance.deserialize_from_dict(payload)

    assert payload["__type__"] == _TYPE_ID
    assert payload["__data__"] == edge.serialize_data_to_dict()
    assert type(restored) is EdgePropagationProvenance
    assert restored == edge


# ===========================================================================
# Pickle and copy
# ===========================================================================


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_a_subclass_pickles_to_an_equal_instance(protocol: int) -> None:
    """Test a subclass instance pickles under every protocol to an equal one."""
    edge = _edge()

    restored = pickle.loads(pickle.dumps(edge, protocol=protocol))

    assert type(restored) is EdgePropagationProvenance
    assert restored == edge
    assert restored is not edge
    assert hash(restored) == hash(edge)


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_a_variant_over_a_subclass_pickles_to_an_equal_variant(protocol: int) -> None:
    """Test a variant holding a subclass child pickles with the child intact."""
    edge = _edge()
    named = NamedProvenance("n", edge)

    restored = pickle.loads(pickle.dumps(named, protocol=protocol))

    assert type(restored) is NamedProvenance
    assert type(restored.child) is EdgePropagationProvenance
    assert restored == named


def test_a_subclass_deep_copies_to_an_equal_new_object() -> None:
    """Test `copy.deepcopy` and `copy.copy` build equal subclass instances."""
    edge = _edge()

    deep, shallow = copy.deepcopy(edge), copy.copy(edge)

    assert type(deep) is EdgePropagationProvenance
    assert deep == edge
    assert deep is not edge
    assert shallow == edge


def test_a_variant_over_a_subclass_deep_copies_the_child() -> None:
    """Test `copy.deepcopy` of a variant copies its subclass child."""
    edge = _edge()
    named = NamedProvenance("n", edge)

    copied = copy.deepcopy(named)

    assert copied == named
    assert type(copied.child) is EdgePropagationProvenance
    assert copied.child is not edge


# ===========================================================================
# Exceptions
# ===========================================================================


def test_an_exception_from_a_subclass_equality_surfaces() -> None:
    """Test comparing variants over subclass values raises what `__eq__` raised."""
    left = NamedProvenance("n", _ExplodingEqualityProvenance("a"))
    right = NamedProvenance("n", _ExplodingEqualityProvenance("b"))

    with pytest.raises(_Boom, match="eq"):
        left == right  # noqa: B015  # test: the comparison is the call

    with pytest.raises(_Boom, match="eq"):
        left != right  # noqa: B015  # test: the comparison is the call


def test_an_exception_from_a_subclass_hash_surfaces() -> None:
    """Test hashing a variant over a subclass value raises what `__hash__` raised."""
    named = NamedProvenance("n", _ExplodingHashProvenance("a"))

    with pytest.raises(_Boom, match="hash"):
        hash(named)


def test_a_failed_comparison_leaves_no_pending_error() -> None:
    """Test a comparison after a raised one runs as usual."""
    left = NamedProvenance("n", _ExplodingEqualityProvenance("a"))
    right = NamedProvenance("n", _ExplodingEqualityProvenance("b"))
    with pytest.raises(_Boom):
        left == right  # noqa: B015  # test: the comparison is the call

    assert NamedProvenance("n", _edge("x")) != NamedProvenance("n", _edge("x"))
    assert "explodes<a>" in str(left)


# ===========================================================================
# Garbage collection
# ===========================================================================


def test_a_cycle_through_a_subclass_child_is_collected() -> None:
    """Test a variant, its subclass child and a pointer back from the child die."""

    def build() -> object:
        edge = _edge()
        named = NamedProvenance("n", edge)
        object.__setattr__(edge, "owner", named)
        return edge

    assert _collects(build)


def test_a_cycle_through_a_compared_subclass_child_is_collected() -> None:
    """Test a cycle stays collectable after the child was compared and hashed."""

    def build() -> object:
        edge_id = Identifier("e")
        first = EdgePropagationProvenance(edge_id)
        left = NamedProvenance("n", first)
        right = NamedProvenance("n", EdgePropagationProvenance(edge_id))
        assert left == right
        hash(left)
        object.__setattr__(first, "owner", left)
        return first

    assert _collects(build)


def test_a_cycle_through_a_fused_subclass_source_is_collected() -> None:
    """Test a fused provenance and its subclass source that points back die."""

    def build() -> object:
        edge = _edge()
        fused = Provenance.fuse(FileProvenance(Path("a.fhy")), edge, metadata="m")
        object.__setattr__(edge, "owner", fused)
        return edge

    assert _collects(build)


# ===========================================================================
# Subclassing a variant still works
# ===========================================================================


class _TaggedNamed(NamedProvenance):
    """A subclass of a variant, as downstream code may define."""

    __slots__ = ()


def test_a_subclass_of_a_variant_still_constructs_with_the_variant_fields() -> None:
    """Test a subclass of `NamedProvenance` builds and reads like its base."""
    child = UnknownProvenance()

    tagged = _TaggedNamed("n", child)

    assert isinstance(tagged, NamedProvenance)
    assert type(tagged) is _TaggedNamed
    assert tagged.name == "n"
    assert tagged.child is child
    assert str(tagged) == str(NamedProvenance("n", child))


def test_a_subclass_of_a_variant_keeps_exact_class_equality() -> None:
    """Test a variant subclass and its base compare unequal, as before."""
    child = UnknownProvenance()

    assert _TaggedNamed("n", child) != NamedProvenance("n", child)
    assert _TaggedNamed("n", child) == _TaggedNamed("n", child)


def test_a_subclass_of_a_variant_pickles_as_itself() -> None:
    """Test a variant subclass pickles to an instance of the subclass."""
    tagged = _TaggedNamed("n", UnknownProvenance())

    restored: Any = pickle.loads(pickle.dumps(tagged))

    assert type(restored) is _TaggedNamed
    assert restored == tagged


def test_a_subclass_of_a_variant_nests_as_a_child() -> None:
    """Test a variant subclass is the very child of another variant."""
    tagged = _TaggedNamed("n", UnknownProvenance())

    assert NamedProvenance("outer", tagged).child is tagged
