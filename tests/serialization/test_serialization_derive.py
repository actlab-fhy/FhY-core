"""Tests for schema-derived serialization on the ``Serializable`` traits.

Covers the ``derive=`` class keyword, the field-codec inference engine, the
``field(metadata={"serialize_codec": ...})`` override channel, the
``construct_from_fields`` reconstruction hook, and the instantiation-time
``SerializationDerivationError`` (ABC-style) for classes that opt in but cannot
be derived.
"""

import enum
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, cast

import pytest

from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    Serializable,
    SerializationDerivationError,
    SerializationError,
    SerializationFormat,
    SerializationValueError,
    SerializedDict,
    WrappedFamilySerializable,
    make_enum_field_codec,
    make_field_codec,
    make_labeled_enum_field_codec,
    register_field_codec,
    register_serializable,
)
from fhy_core.traits.frozen import FrozenMixin
from fhy_core.utils import IntEnum, StrEnum
from fhy_core.utils.override import override

# ============================================================================
# Representative derived classes (derive=True is the default)
# ============================================================================


@register_serializable(type_id="_test_derive_point")
@dataclass(frozen=True)
class _Point(Serializable, FrozenMixin):
    x: int
    y: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("coordinates must be non-negative")


@register_serializable(type_id="_test_derive_optbox")
@dataclass(frozen=True)
class _OptBox(Serializable, FrozenMixin):
    value: int | None = None
    point: _Point | None = None


@register_serializable(type_id="_test_derive_bag")
@dataclass(frozen=True)
class _Bag(Serializable, FrozenMixin):
    points: tuple[_Point, ...]


@register_serializable(type_id="_test_derive_empty")
@dataclass(frozen=True)
class _Empty(Serializable, FrozenMixin):
    pass


# ============================================================================
# Basic round-trip
# ============================================================================


def test_derived_dict_round_trip() -> None:
    """Test a derived dataclass round-trips through the dict form."""
    point = _Point(2, 8)

    assert _Point.deserialize_from_dict(point.serialize_to_dict()) == point


def test_derived_serialize_emits_plain_field_dict() -> None:
    """Test the derived dict is the bare field mapping (format preservation)."""
    assert _Point(2, 8).serialize_to_dict() == {"x": 2, "y": 8}


def test_derived_empty_dataclass_round_trips_through_empty_dict() -> None:
    """Test a field-less dataclass serializes to an empty dict and back."""
    assert _Empty().serialize_to_dict() == {}
    assert _Empty.deserialize_from_dict({}) == _Empty()


def test_derived_optional_fields_round_trip_when_present_and_absent() -> None:
    """Test optional scalar and nested fields round-trip in both states."""
    empty = _OptBox()
    full = _OptBox(value=5, point=_Point(1, 2))

    assert _OptBox.deserialize_from_dict(empty.serialize_to_dict()) == empty
    assert _OptBox.deserialize_from_dict(full.serialize_to_dict()) == full


def test_derived_nested_serializable_is_encoded_as_plain_dict() -> None:
    """Test a nested concrete serializable field encodes as its bare dict."""
    box = _OptBox(value=1, point=_Point(3, 4))

    assert box.serialize_to_dict() == {"value": 1, "point": {"x": 3, "y": 4}}


def test_derived_homogeneous_tuple_round_trips() -> None:
    """Test a homogeneous tuple of serializables round-trips."""
    bag = _Bag(points=(_Point(1, 1), _Point(2, 2)))

    assert _Bag.deserialize_from_dict(bag.serialize_to_dict()) == bag


# ============================================================================
# Format-preserving error behavior
# ============================================================================


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param({"x": 1}, id="missing_key"),
        pytest.param({"x": 1, "y": 2, "z": 3}, id="extra_key"),
        pytest.param({"x": 1, "y": "two"}, id="wrong_type"),
        pytest.param({"x": True, "y": 1}, id="bool_for_int"),
    ],
)
def test_derived_deserialize_rejects_malformed_dict(malformed: SerializedDict) -> None:
    """Test a malformed dict raises a structure error."""
    with pytest.raises(DeserializationDictStructureError):
        _Point.deserialize_from_dict(malformed)


def test_derived_deserialize_surfaces_post_init_value_error() -> None:
    """Test a constructor ``ValueError`` surfaces as a deserialization value error."""
    with pytest.raises(DeserializationValueError):
        _Point.deserialize_from_dict({"x": -1, "y": 2})


# ============================================================================
# Cross-format integration (rides the existing JSON / BINARY envelopes)
# ============================================================================


@pytest.mark.parametrize(
    "fmt",
    [SerializationFormat.DICT, SerializationFormat.JSON, SerializationFormat.BINARY],
)
def test_derived_round_trips_in_all_formats(fmt: SerializationFormat) -> None:
    """Test a derived class round-trips through dict, json, and binary."""
    point = _Point(7, 3)

    assert _Point.deserialize(point.serialize(fmt), fmt) == point


# ============================================================================
# derive=False opt-out
# ============================================================================


def test_derive_false_without_methods_raises_at_instantiation() -> None:
    """Test ``derive=False`` without hand-written methods fails on construction."""

    @dataclass(frozen=True)
    class _OptOut(Serializable, FrozenMixin, derive=False):
        x: int

    with pytest.raises(SerializationDerivationError, match="derive=False"):
        _OptOut(1)


def test_derive_false_with_manual_methods_works() -> None:
    """Test ``derive=False`` with hand-written methods behaves as before."""

    @register_serializable(type_id="_test_derive_optout_manual")
    @dataclass(frozen=True)
    class _OptOutManual(Serializable, FrozenMixin, derive=False):
        x: int

        @override
        def serialize_to_dict(self) -> SerializedDict:
            return {"x": self.x}

        @classmethod
        @override
        def deserialize_from_dict(cls, data: SerializedDict) -> "_OptOutManual":
            return cls(cast(int, data["x"]))

    obj = _OptOutManual(5)
    assert _OptOutManual.deserialize_from_dict(obj.serialize_to_dict()) == obj


# ============================================================================
# Instantiation-time SerializationDerivationError (ABC-style)
# ============================================================================


def test_underivable_field_raises_at_instantiation() -> None:
    """Test a derive=True class with an uncodecable field fails on construction."""

    @dataclass
    class _BadField(Serializable):
        payload: complex  # immutable but has no serialization codec

    with pytest.raises(SerializationDerivationError):
        _BadField(1 + 2j)


def test_non_dataclass_derive_raises_at_instantiation() -> None:
    """Test a derive=True non-dataclass cannot be derived and fails on construction."""

    class _NotADataclass(Serializable):
        pass

    with pytest.raises(SerializationDerivationError, match="dataclass"):
        _NotADataclass()


def test_underivable_error_names_the_offending_field() -> None:
    """Test the derivation error message identifies the field and type."""

    @dataclass
    class _BadNamed(Serializable):
        payload: complex

    with pytest.raises(SerializationDerivationError, match="payload"):
        _BadNamed(1j)


def test_unresolvable_annotation_raises_clear_derivation_error() -> None:
    """Test an unresolvable field annotation reports the true cause.

    A forward reference that cannot be resolved is dropped by annotation
    resolution; the derivation error must name the field and say the annotation
    could not be resolved, rather than misreporting it as an unsupported type.
    """

    @dataclass
    class _BadRef(Serializable):
        value: "DefinitelyNotARealType"  # type: ignore[name-defined]  # noqa: F821

    with pytest.raises(
        SerializationDerivationError, match="value.*could not be resolved"
    ):
        _BadRef(1)


# ============================================================================
# Hand-written methods win over derivation
# ============================================================================


def test_hand_written_methods_take_precedence_over_derivation() -> None:
    """Test hand-written methods take precedence over derivation."""

    @register_serializable(type_id="_test_derive_manual")
    @dataclass(frozen=True)
    class _Manual(Serializable, FrozenMixin):
        x: int

        @override
        def serialize_to_dict(self) -> SerializedDict:
            return {"x": self.x, "extra": "manual"}

        @classmethod
        @override
        def deserialize_from_dict(cls, data: SerializedDict) -> "_Manual":
            return cls(cast(int, data["x"]))

    assert _Manual(3).serialize_to_dict() == {"x": 3, "extra": "manual"}


def test_one_manual_one_derived_method_mix() -> None:
    """Test a class may hand-write one method and derive the other."""

    @register_serializable(type_id="_test_derive_halfmanual")
    @dataclass(frozen=True)
    class _HalfManual(Serializable, FrozenMixin):
        x: int

        @override
        def serialize_to_dict(self) -> SerializedDict:
            return {"x": self.x * 10}

    # Manual serialize is used...
    assert _HalfManual(3).serialize_to_dict() == {"x": 30}
    # ...while deserialize is derived.
    assert _HalfManual.deserialize_from_dict({"x": 5}) == _HalfManual(5)


# ============================================================================
# construct_from_fields hook
# ============================================================================


def test_construct_from_fields_override_is_honored() -> None:
    """Test a custom ``construct_from_fields`` drives derived reconstruction."""

    @register_serializable(type_id="_test_derive_constructed")
    @dataclass(frozen=True)
    class _Constructed(Serializable, FrozenMixin):
        x: int

        @classmethod
        @override
        def construct_from_fields(cls, fields: dict[str, Any]) -> "_Constructed":
            return cls(fields["x"] + 100)

    assert _Constructed.deserialize_from_dict({"x": 5}).x == 105


def test_construct_from_fields_builds_from_decoded_fields() -> None:
    """Test the default hook reconstructs directly from a well-formed field map."""
    assert _Point.construct_from_fields({"x": 2, "y": 8}) == _Point(2, 8)


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param({"x": 1}, id="missing_field"),
        pytest.param({"x": 1, "y": 2, "z": 3}, id="extra_field"),
    ],
)
def test_construct_from_fields_rejects_malformed_field_map(
    malformed: dict[str, Any],
) -> None:
    """Test the hook surfaces a ``TypeError`` when ``fields`` is not one-per-field.

    The hook's precondition is one decoded entry per dataclass field; the
    validated ``deserialize_from_dict`` path guarantees this, so a malformed
    map only reaches the hook on direct misuse and fails loudly rather than
    silently constructing a wrong instance.
    """
    with pytest.raises(TypeError):
        _Point.construct_from_fields(malformed)


def test_construct_from_fields_propagates_constructor_value_error() -> None:
    """Test constructor validation errors propagate raw from a direct hook call.

    Wrapping into ``DeserializationValueError`` is the deserialize path's job
    (see ``test_derived_deserialize_surfaces_post_init_value_error``); the hook
    itself does not catch, so misuse is not disguised as a serialization fault.
    """
    with pytest.raises(ValueError, match="non-negative"):
        _Point.construct_from_fields({"x": -1, "y": 2})


# ============================================================================
# Per-field codec overrides via field(metadata=...)
# ============================================================================


def test_path_field_via_make_field_codec_round_trips() -> None:
    """Test a bespoke ``Path`` codec supplied through field metadata round-trips."""
    path_codec = make_field_codec(encode=str, decode=Path)

    @register_serializable(type_id="_test_derive_pathy")
    @dataclass(frozen=True)
    class _Pathy(Serializable, FrozenMixin):
        location: Path = field(metadata={"serialize_codec": path_codec})

    obj = _Pathy(Path("/tmp/example.fhy"))
    # The bespoke codec encodes with ``str``, whose separator is OS-dependent;
    # compare against the platform-native form rather than a hard-coded one.
    expected = str(Path("/tmp/example.fhy"))
    assert obj.serialize_to_dict() == {"location": expected}
    assert _Pathy.deserialize_from_dict(obj.serialize_to_dict()) == obj


def test_make_enum_field_codec_round_trips_by_name() -> None:
    """Test an enum field encoded by name through field metadata round-trips."""

    class _Color(enum.Enum):
        RED = 1
        BLUE = 2

    @register_serializable(type_id="_test_derive_colored")
    @dataclass(frozen=True)
    class _Colored(Serializable, FrozenMixin):
        color: _Color = field(
            metadata={"serialize_codec": make_enum_field_codec(_Color, by="name")}
        )

    obj = _Colored(_Color.BLUE)
    assert obj.serialize_to_dict() == {"color": "BLUE"}
    assert _Colored.deserialize_from_dict({"color": "BLUE"}) == obj


def test_register_field_codec_teaches_inference_a_new_leaf() -> None:
    """Test a registered leaf codec lets a field derive without a metadata override."""

    class _Celsius:
        def __init__(self, degrees: int) -> None:
            self.degrees = degrees

        @override
        def __eq__(self, other: object) -> bool:
            return isinstance(other, _Celsius) and other.degrees == self.degrees

        @override
        def __hash__(self) -> int:
            return hash(self.degrees)

    register_field_codec(
        _Celsius, make_field_codec(encode=lambda c: c.degrees, decode=_Celsius)
    )

    @register_serializable(type_id="_test_derive_weather")
    @dataclass
    class _Weather(Serializable):
        temperature: _Celsius

    obj = _Weather(_Celsius(21))
    assert obj.serialize_to_dict() == {"temperature": 21}
    assert _Weather.deserialize_from_dict({"temperature": 21}) == obj


# ============================================================================
# WrappedFamilySerializable derivation (fills the *_data_* methods)
# ============================================================================


def test_wrapped_family_member_derives_data_methods() -> None:
    """Test a wrapped-family member derives its data serialization."""

    class _Shape(WrappedFamilySerializable, FrozenMixin):
        pass

    @register_serializable(type_id="_test_derive_circle")
    @dataclass(frozen=True)
    class _Circle(_Shape):
        radius: int

    circle = _Circle(4)
    wrapped = circle.serialize_to_dict()

    assert wrapped == {"__type__": "_test_derive_circle", "__data__": {"radius": 4}}
    assert _Shape.deserialize_from_dict(wrapped) == circle


# ============================================================================
# Edge cases and adversarial inputs
# ============================================================================


def test_inferred_scalar_fields_round_trip() -> None:
    """Test a class with inferred string and boolean fields round-trips."""

    @dataclass(frozen=True)
    class _Flags(Serializable, FrozenMixin):
        name: str
        active: bool

    obj = _Flags("widget", True)
    assert obj.serialize_to_dict() == {"name": "widget", "active": True}
    assert _Flags.deserialize_from_dict({"name": "widget", "active": True}) == obj


def test_inferred_path_field_round_trips() -> None:
    """Test a ``Path`` field is inferred (no metadata) and round-trips as a string."""

    @dataclass(frozen=True)
    class _Located(Serializable, FrozenMixin):
        path: Path

    obj = _Located(Path("/tmp/example.fhy"))
    assert obj.serialize_to_dict() == {"path": "/tmp/example.fhy"}
    assert _Located.deserialize_from_dict({"path": "/tmp/example.fhy"}) == obj


def test_inferred_path_field_encodes_in_posix_form() -> None:
    """Test an inferred path field encodes with forward slashes on every OS.

    Uses ``PureWindowsPath`` so the input's separator semantics are fixed
    regardless of the host OS; the encoded form must still be POSIX so blobs are
    portable across platforms.
    """

    @dataclass(frozen=True)
    class _WinLocated(Serializable, FrozenMixin):
        path: PureWindowsPath

    obj = _WinLocated(PureWindowsPath("dir/sub/file.fhy"))
    assert obj.serialize_to_dict() == {"path": "dir/sub/file.fhy"}
    assert _WinLocated.deserialize_from_dict({"path": "dir/sub/file.fhy"}) == obj


def test_float_field_round_trips() -> None:
    """Test a float field round-trips through the dict form."""

    @dataclass
    class _Temp(Serializable):
        celsius: float

    assert _Temp(98.6).serialize_to_dict() == {"celsius": 98.6}
    assert _Temp.deserialize_from_dict({"celsius": 98.6}) == _Temp(98.6)


def test_float_field_coerces_integer_payload() -> None:
    """Test a float field accepts an integer-valued payload, coercing to float."""

    @dataclass
    class _Temp(Serializable):
        celsius: float

    assert _Temp.deserialize_from_dict({"celsius": 98}) == _Temp(98.0)


def test_frozenset_field_round_trips() -> None:
    """Test a homogeneous frozenset field round-trips regardless of order."""

    @dataclass(frozen=True)
    class _Tags(Serializable, FrozenMixin):
        tags: frozenset[int]

    obj = _Tags(frozenset({3, 1, 2}))

    assert _Tags.deserialize_from_dict(obj.serialize_to_dict()) == obj


def test_frozenset_field_serializes_deterministically() -> None:
    """Test a frozenset field emits its elements in a stable, sorted order."""

    @dataclass(frozen=True)
    class _Names(Serializable, FrozenMixin):
        names: frozenset[str]

    # str hashing is randomized per process, so a frozenset's iteration order is
    # not stable; the encoded form must be sorted so the wire output is fixed.
    obj = _Names(frozenset({"gamma", "alpha", "delta", "beta"}))

    assert obj.serialize_to_dict() == {"names": ["alpha", "beta", "delta", "gamma"]}


def test_list_field_round_trips() -> None:
    """Test a homogeneous list field round-trips."""

    @dataclass
    class _ListBox(Serializable):
        items: list[int]

    assert _ListBox([1, 2, 3]).serialize_to_dict() == {"items": [1, 2, 3]}
    assert _ListBox.deserialize_from_dict({"items": [1, 2, 3]}) == _ListBox([1, 2, 3])


def test_heterogeneous_tuple_is_not_derivable() -> None:
    """Test a heterogeneous tuple field cannot be derived."""

    @dataclass
    class _Pair(Serializable):
        pair: tuple[int, str]

    with pytest.raises(SerializationDerivationError, match="pair"):
        _Pair((1, "a"))


def test_fixed_length_single_tuple_is_not_derivable() -> None:
    """Test a fixed-length ``tuple[T]`` field cannot be derived.

    Only the variable-length ``tuple[T, ...]`` form is derivable; the sequence
    codec does not enforce length, so a fixed-length tuple must be rejected
    rather than silently accepting payloads of any arity.
    """

    @dataclass
    class _Single(Serializable):
        only: tuple[int]

    with pytest.raises(SerializationDerivationError, match="only"):
        _Single((1,))


def test_make_enum_field_codec_round_trips_by_value() -> None:
    """Test an enum field encoded by value round-trips."""

    class _Level(enum.Enum):
        LOW = 1
        HIGH = 2

    @dataclass
    class _Setting(Serializable):
        level: _Level = field(
            metadata={"serialize_codec": make_enum_field_codec(_Level, by="value")}
        )

    obj = _Setting(_Level.HIGH)
    assert obj.serialize_to_dict() == {"level": 2}
    assert _Setting.deserialize_from_dict({"level": 2}) == obj


def test_make_enum_field_codec_rejects_invalid_by_argument() -> None:
    """Test ``make_enum_field_codec`` rejects an unsupported ``by`` argument."""

    class _Level(enum.Enum):
        LOW = 1

    with pytest.raises(ValueError, match='"name" or "value"'):
        make_enum_field_codec(_Level, by="ordinal")


def test_make_enum_field_codec_rejects_unknown_name() -> None:
    """Test deserializing an unknown enum name raises a value error."""

    class _Color(enum.Enum):
        RED = 1

    @dataclass
    class _Painted(Serializable):
        color: _Color = field(
            metadata={"serialize_codec": make_enum_field_codec(_Color)}
        )

    with pytest.raises(DeserializationValueError, match="color"):
        _Painted.deserialize_from_dict({"color": "MAGENTA"})


class _ByValueLevel(enum.Enum):
    LOW = 1
    HIGH = 2


def test_make_enum_field_codec_rejects_out_of_range_value() -> None:
    """Test deserializing an out-of-range by-value enum payload raises."""

    @dataclass
    class _Setting(Serializable):
        level: int = field(
            metadata={
                "serialize_codec": make_enum_field_codec(_ByValueLevel, by="value")
            }
        )

    with pytest.raises(DeserializationValueError, match="level"):
        _Setting.deserialize_from_dict({"level": 999})


def test_make_enum_field_codec_rejects_unhashable_by_value_payload() -> None:
    """Test an unhashable by-value enum payload surfaces as a value error.

    ``Enum(value)`` raises ``TypeError`` (not ``ValueError``) for an unhashable
    payload such as a dict; the codec must still wrap it in the serialization
    error hierarchy rather than letting the raw ``TypeError`` escape.
    """

    @dataclass
    class _Setting(Serializable):
        level: int = field(
            metadata={
                "serialize_codec": make_enum_field_codec(_ByValueLevel, by="value")
            }
        )

    with pytest.raises(DeserializationValueError, match="level"):
        _Setting.deserialize_from_dict({"level": {"unhashable": True}})


def test_int_enum_field_rejects_out_of_range_value_without_metadata() -> None:
    """Test the auto-derived ``IntEnum`` by-value path rejects an out-of-range value."""

    class _Code(IntEnum):
        OK = 0
        FAIL = 1

    @dataclass
    class _Result(Serializable):
        code: _Code

    with pytest.raises(DeserializationValueError, match="code"):
        _Result.deserialize_from_dict({"code": 999})


def test_str_enum_field_derives_by_value_without_metadata() -> None:
    """Test a ``StrEnum`` field auto-serializes by value -- no metadata override."""

    class _Color(StrEnum):
        RED = "red"
        BLUE = "blue"

    @register_serializable(type_id="_test_derive_strenum")
    @dataclass(frozen=True)
    class _Painted(Serializable, FrozenMixin):
        color: _Color

    obj = _Painted(_Color.BLUE)
    assert obj.serialize_to_dict() == {"color": "blue"}
    assert _Painted.deserialize_from_dict({"color": "blue"}) == obj


def test_int_enum_field_derives_by_value_without_metadata() -> None:
    """Test an ``IntEnum`` field auto-serializes by value -- no metadata override."""

    class _Status(IntEnum):
        OK = 200
        TEAPOT = 418

    @register_serializable(type_id="_test_derive_intenum")
    @dataclass(frozen=True)
    class _Response(Serializable, FrozenMixin):
        status: _Status

    obj = _Response(_Status.TEAPOT)
    assert obj.serialize_to_dict() == {"status": 418}
    assert _Response.deserialize_from_dict({"status": 418}) == obj


def test_plain_enum_field_derives_by_name_without_metadata() -> None:
    """Test a plain ``Enum`` field auto-serializes by name (the always-safe form)."""

    class _Direction(enum.Enum):
        NORTH = enum.auto()
        SOUTH = enum.auto()

    @register_serializable(type_id="_test_derive_plainenum")
    @dataclass(frozen=True)
    class _Heading(Serializable, FrozenMixin):
        direction: _Direction

    obj = _Heading(_Direction.NORTH)
    assert obj.serialize_to_dict() == {"direction": "NORTH"}
    assert _Heading.deserialize_from_dict({"direction": "NORTH"}) == obj


class _Op(enum.Enum):
    ADD = enum.auto()
    SUBTRACT = enum.auto()


_OP_LABELS = {_Op.ADD: "add", _Op.SUBTRACT: "subtract"}


def test_make_labeled_enum_field_codec_round_trips_through_labels() -> None:
    """Test an enum field encoded through bespoke string labels round-trips."""

    @register_serializable(type_id="_test_derive_labeled")
    @dataclass(frozen=True)
    class _Labeled(Serializable, FrozenMixin):
        op: _Op = field(
            metadata={"serialize_codec": make_labeled_enum_field_codec(_Op, _OP_LABELS)}
        )

    obj = _Labeled(_Op.SUBTRACT)
    assert obj.serialize_to_dict() == {"op": "subtract"}
    assert _Labeled.deserialize_from_dict({"op": "subtract"}) == obj


def test_make_labeled_enum_field_codec_rejects_unknown_label() -> None:
    """Test deserializing an unknown label raises a value error."""

    @register_serializable(type_id="_test_derive_labeled_unknown")
    @dataclass(frozen=True)
    class _Labeled(Serializable, FrozenMixin):
        op: _Op = field(
            metadata={"serialize_codec": make_labeled_enum_field_codec(_Op, _OP_LABELS)}
        )

    with pytest.raises(DeserializationValueError, match="op"):
        _Labeled.deserialize_from_dict({"op": "multiply"})


def test_make_labeled_enum_field_codec_rejects_non_string_label() -> None:
    """Test a non-string payload is rejected as a structure error, not a value error."""

    @register_serializable(type_id="_test_derive_labeled_nonstr")
    @dataclass(frozen=True)
    class _Labeled(Serializable, FrozenMixin):
        op: _Op = field(
            metadata={"serialize_codec": make_labeled_enum_field_codec(_Op, _OP_LABELS)}
        )

    with pytest.raises(DeserializationDictStructureError):
        _Labeled.deserialize_from_dict({"op": 7})


def test_make_labeled_enum_field_codec_encode_rejects_out_of_domain_member() -> None:
    """Test encoding a member outside the label domain raises a value error.

    The labels are verified to cover every member at build time, so this only
    fires for a value not in the codec's enum (e.g. a member of a different
    enum). It must surface as a ``SerializationValueError`` rather than a bare
    ``KeyError``.
    """
    codec = make_labeled_enum_field_codec(_Op, _OP_LABELS)

    class _OtherOp(enum.Enum):
        MULTIPLY = enum.auto()

    with pytest.raises(SerializationValueError):
        codec.encode(_OtherOp.MULTIPLY)


def test_make_labeled_enum_field_codec_rejects_incomplete_labels() -> None:
    """Test labels that omit a member are rejected when the codec is built."""
    with pytest.raises(ValueError, match="every .* member"):
        make_labeled_enum_field_codec(_Op, {_Op.ADD: "add"})


def test_make_labeled_enum_field_codec_rejects_duplicate_labels() -> None:
    """Test labels that collide on the same string are rejected when built."""
    with pytest.raises(ValueError, match="distinct"):
        make_labeled_enum_field_codec(_Op, {_Op.ADD: "same", _Op.SUBTRACT: "same"})


def test_register_field_codec_rejects_conflicting_codec() -> None:
    """Test registering a second, different codec for a type raises."""

    class _Widget:
        pass

    register_field_codec(
        _Widget, make_field_codec(encode=lambda w: 0, decode=lambda d: _Widget())
    )

    with pytest.raises(SerializationError, match="already registered"):
        register_field_codec(
            _Widget, make_field_codec(encode=lambda w: 1, decode=lambda d: _Widget())
        )


def test_make_field_codec_wraps_decode_errors_as_value_error() -> None:
    """Test a bespoke codec's decode failure surfaces as a value error."""

    def _failing_decode(value: object) -> int:
        raise ValueError("bad value")

    @dataclass
    class _Wrapped(Serializable):
        x: int = field(
            metadata={
                "serialize_codec": make_field_codec(
                    encode=lambda v: v, decode=_failing_decode
                )
            }
        )

    with pytest.raises(DeserializationValueError, match="decodable"):
        _Wrapped.deserialize_from_dict({"x": 5})
