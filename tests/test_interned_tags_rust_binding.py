"""Tests the Python interface of the Rust-backed interned tags.

``OpAttribute``, ``NoteKind`` and ``ValueDomain`` are thin Python subclasses
of the ``fhy_core._rs`` classes over the Rust registries (slice S2 of
``docs/design/python-switch.md``). Their behavioral suites cover the tags'
semantics; this suite covers what the binding adds: canonical identity,
lookup, payloads and pickles, frozen errors, and the decisions D-S2-1,
D-S2-3 and D-S2-4.
"""

import base64
import copy
import io
import operator
import pickle
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pytest

import fhy_core
from fhy_core.diagnostic import (
    OTHER_NOTE_KIND,
    RATIONALE_NOTE_KIND,
    REMARK_NOTE_KIND,
    SUGGESTION_NOTE_KIND,
    Note,
    NoteKind,
)
from fhy_core.identifier import Identifier
from fhy_core.op_attribute import (
    ASSOCIATIVE,
    COMMUTATIVE,
    ELEMENTWISE,
    PURE,
    OpAttribute,
)
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    Serializable,
    SerializedDict,
)
from fhy_core.traits import FrozenMixin, FrozenMutationError, InternedMixin
from fhy_core.utils.override import override
from fhy_core.value_domain import ADDRESS_DOMAIN, DATA_DOMAIN, ValueDomain

_Tag = OpAttribute | NoteKind | ValueDomain

_TAG_CLASSES = pytest.mark.parametrize(
    "cls", [OpAttribute, NoteKind, ValueDomain], ids=lambda cls: cls.__name__
)

# Each shipped tag with its module, and the fixed id and name hint of
# `fhy_core::identifier::reserved`.
_SHIPPED_TAGS: list[tuple[_Tag, str, int, str]] = [
    (RATIONALE_NOTE_KIND, "fhy_core.diagnostic", 0, "rationale"),
    (SUGGESTION_NOTE_KIND, "fhy_core.diagnostic", 1, "suggestion"),
    (REMARK_NOTE_KIND, "fhy_core.diagnostic", 2, "remark"),
    (OTHER_NOTE_KIND, "fhy_core.diagnostic", 3, "other"),
    (COMMUTATIVE, "fhy_core.op_attribute", 16, "commutative"),
    (ASSOCIATIVE, "fhy_core.op_attribute", 17, "associative"),
    (PURE, "fhy_core.op_attribute", 18, "pure"),
    (ELEMENTWISE, "fhy_core.op_attribute", 19, "elementwise"),
    (DATA_DOMAIN, "fhy_core.value_domain", 32, "data"),
    (ADDRESS_DOMAIN, "fhy_core.value_domain", 33, "address"),
]
_SHIPPED = pytest.mark.parametrize(
    ("tag", "module_name", "reserved_id", "name_hint"),
    _SHIPPED_TAGS,
    ids=[name_hint for _, _, _, name_hint in _SHIPPED_TAGS],
)


def _build_payload(
    name: Identifier, description: str, cls: type[_Tag]
) -> SerializedDict:
    """Return the Python payload of a root tag of `cls`."""
    payload: SerializedDict = {
        "name": name.serialize_to_dict(),
        "description": description,
    }
    if cls is ValueDomain:
        payload["parent"] = None
    return payload


# =============================================================================
# Class structure
# =============================================================================


@_TAG_CLASSES
def test_public_class_is_a_thin_subclass_of_the_rust_class(cls: type[_Tag]) -> None:
    """Test the public class subclasses its `_rs` class and mixes in protocols."""
    rust_class = getattr(fhy_core._rs, cls.__name__)

    assert cls.__bases__[0] is rust_class
    assert issubclass(cls, Serializable)
    assert issubclass(cls, InternedMixin)
    assert issubclass(cls, FrozenMixin)
    assert InternedMixin not in cls.__mro__
    assert FrozenMixin not in cls.__mro__


@_TAG_CLASSES
def test_rust_class_cannot_be_constructed_without_the_binding(
    cls: type[_Tag],
) -> None:
    """Test the `_rs` class builds instances only from the binding's seeds."""
    rust_class = getattr(fhy_core._rs, cls.__name__)

    with pytest.raises(TypeError):
        rust_class(Identifier("no-seed"), "desc")


# =============================================================================
# Construction and canonical identity (D-S2-4)
# =============================================================================


@_TAG_CLASSES
def test_construction_of_a_registered_key_returns_the_canonical_instance(
    cls: type[_Tag],
) -> None:
    """Test constructing a registered key returns the first tag and description."""
    name = Identifier("binding-canonical")
    first = cls(name, "first description")

    second = cls(name, "second description")

    assert second is first
    assert second.description == "first description"


@_TAG_CLASSES
def test_construction_holds_the_given_name_object(cls: type[_Tag]) -> None:
    """Test a new tag returns the given `Identifier` object as its name."""
    name = Identifier("binding-name-object")

    tag = cls(name, "desc")

    assert tag.name is name
    assert tag.get_identifier() is name
    assert tag.get_intern_key() is name


@_TAG_CLASSES
def test_construction_accepts_keyword_arguments(cls: type[_Tag]) -> None:
    """Test the fields can be passed by keyword, as to the dataclasses."""
    name = Identifier("binding-keywords")

    tag = cls(name=name, description="desc")

    assert cls.get_interned(name) is tag


@_TAG_CLASSES
def test_construction_rejects_a_name_that_is_not_an_identifier(
    cls: type[_Tag],
) -> None:
    """Test a name must be an `Identifier`, the only key the Rust registry takes."""
    with pytest.raises(TypeError, match=f"{cls.__name__} name must be an Identifier"):
        cls("not-an-identifier", "desc")  # type: ignore[arg-type]  # test: wrong type


@_SHIPPED
def test_shipped_tag_is_the_canonical_rust_tag(
    tag: _Tag, module_name: str, reserved_id: int, name_hint: str
) -> None:
    """Test each shipped constant is the canonical tag for its reserved id."""
    del module_name
    reserved_name = Identifier.deserialize_from_dict(
        {"id": reserved_id, "name_hint": name_hint}
    )

    assert (tag.name.id, tag.name.name_hint) == (reserved_id, name_hint)
    assert type(tag).get_interned(reserved_name) is tag
    assert type(tag)(tag.name, "another description") is tag


# =============================================================================
# Lookup
# =============================================================================


@_TAG_CLASSES
def test_get_interned_returns_none_for_an_unregistered_key(cls: type[_Tag]) -> None:
    """Test `get_interned` returns `None` for an identifier no tag is named by."""
    assert cls.get_interned(Identifier("binding-unregistered")) is None


@_TAG_CLASSES
def test_get_interned_returns_none_for_a_key_that_is_not_an_identifier(
    cls: type[_Tag],
) -> None:
    """Test `get_interned` returns `None` for a hashable non-identifier key."""
    assert cls.get_interned("commutative") is None  # type: ignore[arg-type]


@_TAG_CLASSES
def test_get_interned_raises_for_an_unhashable_key(cls: type[_Tag]) -> None:
    """Test an unhashable key raises `TypeError`, as a dict lookup does."""
    with pytest.raises(TypeError, match="unhashable"):
        cls.get_interned([])  # type: ignore[arg-type]  # test: unhashable key


@_TAG_CLASSES
def test_get_interned_finds_a_tag_by_an_equal_identifier(cls: type[_Tag]) -> None:
    """Test lookup keys on the identifier's id, not on the identifier object."""
    name = Identifier("binding-equal-key")
    tag = cls(name, "desc")
    equal_name = Identifier.deserialize_from_dict(name.serialize_to_dict())

    assert equal_name is not name
    assert cls.get_interned(equal_name) is tag


@_TAG_CLASSES
def test_require_interned_raises_the_python_key_error(cls: type[_Tag]) -> None:
    """Test `require_interned` raises the Python implementation's `KeyError`."""
    key = Identifier("binding-missing")

    with pytest.raises(KeyError) as exc_info:
        cls.require_interned(key)

    assert exc_info.value.args == (
        f'No registered "{cls.__name__}" instance for key {key!r}.',
    )


# =============================================================================
# Payloads and pickles
# =============================================================================


@pytest.mark.usefixtures("v1_wire")
@_TAG_CLASSES
def test_serialize_to_dict_keeps_the_python_payload_shape(cls: type[_Tag]) -> None:
    """Test the payload keeps the Python `serialize_to_dict` shape."""
    name = Identifier("binding-payload-shape")
    tag = cls(name, "desc")

    assert tag.serialize_to_dict() == _build_payload(name, "desc", cls)


@pytest.mark.usefixtures("v1_wire")
def test_value_domain_payload_nests_its_parent_chain() -> None:
    """Test a domain's payload nests its parent's payload, up to the root."""
    middle = ValueDomain(Identifier("binding-middle"), "middle", parent=DATA_DOMAIN)
    leaf = ValueDomain(Identifier("binding-leaf"), "leaf", parent=middle)

    assert leaf.serialize_to_dict() == {
        "name": leaf.name.serialize_to_dict(),
        "description": "leaf",
        "parent": {
            "name": middle.name.serialize_to_dict(),
            "description": "middle",
            "parent": DATA_DOMAIN.serialize_to_dict(),
        },
    }


@_TAG_CLASSES
@pytest.mark.parametrize(
    "round_trip",
    [
        lambda tag: type(tag).deserialize_from_dict(tag.serialize_to_dict()),
        lambda tag: type(tag).from_json(tag.to_json()),
        lambda tag: Serializable.from_bytes(tag.to_bytes()),
        lambda tag: pickle.loads(pickle.dumps(tag)),
        copy.copy,
        copy.deepcopy,
    ],
    ids=["dict", "json", "bytes", "pickle", "copy", "deepcopy"],
)
def test_round_trip_returns_the_canonical_instance(
    cls: type[_Tag], round_trip: Callable[[_Tag], object]
) -> None:
    """Test every round trip of a tag returns the canonical tag itself."""
    tag = cls(Identifier("binding-round-trip"), "desc")

    assert round_trip(tag) is tag


def test_value_domain_round_trip_registers_a_new_chain_root_first() -> None:
    """Test decoding an unseen chain registers each level under its parent."""
    root_name = Identifier("binding-decoded-root")
    leaf_name = Identifier("binding-decoded-leaf")
    payload: SerializedDict = {
        "name": leaf_name.serialize_to_dict(),
        "description": "leaf",
        "parent": _build_payload(root_name, "root", ValueDomain),
    }

    leaf = ValueDomain.deserialize_from_dict(payload)

    root = ValueDomain.get_interned(root_name)
    assert root is not None
    assert leaf.parent is root
    assert leaf.is_subdomain_of(root)
    assert ValueDomain.get_interned(leaf_name) is leaf


def test_note_round_trip_returns_the_canonical_kind() -> None:
    """Test a `Note` holds the canonical kind after a round trip."""
    note = Note("message", SUGGESTION_NOTE_KIND)

    restored = Note.deserialize_from_dict(note.serialize_to_dict())

    assert restored.kind is SUGGESTION_NOTE_KIND


class _GlobalRecordingUnpickler(pickle.Unpickler):
    """Unpickler that records every global a pickle refers to."""

    found_globals: list[tuple[str, str]]

    def __init__(self, data: bytes) -> None:
        super().__init__(io.BytesIO(data))
        self.found_globals = []

    @override
    def find_class(self, module_name: str, global_name: str) -> Any:
        self.found_globals.append((module_name, global_name))
        return super().find_class(module_name, global_name)


@_SHIPPED
def test_pickle_refers_only_to_the_public_class(
    tag: _Tag, module_name: str, reserved_id: int, name_hint: str
) -> None:
    """Test a pickle names the public class, never `fhy_core._rs`."""
    del reserved_id, name_hint
    unpickler = _GlobalRecordingUnpickler(pickle.dumps(tag))

    assert unpickler.load() is tag
    assert unpickler.found_globals == [
        ("builtins", "getattr"),
        (module_name, type(tag).__name__),
    ]


def _run_python_in_a_fresh_process(source: str, *, stdin: str | None = None) -> str:
    """Run a program in a fresh interpreter and return its output."""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@pytest.mark.slow
@pytest.mark.subprocess
def test_pickles_of_shipped_tags_are_identical_in_every_process() -> None:
    """Test each shipped tag pickles to the same bytes in a fresh process (D-S2-2)."""
    this_process_pickles = [
        base64.b64encode(pickle.dumps(tag)).decode("ascii")
        for tag, _, _, _ in _SHIPPED_TAGS
    ]
    shipped_keys = [
        f"{module_name}:{type(tag).__name__}:{reserved_id}:{name_hint}"
        for tag, module_name, reserved_id, name_hint in _SHIPPED_TAGS
    ]

    fresh_process_pickles = _run_python_in_a_fresh_process(
        "import base64, importlib, pickle, sys\n"
        "from fhy_core.identifier import Identifier\n"
        "for key in sys.stdin.read().split():\n"
        "    module_name, class_name, reserved_id, name_hint = key.split(':')\n"
        "    cls = getattr(importlib.import_module(module_name), class_name)\n"
        "    name = Identifier.deserialize_from_dict(\n"
        "        {'id': int(reserved_id), 'name_hint': name_hint}\n"
        "    )\n"
        "    tag = cls.require_interned(name)\n"
        "    print(base64.b64encode(pickle.dumps(tag)).decode('ascii'))",
        stdin=" ".join(shipped_keys),
    ).split()

    assert fresh_process_pickles == this_process_pickles


@pytest.mark.slow
@pytest.mark.subprocess
def test_pickle_of_a_new_domain_loads_in_a_fresh_process() -> None:
    """Test a domain chain pickled in this process loads in a fresh one."""
    child = ValueDomain(Identifier("binding-pickled"), "child", parent=ADDRESS_DOMAIN)
    payload = base64.b64encode(pickle.dumps(child)).decode("ascii")

    output = _run_python_in_a_fresh_process(
        "import base64, pickle, sys\n"
        "from fhy_core.value_domain import ADDRESS_DOMAIN\n"
        "child = pickle.loads(base64.b64decode(sys.stdin.read()))\n"
        "print(child.name.id, child.description, child.parent is ADDRESS_DOMAIN)",
        stdin=payload,
    )

    assert output.split() == [str(child.name.id), "child", "True"]


@pytest.mark.slow
@pytest.mark.subprocess
def test_pickle_from_a_fresh_process_loads_as_the_canonical_tag() -> None:
    """Test a pickle written in a fresh process loads as the canonical tag."""
    payload = _run_python_in_a_fresh_process(
        "import base64, pickle\n"
        "from fhy_core.op_attribute import PURE\n"
        "print(base64.b64encode(pickle.dumps(PURE)).decode('ascii'))"
    )

    assert pickle.loads(base64.b64decode(payload)) is PURE


# =============================================================================
# Malformed payloads
# =============================================================================


@_TAG_CLASSES
def test_malformed_payload_raises_the_python_structure_error(cls: type[_Tag]) -> None:
    """Test a malformed payload raises the framework's structure error."""
    data: SerializedDict = {"name": 1, "description": "desc"}
    expected_structure: dict[str, Any] = {"name": dict, "description": str}
    if cls is ValueDomain:
        data["parent"] = None
        expected_structure["parent"] = dict | None

    with pytest.raises(DeserializationDictStructureError) as exc_info:
        cls.deserialize_from_dict(data)

    expected = DeserializationDictStructureError(cls, expected_structure, data)
    assert str(exc_info.value) == str(expected)


@_TAG_CLASSES
def test_malformed_name_raises_the_identifier_error(cls: type[_Tag]) -> None:
    """Test a malformed name raises the error `Identifier` deserialization raises."""
    data = _build_payload(Identifier("binding-bad-id"), "desc", cls)
    data["name"] = {"id": -1, "name_hint": "negative"}

    with pytest.raises(DeserializationValueError, match='field "id"'):
        cls.deserialize_from_dict(data)


@_TAG_CLASSES
def test_construct_from_fields_returns_the_canonical_instance(
    cls: type[_Tag],
) -> None:
    """Test `construct_from_fields` returns the tag registered for the name."""
    name = Identifier("binding-from-fields")
    tag = cls(name, "desc")

    assert cls.construct_from_fields({"name": name, "description": "other"}) is tag


@_TAG_CLASSES
def test_construct_from_fields_rejects_an_unknown_field(cls: type[_Tag]) -> None:
    """Test an unknown field raises `TypeError`, as `cls(**fields)` would."""
    fields = {"name": Identifier("binding-extra-field"), "description": "d", "x": 1}

    with pytest.raises(TypeError, match="construct_from_fields"):
        cls.construct_from_fields(fields)


# =============================================================================
# Frozen instances
# =============================================================================


@_TAG_CLASSES
def test_mutation_raises_the_frozen_mixin_error(cls: type[_Tag]) -> None:
    """Test assigning or deleting an attribute raises `FrozenMixin`'s error."""
    tag = cls(Identifier("binding-frozen"), "desc")

    with pytest.raises(FrozenMutationError) as set_info:
        tag.description = "rewritten"  # type: ignore[misc]  # test: frozen
    with pytest.raises(FrozenMutationError) as new_info:
        tag.extra = 1  # type: ignore[union-attr]  # test: no such attribute
    with pytest.raises(FrozenMutationError) as delete_info:
        del tag.name

    frozen = f"on frozen {cls.__name__}."
    assert str(set_info.value) == f'Cannot modify "description" {frozen}'
    assert str(new_info.value) == f'Cannot modify "extra" {frozen}'
    assert str(delete_info.value) == f'Cannot delete "name" {frozen}'
    assert tag.description == "desc"


@_TAG_CLASSES
def test_tag_reports_itself_frozen(cls: type[_Tag]) -> None:
    """Test the `Frozen` protocol members report an always-frozen tag."""
    tag = cls(Identifier("binding-is-frozen"), "desc")

    tag.freeze()
    tag.assert_frozen()

    assert tag.is_frozen


# =============================================================================
# Append-only registries (D-S2-1)
# =============================================================================


@_TAG_CLASSES
@pytest.mark.parametrize(
    "method_name", ["clear_interned_registry", "register_default_instances"]
)
def test_registry_reset_raises_not_implemented(
    cls: type[_Tag], method_name: str
) -> None:
    """Test the registry operations that need a clearable registry raise.

    The registry is left as it was, so a shipped tag stays canonical.
    """
    shipped: dict[type[_Tag], _Tag] = {
        OpAttribute: PURE,
        NoteKind: REMARK_NOTE_KIND,
        ValueDomain: DATA_DOMAIN,
    }

    with pytest.raises(NotImplementedError, match="append-only") as exc_info:
        getattr(cls, method_name)()

    assert str(exc_info.value).startswith(f"{cls.__name__}.{method_name} is not")
    assert cls.get_interned(shipped[cls].name) is shipped[cls]


# =============================================================================
# Equality, hashing and rendering
# =============================================================================


@_TAG_CLASSES
def test_tags_compare_and_hash_by_name(cls: type[_Tag]) -> None:
    """Test tags compare by name, hash equal when equal, and never equal others."""
    first = cls(Identifier("binding-eq-a"), "desc")
    second = cls(Identifier("binding-eq-b"), "desc")

    assert first == cls(first.name, "another description")
    assert first != second
    assert hash(first) != hash(second)
    assert first != first.name
    assert first.is_structurally_equivalent(first)
    assert not first.is_structurally_equivalent(second)
    assert first.is_alpha_equivalent(first)
    assert not first.is_alpha_equivalent(second)


def test_tags_of_different_classes_are_not_equivalent() -> None:
    """Test one identifier names independent tags in different classes."""
    name = Identifier("binding-two-classes")
    attribute = OpAttribute(name, "an attribute")
    kind = NoteKind(name, "a note kind")

    assert not operator.eq(attribute, kind)
    assert not attribute.is_structurally_equivalent(kind)
    assert kind.description == "a note kind"


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        (
            COMMUTATIVE,
            "OpAttribute(name=commutative::16, "
            "description='Op output is invariant under operand swap.')",
        ),
        (OTHER_NOTE_KIND, "NoteKind(name=other::3, description='Uncategorized note.')"),
        (
            ADDRESS_DOMAIN,
            "ValueDomain(name=address::33, description='Index, offset, or address "
            "values used to access data.', parent=None)",
        ),
    ],
    ids=["OpAttribute", "NoteKind", "ValueDomain"],
)
def test_repr_matches_the_dataclass_repr(tag: _Tag, expected: str) -> None:
    """Test `repr` renders in the dataclass `repr` form."""
    assert repr(tag) == expected


def test_note_kind_str_is_its_name_hint() -> None:
    """Test `str` of a note kind renders its name hint, as `Note` shows it."""
    assert str(RATIONALE_NOTE_KIND) == "rationale"
    assert str(Note("tiled", RATIONALE_NOTE_KIND)) == "rationale: tiled"


# =============================================================================
# Value domains follow the Rust semantics (D-S2-3)
# =============================================================================


def test_value_domain_construction_under_another_parent_raises() -> None:
    """Test constructing a name under another parent raises `ValueError`."""
    name = Identifier("binding-conflict")
    domain = ValueDomain(name, "desc", parent=DATA_DOMAIN)

    with pytest.raises(ValueError, match="already registered with parent `data`"):
        ValueDomain(name, "desc", parent=ADDRESS_DOMAIN)
    with pytest.raises(ValueError, match="not as a root"):
        ValueDomain(name, "desc")

    assert ValueDomain.get_interned(name) is domain


def test_value_domain_construction_of_a_root_under_a_parent_raises() -> None:
    """Test a registered root cannot be constructed again as a child."""
    name = Identifier("binding-root-conflict")
    ValueDomain(name, "desc")

    with pytest.raises(ValueError, match="with no parent, not `data`"):
        ValueDomain(name, "desc", parent=DATA_DOMAIN)


def test_value_domain_construct_from_fields_raises_the_conflict_error() -> None:
    """Test fields under another parent raise `DeserializationValueError`."""
    name = Identifier("binding-fields-conflict")
    ValueDomain(name, "desc", parent=DATA_DOMAIN)

    with pytest.raises(DeserializationValueError) as exc_info:
        ValueDomain.construct_from_fields(
            {"name": name, "description": "desc", "parent": ADDRESS_DOMAIN}
        )

    assert str(exc_info.value) == (
        f'Payload for "ValueDomain" key {name!r} conflicts with the canonical '
        f"instance on parent (canonical {DATA_DOMAIN!r}, payload "
        f"{ADDRESS_DOMAIN!r})."
    )


def test_value_domain_rejects_a_parent_that_is_not_a_domain() -> None:
    """Test a parent must be a `ValueDomain` or `None`."""
    with pytest.raises(TypeError, match="parent must be a ValueDomain or None"):
        ValueDomain(Identifier("binding-bad-parent"), "desc", parent=COMMUTATIVE)  # type: ignore[arg-type]


def test_value_domain_parent_is_the_canonical_parent_object() -> None:
    """Test `parent` returns the parent's single Python object."""
    name = Identifier("binding-parent-object")

    child = ValueDomain(name, "child", parent=DATA_DOMAIN)

    assert child.parent is DATA_DOMAIN


def test_value_domain_is_not_a_subdomain_of_a_non_domain() -> None:
    """Test `is_subdomain_of` returns `False` for anything but a domain."""
    assert not DATA_DOMAIN.is_subdomain_of(COMMUTATIVE)
