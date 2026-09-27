"""Interface tests of the symbol table over the Rust core (S15).

The behavior of the table itself is specified by the Rust tests of
``fhy_core::symbol_table``; these tests cover what the binding adds: the
Python classes and their protocols, argument checks, the objects the table
and the frames hand back, payloads, pickling, a frame Python defines, the
errors' texts, the report of ``verify`` and the DEBUG lines.
"""

import copy
import dataclasses
import json
import logging
import pickle
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from fhy_core import _rs
from fhy_core.diagnostic import DiagnosticLevel
from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    SerializedDict,
    register_serializable,
)
from fhy_core.symbol_table import (
    FunctionKeyword,
    FunctionSymbolTableFrame,
    ImportSymbolTableFrame,
    SymbolTable,
    SymbolTableError,
    SymbolTableFrame,
    VariableSymbolTableFrame,
)
from fhy_core.term import AlphaRenaming
from fhy_core.traits import FrozenMixin, FrozenMutationError, VerifiableMixin
from fhy_core.types import (
    CoreDataType,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
)
from fhy_core.utils.override import override

# The identifiers this file builds, by name hint and number, so a test that
# names one twice gets the same identifier. They are real identifiers, since
# the tests pickle them and log them.
_IDENTIFIERS: dict[tuple[str, int], Identifier] = {}


def _int32() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32))


def _identifier(name: str, number: int) -> Identifier:
    return _IDENTIFIERS.setdefault((name, number), Identifier(name))


@register_serializable(type_id="test_symbol_table_binding_note_frame")
@dataclasses.dataclass(frozen=True)
class NoteFrame(SymbolTableFrame):
    """A frame a third party defines, with a field of its own."""

    note: str


@dataclasses.dataclass(frozen=True)
class RaisingFrame(SymbolTableFrame):
    """A frame whose structural equivalence raises."""

    @property
    def marker(self) -> int:
        return 1

    @override
    def is_structurally_equivalent(self, other: object) -> bool:
        raise RuntimeError("cannot compare")


class _NamedLikeFrame(SymbolTableFrame):
    """A frame whose name is not an identifier."""


def _pair_table(frame: SymbolTableFrame | None = None) -> SymbolTable:
    namespace = _identifier("ns", 0)
    symbol = _identifier("x", 1)
    table = SymbolTable()
    table.add_namespace(namespace)
    table.add_symbol(
        namespace,
        symbol,
        frame if frame is not None else ImportSymbolTableFrame(symbol),
    )
    return table


# ---------------------------------------------------------------------------
# Classes and protocols
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("public", "rust"),
    [
        (ImportSymbolTableFrame, _rs.ImportSymbolTableFrame),
        (VariableSymbolTableFrame, _rs.VariableSymbolTableFrame),
        (FunctionSymbolTableFrame, _rs.FunctionSymbolTableFrame),
        (SymbolTable, _rs.SymbolTable),
    ],
)
def test_public_classes_are_thin_subclasses_of_the_rust_classes(
    public: type, rust: type
) -> None:
    """Test each public class subclasses its `_rs` class."""
    assert issubclass(public, rust)


@pytest.mark.parametrize(
    "frame_class",
    [ImportSymbolTableFrame, VariableSymbolTableFrame, FunctionSymbolTableFrame],
)
def test_built_in_frames_are_virtual_frames_and_frozen_mixins(
    frame_class: type,
) -> None:
    """Test the built-in frames register with `SymbolTableFrame` and `FrozenMixin`."""
    assert issubclass(frame_class, SymbolTableFrame)
    assert issubclass(frame_class, FrozenMixin)
    assert SymbolTableFrame not in frame_class.__mro__
    assert not dataclasses.is_dataclass(frame_class)


def test_symbol_table_is_a_virtual_verifiable_mixin() -> None:
    """Test `SymbolTable` registers with `VerifiableMixin`."""
    assert isinstance(SymbolTable(), VerifiableMixin)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def test_frames_keep_their_field_objects() -> None:
    """Test each frame's attributes are the objects it was given."""
    name, int32 = _identifier("x", 2), _int32()
    variable = VariableSymbolTableFrame(name, int32, TypeQualifier.STATE)
    function = FunctionSymbolTableFrame(
        name, FunctionKeyword.OPERATION, [(TypeQualifier.INPUT, int32)]
    )

    assert variable.name is name
    assert variable.type is int32
    assert variable.type_qualifier is TypeQualifier.STATE
    assert function.keyword is FunctionKeyword.OPERATION
    assert function.signature[0][1] is int32
    assert ImportSymbolTableFrame(name).name is name


def test_a_signature_is_stored_as_a_tuple_of_pairs() -> None:
    """Test a list signature becomes a tuple, so the frame hashes."""
    name, int32 = _identifier("f", 3), _int32()
    frame = FunctionSymbolTableFrame(
        name,
        FunctionKeyword.PROCEDURE,
        [[TypeQualifier.INPUT, int32]],  # type: ignore[list-item]
    )

    assert frame.signature == ((TypeQualifier.INPUT, int32),)
    assert hash(frame) == hash(
        FunctionSymbolTableFrame(
            name, FunctionKeyword.PROCEDURE, ((TypeQualifier.INPUT, _int32()),)
        )
    )
    assert FunctionSymbolTableFrame(name, FunctionKeyword.NATIVE).signature == ()


def test_frame_repr_is_the_dataclass_repr() -> None:
    """Test `repr` lists every field as the dataclass did."""
    name = _identifier("x", 4)
    frame = VariableSymbolTableFrame(name, _int32(), TypeQualifier.INPUT)

    assert repr(frame) == (
        f"VariableSymbolTableFrame(name={name!r}, type={_int32()!r}, "
        f"type_qualifier={TypeQualifier.INPUT!r})"
    )
    assert (
        repr(ImportSymbolTableFrame(name)) == f"ImportSymbolTableFrame(name={name!r})"
    )


def test_frames_are_frozen() -> None:
    """Test mutating a frame raises `FrozenMutationError`."""
    frame = ImportSymbolTableFrame(_identifier("x", 5))

    assert frame.is_frozen
    frame.assert_frozen()
    with pytest.raises(FrozenMutationError, match='Cannot modify "name"'):
        frame.name = _identifier("y", 6)  # type: ignore[misc]
    with pytest.raises(FrozenMutationError, match='Cannot delete "name"'):
        del frame.name


@pytest.mark.parametrize("round_trip", [pickle.loads, copy.copy, copy.deepcopy])
def test_frames_pickle_and_copy_to_equal_frames(round_trip: Any) -> None:
    """Test pickling and copying a frame gives an equal frame."""
    int32 = _int32()
    frame = FunctionSymbolTableFrame(
        _identifier("f", 7), FunctionKeyword.PROCEDURE, ((TypeQualifier.INPUT, int32),)
    )
    argument = pickle.dumps(frame) if round_trip is pickle.loads else frame

    restored = round_trip(argument)

    assert restored == frame
    assert type(restored) is FunctionSymbolTableFrame


def test_eq_needs_the_same_class_and_agrees_with_hash() -> None:
    """Test `==` requires the exact class, and equal frames hash alike."""
    name = _identifier("x", 8)

    class LocalImport(ImportSymbolTableFrame):
        __slots__ = ()

    assert ImportSymbolTableFrame(name) == ImportSymbolTableFrame(name)
    assert hash(ImportSymbolTableFrame(name)) == hash(ImportSymbolTableFrame(name))
    assert ImportSymbolTableFrame(name) != LocalImport(name)
    assert ImportSymbolTableFrame(name) != VariableSymbolTableFrame(
        name, _int32(), TypeQualifier.STATE
    )
    assert ImportSymbolTableFrame(name) != name


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: ImportSymbolTableFrame("x"),  # type: ignore[arg-type]
            "ImportSymbolTableFrame name must be an Identifier, got str.",
        ),
        (
            lambda: VariableSymbolTableFrame(
                _identifier("x", 9),
                3,  # type: ignore[arg-type]
                TypeQualifier.STATE,
            ),
            "VariableSymbolTableFrame type must be a Type, got int.",
        ),
        (
            lambda: VariableSymbolTableFrame(_identifier("x", 10), _int32(), "state"),  # type: ignore[arg-type]
            "VariableSymbolTableFrame type_qualifier must be a TypeQualifier, got str.",
        ),
        (
            lambda: FunctionSymbolTableFrame(_identifier("f", 11), "proc"),  # type: ignore[arg-type]
            "FunctionSymbolTableFrame keyword must be a FunctionKeyword, got str.",
        ),
        (
            lambda: FunctionSymbolTableFrame(
                _identifier("f", 12),
                FunctionKeyword.PROCEDURE,
                [TypeQualifier.INPUT],  # type: ignore[list-item]
            ),
            "FunctionSymbolTableFrame signature entry must be a "
            "(TypeQualifier, Type) pair",
        ),
        (
            lambda: FunctionSymbolTableFrame(
                _identifier("f", 13),
                FunctionKeyword.PROCEDURE,
                [(_int32(), TypeQualifier.INPUT)],  # type: ignore[list-item]
            ),
            "FunctionSymbolTableFrame signature qualifier must be a TypeQualifier",
        ),
    ],
)
def test_frame_arguments_are_checked(build: Any, message: str) -> None:
    """Test a frame refuses arguments of the wrong type."""
    with pytest.raises(
        TypeError, match=message.replace("(", r"\(").replace(")", r"\)")
    ):
        build()


@pytest.mark.usefixtures("v1_wire")
def test_frame_payloads_are_todays_wire_format() -> None:
    """Test each frame's payload, pinned as the dataclass wrote it."""
    name = _identifier("x", 14)
    int32 = _int32()
    type_payload = int32.serialize_to_dict()

    assert ImportSymbolTableFrame(name).serialize_to_dict() == {
        "__type__": "import_symbol_table_frame",
        "__data__": {"name": {"id": name.id, "name_hint": "x"}},
    }
    assert VariableSymbolTableFrame(
        name, int32, TypeQualifier.INPUT
    ).serialize_to_dict() == {
        "__type__": "variable_symbol_table_frame",
        "__data__": {
            "name": {"id": name.id, "name_hint": "x"},
            "type": type_payload,
            "type_qualifier": "input",
        },
    }
    function = FunctionSymbolTableFrame(
        name, FunctionKeyword.PROCEDURE, ((TypeQualifier.OUTPUT, int32),)
    )
    assert function.serialize_to_dict() == {
        "__type__": "function_symbol_table_frame",
        "__data__": {
            "name": {"id": name.id, "name_hint": "x"},
            "keyword": "proc",
            "signature": [{"type_qualifier": "output", "type": type_payload}],
        },
    }
    for frame in (ImportSymbolTableFrame(name), function):
        payload = json.loads(json.dumps(frame.serialize_to_dict()))
        assert SymbolTableFrame.deserialize_from_dict(payload) == frame


@pytest.mark.parametrize(
    ("data", "error"),
    [
        (
            {
                "__type__": "import_symbol_table_frame",
                "__data__": {"name": {"id": 1, "name_hint": "a"}, "z": 1},
            },
            DeserializationDictStructureError,
        ),
        (
            {
                "__type__": "variable_symbol_table_frame",
                "__data__": {"name": {"id": 1, "name_hint": "a"}},
            },
            DeserializationDictStructureError,
        ),
        (
            {
                "__type__": "variable_symbol_table_frame",
                "__data__": {
                    "name": {"id": 1, "name_hint": "a"},
                    "type": NumericalType(
                        PrimitiveDataType(CoreDataType.INT8)
                    ).serialize_to_dict(),
                    "type_qualifier": "bogus",
                },
            },
            DeserializationValueError,
        ),
        (
            {
                "__type__": "function_symbol_table_frame",
                "__data__": {
                    "name": {"id": 1, "name_hint": "a"},
                    "keyword": "bogus",
                    "signature": [],
                },
            },
            DeserializationValueError,
        ),
        (
            {
                "__type__": "function_symbol_table_frame",
                "__data__": {
                    "name": {"id": 1, "name_hint": "a"},
                    "keyword": "proc",
                    "signature": [{"x": 1}],
                },
            },
            DeserializationDictStructureError,
        ),
    ],
)
def test_malformed_frame_payloads_are_refused(
    data: SerializedDict, error: type[Exception]
) -> None:
    """Test a malformed frame payload raises the framework's error class."""
    with pytest.raises(error):
        SymbolTableFrame.deserialize_from_dict(data)


def test_a_bad_function_keyword_names_the_value() -> None:
    """Test the function frame keeps its message for a bad keyword."""
    data: SerializedDict = {
        "__type__": "function_symbol_table_frame",
        "__data__": {
            "name": {"id": 1, "name_hint": "a"},
            "keyword": "bogus",
            "signature": [],
        },
    }

    with pytest.raises(
        DeserializationValueError,
        match="Invalid function frame values: 'bogus' is not a valid FunctionKeyword",
    ):
        SymbolTableFrame.deserialize_from_dict(data)


def test_frame_equivalence_methods_ignore_the_renaming() -> None:
    """Test the three equivalence methods compare structurally, renaming unused."""
    x, y = _identifier("x", 15), _identifier("y", 16)
    left = VariableSymbolTableFrame(x, _int32(), TypeQualifier.STATE)
    same = VariableSymbolTableFrame(x, _int32(), TypeQualifier.STATE)
    renamed = VariableSymbolTableFrame(y, _int32(), TypeQualifier.STATE)
    renaming = AlphaRenaming.empty().with_free_renaming({x: y})

    assert left.is_structurally_equivalent(same)
    assert left.is_alpha_equivalent(same)
    assert left.is_alpha_equivalent_under(same, renaming)
    assert not left.is_alpha_equivalent_under(renamed, renaming)
    assert not left.is_structurally_equivalent(ImportSymbolTableFrame(x))
    with pytest.raises(TypeError, match="renaming must be an AlphaRenaming"):
        left.is_alpha_equivalent_under(same, {x: y})  # type: ignore[arg-type]


def test_a_python_defined_type_in_a_frame_compares_through_its_eq() -> None:
    """Test a frame whose type Python defines compares and hashes through it."""

    class Opaque(Type):
        def __init__(self, tag: int) -> None:
            super().__init__()
            self.tag = tag

        @override
        def __eq__(self, other: object) -> bool:
            return isinstance(other, Opaque) and other.tag == self.tag

        @override
        def __hash__(self) -> int:
            return hash(self.tag)

    name = _identifier("x", 17)
    left = VariableSymbolTableFrame(name, Opaque(1), TypeQualifier.STATE)

    assert left == VariableSymbolTableFrame(name, Opaque(1), TypeQualifier.STATE)
    assert left != VariableSymbolTableFrame(name, Opaque(2), TypeQualifier.STATE)
    assert hash(left) == hash(
        VariableSymbolTableFrame(name, Opaque(1), TypeQualifier.STATE)
    )


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_the_table_returns_the_objects_it_was_given() -> None:
    """Test lookups return the frame objects and symbol objects added."""
    namespace, child, symbol = (
        _identifier("ns", 18),
        _identifier("c", 19),
        _identifier("x", 20),
    )
    frame = ImportSymbolTableFrame(symbol)
    table = SymbolTable()
    table.add_namespace(namespace)
    table.add_namespace(child, namespace)
    table.add_symbol(namespace, symbol, frame)

    assert table.get_frame(symbol) is frame
    assert table.get_frame_from_namespace(child, symbol) is frame
    (key,) = table.get_namespace(namespace)
    assert key is symbol


def test_get_namespace_returns_a_new_dict() -> None:
    """Test mutating `get_namespace`'s dict leaves the table unchanged."""
    table = _pair_table()
    namespace, symbol = _identifier("ns", 0), _identifier("x", 1)

    symbols = table.get_namespace(namespace)
    symbols.clear()

    assert table.get_namespace(namespace) is not symbols
    assert table.is_symbol_defined_in_namespace(namespace, symbol)


def test_get_namespace_reflects_each_change() -> None:
    """Test `get_namespace` answers from the table after every kind of change."""
    namespace, first, second = (
        _identifier("ns", 47),
        _identifier("a", 48),
        _identifier("b", 49),
    )
    table = SymbolTable()
    table.add_namespace(namespace)
    assert table.get_namespace(namespace) == {}

    table.add_symbol(namespace, second, ImportSymbolTableFrame(second))
    table.add_symbol(namespace, first, ImportSymbolTableFrame(first))
    assert list(table.get_namespace(namespace)) == [second, first]

    table.canonicalize()
    assert list(table.get_namespace(namespace)) == [first, second]

    table.remove_symbol(namespace, first)
    assert list(table.get_namespace(namespace)) == [second]

    source = SymbolTable()
    source.add_namespace(namespace)
    table.update_namespaces(source)
    assert table.get_namespace(namespace) == {}

    table.remove_namespace(namespace)
    table.add_namespace(namespace)
    table.add_symbol(namespace, first, ImportSymbolTableFrame(first))
    assert list(table.get_namespace(namespace)) == [first]
    restored = pickle.loads(pickle.dumps(table))
    assert list(restored.get_namespace(namespace)) == [first]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda t: t.add_namespace("ns"),
            "SymbolTable namespace_name must be an Identifier, got str.",
        ),
        (
            lambda t: t.add_namespace(_identifier("a", 21), "p"),
            "SymbolTable parent_namespace_name must be an Identifier, got str.",
        ),
        (
            lambda t: t.is_namespace_defined(1),
            "SymbolTable namespace_name must be an Identifier, got int.",
        ),
        (
            lambda t: t.add_symbol(
                _identifier("ns", 0), _identifier("y", 22), object()
            ),
            "SymbolTable frame must be a SymbolTableFrame, got object.",
        ),
        (
            lambda t: t.get_frame("x"),
            "SymbolTable symbol_name must be an Identifier, got str.",
        ),
        (
            lambda t: t.update_namespaces({}),
            "SymbolTable other_symbol_table must be a SymbolTable, got dict.",
        ),
    ],
)
def test_table_arguments_are_checked(call: Any, message: str) -> None:
    """Test the table refuses arguments of the wrong type."""
    with pytest.raises(TypeError, match=message):
        call(_pair_table())


def test_a_lookup_through_a_missing_parent_raises_a_symbol_table_error() -> None:
    """Test a missing parent fails a lookup with `SymbolTableError`."""
    child, missing = _identifier("child", 23), _identifier("missing", 24)
    table = SymbolTable()
    table.add_namespace(child, missing)

    with pytest.raises(SymbolTableError, match="references missing parent namespace"):
        table.is_symbol_defined_in_namespace(child, _identifier("x", 25))


def test_errors_carry_the_cores_text() -> None:
    """Test each error's text names the identifiers with their ids."""
    root, child, symbol = (
        _identifier("root", 26),
        _identifier("child", 27),
        _identifier("x", 28),
    )
    table = SymbolTable()
    table.add_namespace(root)
    table.add_namespace(child, root)
    table.add_symbol(root, symbol, ImportSymbolTableFrame(symbol))
    cases = [
        (
            lambda: table.add_namespace(root),
            f"namespace {root!r} already defined in the symbol table",
        ),
        (
            lambda: table.add_symbol(child, symbol, ImportSymbolTableFrame(symbol)),
            f"symbol {symbol!r} already defined in namespace {root!r}, "
            f"an ancestor of namespace {child!r}",
        ),
        (
            lambda: table.get_frame(child),
            f"symbol {child!r} not found in the symbol table",
        ),
        (
            lambda: table.get_frame_from_namespace(child, child),
            f"symbol {child!r} not found in namespace {child!r}",
        ),
        (
            lambda: table.remove_namespace(root),
            f"namespace {root!r} cannot be removed because it is the parent of "
            f"{child!r}",
        ),
        (
            lambda: table.get_namespace(symbol),
            f"namespace {symbol!r} not found in the symbol table",
        ),
    ]
    checked: Sequence[tuple[Callable[[], object], str]] = cases
    for call, text in checked:
        with pytest.raises(SymbolTableError) as caught:
            call()
        assert str(caught.value) == text


def test_update_namespaces_of_the_table_itself_changes_nothing() -> None:
    """Test merging a table into itself leaves it as it was."""
    table = _pair_table()
    before = table.serialize_to_dict()

    table.update_namespaces(table)

    assert table.serialize_to_dict() == before


def test_the_table_payload_is_todays_wire_format() -> None:
    """Test the table's payload, pinned as the Python class wrote it."""
    root, child, symbol = (
        _identifier("root", 29),
        _identifier("child", 30),
        _identifier("x", 31),
    )
    frame = ImportSymbolTableFrame(symbol)
    table = SymbolTable()
    table.add_namespace(root)
    table.add_namespace(child, root)
    table.add_symbol(child, symbol, frame)

    assert table.serialize_to_dict() == {
        "namespaces": [
            {
                "namespace_name": {"id": root.id, "name_hint": "root"},
                "parent_namespace_name": None,
                "symbols": [],
            },
            {
                "namespace_name": {"id": child.id, "name_hint": "child"},
                "parent_namespace_name": {"id": root.id, "name_hint": "root"},
                "symbols": [
                    {
                        "symbol_name": {"id": symbol.id, "name_hint": "x"},
                        "frame": frame.serialize_to_dict(),
                    }
                ],
            },
        ]
    }


def test_a_payload_that_cannot_be_rebuilt_raises_a_symbol_table_error() -> None:
    """Test a duplicate namespace in a payload fails the replay."""
    entry: SerializedDict = {
        "namespace_name": {"id": _identifier("a", 32).id, "name_hint": "a"},
        "parent_namespace_name": None,
        "symbols": [],
    }

    with pytest.raises(SymbolTableError, match="already defined"):
        SymbolTable.deserialize_from_dict({"namespaces": [entry, entry]})
    with pytest.raises(DeserializationDictStructureError):
        SymbolTable.deserialize_from_dict(
            {"namespaces": [{"namespace_name": entry["namespace_name"]}]}
        )


@pytest.mark.parametrize("round_trip", ["pickle", "copy", "deepcopy"])
def test_a_table_pickles_and_copies_to_an_independent_equivalent_table(
    round_trip: str,
) -> None:
    """Test pickled and copied tables are equivalent and independent."""
    table = _pair_table(
        VariableSymbolTableFrame(_identifier("x", 1), _int32(), TypeQualifier.STATE)
    )
    restored: SymbolTable
    if round_trip == "pickle":
        restored = pickle.loads(pickle.dumps(table))
    elif round_trip == "copy":
        restored = copy.copy(table)
    else:
        restored = copy.deepcopy(table)

    assert restored.is_structurally_equivalent(table)
    restored.add_symbol(
        _identifier("ns", 0),
        _identifier("y", 33),
        ImportSymbolTableFrame(_identifier("y", 33)),
    )
    assert not table.is_symbol_defined(_identifier("y", 33))


def test_a_subclass_keeps_its_class_and_attributes_through_a_pickle() -> None:
    """Test a subclass with its own `__init__` constructs and pickles."""

    class LabelledTable(SymbolTable):
        def __init__(self, label: str) -> None:
            super().__init__()
            self.label = label

    table = LabelledTable("mine")
    table.add_namespace(_identifier("ns", 34))

    restored = copy.deepcopy(table)

    assert type(restored) is LabelledTable
    assert restored.label == "mine"
    assert restored.is_namespace_defined(_identifier("ns", 34))


def test_a_state_only_a_merge_reaches_round_trips_through_pickle() -> None:
    """Test a shadowing symbol, which `add_symbol` refuses, survives a pickle."""
    root, child, symbol = (
        _identifier("root", 35),
        _identifier("child", 36),
        _identifier("x", 37),
    )
    table = SymbolTable()
    table.add_namespace(root)
    table.add_namespace(child, root)
    table.add_symbol(root, symbol, ImportSymbolTableFrame(symbol))
    source = SymbolTable()
    source.add_namespace(child)
    source.add_symbol(child, symbol, ImportSymbolTableFrame(symbol))
    table.update_namespaces(source)

    restored = pickle.loads(pickle.dumps(table))

    assert restored.is_structurally_equivalent(table)
    assert restored.get_namespace(child) == table.get_namespace(child)


def test_verify_reports_each_violation_as_an_error() -> None:
    """Test `verify`'s diagnostics: messages, source and level."""
    namespace, symbol, other = (
        _identifier("ns", 38),
        _identifier("x", 39),
        _identifier("y", 40),
    )
    own = _identifier("own", 41)
    table = SymbolTable()
    table.add_namespace(namespace)
    table.add_symbol(namespace, symbol, ImportSymbolTableFrame(other))
    table.add_namespace(own, own)

    report = table.verify()

    assert [str(diagnostic.message.message) for diagnostic in report.diagnostics] == [
        f"namespace {own!r} cannot be its own parent",
        f"namespace {own!r} has a cyclic parent chain",
        f"namespace {namespace!r} has symbol entry {symbol!r} "
        f"whose frame name is {other!r}",
    ]
    assert {diagnostic.level for diagnostic in report.diagnostics} == {
        DiagnosticLevel.ERROR
    }
    assert {diagnostic.source for diagnostic in report.diagnostics} == {
        "fhy_core.symbol_table.SymbolTable.verify"
    }


def test_the_debug_lines_are_kept(caplog: pytest.LogCaptureFixture) -> None:
    """Test the four DEBUG lines of the mutating methods."""
    namespace, symbol = _identifier("ns", 42), _identifier("x", 43)
    table = SymbolTable()

    with caplog.at_level(logging.DEBUG, logger="fhy_core.symbol_table"):
        table.add_namespace(namespace)
        table.add_symbol(namespace, symbol, ImportSymbolTableFrame(symbol))
        table.remove_symbol(namespace, symbol)
        table.remove_namespace(namespace)

    assert [record.getMessage() for record in caplog.records] == [
        "added namespace ns (parent=None)",
        "added symbol x to namespace ns (frame=ImportSymbolTableFrame)",
        "removed symbol x from namespace ns",
        "removed namespace ns",
    ]
    assert {record.name for record in caplog.records} == {"fhy_core.symbol_table"}


# ---------------------------------------------------------------------------
# A frame Python defines
# ---------------------------------------------------------------------------


def test_a_python_defined_frame_takes_part_in_a_table() -> None:
    """Test a third-party frame is added, found, compared and round-trips."""
    namespace, symbol = _identifier("ns", 44), _identifier("x", 45)
    frame = NoteFrame(symbol, "hello")
    table = SymbolTable()
    table.add_namespace(namespace)
    table.add_symbol(namespace, symbol, frame)
    other = SymbolTable()
    other.add_namespace(namespace)
    other.add_symbol(namespace, symbol, NoteFrame(symbol, "hello"))
    different = SymbolTable()
    different.add_namespace(namespace)
    different.add_symbol(namespace, symbol, NoteFrame(symbol, "bye"))

    assert table.get_frame(symbol) is frame
    assert table.verify().has_errors() is False
    assert table.is_structurally_equivalent(other)
    assert not table.is_structurally_equivalent(different)
    restored = SymbolTable.deserialize_from_dict(table.serialize_to_dict())
    assert restored.get_frame(symbol) == frame
    assert restored.is_structurally_equivalent(table)


def test_a_python_defined_frame_needs_an_identifier_name() -> None:
    """Test a frame whose `name` is no `Identifier` is refused."""
    table = _pair_table()

    with pytest.raises(
        TypeError, match=r"SymbolTableFrame name must be an Identifier, got str\."
    ):
        table.add_symbol(
            _identifier("ns", 0),
            _identifier("y", 46),
            NoteFrame("y", "n"),  # type: ignore[arg-type]
        )


def test_a_raising_frame_comparison_propagates() -> None:
    """Test an exception from a frame's own comparison reaches the caller."""
    symbol = _identifier("x", 1)
    left = _pair_table(RaisingFrame(symbol))
    right = _pair_table(RaisingFrame(symbol))

    with pytest.raises(RuntimeError, match="cannot compare"):
        left.is_structurally_equivalent(right)
