"""Tests the symbol table."""

import pytest
from fhy_core.serialization import DeserializationDictStructureError
from fhy_core.symbol_table import (
    FunctionKeyword,
    FunctionSymbolTableFrame,
    ImportSymbolTableFrame,
    SymbolTable,
    SymbolTableError,
    VariableSymbolTableFrame,
)
from fhy_core.trait import (
    Canonicalizable,
    StructuralEquivalence,
    Verifiable,
    VerificationError,
)
from fhy_core.types import CoreDataType, NumericalType, PrimitiveDataType, TypeQualifier

from .conftest import mock_identifier


@pytest.fixture
def empty_symbol_table() -> SymbolTable:
    return SymbolTable()


def test_add_and_check_namespace(empty_symbol_table: SymbolTable):
    """Test that a namespace can be added and checked."""
    namespace = mock_identifier("test_namespace", 0)

    empty_symbol_table.add_namespace(namespace)

    assert empty_symbol_table.is_namespace_defined(namespace)
    assert empty_symbol_table.get_number_of_namespaces() == 1


def test_add_duplicate_namespace_fails(empty_symbol_table: SymbolTable):
    """Test that adding a duplicate namespace raises a SymbolTableError."""
    namespace = mock_identifier("test_namespace", 0)

    empty_symbol_table.add_namespace(namespace)

    with pytest.raises(SymbolTableError):
        empty_symbol_table.add_namespace(namespace)


def test_get_undefined_namespace_fails(empty_symbol_table: SymbolTable):
    """Test that an undefined namespace raises a SymbolTableError."""
    undefined_namespace = mock_identifier("undefined_namespace", 0)

    with pytest.raises(SymbolTableError):
        empty_symbol_table.get_namespace(undefined_namespace)


def test_add_and_get_symbol(empty_symbol_table: SymbolTable):
    """Test that a symbol can be added and retrieved."""
    namespace = mock_identifier("test_namespace", 0)
    symbol_name = mock_identifier("test_symbol", 1)
    frame = ImportSymbolTableFrame(symbol_name)

    empty_symbol_table.add_namespace(namespace)
    empty_symbol_table.add_symbol(namespace, symbol_name, frame)

    assert empty_symbol_table.is_symbol_defined(symbol_name)
    assert empty_symbol_table.get_frame(symbol_name) == frame


def test_add_and_get_symbol_in_namespace(empty_symbol_table: SymbolTable):
    """Test that a symbol can be added and retrieved from a namespace."""
    namespace = mock_identifier("test_namespace", 0)
    symbol_name = mock_identifier("test_symbol", 1)
    frame = ImportSymbolTableFrame(symbol_name)

    empty_symbol_table.add_namespace(namespace)
    empty_symbol_table.add_symbol(namespace, symbol_name, frame)

    assert empty_symbol_table.is_symbol_defined_in_namespace(namespace, symbol_name)
    assert empty_symbol_table.get_frame_from_namespace(namespace, symbol_name) == frame


def test_add_duplicate_symbol_fails(empty_symbol_table: SymbolTable):
    """Test that adding a duplicate symbol raises a SymbolTableError."""
    namespace = mock_identifier("test_namespace", 0)
    symbol_name = mock_identifier("test_symbol", 1)
    frame = ImportSymbolTableFrame(symbol_name)

    empty_symbol_table.add_namespace(namespace)
    empty_symbol_table.add_symbol(namespace, symbol_name, frame)

    with pytest.raises(SymbolTableError):
        empty_symbol_table.add_symbol(namespace, symbol_name, frame)


def test_get_undefined_symbol_from_namespace_fails(empty_symbol_table: SymbolTable):
    """Test that getting an undefined symbol from a namespace raises a
    SymbolTableError.
    """
    namespace = mock_identifier("test_namespace", 0)
    symbol_name = mock_identifier("undefined_symbol", 1)

    empty_symbol_table.add_namespace(namespace)

    with pytest.raises(SymbolTableError):
        empty_symbol_table.get_frame_from_namespace(namespace, symbol_name)


def test_add_namespace_with_parent(empty_symbol_table: SymbolTable):
    """Test that a namespace can be added with a parent namespace."""
    parent_namespace = mock_identifier("parent_namespace", 0)
    child_namespace = mock_identifier("child_namespace", 1)

    empty_symbol_table.add_namespace(parent_namespace)
    empty_symbol_table.add_namespace(child_namespace, parent_namespace)

    assert empty_symbol_table.is_namespace_defined(child_namespace)


def test_get_symbol_from_namespace_inherited_from_parent(
    empty_symbol_table: SymbolTable,
):
    """Test that a child namespace can access a symbol from a parent namespace."""
    parent_namespace = mock_identifier("parent_namespace", 0)
    child_namespace = mock_identifier("child_namespace", 1)
    symbol_name = mock_identifier("test_symbol", 2)
    frame = ImportSymbolTableFrame(symbol_name)

    empty_symbol_table.add_namespace(parent_namespace)
    empty_symbol_table.add_namespace(child_namespace, parent_namespace)
    empty_symbol_table.add_symbol(parent_namespace, symbol_name, frame)

    assert empty_symbol_table.is_symbol_defined_in_namespace(
        child_namespace, symbol_name
    )
    assert (
        empty_symbol_table.get_frame_from_namespace(child_namespace, symbol_name)
        == frame
    )


def test_cyclic_namespace_fails(empty_symbol_table: SymbolTable):
    """Test that adding a cyclic namespace raises a RuntimeError."""
    namespace_a = mock_identifier("namespace_a", 0)
    namespace_b = mock_identifier("namespace_b", 1)

    empty_symbol_table.add_namespace(namespace_a, namespace_b)
    empty_symbol_table.add_namespace(namespace_b, namespace_a)

    with pytest.raises(RuntimeError):
        empty_symbol_table.is_symbol_defined_in_namespace(
            namespace_a, mock_identifier("some_symbol", 2)
        )


def test_update_namespaces():
    """Test that namespaces can be merged."""
    symbol_table_1 = SymbolTable()
    symbol_table_2 = SymbolTable()
    namespace = mock_identifier("shared_namespace", 0)
    symbol_name = mock_identifier("shared_symbol", 1)
    frame = ImportSymbolTableFrame(symbol_name)

    symbol_table_1.add_namespace(namespace)
    symbol_table_1.add_symbol(namespace, symbol_name, frame)

    symbol_table_2.update_namespaces(symbol_table_1)

    assert symbol_table_2.is_namespace_defined(namespace)
    assert symbol_table_2.is_symbol_defined_in_namespace(namespace, symbol_name)
    assert symbol_table_2.get_frame_from_namespace(namespace, symbol_name) == frame


def test_symbol_table_dict_serialization_with_import_frame():
    """Test `SymbolTable` can serialize/deserialize import frames."""
    symbol_table = SymbolTable()
    namespace = mock_identifier("namespace", 0)
    symbol_name = mock_identifier("imported", 1)
    frame = ImportSymbolTableFrame(symbol_name)
    symbol_table.add_namespace(namespace)
    symbol_table.add_symbol(namespace, symbol_name, frame)

    restored = SymbolTable.deserialize_from_dict(symbol_table.serialize_to_dict())

    restored_frame = restored.get_frame_from_namespace(namespace, symbol_name)
    assert isinstance(restored_frame, ImportSymbolTableFrame)


def test_symbol_table_dict_serialization_with_variable_frame():
    """Test `SymbolTable` can serialize/deserialize variable frames."""
    symbol_table = SymbolTable()
    namespace = mock_identifier("namespace", 0)
    symbol_name = mock_identifier("var", 1)
    frame = VariableSymbolTableFrame(
        symbol_name,
        NumericalType(PrimitiveDataType(CoreDataType.INT32)),
        TypeQualifier.STATE,
    )
    symbol_table.add_namespace(namespace)
    symbol_table.add_symbol(namespace, symbol_name, frame)

    restored = SymbolTable.deserialize_from_dict(symbol_table.serialize_to_dict())

    restored_frame = restored.get_frame_from_namespace(namespace, symbol_name)
    assert isinstance(restored_frame, VariableSymbolTableFrame)


def test_symbol_table_dict_serialization_with_function_frame():
    """Test `SymbolTable` can serialize/deserialize function frames."""
    symbol_table = SymbolTable()
    namespace = mock_identifier("namespace", 0)
    symbol_name = mock_identifier("fn", 1)
    frame = FunctionSymbolTableFrame(
        symbol_name,
        FunctionKeyword.PROCEDURE,
        signature=[
            (TypeQualifier.INPUT, NumericalType(PrimitiveDataType(CoreDataType.INT32))),
            (
                TypeQualifier.OUTPUT,
                NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            ),
        ],
    )
    symbol_table.add_namespace(namespace)
    symbol_table.add_symbol(namespace, symbol_name, frame)

    restored = SymbolTable.deserialize_from_dict(symbol_table.serialize_to_dict())

    restored_frame = restored.get_frame_from_namespace(namespace, symbol_name)
    assert isinstance(restored_frame, FunctionSymbolTableFrame)


def test_symbol_table_deserialization_structure_rejected():
    """Test invalid `SymbolTable` serialization structures are rejected."""
    with pytest.raises(DeserializationDictStructureError):
        SymbolTable.deserialize_from_dict({"bad": "data"})


def test_symbol_table_is_canonicalizable_runtime_protocol():
    """Test `SymbolTable` satisfies `Canonicalizable` runtime protocol."""
    symbol_table = SymbolTable()
    assert isinstance(symbol_table, Canonicalizable)


def test_symbol_table_is_verifiable_runtime_protocol():
    """Test `SymbolTable` satisfies `Verifiable` runtime protocol."""
    symbol_table = SymbolTable()
    assert isinstance(symbol_table, Verifiable)


def test_symbol_table_is_structural_equivalence_runtime_protocol():
    """Test `SymbolTable` satisfies `StructuralEquivalence` runtime protocol."""
    symbol_table = SymbolTable()
    assert isinstance(symbol_table, StructuralEquivalence)


def test_symbol_table_verify_returns_none_for_valid_table():
    """Test symbol table verification passes for valid tables."""
    symbol_table = SymbolTable()
    namespace = mock_identifier("namespace", 0)
    symbol_name = mock_identifier("symbol", 1)
    symbol_table.add_namespace(namespace)
    symbol_table.add_symbol(namespace, symbol_name, ImportSymbolTableFrame(symbol_name))

    assert symbol_table.verify() is None


def test_symbol_table_verify_raises_for_missing_parent_namespace():
    """Test verification raises when a parent namespace is missing."""
    symbol_table = SymbolTable()
    child_namespace = mock_identifier("child", 0)
    missing_parent = mock_identifier("missing_parent", 1)
    symbol_table.add_namespace(child_namespace, missing_parent)

    with pytest.raises(VerificationError):
        symbol_table.verify()


def test_symbol_table_verify_raises_for_cyclic_parent_chain():
    """Test verification raises when parent namespace graph has a cycle."""
    symbol_table = SymbolTable()
    namespace_a = mock_identifier("namespace_a", 0)
    namespace_b = mock_identifier("namespace_b", 1)
    symbol_table.add_namespace(namespace_a, namespace_b)
    symbol_table.add_namespace(namespace_b, namespace_a)

    with pytest.raises(VerificationError):
        symbol_table.verify()


def test_symbol_table_verify_raises_for_mismatched_frame_identifier():
    """Test verification raises when symbol key and frame identifier mismatch."""
    symbol_table = SymbolTable()
    namespace = mock_identifier("namespace", 0)
    symbol_name = mock_identifier("symbol", 1)
    frame_name = mock_identifier("other_symbol", 2)
    symbol_table.add_namespace(namespace)
    symbol_table.add_symbol(namespace, symbol_name, ImportSymbolTableFrame(frame_name))

    with pytest.raises(VerificationError):
        symbol_table.verify()


def test_symbol_table_canonicalize_reports_change_when_order_unsorted():
    """Test canonicalization reports change when namespace order is unsorted."""
    symbol_table = SymbolTable()
    namespace_high = mock_identifier("high", 2)
    namespace_low = mock_identifier("low", 1)
    symbol_table.add_namespace(namespace_high)
    symbol_table.add_namespace(namespace_low)

    assert symbol_table.canonicalize()


def test_symbol_table_canonicalize_reports_no_change_when_already_sorted():
    """Test canonicalization reports no change when table is already sorted."""
    symbol_table = SymbolTable()
    namespace_low = mock_identifier("low", 1)
    namespace_high = mock_identifier("high", 2)
    symbol_table.add_namespace(namespace_low)
    symbol_table.add_namespace(namespace_high)

    assert not symbol_table.canonicalize()


def test_symbol_table_structural_equivalence_true_for_same_content():
    """Test structural equivalence is true for symbol tables with same content."""
    namespace = mock_identifier("namespace", 0)
    symbol_name = mock_identifier("symbol", 1)
    frame = ImportSymbolTableFrame(symbol_name)

    left = SymbolTable()
    left.add_namespace(namespace)
    left.add_symbol(namespace, symbol_name, frame)

    right = SymbolTable()
    right.add_namespace(namespace)
    right.add_symbol(namespace, symbol_name, frame)

    assert left.is_structurally_equivalent(right)


def test_symbol_table_structural_equivalence_false_for_different_parent_graph():
    """Test structural equivalence is false for different namespace parent graphs."""
    root = mock_identifier("root", 0)
    child = mock_identifier("child", 1)
    symbol_name = mock_identifier("symbol", 2)
    frame = ImportSymbolTableFrame(symbol_name)

    left = SymbolTable()
    left.add_namespace(root)
    left.add_namespace(child, root)
    left.add_symbol(child, symbol_name, frame)

    right = SymbolTable()
    right.add_namespace(root)
    right.add_namespace(child)
    right.add_symbol(child, symbol_name, frame)

    assert not left.is_structurally_equivalent(right)


def test_symbol_table_structural_equivalence_false_for_other_python_type():
    """Test structural equivalence is false for non-symbol-table objects."""
    symbol_table = SymbolTable()
    assert not symbol_table.is_structurally_equivalent({})
