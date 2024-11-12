"""Tests the symbol table."""

import pytest
from fhy_core.error import SymbolTableError
from fhy_core.identifier import Identifier
from fhy_core.symbol_table import (
    ImportSymbolTableFrame,
    SymbolTable,
)


@pytest.fixture
def empty_symbol_table() -> SymbolTable:
    return SymbolTable()


def test_add_and_check_namespace(empty_symbol_table: SymbolTable):
    """Test that a namespace can be added and checked."""
    namespace = Identifier("test_namespace")

    empty_symbol_table.add_namespace(namespace)

    assert empty_symbol_table.is_namespace_defined(namespace)
    assert empty_symbol_table.get_number_of_namespaces() == 1


def test_add_duplicate_namespace_fails(empty_symbol_table: SymbolTable):
    """Test that adding a duplicate namespace raises a SymbolTableError."""
    namespace = Identifier("test_namespace")

    empty_symbol_table.add_namespace(namespace)

    with pytest.raises(SymbolTableError):
        empty_symbol_table.add_namespace(namespace)


def test_get_undefined_namespace_fails(empty_symbol_table: SymbolTable):
    """Test that an undefined namespace raises a SymbolTableError."""
    undefined_namespace = Identifier("undefined_namespace")

    with pytest.raises(SymbolTableError):
        empty_symbol_table.get_namespace(undefined_namespace)


def test_add_and_get_symbol(empty_symbol_table: SymbolTable):
    """Test that a symbol can be added and retrieved."""
    namespace = Identifier("test_namespace")
    symbol_name = Identifier("test_symbol")
    frame = ImportSymbolTableFrame(symbol_name)

    empty_symbol_table.add_namespace(namespace)
    empty_symbol_table.add_symbol(namespace, symbol_name, frame)

    assert empty_symbol_table.is_symbol_defined(symbol_name)
    assert empty_symbol_table.get_frame(symbol_name) == frame


def test_add_and_get_symbol_in_namespace(empty_symbol_table: SymbolTable):
    """Test that a symbol can be added and retrieved from a namespace."""
    namespace = Identifier("test_namespace")
    symbol_name = Identifier("test_symbol")
    frame = ImportSymbolTableFrame(symbol_name)

    empty_symbol_table.add_namespace(namespace)
    empty_symbol_table.add_symbol(namespace, symbol_name, frame)

    assert empty_symbol_table.is_symbol_defined_in_namespace(namespace, symbol_name)
    assert empty_symbol_table.get_frame_from_namespace(namespace, symbol_name) == frame


def test_add_duplicate_symbol_fails(empty_symbol_table: SymbolTable):
    """Test that adding a duplicate symbol raises a SymbolTableError."""
    namespace = Identifier("test_namespace")
    symbol_name = Identifier("test_symbol")
    frame = ImportSymbolTableFrame(symbol_name)

    empty_symbol_table.add_namespace(namespace)
    empty_symbol_table.add_symbol(namespace, symbol_name, frame)

    with pytest.raises(SymbolTableError):
        empty_symbol_table.add_symbol(namespace, symbol_name, frame)


def test_get_undefined_symbol_from_namespace_fails(empty_symbol_table: SymbolTable):
    """Test that getting an undefined symbol from a namespace raises a
    SymbolTableError.
    """
    namespace = Identifier("test_namespace")
    symbol_name = Identifier("undefined_symbol")

    empty_symbol_table.add_namespace(namespace)

    with pytest.raises(SymbolTableError):
        empty_symbol_table.get_frame_from_namespace(namespace, symbol_name)


def test_add_namespace_with_parent(empty_symbol_table: SymbolTable):
    """Test that a namespace can be added with a parent namespace."""
    parent_namespace = Identifier("parent_namespace")
    child_namespace = Identifier("child_namespace")

    empty_symbol_table.add_namespace(parent_namespace)
    empty_symbol_table.add_namespace(child_namespace, parent_namespace)

    assert empty_symbol_table.is_namespace_defined(child_namespace)


def test_get_symbol_from_namespace_inherited_from_parent(
    empty_symbol_table: SymbolTable,
):
    """Test that a child namespace can access a symbol from a parent namespace."""
    parent_namespace = Identifier("parent_namespace")
    child_namespace = Identifier("child_namespace")
    symbol_name = Identifier("test_symbol")
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
    namespace_a = Identifier("namespace_a")
    namespace_b = Identifier("namespace_b")

    empty_symbol_table.add_namespace(namespace_a, namespace_b)
    empty_symbol_table.add_namespace(namespace_b, namespace_a)

    with pytest.raises(RuntimeError):
        empty_symbol_table.is_symbol_defined_in_namespace(
            namespace_a, Identifier("some_symbol")
        )


def test_update_namespaces():
    """Test that namespaces can be merged."""
    symbol_table_1 = SymbolTable()
    symbol_table_2 = SymbolTable()
    namespace = Identifier("shared_namespace")
    symbol_name = Identifier("shared_symbol")
    frame = ImportSymbolTableFrame(symbol_name)

    symbol_table_1.add_namespace(namespace)
    symbol_table_1.add_symbol(namespace, symbol_name, frame)

    symbol_table_2.update_namespaces(symbol_table_1)

    assert symbol_table_2.is_namespace_defined(namespace)
    assert symbol_table_2.is_symbol_defined_in_namespace(namespace, symbol_name)
    assert symbol_table_2.get_frame_from_namespace(namespace, symbol_name) == frame
