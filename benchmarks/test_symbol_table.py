"""Benchmarks of the symbol table and its frames.

They measure ``fhy_core.symbol_table`` before and after it switches to the
Rust core (S15 of ``docs/design/python-switch.md``), through the public API
only. The variable frame's construction and hash are measured in
``test_types.py``, and the table's structural equivalence in
``test_term.py``.
"""

import itertools
import pickle
from collections.abc import Sequence

import pytest

from fhy_core.identifier import Identifier
from fhy_core.serialization import SerializedDict
from fhy_core.symbol_table import (
    FunctionKeyword,
    FunctionSymbolTableFrame,
    ImportSymbolTableFrame,
    SymbolTable,
    SymbolTableFrame,
    VariableSymbolTableFrame,
)
from fhy_core.types import CoreDataType, NumericalType, PrimitiveDataType, TypeQualifier

from .conftest import Benchmark

pytestmark = pytest.mark.benchmark(group="symbol_table")

# How many symbols a benchmarked namespace holds.
_SYMBOL_COUNT = 20
# How many namespaces the wide table and the parent chain hold.
_NAMESPACE_COUNT = 10


def _int32() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32))


def _build_variable_frames(count: int, prefix: str) -> list[VariableSymbolTableFrame]:
    """Return `count` variable frames, each named by a new identifier."""
    int32 = _int32()
    return [
        VariableSymbolTableFrame(
            Identifier(f"{prefix}{index}"), int32, TypeQualifier.STATE
        )
        for index in range(count)
    ]


def _build_table(
    namespace: Identifier, frames: Sequence[SymbolTableFrame]
) -> SymbolTable:
    """Return a table of one namespace holding each frame under its name."""
    table = SymbolTable()
    table.add_namespace(namespace)
    for frame in frames:
        table.add_symbol(namespace, frame.name, frame)
    return table


def _build_wide_table(*, reverse: bool = False) -> tuple[SymbolTable, list[Identifier]]:
    """Return a table of 10 root namespaces of 20 variables each, and its symbols.

    With ``reverse``, the namespaces and each namespace's symbols are added
    in the reverse of their identifiers' order.
    """
    namespaces = [Identifier(f"ns{index}") for index in range(_NAMESPACE_COUNT)]
    contents = [
        _build_variable_frames(_SYMBOL_COUNT, f"ns{index}_s")
        for index in range(_NAMESPACE_COUNT)
    ]
    order = range(_NAMESPACE_COUNT - 1, -1, -1) if reverse else range(_NAMESPACE_COUNT)
    table = SymbolTable()
    for index in order:
        table.add_namespace(namespaces[index])
        frames = reversed(contents[index]) if reverse else contents[index]
        for frame in frames:
            table.add_symbol(namespaces[index], frame.name, frame)
    symbols = [frame.name for frames in contents for frame in frames]
    return table, symbols


def _build_chain() -> tuple[SymbolTable, Identifier, Identifier]:
    """Return a chain of 10 namespaces, its innermost one, and the root's symbol."""
    namespaces = [Identifier(f"chain{index}") for index in range(_NAMESPACE_COUNT)]
    table = SymbolTable()
    table.add_namespace(namespaces[0])
    for parent, child in itertools.pairwise(namespaces):
        table.add_namespace(child, parent)
    symbol = Identifier("root_symbol")
    table.add_symbol(namespaces[0], symbol, ImportSymbolTableFrame(symbol))
    return table, namespaces[-1], symbol


def _function_frame() -> FunctionSymbolTableFrame:
    int32 = _int32()
    return FunctionSymbolTableFrame(
        Identifier("f"),
        FunctionKeyword.PROCEDURE,
        ((TypeQualifier.INPUT, int32), (TypeQualifier.OUTPUT, int32)),
    )


# ---------------------------------------------------------------------------
# The frames
# ---------------------------------------------------------------------------


def test_import_frame_construction(benchmark: Benchmark) -> None:
    """Construct an import frame."""
    benchmark(ImportSymbolTableFrame, Identifier("m"))


def test_function_frame_construction(benchmark: Benchmark) -> None:
    """Construct a function frame of two parameters."""
    name, int32 = Identifier("f"), _int32()
    signature = ((TypeQualifier.INPUT, int32), (TypeQualifier.OUTPUT, int32))
    benchmark(FunctionSymbolTableFrame, name, FunctionKeyword.PROCEDURE, signature)


def test_frame_name_access(benchmark: Benchmark) -> None:
    """Read a variable frame's name."""
    (frame,) = _build_variable_frames(1, "x")
    benchmark(getattr, frame, "name")


def test_frame_eq(benchmark: Benchmark) -> None:
    """Compare two equal variable frames built apart."""
    name, int32 = Identifier("x"), _int32()
    left = VariableSymbolTableFrame(name, int32, TypeQualifier.STATE)
    right = VariableSymbolTableFrame(name, _int32(), TypeQualifier.STATE)
    benchmark(left.__eq__, right)


def test_frame_structural_equivalence(benchmark: Benchmark) -> None:
    """Compare two equal variable frames built apart, structurally."""
    name = Identifier("x")
    left = VariableSymbolTableFrame(name, _int32(), TypeQualifier.STATE)
    right = VariableSymbolTableFrame(name, _int32(), TypeQualifier.STATE)
    benchmark(left.is_structurally_equivalent, right)


def test_frame_serialize_to_dict(benchmark: Benchmark) -> None:
    """Serialize a function frame of two parameters."""
    benchmark(_function_frame().serialize_to_dict)


def test_frame_deserialize_from_dict(benchmark: Benchmark) -> None:
    """Deserialize a function frame of two parameters through the family."""
    payload = _function_frame().serialize_to_dict()
    benchmark(SymbolTableFrame.deserialize_from_dict, payload)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_symbol_table_construction(benchmark: Benchmark) -> None:
    """Construct an empty table."""
    benchmark(SymbolTable)


def test_symbol_table_build_of_20_symbols(benchmark: Benchmark) -> None:
    """Build a table of one namespace holding 20 prebuilt variable frames."""
    frames = _build_variable_frames(_SYMBOL_COUNT, "s")
    benchmark(_build_table, Identifier("ns"), frames)


def test_symbol_table_add_and_remove_symbol(benchmark: Benchmark) -> None:
    """Add a symbol to a 20-symbol namespace, then remove it."""
    namespace = Identifier("ns")
    table = _build_table(namespace, _build_variable_frames(_SYMBOL_COUNT, "s"))
    (frame,) = _build_variable_frames(1, "extra")

    def add_and_remove() -> None:
        table.add_symbol(namespace, frame.name, frame)
        table.remove_symbol(namespace, frame.name)

    benchmark(add_and_remove)


def test_is_symbol_defined_in_namespace_through_a_chain_of_10(
    benchmark: Benchmark,
) -> None:
    """Find the root's symbol from the innermost of 10 nested namespaces."""
    table, innermost, symbol = _build_chain()
    benchmark(table.is_symbol_defined_in_namespace, innermost, symbol)


def test_get_frame_from_namespace_through_a_chain_of_10(benchmark: Benchmark) -> None:
    """Get the root's frame from the innermost of 10 nested namespaces."""
    table, innermost, symbol = _build_chain()
    benchmark(table.get_frame_from_namespace, innermost, symbol)


def test_get_frame_of_the_last_symbol(benchmark: Benchmark) -> None:
    """Search 10 namespaces of 20 symbols for the last symbol's frame."""
    table, symbols = _build_wide_table()
    benchmark(table.get_frame, symbols[-1])


def test_is_symbol_defined(benchmark: Benchmark) -> None:
    """Search 10 namespaces of 20 symbols for the last symbol."""
    table, symbols = _build_wide_table()
    benchmark(table.is_symbol_defined, symbols[-1])


def test_get_namespace_of_20_symbols(benchmark: Benchmark) -> None:
    """Get the symbols of a 20-symbol namespace."""
    namespace = Identifier("ns")
    table = _build_table(namespace, _build_variable_frames(_SYMBOL_COUNT, "s"))
    benchmark(table.get_namespace, namespace)


def test_symbol_table_verify(benchmark: Benchmark) -> None:
    """Verify a well-formed table of 10 namespaces of 20 symbols."""
    table, _ = _build_wide_table()
    benchmark(table.verify)


def test_symbol_table_canonicalize(benchmark: Benchmark) -> None:
    """Canonicalize a table of 10 namespaces of 20 symbols built in reverse."""
    table, _ = _build_wide_table(reverse=True)
    benchmark(table.canonicalize)


def test_symbol_table_serialize_to_dict(benchmark: Benchmark) -> None:
    """Serialize a table of one namespace of 20 variables."""
    table = _build_table(Identifier("ns"), _build_variable_frames(_SYMBOL_COUNT, "s"))
    benchmark(table.serialize_to_dict)


def test_symbol_table_deserialize_from_dict(benchmark: Benchmark) -> None:
    """Deserialize a table of one namespace of 20 variables."""
    table = _build_table(Identifier("ns"), _build_variable_frames(_SYMBOL_COUNT, "s"))
    payload: SerializedDict = table.serialize_to_dict()
    benchmark(SymbolTable.deserialize_from_dict, payload)


def test_symbol_table_pickle_round_trip(benchmark: Benchmark) -> None:
    """Pickle and unpickle a table of one namespace of 20 variables."""
    table = _build_table(Identifier("ns"), _build_variable_frames(_SYMBOL_COUNT, "s"))
    benchmark(lambda: pickle.loads(pickle.dumps(table)))


def test_symbol_table_update_namespaces(benchmark: Benchmark) -> None:
    """Merge a table of 10 namespaces of 20 symbols into another table."""
    source, _ = _build_wide_table()
    destination = SymbolTable()
    benchmark(destination.update_namespaces, source)
