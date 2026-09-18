"""Hypothesis property tests for `SymbolTable`.

`SymbolTable` is not stack-shaped: it is a set of namespaces linked by an
acyclic parent chain, each holding a flat map of symbols to frames. There is
no single push/pop protocol to model as a state machine, so this file
instead states the laws its public API documents for a well-formed table:
`canonicalize` is idempotent, a dict round trip is structurally equivalent
to the original, and `verify()` reports no errors on a table built to
satisfy every invariant it checks (acyclic parent chain, every referenced
parent present, and `frame.name == symbol_name` for every symbol).
"""

from typing import Final

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbol_table import ImportSymbolTableFrame, SymbolTable

from .conftest import mock_identifier

pytestmark = pytest.mark.property

# No native constant identifier reaches this id, so mock_identifier never
# raises MockIdentifierAliasError for identifiers this file builds.
_MOCK_ID_BASE: Final = 20_000


@st.composite
def draw_symbol_table(draw: st.DrawFn) -> SymbolTable:
    """Draw a well-formed SymbolTable: a parent-linked tree of namespaces.

    Each namespace's parent (when present) is drawn only from namespaces
    already added, so the parent chain is acyclic and every parent
    reference resolves, by construction. Every symbol's frame is built with
    `frame.name` equal to the symbol's own name, satisfying the
    `frame.name == symbol_name` invariant `verify()` checks. Identifiers are
    mock identifiers with strictly increasing ids, so no two are
    accidentally equal.
    """
    next_id = [_MOCK_ID_BASE]

    def draw_identifier(prefix: str) -> Identifier:
        identifier = mock_identifier(prefix, next_id[0])
        next_id[0] += 1
        return identifier

    symbol_table = SymbolTable()
    namespace_count = draw(st.integers(min_value=1, max_value=4))
    namespace_names: list[Identifier] = []
    for namespace_index in range(namespace_count):
        namespace_name = draw_identifier(f"ns{namespace_index}")
        should_pick_parent = bool(namespace_names) and draw(st.booleans())
        parent_name: Identifier | None = None
        if should_pick_parent:
            parent_name = draw(st.sampled_from(namespace_names))
        symbol_table.add_namespace(namespace_name, parent_name)
        namespace_names.append(namespace_name)

        symbol_count = draw(st.integers(min_value=0, max_value=3))
        for symbol_index in range(symbol_count):
            symbol_name = draw_identifier(f"ns{namespace_index}_sym{symbol_index}")
            symbol_table.add_symbol(
                namespace_name, symbol_name, ImportSymbolTableFrame(symbol_name)
            )

    return symbol_table


@given(symbol_table=draw_symbol_table())
def test_canonicalize_is_idempotent(symbol_table: SymbolTable) -> None:
    """Test a second canonicalize() leaves the serialized layout unchanged.

    Oracle: canonicalize's own documented contract (sort namespaces and
    symbols by id/name-hint key); a table already in that order has nothing
    left to reorder, so a second pass must be a no-op.
    """
    symbol_table.canonicalize()
    first = symbol_table.serialize_to_dict()
    symbol_table.canonicalize()
    second = symbol_table.serialize_to_dict()

    assert first == second


@given(symbol_table=draw_symbol_table())
def test_dict_round_trip_is_structurally_equivalent(symbol_table: SymbolTable) -> None:
    """Test a serialize/deserialize round trip reconstructs a structurally equal table.

    Oracle: SymbolTable.is_structurally_equivalent, the type's own documented
    equivalence contract.
    """
    round_tripped = SymbolTable.deserialize_from_dict(symbol_table.serialize_to_dict())

    assert symbol_table.is_structurally_equivalent(round_tripped)


@given(symbol_table=draw_symbol_table())
def test_verify_passes_on_well_formed_tables(symbol_table: SymbolTable) -> None:
    """Test verify() reports no errors for a table built to satisfy every invariant.

    Oracle: the invariants verify() checks, restated as the strategy's own
    construction rules (acyclic parent chain, every parent present,
    frame.name == symbol_name), so a table this strategy draws can never
    violate them.
    """
    report = symbol_table.verify()

    assert not report.has_errors()
