"""Tests of the reusable serialization test contract.

The contract helpers collapse the per-class round-trip / structure-rejection
battery -- previously copy-pasted for every ``Serializable`` -- into a single
parametrized form: a migrated class contributes only representative instances
and a few malformed dicts, not four near-identical test functions.

Classes whose equality is by-value (provenance, notes, interned domains/
attributes, dataclass frames) round-trip here. Expressions are ``eq=False``
(identity equality), so their round-trips stay in the expression test suite
where structural equivalence is asserted.
"""

from pathlib import Path

import pytest

from fhy_core.diagnostic import Note, NoteKind
from fhy_core.identifier import Identifier
from fhy_core.op_attribute import OpAttribute
from fhy_core.provenance import (
    CallSiteProvenance,
    FileProvenance,
    FusedProvenance,
    NamedProvenance,
    Position,
    Span,
    UnknownProvenance,
)
from fhy_core.serialization import (
    DeserializationDictStructureError,
    Serializable,
    SerializedDict,
)
from fhy_core.symbol_table import ImportSymbolTableFrame
from fhy_core.value_domain import ValueDomain

from .conftest import (
    assert_rejects_malformed_dict,
    assert_round_trips_in_all_formats,
)

_SOURCE = FileProvenance(Path("a.fhy"), Span(start_offset=0, end_offset=3))
_OTHER = FileProvenance(Path("b.fhy"))


@pytest.mark.parametrize(
    "instance",
    [
        pytest.param(Position(2, 8), id="position"),
        pytest.param(Span(0, 3, Position(1, 1), Position(1, 4)), id="span_full"),
        pytest.param(Span(), id="span_unknown"),
        pytest.param(UnknownProvenance(), id="provenance_unknown"),
        pytest.param(_SOURCE, id="provenance_file"),
        pytest.param(
            NamedProvenance("lib", UnknownProvenance()), id="provenance_named"
        ),
        pytest.param(
            CallSiteProvenance(callee=_SOURCE, caller=_OTHER), id="provenance_call_site"
        ),
        pytest.param(
            FusedProvenance(sources=(_SOURCE, _OTHER), metadata="fuse"),
            id="provenance_fused",
        ),
        pytest.param(Note("hello", NoteKind.OTHER), id="note"),
        pytest.param(
            OpAttribute(Identifier("contract_attr"), "desc"), id="op_attribute"
        ),
        pytest.param(ValueDomain(Identifier("contract_domain"), "desc"), id="domain"),
        pytest.param(
            ImportSymbolTableFrame(Identifier("contract_import")), id="import_frame"
        ),
    ],
)
def test_serializable_round_trips_in_all_formats(instance: Serializable) -> None:
    """Test representative serializables round-trip through every format."""
    assert_round_trips_in_all_formats(instance)


@pytest.mark.parametrize(
    "cls,data",
    [
        pytest.param(Position, {"line": 1}, id="position_missing_key"),
        pytest.param(Position, {"line": True, "column": 1}, id="position_bool"),
        pytest.param(Position, {"line": 1, "column": 2, "z": 3}, id="position_extra"),
        pytest.param(Span, {"start_offset": 0}, id="span_missing_keys"),
        pytest.param(Note, {"message": "x"}, id="note_missing_kind"),
    ],
)
def test_serializable_rejects_malformed_dict(
    cls: type[Serializable], data: SerializedDict
) -> None:
    """Test malformed dicts are rejected with a structure error."""
    assert_rejects_malformed_dict(
        cls, data, expected_error=DeserializationDictStructureError
    )
