"""Core symbol table.

Backed by the Rust implementation: ``SymbolTable`` and the three built-in
frames are thin subclasses of their ``fhy_core._rs`` classes. The table
keeps its namespaces, their parents and their symbols in the Rust core,
which walks the parent chains, refuses shadowing, merges, canonicalizes,
compares and verifies; the frames hold their Rust values and the objects
each was built from. ``FunctionKeyword`` stays a Python enum, converted by
value at the boundary.

``SymbolTableFrame`` stays the abstract, frozen dataclass base third parties
subclass. The built-in frames are its virtual subclasses. A table holds a
frame Python defines as it is given: it reads the frame's ``name`` once,
when the frame is added, and asks the frame's own methods to compare and
serialize it.
"""

__all__ = [
    "FunctionSymbolTableFrame",
    "ImportSymbolTableFrame",
    "SymbolTable",
    "SymbolTableError",
    "SymbolTableFrame",
    "VariableSymbolTableFrame",
]

from abc import ABC
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.serialization import (
    Serializable,
    WrappedFamilySerializable,
    register_serializable,
)
from fhy_core.term import DerivedEquivalenceMixin
from fhy_core.traits import (
    Canonicalizable,
    FrozenMixin,
    StructuralEquivalence,
    VerifiableMixin,
)

from .error import register_error

_LOGGER = get_logger(__name__)
"""The logger the Rust binding writes this module's DEBUG lines to."""


@dataclass(frozen=True)
class SymbolTableFrame(
    WrappedFamilySerializable, FrozenMixin, DerivedEquivalenceMixin, ABC
):
    """Base symbol table frame.

    The serialization family of the frames. The built-in frames run on the
    Rust core and are registered as virtual subclasses; a frame a third
    party defines subclasses this dataclass, and inherits its ``name``
    field and derived equivalence.
    """

    _WIRE_FAMILY: ClassVar[str | None] = "symbol_frame"

    name: Identifier


class FunctionKeyword(StrEnum):
    """Function keyword."""

    PROCEDURE = "proc"
    OPERATION = "op"
    NATIVE = "native"


@register_serializable(type_id="import_symbol_table_frame")
class ImportSymbolTableFrame(_rs.ImportSymbolTableFrame, WrappedFamilySerializable):
    """Imported symbol frame.

    Attributes:
        name: The symbol's ``Identifier``, as given.

    """

    _WIRE_FAMILY: ClassVar[str | None] = "symbol_frame"

    __slots__ = ()


@register_serializable(type_id="variable_symbol_table_frame")
class VariableSymbolTableFrame(_rs.VariableSymbolTableFrame, WrappedFamilySerializable):
    """Variable symbol frame.

    Attributes:
        name: The variable's ``Identifier``, as given.
        type: Its ``Type``, as given.
        type_qualifier: Its ``TypeQualifier``.

    """

    _WIRE_FAMILY: ClassVar[str | None] = "symbol_frame"

    __slots__ = ()


@register_serializable(type_id="function_symbol_table_frame")
class FunctionSymbolTableFrame(_rs.FunctionSymbolTableFrame, WrappedFamilySerializable):
    """Functions symbol frame.

    Attributes:
        name: The function's ``Identifier``, as given.
        keyword: Its ``FunctionKeyword``.
        signature: A tuple of each parameter's ``(TypeQualifier, Type)``
            pair, from whatever iterable was given; empty by default.

    """

    _WIRE_FAMILY: ClassVar[str | None] = "symbol_frame"

    __slots__ = ()


# The built-in frames are registered, not derived: `SymbolTableFrame`'s bases
# carry an instance layout a Rust-backed class cannot share.
for _frame_class in (
    ImportSymbolTableFrame,
    VariableSymbolTableFrame,
    FunctionSymbolTableFrame,
):
    SymbolTableFrame.register(_frame_class)
    FrozenMixin.register(_frame_class)
del _frame_class


@register_error
class SymbolTableError(Exception):
    """Symbol table error."""


@register_serializable(type_id="symbol_table")
class SymbolTable(
    _rs.SymbolTable, Serializable, Canonicalizable, StructuralEquivalence
):
    """Core nested symbol table comprised of various frames.

    Namespaces keep the order they were added in, and so do each
    namespace's symbols. A namespace may name a parent namespace, which is
    not checked when it is added: a lookup in a namespace
    (``is_symbol_defined_in_namespace``, ``get_frame_from_namespace``) walks
    up its parents, and ``add_symbol`` refuses a symbol the namespace or one
    of its ancestors already defines. ``get_frame`` and
    ``is_symbol_defined`` search every namespace in order, ignoring parents.

    Methods:
        add_namespace(namespace_name, parent_namespace_name=None),
            remove_namespace(namespace_name), is_namespace_defined,
            get_namespace (a new dict of the symbols to their frames),
            get_number_of_namespaces: The namespaces.
        add_symbol(namespace_name, symbol_name, frame),
            remove_symbol(namespace_name, symbol_name), is_symbol_defined,
            is_symbol_defined_in_namespace, get_frame,
            get_frame_from_namespace: The symbols. A lookup raises
            ``SymbolTableError`` for an undefined namespace, and when its walk
            meets a cycle or a parent that is not defined.
        update_namespaces(other_symbol_table): Copy the other table's
            namespaces in, replacing a defined namespace's symbols in place;
            a namespace takes the other's parent when the other names one.
        canonicalize(): Reorder the namespaces and symbols by identifier, in
            place.
        verify(): A :class:`~fhy_core.diagnostic.ValidationReport` with one
            ERROR diagnostic per missing or self parent, cyclic parent chain,
            and frame whose name is not its symbol.
        is_structurally_equivalent(other): The same namespaces, parents and
            symbols, with structurally equivalent frames; order does not
            count.

    It is a :class:`~fhy_core.traits.verifiable.VerifiableMixin` by
    registration, and defines :meth:`verify` itself. ``==`` and ``hash`` are
    identity. A pickle or a copy is an independent table with the same
    namespaces, symbols and frames.
    """


VerifiableMixin.register(SymbolTable)
