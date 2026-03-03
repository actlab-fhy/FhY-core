"""Core symbol table."""

__all__ = [
    "FunctionSymbolTableFrame",
    "ImportSymbolTableFrame",
    "SymbolTable",
    "SymbolTableError",
    "SymbolTableFrame",
    "VariableSymbolTableFrame",
]


from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, NoReturn, TypedDict, TypeGuard, TypeVar

from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    Serializable,
    SerializedDict,
    WrappedFamilySerializable,
    is_serialized_dict,
    register_serializable,
)

from .error import register_error
from .types import Type, TypeQualifier
from .utils import StrEnum


@dataclass(frozen=True)
class SymbolTableFrame(WrappedFamilySerializable, ABC):
    """Base symbol table frame."""

    name: Identifier


@register_serializable(type_id="import_symbol_table_frame")
class ImportSymbolTableFrame(SymbolTableFrame):
    """Imported symbol frame."""

    def serialize_data_to_dict(self) -> SerializedDict:
        return {"name": self.name.serialize_to_dict()}

    @classmethod
    def deserialize_data_from_dict(
        cls, data: SerializedDict
    ) -> "ImportSymbolTableFrame":
        if not _is_valid_import_symbol_table_frame_data(data):
            raise DeserializationDictStructureError(
                cls, _ImportSymbolTableFrameData.__annotations__, data
            )
        return cls(Identifier.deserialize_from_dict(data["name"]))


@dataclass(frozen=True)
@register_serializable(type_id="variable_symbol_table_frame")
class VariableSymbolTableFrame(SymbolTableFrame):
    """Variable symbol frame."""

    type: Type
    type_qualifier: TypeQualifier

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "name": self.name.serialize_to_dict(),
            "type": self.type.serialize_to_dict(),
            "type_qualifier": self.type_qualifier.value,
        }

    @classmethod
    def deserialize_data_from_dict(
        cls, data: SerializedDict
    ) -> "VariableSymbolTableFrame":
        if not _is_valid_variable_symbol_table_frame_data(data):
            raise DeserializationDictStructureError(
                cls, _VariableSymbolTableFrameData.__annotations__, data
            )
        try:
            return cls(
                Identifier.deserialize_from_dict(data["name"]),
                Type.deserialize_from_dict(data["type"]),
                type_qualifier=TypeQualifier(data["type_qualifier"]),
            )
        except ValueError as exc:
            raise DeserializationValueError(
                f"Invalid variable frame values: {exc}"
            ) from exc


class FunctionKeyword(StrEnum):
    """Function keyword."""

    PROCEDURE = "proc"
    OPERATION = "op"
    NATIVE = "native"


@dataclass(frozen=True)
@register_serializable(type_id="function_symbol_table_frame")
class FunctionSymbolTableFrame(SymbolTableFrame):
    """Functions symbol frame."""

    keyword: FunctionKeyword
    signature: list[tuple[TypeQualifier, Type]] = field(default_factory=list)

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "name": self.name.serialize_to_dict(),
            "keyword": self.keyword.value,
            "signature": [
                {
                    "type_qualifier": type_qualifier.value,
                    "type": ty.serialize_to_dict(),
                }
                for type_qualifier, ty in self.signature
            ],
        }

    @classmethod
    def deserialize_data_from_dict(
        cls, data: SerializedDict
    ) -> "FunctionSymbolTableFrame":
        if not _is_valid_function_symbol_table_frame_data(data):
            raise DeserializationDictStructureError(
                cls, _FunctionSymbolTableFrameData.__annotations__, data
            )

        try:
            signature = [
                (
                    TypeQualifier(signature_entry["type_qualifier"]),
                    Type.deserialize_from_dict(signature_entry["type"]),
                )
                for signature_entry in data["signature"]
            ]
            return cls(
                Identifier.deserialize_from_dict(data["name"]),
                FunctionKeyword(data["keyword"]),
                signature=signature,
            )
        except ValueError as exc:
            raise DeserializationValueError(
                f"Invalid function frame values: {exc}"
            ) from exc


@register_error
class SymbolTableError(Exception):
    """Symbol table error."""


_T = TypeVar("_T")
_SUCC_T = TypeVar("_SUCC_T")
_FAIL_T = TypeVar("_FAIL_T")


@dataclass(frozen=True)
class _SymbolTableSearchResult(Generic[_T]):
    value: _T


class _ImportSymbolTableFrameData(TypedDict):
    name: SerializedDict


def _is_valid_import_symbol_table_frame_data(
    data: SerializedDict,
) -> TypeGuard[_ImportSymbolTableFrameData]:
    return "name" in data and is_serialized_dict(data["name"])


class _VariableSymbolTableFrameData(TypedDict):
    name: SerializedDict
    type: SerializedDict
    type_qualifier: str


def _is_valid_variable_symbol_table_frame_data(
    data: SerializedDict,
) -> TypeGuard[_VariableSymbolTableFrameData]:
    return (
        "name" in data
        and is_serialized_dict(data["name"])
        and "type" in data
        and is_serialized_dict(data["type"])
        and "type_qualifier" in data
        and isinstance(data["type_qualifier"], str)
    )


class _FunctionSignatureEntryData(TypedDict):
    type_qualifier: str
    type: SerializedDict


def _is_valid_function_signature_entry_data(
    data: SerializedDict,
) -> TypeGuard[_FunctionSignatureEntryData]:
    return (
        "type_qualifier" in data
        and isinstance(data["type_qualifier"], str)
        and "type" in data
        and is_serialized_dict(data["type"])
    )


class _FunctionSymbolTableFrameData(TypedDict):
    name: SerializedDict
    keyword: str
    signature: list[_FunctionSignatureEntryData]


def _is_valid_function_symbol_table_frame_data(
    data: SerializedDict,
) -> TypeGuard[_FunctionSymbolTableFrameData]:
    if (
        "name" not in data
        or not is_serialized_dict(data["name"])
        or "keyword" not in data
        or not isinstance(data["keyword"], str)
        or "signature" not in data
        or not isinstance(data["signature"], list)
    ):
        return False
    return all(
        is_serialized_dict(signature_entry)
        and _is_valid_function_signature_entry_data(signature_entry)
        for signature_entry in data["signature"]
    )


class _SymbolEntryData(TypedDict):
    symbol_name: SerializedDict
    frame: SerializedDict


def _is_valid_symbol_entry_data(data: SerializedDict) -> TypeGuard[_SymbolEntryData]:
    return (
        "symbol_name" in data
        and is_serialized_dict(data["symbol_name"])
        and "frame" in data
        and is_serialized_dict(data["frame"])
    )


class _NamespaceEntryData(TypedDict):
    namespace_name: SerializedDict
    parent_namespace_name: SerializedDict | None
    symbols: list[_SymbolEntryData]


def _is_valid_namespace_entry_data(
    data: SerializedDict,
) -> TypeGuard[_NamespaceEntryData]:
    if (
        "namespace_name" not in data
        or not is_serialized_dict(data["namespace_name"])
        or "parent_namespace_name" not in data
        or (
            data["parent_namespace_name"] is not None
            and not is_serialized_dict(data["parent_namespace_name"])
        )
        or "symbols" not in data
        or not isinstance(data["symbols"], list)
    ):
        return False
    return all(
        is_serialized_dict(symbol_entry) and _is_valid_symbol_entry_data(symbol_entry)
        for symbol_entry in data["symbols"]
    )


class _SymbolTableData(TypedDict):
    namespaces: list[_NamespaceEntryData]


def _is_valid_symbol_table_data(data: SerializedDict) -> TypeGuard[_SymbolTableData]:
    if "namespaces" not in data or not isinstance(data["namespaces"], list):
        return False
    return all(
        is_serialized_dict(namespace_entry)
        and _is_valid_namespace_entry_data(namespace_entry)
        for namespace_entry in data["namespaces"]
    )


@register_serializable(type_id="symbol_table")
class SymbolTable(Serializable):
    """Core nested symbol table comprised of various frames."""

    _table: dict[Identifier, dict[Identifier, SymbolTableFrame]]
    _parent_namespace: dict[Identifier, Identifier]

    def __init__(self) -> None:
        super().__init__()
        self._table = {}
        self._parent_namespace = {}

    def serialize_to_dict(self) -> SerializedDict:
        return {
            "namespaces": [
                {
                    "namespace_name": namespace_name.serialize_to_dict(),
                    "parent_namespace_name": (
                        self._parent_namespace[namespace_name].serialize_to_dict()
                        if namespace_name in self._parent_namespace
                        else None
                    ),
                    "symbols": [
                        {
                            "symbol_name": symbol_name.serialize_to_dict(),
                            "frame": frame.serialize_to_dict(),
                        }
                        for symbol_name, frame in namespace_table.items()
                    ],
                }
                for namespace_name, namespace_table in self._table.items()
            ]
        }

    @classmethod
    def deserialize_from_dict(cls, data: SerializedDict) -> "SymbolTable":
        if not _is_valid_symbol_table_data(data):
            raise DeserializationDictStructureError(
                cls, _SymbolTableData.__annotations__, data
            )

        symbol_table = cls()
        for namespace_entry in data["namespaces"]:
            namespace_name = Identifier.deserialize_from_dict(
                namespace_entry["namespace_name"]
            )
            parent_namespace_name_data = namespace_entry["parent_namespace_name"]
            parent_namespace_name = (
                Identifier.deserialize_from_dict(parent_namespace_name_data)
                if is_serialized_dict(parent_namespace_name_data)
                else None
            )
            symbol_table.add_namespace(namespace_name, parent_namespace_name)

        for namespace_entry in data["namespaces"]:
            namespace_name = Identifier.deserialize_from_dict(
                namespace_entry["namespace_name"]
            )
            for symbol_entry in namespace_entry["symbols"]:
                symbol_name = Identifier.deserialize_from_dict(
                    symbol_entry["symbol_name"]
                )
                frame = SymbolTableFrame.deserialize_from_dict(symbol_entry["frame"])
                symbol_table.add_symbol(namespace_name, symbol_name, frame)

        return symbol_table

    def get_number_of_namespaces(self) -> int:
        """Return the number of namespaces in the symbol table."""
        return len(self._table)

    def add_namespace(
        self,
        namespace_name: Identifier,
        parent_namespace_name: Identifier | None = None,
    ) -> None:
        """Add a new namespace to the symbol table.

        Args:
            namespace_name: Name of the namespace to be added to the symbol table.
            parent_namespace_name: Name of the parent namespace.

        Raises:
            SymbolTableError: If the namespace is already defined in the symbol table.

        """
        if self.is_namespace_defined(namespace_name):
            raise SymbolTableError(
                f"Namespace {namespace_name} already defined in the symbol table."
            )
        self._table[namespace_name] = {}
        if parent_namespace_name:
            self._parent_namespace[namespace_name] = parent_namespace_name

    def is_namespace_defined(self, namespace_name: Identifier) -> bool:
        """Check if a namespace exists in the symbol table.

        Args:
            namespace_name: Name of the namespace to be checked.

        Returns:
            True if the namespace exists in the symbol table, False otherwise.

        """
        return namespace_name in self._table

    def get_namespace(
        self, namespace_name: Identifier
    ) -> dict[Identifier, SymbolTableFrame]:
        """Retrieve a namespace from the symbol table.

        Args:
            namespace_name: Name of the namespace to be retrieved from
                the symbol table.

        Returns:
            Namespace table.

        Raises:
            SymbolTableError: If the namespace does not exist in the symbol table.

        """
        if namespace_name not in self._table:
            raise SymbolTableError(
                f"Namespace {namespace_name} not found in the symbol table."
            )

        return self._table[namespace_name]

    def update_namespaces(self, other_symbol_table: "SymbolTable") -> None:
        """Update the symbol table with new namespaces from another symbol table.

        Args:
            other_symbol_table: Symbol table to update with.

        """
        self._table.update(other_symbol_table._table)
        self._parent_namespace.update(other_symbol_table._parent_namespace)

    def add_symbol(
        self,
        namespace_name: Identifier,
        symbol_name: Identifier,
        frame: SymbolTableFrame,
    ) -> None:
        """Add a symbol to the symbol table.

        Args:
            namespace_name: Name of the namespace to add the symbol to.
            symbol_name: Name of the symbol to be added.
            frame: Frame to be added to the symbol table.

        Raises:
            SymbolTableError: If the symbol is already defined in the namespace.

        """
        if self.is_symbol_defined_in_namespace(namespace_name, symbol_name):
            raise SymbolTableError(
                f"Symbol {symbol_name} already defined in namespace {namespace_name}."
            )
        self._table[namespace_name][symbol_name] = frame

    def is_symbol_defined(self, symbol_name: Identifier) -> bool:
        """Check if a symbol exists in the symbol table.

        Args:
            symbol_name: Name of the symbol to check.

        Returns:
            True if the symbol exists in the symbol table, False otherwise.

        """
        return self._search_table_with_action(
            lambda _, candidate_symbol_name: (
                _SymbolTableSearchResult(True)
                if symbol_name == candidate_symbol_name
                else None
            ),
            lambda: False,
        )

    def is_symbol_defined_in_namespace(
        self, namespace_name: Identifier, symbol_name: Identifier
    ) -> bool:
        """Check if a symbol exists in a specific namespace.

        Args:
            namespace_name: Name of the namespace to check.
            symbol_name: Name of the symbol to check.

        Returns:
            True if the symbol exists in the namespace, False otherwise.

        """

        def is_symbol_defined_in_namespace(
            namespace_name: Identifier,
        ) -> _SymbolTableSearchResult[bool] | None:
            if symbol_name in self._table[namespace_name]:
                return _SymbolTableSearchResult(True)
            else:
                return None

        return self._search_namespace_with_action(
            namespace_name, is_symbol_defined_in_namespace, lambda: False
        )

    def get_frame(self, symbol_name: Identifier) -> SymbolTableFrame:
        """Retrieve a frame from the symbol table.

        Args:
            symbol_name: Name of the symbol to retrieve the frame for.

        Returns:
            The frame for the given symbol.

        Raises:
            SymbolTableError: If the symbol is not found in the symbol table or
                there is a cyclic namespace dependency.

        """

        def get_frame(
            namespace_name: Identifier, candidate_symbol_name: Identifier
        ) -> _SymbolTableSearchResult[SymbolTableFrame] | None:
            if symbol_name == candidate_symbol_name:
                return _SymbolTableSearchResult(
                    self._table[namespace_name][symbol_name]
                )
            else:
                return None

        def raise_symbol_not_found() -> NoReturn:
            raise SymbolTableError(
                f"Symbol {symbol_name} not found in the symbol table."
            )

        return self._search_table_with_action(get_frame, raise_symbol_not_found)

    def get_frame_from_namespace(
        self, namespace_name: Identifier, symbol_name: Identifier
    ) -> SymbolTableFrame:
        """Retrieve a frame from a specific namespace.

        Args:
            namespace_name: Name of the namespace.
            symbol_name: Name of the symbol to retrieve the frame for.

        Returns:
            The frame for the given symbol in the given namespace.

        Raises:
            SymbolTableError: If the symbol is not found in the namespace or
                there is a cyclic namespace dependency.

        """

        def get_frame_in_namespace(
            namespace_name: Identifier,
        ) -> _SymbolTableSearchResult[SymbolTableFrame] | None:
            if symbol_name in self._table[namespace_name]:
                return _SymbolTableSearchResult(
                    self._table[namespace_name][symbol_name]
                )
            else:
                return None

        def raise_symbol_not_found() -> NoReturn:
            raise SymbolTableError(
                f"Symbol {symbol_name} not found in namespace {namespace_name}."
            )

        return self._search_namespace_with_action(
            namespace_name, get_frame_in_namespace, raise_symbol_not_found
        )

    def _search_table_with_action(
        self,
        action: Callable[
            [Identifier, Identifier], _SymbolTableSearchResult[_SUCC_T] | None
        ],
        action_fail_func: Callable[[], _FAIL_T],
    ) -> _SUCC_T | _FAIL_T:
        for namespace_name, namespace_table in self._table.items():
            for symbol_name in namespace_table:
                result = action(namespace_name, symbol_name)
                if result is not None:
                    return result.value

        return action_fail_func()

    def _search_namespace_with_action(
        self,
        namespace_name: Identifier,
        action: Callable[[Identifier], _SymbolTableSearchResult[_SUCC_T] | None],
        action_fail_func: Callable[[], _FAIL_T],
    ) -> _SUCC_T | _FAIL_T:
        if not self.is_namespace_defined(namespace_name):
            raise SymbolTableError(
                f"Namespace {namespace_name} not found in the symbol table."
            )

        current_namespace_name: Identifier | None = namespace_name
        seen_namespace_names = set()
        while current_namespace_name is not None:
            if current_namespace_name in seen_namespace_names:
                raise RuntimeError(f"Namespace {current_namespace_name} is cyclic.")
            seen_namespace_names.add(current_namespace_name)

            result = action(current_namespace_name)
            if result is not None:
                return result.value

            current_namespace_name = self._parent_namespace.get(
                current_namespace_name, None
            )

        return action_fail_func()
