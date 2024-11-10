"""Core symbol table."""

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, NoReturn, TypeVar

from fhy_core.identifier import Identifier

from .error import SymbolTableError
from .memory_instance import MemoryInstance
from .types import Type, TypeQualifier
from .utils import StrEnum


class SymbolMemoryTracker:
    """Symbol memory tracker."""

    _instances: dict[Identifier, MemoryInstance]

    def __init__(self) -> None:
        self._instances = {}

    def get_number_of_instances(self) -> int:
        """Return the number of memory instances in the tracker."""
        return len(self._instances)

    def add_instance(self, instance_name: Identifier, instance: MemoryInstance) -> None:
        """Add a memory instance to the tracker.

        Args:
            instance_name: Name of the memory instance.
            instance: Memory instance to be added.

        Raises:
            SymbolTableError: If the memory instance already exists in the tracker.

        """
        if instance_name in self._instances:
            raise SymbolTableError(
                f"Memory instance {instance_name} already exists in the memory tracker."
            )
        self._instances[instance_name] = instance

    def is_instance_defined(self, instance_name: Identifier) -> bool:
        """Check if a memory instance exists in the tracker.

        Args:
            instance_name: Name of the memory instance to check.

        Returns:
            True if the memory instance exists in the tracker, False otherwise.

        """
        return instance_name in self._instances

    def get_instance(self, instance_name: Identifier) -> MemoryInstance:
        """Retrieve a memory instance from the tracker.

        Args:
            instance_name: Name of the memory instance to retrieve.

        Returns:
            Memory instance.

        Raises:
            SymbolTableError: If the memory instance does not exist in the tracker.

        """
        if instance_name not in self._instances:
            raise SymbolTableError(
                f"Memory instance {instance_name} not found in the memory tracker."
            )

        return self._instances[instance_name]

    def remove_instance(self, instance_name: Identifier) -> None:
        """Remove a memory instance from the tracker.

        Args:
            instance_name: Name of the memory instance to remove.

        Raises:
            SymbolTableError: If the memory instance does not exist in the tracker.

        """
        if instance_name not in self._instances:
            raise SymbolTableError(
                f"Memory instance {instance_name} not found in the memory tracker."
            )
        self._instances.pop(instance_name)


@dataclass(frozen=True)
class SymbolTableFrame(ABC):
    """Base symbol table frame."""

    name: Identifier


class ImportSymbolTableFrame(SymbolTableFrame):
    """Imported symbol frame."""


@dataclass(frozen=True)
class VariableSymbolTableFrame(SymbolTableFrame):
    """Variable symbol frame."""

    type: Type
    type_qualifier: TypeQualifier
    memory_tracker: SymbolMemoryTracker


class FunctionKeyword(StrEnum):
    """Function keyword."""

    PROCEDURE = "proc"
    OPERATION = "op"
    NATIVE = "native"


@dataclass(frozen=True)
class FunctionSymbolTableFrame(SymbolTableFrame):
    """Functions symbol frame."""

    keyword: FunctionKeyword
    signature: list[tuple[TypeQualifier, Type]] = field(default_factory=list)


_T = TypeVar("_T")
_SUCC_T = TypeVar("_SUCC_T")
_FAIL_T = TypeVar("_FAIL_T")


@dataclass(frozen=True)
class _SymbolTableSearchResult(Generic[_T]):
    value: _T


class SymbolTable:
    """Core nested symbol table comprised of various frames."""

    _table: dict[Identifier, dict[Identifier, SymbolTableFrame]]
    _parent_namespace: dict[Identifier, Identifier]

    def __init__(self) -> None:
        super().__init__()
        self._table = {}
        self._parent_namespace = {}

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
        if self.is_symbol_defined(namespace_name, symbol_name):
            raise SymbolTableError(
                f"Symbol {symbol_name} already defined in namespace {namespace_name}."
            )
        self._table[namespace_name][symbol_name] = frame

    def is_symbol_defined(
        self, namespace_name: Identifier, symbol_name: Identifier
    ) -> bool:
        """Check if a symbol exists in the symbol table.

        Args:
            namespace_name: Name of the namespace to check.
            symbol_name: Name of the symbol to check.

        Returns:
            True if the symbol exists in the symbol table, False otherwise.

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

    def get_frame(
        self, namespace_name: Identifier, symbol_name: Identifier
    ) -> SymbolTableFrame:
        """Retrieve a frame from the symbol table.

        Args:
            namespace_name: Name of the current namespace.
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
