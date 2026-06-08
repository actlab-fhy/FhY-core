"""`PartialEqual`/`Equal` traits and mixins."""

from fhy_core.utils.override import override

__all__ = ["Equal", "EqualMixin", "PartialEqual", "PartialEqualMixin"]

from dataclasses import is_dataclass
from types import NotImplementedType
from typing import Protocol, runtime_checkable


@runtime_checkable
class PartialEqual(Protocol):  # noqa: PLW1641
    """Protocol for objects that define `==` semantics."""

    @property
    def supports_partial_equality(self) -> bool:
        """Whether ``==`` is reliable (possibly partial) for this object."""

    @override
    def __eq__(self, other: object) -> bool | NotImplementedType: ...


@runtime_checkable
class Equal(PartialEqual, Protocol):
    """Protocol for objects with total equality semantics.

    Total equality is a strengthening of partial equality:
    ``supports_equality`` implies ``supports_partial_equality`` and that
    the object is hashable. Generic callers may rely on both ``==`` and
    ``hash()`` when ``supports_equality`` is ``True``.
    """

    @property
    def supports_equality(self) -> bool:
        """Whether ``==`` and ``hash()`` are both reliable for this object.

        When ``True``, ``supports_partial_equality`` is also ``True``.
        """

    @override
    def __hash__(self) -> int: ...


class PartialEqualMixin:
    """Mixin for objects that define `==` semantics."""

    @property
    def supports_partial_equality(self) -> bool:
        """Whether ``==`` is reliable (possibly partial) for this object."""
        if self._is_native_equatable_dataclass():
            return True
        type_eq = getattr(type(self), "__eq__")
        return type_eq is not object.__eq__ and type_eq is not PartialEqualMixin.__eq__

    @override
    def __eq__(self, other: object) -> bool | NotImplementedType:
        return NotImplemented

    @override
    def __hash__(self) -> int:
        raise TypeError(f'Unhashable type: "{type(self).__name__}"')

    def _is_native_equatable_dataclass(self) -> bool:
        if not is_dataclass(self):
            return False
        params = getattr(type(self), "__dataclass_params__", None)
        return bool(params and params.eq)


class EqualMixin(PartialEqualMixin):
    """Mixin for objects with total equality semantics."""

    @property
    def supports_equality(self) -> bool:
        """Whether ``==`` and ``hash()`` are both reliable for this object."""
        if not self.supports_partial_equality:
            return False
        type_hash = getattr(type(self), "__hash__", None)
        if type_hash is None:
            return False
        return (
            type_hash is not EqualMixin.__hash__
            and type_hash is not PartialEqualMixin.__hash__
        )

    @override
    def __hash__(self) -> int:
        raise NotImplementedError(
            f'"{type(self).__name__}" declares total equality via '
            f'"{type(self).__name__}" but does not implement "__hash__". '
            'Implement "__hash__", or use "@dataclass(frozen=True)" to derive one.'
        )
