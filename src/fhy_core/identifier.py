"""Unique identifier for named compiler objects."""

__all__ = ["Identifier"]

from typing import Any, ClassVar, TypedDict, TypeGuard

from .serialization import (
    InvalidDeserializationDataValueError,
    InvalidDeserializationDictStructureError,
    Serializable,
    SerializedDict,
    register_serializable,
)


class _IdentifierData(TypedDict):
    id: int
    name_hint: str


def _is_valid_identifier_data(data: SerializedDict) -> TypeGuard[_IdentifierData]:
    return (
        "id" in data
        and isinstance(data["id"], int)
        and "name_hint" in data
        and isinstance(data["name_hint"], str)
    )


@register_serializable(type_id="id")
class Identifier(Serializable):
    """Unique name."""

    _next_id: ClassVar[int] = 0
    _id: int
    _name_hint: str

    def __init__(self, name_hint: str) -> None:
        # TODO: Add a lock to implement RMW atomicity for _next_id.
        self._id = Identifier._next_id
        Identifier._next_id += 1
        self._name_hint = name_hint

    @property
    def name_hint(self) -> str:
        return self._name_hint

    @property
    def id(self) -> int:
        return self._id

    def serialize_to_dict(self) -> SerializedDict:
        return {"id": self._id, "name_hint": self._name_hint}

    @classmethod
    def deserialize_from_dict(cls, data: SerializedDict) -> "Identifier":
        if not _is_valid_identifier_data(data):
            raise InvalidDeserializationDictStructureError(
                cls, _IdentifierData.__annotations__, data
            )
        if data["id"] < 0:
            raise InvalidDeserializationDataValueError(
                cls, "id", "a non-negative integer", data["id"]
            )
        identifier = cls.__new__(cls)
        identifier._id = data["id"]
        identifier._name_hint = data["name_hint"]
        if identifier._id >= cls._next_id:
            cls._next_id = identifier._id + 1
        return identifier

    def __copy__(self) -> "Identifier":
        identifier = Identifier.__new__(Identifier)
        identifier._id = self._id
        identifier._name_hint = self._name_hint

        return identifier

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Identifier) and self._id == other._id

    def __hash__(self) -> int:
        return hash(self._id)

    def __str__(self) -> str:
        return self._name_hint

    def __repr__(self) -> str:
        return f"{self._name_hint}::{self._id}"
