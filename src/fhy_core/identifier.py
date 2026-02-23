"""Unique identifier for named compiler objects."""

__all__ = ["Identifier"]

from typing import Any, ClassVar

from .serialization import Serializable, SerializedMapping, register_serializable


@register_serializable
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

    def serialize_to_dict(self) -> SerializedMapping:
        return {"id": self._id, "name_hint": self._name_hint}

    @classmethod
    def deserialize_from_dict(cls, data: SerializedMapping) -> "Identifier":
        cls.raise_error_if_deserialization_from_dict_data_invalid(
            data, {"id": lambda x: isinstance(x, int) and x >= 0, "name_hint": str}
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
