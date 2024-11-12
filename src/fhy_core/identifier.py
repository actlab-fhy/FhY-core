"""Unique identifier for named compiler objects."""

__all__ = ["Identifier"]

__all__ = ["Identifier"]

from typing import Any


class Identifier:
    """Unique name."""

    _next_id: int = 0
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
