"""Open, registry-backed classification of value kinds in a compiler IR.

A compiler intermediate representation often needs to classify the kind of
value an operation produces or consumes (concrete data, an address, a
control token, ...) without committing to a closed set of kinds in a
foundational utility package. ``ValueDomain`` provides that classification
as an open class with canonical interning: callers register new domains as
needed without modifying ``fhy_core``.

``DATA_DOMAIN`` and ``ADDRESS_DOMAIN`` are the two canonical domains
shipped here. Callers should import these names rather than constructing
fresh ``ValueDomain`` instances with the same ``name_hint``, because
``Identifier`` uses id-equality and a freshly-constructed
``Identifier("data")`` would not match the canonical one.
"""

__all__ = ["ADDRESS_DOMAIN", "DATA_DOMAIN", "ValueDomain"]

from dataclasses import dataclass, field
from typing import TypedDict, TypeGuard

from .identifier import Identifier
from .serialization import (
    DeserializationDictStructureError,
    Serializable,
    SerializedDict,
    is_serialized_dict,
    register_serializable,
)
from .trait import (
    FrozenMixin,
    HasIdentifierMixin,
    InternedMixin,
    StructuralEquivalenceMixin,
)


class _ValueDomainData(TypedDict):
    name: SerializedDict
    description: str
    parent: SerializedDict | None


_VALUE_DOMAIN_DATA_KEYS: frozenset[str] = frozenset({"name", "description", "parent"})


def _is_valid_value_domain_data(data: SerializedDict) -> TypeGuard[_ValueDomainData]:
    if data.keys() != _VALUE_DOMAIN_DATA_KEYS:
        return False
    if not is_serialized_dict(data["name"]):
        return False
    if not isinstance(data["description"], str):
        return False
    parent = data["parent"]
    if parent is None:
        return True
    return is_serialized_dict(parent)


@register_serializable(type_id="value_domain")
@dataclass(frozen=True)
class ValueDomain(
    HasIdentifierMixin,
    FrozenMixin,
    StructuralEquivalenceMixin,
    InternedMixin[Identifier],
    Serializable,
):
    """Open classification of the kind of value an IR operation handles.

    Each ``ValueDomain`` is uniquely identified by an ``Identifier`` and
    is canonicalized through the ``InternedMixin`` registry: the first
    instance constructed for a given ``Identifier`` becomes the canonical
    entry and subsequent constructions with the same key shadow into that
    entry without replacing it.

    Domains may optionally form a hierarchy through ``parent``; the
    ``is_subdomain_of`` helper walks that chain so callers can ask whether
    one domain is a descendant of another without baking the relationships
    into core.

    Attributes:
        name: Stable, process-global identifier for this domain.
        description: Short human-readable description (surfaced in error
            messages and documentation; excluded from structural equivalence).
        parent: Optional super-domain. ``None`` for root domains.

    """

    name: Identifier
    description: str = field(compare=False)
    parent: "ValueDomain | None" = field(default=None, compare=False)

    def __post_init__(self) -> None:
        self.register_interned_instance()

    def get_identifier(self) -> Identifier:
        return self.name

    def get_intern_key(self) -> Identifier:
        return self.name

    def is_structurally_equivalent(self, other: object) -> bool:
        if not isinstance(other, ValueDomain):
            return False
        if self.name != other.name:
            return False
        if self.parent is None:
            return other.parent is None
        if other.parent is None:
            return False
        return self.parent.is_structurally_equivalent(other.parent)

    def is_subdomain_of(self, other: "ValueDomain") -> bool:
        """Return whether ``other`` is ``self`` or any ancestor via ``parent``.

        Args:
            other: Candidate super-domain.

        Returns:
            True iff ``other`` is structurally equivalent to ``self`` or to
            any domain reachable by following ``parent`` from ``self``.

        """
        current: ValueDomain | None = self
        while current is not None:
            if current.is_structurally_equivalent(other):
                return True
            current = current.parent
        return False

    def serialize_to_dict(self) -> SerializedDict:
        parent_payload: SerializedDict | None = (
            self.parent.serialize_to_dict() if self.parent is not None else None
        )
        return {
            "name": self.name.serialize_to_dict(),
            "description": self.description,
            "parent": parent_payload,
        }

    @classmethod
    def deserialize_from_dict(cls, data: SerializedDict) -> "ValueDomain":
        if not _is_valid_value_domain_data(data):
            raise DeserializationDictStructureError(
                cls, _ValueDomainData.__annotations__, data
            )
        name = Identifier.deserialize_from_dict(data["name"])
        canonical = cls.get_interned(name)
        if canonical is not None:
            return canonical
        parent_payload = data["parent"]
        parent = (
            cls.deserialize_from_dict(parent_payload)
            if parent_payload is not None
            else None
        )
        return cls(name=name, description=data["description"], parent=parent)


DATA_DOMAIN: ValueDomain = ValueDomain(
    Identifier("data"),
    "Concrete data values flowing through the IR.",
)
ADDRESS_DOMAIN: ValueDomain = ValueDomain(
    Identifier("address"),
    "Index, offset, or address values used to access data.",
)
