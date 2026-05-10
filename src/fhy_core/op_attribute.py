"""Open, registry-backed semantic tags for compiler operations.

Every layer of a compiler stack carries semantic attributes on its
operations -- algebraic properties such as commutativity and
associativity, purity, elementwise application, and family-specific tags
contributed by particular IRs. ``OpAttribute`` exposes that classification
as an open class with canonical interning, so layers can share the four
generic algebraic/semantic attributes shipped here while contributing
their own without modifying ``fhy_core``.

``OpAttribute`` is a free-standing tag primitive: it has no dependency on
any particular operation type and is not specialized for any layer.
Callers should import the canonical names rather than constructing fresh
``OpAttribute`` instances with the same ``name_hint``, because
``Identifier`` uses id-equality and a freshly-constructed
``Identifier("commutative")`` would not match the canonical one.
"""

__all__ = [
    "ASSOCIATIVE",
    "COMMUTATIVE",
    "ELEMENTWISE",
    "PURE",
    "OpAttribute",
]

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


class _OpAttributeData(TypedDict):
    name: SerializedDict
    description: str


_OP_ATTRIBUTE_DATA_KEYS: frozenset[str] = frozenset({"name", "description"})


def _is_valid_op_attribute_data(data: SerializedDict) -> TypeGuard[_OpAttributeData]:
    if data.keys() != _OP_ATTRIBUTE_DATA_KEYS:
        return False
    if not is_serialized_dict(data["name"]):
        return False
    return isinstance(data["description"], str)


@register_serializable(type_id="op_attribute")
@dataclass(frozen=True)
class OpAttribute(
    HasIdentifierMixin,
    FrozenMixin,
    StructuralEquivalenceMixin,
    InternedMixin[Identifier],
    Serializable,
):
    """Open semantic tag attached to a compiler operation.

    Each ``OpAttribute`` is uniquely identified by an ``Identifier`` and
    is canonicalized through the ``InternedMixin`` registry: the first
    instance constructed for a given ``Identifier`` becomes the canonical
    entry and subsequent constructions with the same key shadow into that
    entry without replacing it.

    Attributes:
        name: Stable, process-global identifier for this attribute.
        description: Short human-readable description (surfaced in error
            messages, documentation, and pass-author guidance; excluded
            from structural equivalence).

    """

    name: Identifier
    description: str = field(compare=False)

    def __post_init__(self) -> None:
        self.register_interned_instance()

    def get_identifier(self) -> Identifier:
        return self.name

    def get_intern_key(self) -> Identifier:
        return self.name

    def is_structurally_equivalent(self, other: object) -> bool:
        if not isinstance(other, OpAttribute):
            return False
        return self.name == other.name

    def serialize_to_dict(self) -> SerializedDict:
        return {
            "name": self.name.serialize_to_dict(),
            "description": self.description,
        }

    @classmethod
    def deserialize_from_dict(cls, data: SerializedDict) -> "OpAttribute":
        if not _is_valid_op_attribute_data(data):
            raise DeserializationDictStructureError(
                cls, _OpAttributeData.__annotations__, data
            )
        name = Identifier.deserialize_from_dict(data["name"])
        canonical = cls.get_interned(name)
        if canonical is not None:
            return canonical
        return cls(name=name, description=data["description"])


COMMUTATIVE: OpAttribute = OpAttribute(
    Identifier("commutative"),
    "Op output is invariant under operand swap.",
)
ASSOCIATIVE: OpAttribute = OpAttribute(
    Identifier("associative"),
    "Op composes associatively across applications.",
)
PURE: OpAttribute = OpAttribute(
    Identifier("pure"),
    "Op has no side effects and produces deterministic outputs.",
)
ELEMENTWISE: OpAttribute = OpAttribute(
    Identifier("elementwise"),
    "Op acts independently on each element of its operands.",
)
