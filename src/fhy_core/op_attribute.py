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

The shipped attributes hold fixed identifier ids, the same in every process,
so their payloads and pickles are portable. ``OpAttribute`` is backed by the
Rust implementation, whose registry is the only one in the process:
constructing an attribute whose name is registered returns the canonical
instance itself, and the registry cannot be cleared.
"""

__all__ = [
    "ASSOCIATIVE",
    "COMMUTATIVE",
    "ELEMENTWISE",
    "PURE",
    "OpAttribute",
]

from typing import TYPE_CHECKING, Self

from . import _rs
from .identifier import (
    _RESERVED_ASSOCIATIVE,
    _RESERVED_COMMUTATIVE,
    _RESERVED_ELEMENTWISE,
    _RESERVED_PURE,
    HasIdentifier,
    Identifier,
    _build_reserved_identifier,
)
from .serialization import Serializable, register_serializable
from .term import AlphaEquivalenceMixin
from .traits import FrozenMixin, InternedMixin, StructuralEquivalence


@register_serializable(type_id="op_attribute")
class OpAttribute(
    _rs.OpAttribute,
    HasIdentifier,
    StructuralEquivalence,
    AlphaEquivalenceMixin,
    Serializable,
):
    """Open semantic tag attached to a compiler operation.

    Backed by the Rust implementation: the Rust registry holds the
    canonical attributes, and ``fhy_core._rs.OpAttribute`` implements
    the attributes, equality, hashing, interning and payloads. This class
    mixes in the stateless Python protocols, and is registered as a
    virtual subclass of ``InternedMixin`` and ``FrozenMixin``.

    Constructing an attribute whose ``name`` is registered returns the
    canonical instance itself, keeping its description; deserializing a
    payload whose description differs logs a warning. Attributes are
    immutable, compare and hash by ``name``, and pickle as their payload,
    so unpickling returns the canonical instance. The registry is
    append-only: ``clear_interned_registry`` and
    ``register_default_instances`` raise ``NotImplementedError``.

    Attributes:
        name: Stable, process-global identifier for this attribute.
        description: Short human-readable description (excluded from
            equality and structural equivalence).

    """

    __slots__ = ()
    if TYPE_CHECKING:
        # The type checkers' view of `_new_canonical`, which the stub cannot
        # type as this class's constructor.
        def __new__(cls, name: Identifier, description: str) -> Self:
            """Return the canonical attribute named ``name``."""
            ...

    else:
        __new__ = staticmethod(_rs.OpAttribute._new_canonical)


InternedMixin.register(OpAttribute)
FrozenMixin.register(OpAttribute)
OpAttribute._register_public_class()

COMMUTATIVE = OpAttribute.require_interned(
    _build_reserved_identifier(_RESERVED_COMMUTATIVE)
)
ASSOCIATIVE = OpAttribute.require_interned(
    _build_reserved_identifier(_RESERVED_ASSOCIATIVE)
)
PURE = OpAttribute.require_interned(_build_reserved_identifier(_RESERVED_PURE))
ELEMENTWISE = OpAttribute.require_interned(
    _build_reserved_identifier(_RESERVED_ELEMENTWISE)
)
