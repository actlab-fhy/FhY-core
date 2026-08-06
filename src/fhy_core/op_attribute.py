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

from fhy_core.utils.override import override

__all__ = [
    "ASSOCIATIVE",
    "COMMUTATIVE",
    "ELEMENTWISE",
    "PURE",
    "OpAttribute",
]

from dataclasses import dataclass, field

from .identifier import HasIdentifier, Identifier
from .serialization import (
    Serializable,
    register_serializable,
)
from .term import DerivedEquivalenceMixin
from .traits import FrozenMixin, InternedMixin


@register_serializable(type_id="op_attribute")
@dataclass(frozen=True)
class OpAttribute(
    HasIdentifier,
    FrozenMixin,
    DerivedEquivalenceMixin,
    InternedMixin[Identifier],
    Serializable,
):
    """Open semantic tag attached to a compiler operation.

    Each ``OpAttribute`` is uniquely identified by an ``Identifier`` and
    is canonicalized through the ``InternedMixin`` registry: the first
    instance constructed for a given ``Identifier`` becomes the canonical
    entry and subsequent constructions with the same key shadow into that
    entry without replacing it.

    ``description`` is human-readable metadata only -- it does not
    participate in equality, structural equivalence, hashing, or
    interning. The first instance registered for a given ``Identifier``
    becomes canonical; subsequent constructions and deserializations
    with a different description are not rejected but do not update the
    canonical description. Deserializing a payload whose description
    differs from the canonical's emits a warning.

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

    @override
    def get_identifier(self) -> Identifier:
        return self.name

    @override
    def get_intern_key(self) -> Identifier:
        return self.name

    @classmethod
    @override
    def register_default_instances(cls) -> None:
        """Re-register the canonical default ``OpAttribute``s shipped here.

        After :meth:`clear_interned_registry` wipes the registry, call this
        method to restore ``COMMUTATIVE``, ``ASSOCIATIVE``, ``PURE``, and
        ``ELEMENTWISE`` so the module-level constants remain canonical.
        """
        for instance in _DEFAULT_INSTANCES:
            instance.register_interned_instance()


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

_DEFAULT_INSTANCES: tuple[OpAttribute, ...] = (
    COMMUTATIVE,
    ASSOCIATIVE,
    PURE,
    ELEMENTWISE,
)
