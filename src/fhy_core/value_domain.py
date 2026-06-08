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

from fhy_core.utils.override import override

__all__ = ["ADDRESS_DOMAIN", "DATA_DOMAIN", "ValueDomain"]

from dataclasses import dataclass, field

from .identifier import Identifier
from .serialization import (
    Serializable,
    register_serializable,
)
from .traits import (
    DerivedEquivalenceMixin,
    FrozenMixin,
    HasIdentifier,
    InternedMixin,
)


@register_serializable(type_id="value_domain")
@dataclass(frozen=True)
class ValueDomain(
    HasIdentifier,
    FrozenMixin,
    DerivedEquivalenceMixin,
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

    ``description`` is human-readable metadata only -- it does not
    participate in equality, structural equivalence, hashing, or
    interning. The first instance registered for a given ``Identifier``
    becomes canonical; subsequent constructions and deserializations
    with a different description are not rejected but do not update the
    canonical description. Deserializing a payload whose description
    differs from the canonical's emits a warning.

    Attributes:
        name: Stable, process-global identifier for this domain.
        description: Short human-readable description (surfaced in error
            messages and documentation; excluded from structural equivalence).
        parent: Optional super-domain. ``None`` for root domains.

    """

    name: Identifier
    description: str = field(compare=False)
    parent: "ValueDomain | None" = None

    def __post_init__(self) -> None:
        self.register_interned_instance()

    @override
    def get_identifier(self) -> Identifier:
        return self.name

    @override
    def get_intern_key(self) -> Identifier:
        return self.name

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

    @classmethod
    @override
    def register_default_instances(cls) -> None:
        """Re-register the canonical default ``ValueDomain``s shipped here.

        After :meth:`clear_interned_registry` wipes the registry, call this
        method to restore ``DATA_DOMAIN`` and ``ADDRESS_DOMAIN`` so the
        module-level constants remain canonical. Useful for test isolation
        that otherwise desyncs the constants from the registry.
        """
        for instance in _DEFAULT_INSTANCES:
            instance.register_interned_instance()


DATA_DOMAIN: ValueDomain = ValueDomain(
    Identifier("data"),
    "Concrete data values flowing through the IR.",
)
ADDRESS_DOMAIN: ValueDomain = ValueDomain(
    Identifier("address"),
    "Index, offset, or address values used to access data.",
)

_DEFAULT_INSTANCES: tuple[ValueDomain, ...] = (DATA_DOMAIN, ADDRESS_DOMAIN)
