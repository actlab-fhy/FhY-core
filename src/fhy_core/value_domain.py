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

The shipped domains hold fixed identifier ids, the same in every process, so
their payloads and pickles are portable. ``ValueDomain`` is backed by the
Rust implementation, whose registry is the only one in the process: a name
has one parent, constructing a registered name returns the canonical
instance itself, domains compare by name, and the registry cannot be
cleared.
"""

__all__ = ["ADDRESS_DOMAIN", "DATA_DOMAIN", "ValueDomain"]

from typing import TYPE_CHECKING

from . import _rs
from .identifier import (
    _RESERVED_ADDRESS_DOMAIN,
    _RESERVED_DATA_DOMAIN,
    HasIdentifier,
    Identifier,
    _build_reserved_identifier,
)
from .serialization import Serializable, register_serializable
from .term import AlphaEquivalenceMixin
from .traits import FrozenMixin, InternedMixin, StructuralEquivalence
from .utils.self import Self


@register_serializable(type_id="value_domain")
class ValueDomain(
    _rs.ValueDomain,
    HasIdentifier,
    StructuralEquivalence,
    AlphaEquivalenceMixin,
    Serializable,
):
    """Open classification of the kind of value an IR operation handles.

    Backed by the Rust implementation: the Rust registry holds the
    canonical domains, and ``fhy_core._rs.ValueDomain`` implements the
    attributes, ``is_subdomain_of``, equality, hashing, interning and
    payloads. This class mixes in the stateless Python protocols, and is
    registered as a virtual subclass of ``InternedMixin`` and
    ``FrozenMixin``.

    Domains may form a hierarchy through ``parent``; ``is_subdomain_of``
    walks that chain, so callers can ask whether one domain is a descendant
    of another without baking the relationships into core.

    A name has one parent. Constructing a domain whose ``name`` is
    registered under the same parent returns the canonical instance
    itself, keeping its description (deserializing a payload whose
    description differs logs a warning); under another parent it raises
    ``ValueError``, and deserializing such a payload raises
    ``DeserializationValueError``. Domains are immutable, compare and hash
    by ``name``, and pickle as their payload, so unpickling returns the
    canonical instance. The registry is append-only:
    ``clear_interned_registry`` and ``register_default_instances`` raise
    ``NotImplementedError``.

    Attributes:
        name: Stable, process-global identifier for this domain.
        description: Short human-readable description (excluded from
            equality and structural equivalence).
        parent: Optional super-domain. ``None`` for root domains.

    """

    __slots__ = ()
    if TYPE_CHECKING:
        # The type checkers' view of `_new_canonical`, which the stub cannot
        # type as this class's constructor.
        def __new__(
            cls,
            name: Identifier,
            description: str,
            parent: "ValueDomain | None" = None,
        ) -> Self:
            """Return the canonical domain named ``name``."""
            ...

    else:
        __new__ = staticmethod(_rs.ValueDomain._new_canonical)


InternedMixin.register(ValueDomain)
FrozenMixin.register(ValueDomain)

DATA_DOMAIN = ValueDomain.require_interned(
    _build_reserved_identifier(_RESERVED_DATA_DOMAIN)
)
ADDRESS_DOMAIN = ValueDomain.require_interned(
    _build_reserved_identifier(_RESERVED_ADDRESS_DOMAIN)
)
