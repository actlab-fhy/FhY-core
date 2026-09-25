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

The shipped domains hold fixed identifier ids, the same in every process and
on either backend, so their payloads and pickles are portable. On the Rust
backend (``fhy_core.RUST_BACKEND_SELECTED``), ``ValueDomain`` is backed by
the Rust implementation, whose registry is the only one in the process and
follows the Rust semantics: a name has one parent, constructing a registered
name returns the canonical instance itself, domains compare by name, and the
registry cannot be cleared.
"""

from fhy_core.utils.override import override

__all__ = ["ADDRESS_DOMAIN", "DATA_DOMAIN", "ValueDomain"]

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._backend import IS_RUST_BACKEND_SELECTED
from .identifier import (
    _RESERVED_ADDRESS_DOMAIN,
    _RESERVED_DATA_DOMAIN,
    HasIdentifier,
    Identifier,
    _build_reserved_identifier,
)
from .serialization import (
    Serializable,
    SerializedDict,
    register_serializable,
)
from .term import AlphaEquivalenceMixin, DerivedEquivalenceMixin
from .traits import FrozenMixin, InternedMixin, StructuralEquivalence

if TYPE_CHECKING or not IS_RUST_BACKEND_SELECTED:

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

        ``parent`` does participate in equality, compared by value up the
        chain. Deserializing a payload for an already-canonical ``Identifier``
        whose parent differs from the canonical's raises
        ``DeserializationValueError`` rather than returning the canonical.
        Pickling a domain stores its payload, so unpickling returns the
        canonical instance.

        Attributes:
            name: Stable, process-global identifier for this domain.
            description: Short human-readable description (surfaced in error
                messages and documentation; excluded from structural
                equivalence).
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

        @override
        def __reduce__(
            self,
        ) -> tuple[Callable[[SerializedDict], "ValueDomain"], tuple[SerializedDict]]:
            return (ValueDomain.deserialize_from_dict, (self.serialize_to_dict(),))

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
        _build_reserved_identifier(_RESERVED_DATA_DOMAIN),
        "Concrete data values flowing through the IR.",
    )
    ADDRESS_DOMAIN: ValueDomain = ValueDomain(
        _build_reserved_identifier(_RESERVED_ADDRESS_DOMAIN),
        "Index, offset, or address values used to access data.",
    )

else:
    from . import _rs

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

        A name has one parent. Constructing a domain whose ``name`` is
        registered under the same parent returns the canonical instance
        itself, keeping its description; under another parent it raises
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
        __new__ = staticmethod(_rs.ValueDomain._new_canonical)

    InternedMixin.register(ValueDomain)
    FrozenMixin.register(ValueDomain)

    DATA_DOMAIN = ValueDomain.require_interned(
        _build_reserved_identifier(_RESERVED_DATA_DOMAIN)
    )
    ADDRESS_DOMAIN = ValueDomain.require_interned(
        _build_reserved_identifier(_RESERVED_ADDRESS_DOMAIN)
    )

_DEFAULT_INSTANCES: tuple[ValueDomain, ...] = (DATA_DOMAIN, ADDRESS_DOMAIN)
