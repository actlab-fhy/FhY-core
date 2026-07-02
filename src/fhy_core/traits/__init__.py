"""Compiler object traits package.

Each capability is expressed as a structural ``Protocol`` (the contract) paired,
where there is reusable default behavior, with a ``*Mixin`` (the implementation).
Implementers inherit the Protocol explicitly and decorate the implementing
methods with ``@override``.

Most mixins are stateless. Two are stateful and cooperate during construction,
so their relative order in a class's MRO matters:

- ``FrozenMixin`` seals an instance after ``__init__`` completes.
- ``InternedMixin`` finalizes (and, if the instance is ``Frozen``/``Verifiable``,
  freezes and verifies) the instance before registering it in the interning
  registry.

When a class mixes both, list ``FrozenMixin`` before ``InternedMixin`` so the
freeze wrap is the inner wrap and the instance is already frozen by the time the
interner's finalize hook runs. See ``FrozenMixin.__init_subclass__`` and
``InternedMixin`` for the per-site notes.
"""

__all__ = [
    "EQUIVALENCE_METADATA_KEY",
    "AlphaEquivalence",
    "AlphaEquivalenceMixin",
    "AlphaRenaming",
    "BinderMixin",
    "Canonicalizable",
    "DerivedEquivalenceMixin",
    "Equal",
    "EqualMixin",
    "EquivalenceDerivationError",
    "FieldComparator",
    "Frozen",
    "FrozenFieldTypeError",
    "FrozenMixin",
    "FrozenMutationError",
    "FrozenValidationError",
    "HasFreeIdentifiers",
    "HasIdentifier",
    "HasOperands",
    "HasProvenance",
    "HasResults",
    "HasType",
    "Interned",
    "InternedMixin",
    "Orderable",
    "OrderableMixin",
    "PartialEqual",
    "PartialEqualMixin",
    "PartialOrderable",
    "PartialOrderableMixin",
    "Rewritable",
    "RewritableMixin",
    "StructuralEquivalence",
    "Term",
    "Verifiable",
    "VerifiableMixin",
    "VerificationError",
    "Visitable",
    "VisitableMixin",
    "compared_as_binder",
    "compared_as_reference",
    "compared_as_value",
    "compared_with",
    "excluded_from_equivalence",
    "is_identifier_mapping_alpha_equivalent_under",
]

from .alpha_equivalence import (
    AlphaEquivalence,
    AlphaEquivalenceMixin,
    AlphaRenaming,
    is_identifier_mapping_alpha_equivalent_under,
)
from .binder import BinderMixin, HasFreeIdentifiers, Term
from .canonicalizable import Canonicalizable
from .derived_equivalence import (
    EQUIVALENCE_METADATA_KEY,
    DerivedEquivalenceMixin,
    EquivalenceDerivationError,
    FieldComparator,
    compared_as_binder,
    compared_as_reference,
    compared_as_value,
    compared_with,
    excluded_from_equivalence,
)
from .equality import Equal, EqualMixin, PartialEqual, PartialEqualMixin
from .frozen import (
    Frozen,
    FrozenFieldTypeError,
    FrozenMixin,
    FrozenMutationError,
    FrozenValidationError,
)
from .has_identifier import HasIdentifier
from .has_operands import HasOperands
from .has_provenance import HasProvenance
from .has_results import HasResults
from .has_type import HasType
from .interned import Interned, InternedMixin
from .orderable import (
    Orderable,
    OrderableMixin,
    PartialOrderable,
    PartialOrderableMixin,
)
from .rewritable import Rewritable, RewritableMixin
from .structural_equivalence import StructuralEquivalence
from .verifiable import Verifiable, VerifiableMixin, VerificationError
from .visitable import Visitable, VisitableMixin
