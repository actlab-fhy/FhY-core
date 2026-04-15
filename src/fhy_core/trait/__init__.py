"""Compiler object traits package."""

__all__ = [
    "Canonicalizable",
    "CanonicalizableMixin",
    "Equal",
    "EqualMixin",
    "Frozen",
    "FrozenMixin",
    "FrozenMutationError",
    "FrozenValidationError",
    "Foldable",
    "FoldableMixin",
    "HasIdentifier",
    "HasIdentifierMixin",
    "Interned",
    "InternedMixin",
    "HasOperands",
    "HasOperandsMixin",
    "HasProvenance",
    "HasProvenanceMixin",
    "HasResults",
    "HasResultsMixin",
    "HasType",
    "HasTypeMixin",
    "PartialEqual",
    "PartialEqualMixin",
    "Orderable",
    "OrderableMixin",
    "PartialOrderable",
    "PartialOrderableMixin",
    "StructuralEquivalence",
    "StructuralEquivalenceMixin",
    "Verifiable",
    "VerifiableMixin",
    "Visitable",
    "VisitableMixin",
    "VerificationError",
]

from .canonicalizable import Canonicalizable, CanonicalizableMixin
from .equality import Equal, EqualMixin, PartialEqual, PartialEqualMixin
from .foldable import Foldable, FoldableMixin
from .frozen import (
    Frozen,
    FrozenMixin,
    FrozenMutationError,
    FrozenValidationError,
)
from .has_identifier import HasIdentifier, HasIdentifierMixin
from .has_operands import HasOperands, HasOperandsMixin
from .has_provenance import HasProvenance, HasProvenanceMixin
from .has_results import HasResults, HasResultsMixin
from .has_type import HasType, HasTypeMixin
from .interned import Interned, InternedMixin
from .orderable import (
    Orderable,
    OrderableMixin,
    PartialOrderable,
    PartialOrderableMixin,
)
from .structural_equivalence import (
    StructuralEquivalence,
    StructuralEquivalenceMixin,
)
from .verifiable import Verifiable, VerifiableMixin, VerificationError
from .visitable import Visitable, VisitableMixin
