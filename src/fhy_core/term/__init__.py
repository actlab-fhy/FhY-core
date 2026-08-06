"""Traits of the term language.

A term is a syntax tree whose leaves may reference, and whose interior
nodes may bind, :class:`fhy_core.identifier.Identifier` values. This
package holds the traits that concern that binding structure, which
makes them vocabulary of the term language rather than generic
structural contracts:

- :mod:`~fhy_core.term.alpha_equivalence` compares two terms up to a
  consistent renaming of their bound identifiers.
- :mod:`~fhy_core.term.binder` turns "which identifiers do I bind, and
  over what" into alpha-equivalence, free-identifier computation, and
  capture-avoiding substitution.
- :mod:`~fhy_core.term.derived_equivalence` derives both structural and
  alpha equivalence from a dataclass field schema, dispatching on the
  declared identifier role of each field.

The generic, identifier-free traits live in :mod:`fhy_core.traits`.
"""

__all__ = [
    "EQUIVALENCE_METADATA_KEY",
    "AlphaEquivalence",
    "AlphaEquivalenceMixin",
    "AlphaRenaming",
    "BinderMixin",
    "DerivedEquivalenceMixin",
    "EquivalenceDerivationError",
    "FieldComparator",
    "HasFreeIdentifiers",
    "Term",
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
