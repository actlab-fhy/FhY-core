"""`AlphaEquivalence` trait, mixin, ``AlphaRenaming``, and mapping helper.

This module exposes the alpha-equivalence contract for compiler-IR
objects: the binary relation that holds between two terms when they
agree up to a consistent renaming of their bound identifiers. The trait
is parallel to ``StructuralEquivalence``; structurally-equivalent pairs
are always alpha-equivalent, but alpha-equivalent pairs need not be
structurally equivalent.

The renaming state is carried by ``AlphaRenaming``: a stack of binder
frames plus a free-identifier bijection. Binder nodes ``extend`` the
renaming when they recurse into their bodies; identifier-reference
nodes consult the renaming to compare their identifiers. It is the
Rust-backed ``fhy_core._rs.AlphaRenaming``, over the Rust core's
``fhy_core::term::AlphaRenaming``, and so is the mapping helper
(S10 of ``docs/design/python-switch.md``).

For IR nodes whose attributes include ``Mapping[Identifier, V]``
collections (substitutions, parallel let-groups, identifier-keyed
catalogs), the module also exports
``is_identifier_mapping_alpha_equivalent_under``, a free function that
performs the bijection-finding and pairwise value comparison correctly
so implementors do not have to open-code it.
"""

__all__ = [
    "AlphaEquivalence",
    "AlphaEquivalenceMixin",
    "AlphaRenaming",
    "is_identifier_mapping_alpha_equivalent_under",
]

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from fhy_core import _rs
from fhy_core._rs import is_identifier_mapping_alpha_equivalent_under
from fhy_core.traits import FrozenMixin

AlphaRenaming = _rs.AlphaRenaming
"""Stack of binder-frame bijections plus a free-identifier bijection.

The renaming state of a recursive alpha-equivalence comparison: the frames
pushed by enclosing binder nodes, innermost last, and an ambient bijection
on free identifiers supplied by the original caller.

- ``AlphaRenaming.empty()`` is the renaming with no frame and no free
  renaming, one shared instance.
- ``AlphaRenaming.with_free_renaming(mapping)`` seeds the free bijection;
  ``renaming.extend(bindings)`` returns a new renaming with one more
  innermost frame. Both refuse a map that is not injective with
  ``ValueError``, and a key or value that is not an ``Identifier`` with
  ``TypeError``.
- ``renaming.resolve(identifier)`` looks the identifier up in the frames,
  innermost first, then in the free renaming, then falls back to the
  identifier itself; ``renaming.are_identifiers_alpha_equivalent(left,
  right)`` decides correspondence, refusing a capture: an identifier bound
  on one side only corresponds to nothing on the other.

Renamings are immutable (mutation raises ``FrozenMutationError``), equal by
their frames, in order, and their free renaming, hashable, and pickle.
Across frames, images may repeat: that is the shadowing case.
"""

FrozenMixin.register(AlphaRenaming)


@runtime_checkable
class AlphaEquivalence(Protocol):
    """Protocol for objects that support alpha-equivalence comparison.

    Alpha-equivalence treats two binder-containing terms as equal when
    they agree up to a consistent renaming of bound identifiers. For
    objects without binders, the relation reduces to structural
    equivalence threaded through children; for objects that reference
    identifiers, the comparison consults an ``AlphaRenaming``.

    The two methods are the caller surface and the composition surface
    respectively. Callers ask ``is_alpha_equivalent``; recursive
    implementations call ``is_alpha_equivalent_under`` on their children
    so that binder frames installed by enclosing nodes remain in scope.

    Implementations must be reflexive, symmetric, and transitive on the
    domain of well-formed IR objects.
    """

    def is_alpha_equivalent(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are alpha-equivalent.

        Equivalent to ``self.is_alpha_equivalent_under(other,
        AlphaRenaming.empty())``: no binders are in scope and free
        identifiers compare by ``Identifier`` equality.

        Args:
            other: Candidate term.

        Returns:
            ``True`` if the two terms agree up to a consistent renaming
            of bound identifiers, ``False`` otherwise. Returns ``False``
            (not raises) when ``other`` is not of a compatible type.

        """

    def is_alpha_equivalent_under(
        self, other: object, renaming: "AlphaRenaming"
    ) -> bool:
        """Return whether self and other are alpha-equivalent under the renaming.

        Binder-bearing implementations call ``renaming.extend(...)``
        with the pairwise binding of their parameters and recurse on
        bodies. Identifier-reference implementations call
        ``renaming.are_identifiers_alpha_equivalent`` on their stored
        ``Identifier``s. All other implementations thread ``renaming``
        through unchanged into recursive calls on their children.

        Args:
            other: Candidate term.
            renaming: Binder-frame stack plus free-identifier bijection
                accumulated by enclosing comparison calls.

        Returns:
            ``True`` if ``self`` and ``other`` are alpha-equivalent
            under ``renaming``, ``False`` otherwise. Returns ``False``
            (not raises) when ``other`` is not of a compatible type.

        """


class AlphaEquivalenceMixin(ABC):
    """Mixin for objects that support alpha-equivalence comparison.

    Subclasses implement ``is_alpha_equivalent_under`` only.
    ``is_alpha_equivalent`` is provided as a concrete default that
    constructs the empty renaming and delegates.
    """

    def is_alpha_equivalent(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are alpha-equivalent.

        Delegates to ``is_alpha_equivalent_under`` with
        ``AlphaRenaming.empty()``. See :class:`AlphaEquivalence` for the
        contract.

        """
        return self.is_alpha_equivalent_under(other, AlphaRenaming.empty())

    @abstractmethod
    def is_alpha_equivalent_under(
        self, other: object, renaming: "AlphaRenaming"
    ) -> bool:
        """Return whether self and other are alpha-equivalent under the renaming.

        See :class:`AlphaEquivalence` for the contract.

        """
