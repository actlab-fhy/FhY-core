"""`Rewritable` trait and mixin.

The trait is the inverse of :class:`Visitable.get_visit_children`: given
a sequence of replacement children in the same order as
``get_visit_children`` would yield, :meth:`rebuild_with_visit_children`
returns a new node of the same kind with those children substituted.

The pair lets a generic bottom-up rewriter (such as
:class:`fhy_core.pass_infrastructure.RewritablePass`) walk an IR tree,
transform every child, and reconstruct each node without knowing
per-kind field names.
"""

__all__ = ["Rewritable", "RewritableMixin"]

from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar, runtime_checkable

_ChildT_contra = TypeVar("_ChildT_contra", contravariant=True)
_ChildT = TypeVar("_ChildT")


@runtime_checkable
class Rewritable(Protocol[_ChildT_contra]):
    """Protocol for IR nodes that can be rebuilt from a new child sequence."""

    def rebuild_with_visit_children(
        self, new_children: Sequence[_ChildT_contra]
    ) -> "Rewritable[_ChildT_contra]":
        """Return a new node of the same kind rebuilt from new children."""


class RewritableMixin(Generic[_ChildT]):
    """Mixin for IR nodes that can be rebuilt from a new child sequence.

    The type parameter ``_ChildT`` names the kind of child the subclass
    holds (typically the IR node's own base, such as ``Expression``).

    Subclasses with no children inherit the default no-op behavior:
    :meth:`rebuild_with_visit_children` returns ``self`` when
    ``new_children`` is empty and raises ``NotImplementedError`` when
    it is not. Subclasses with children must override the method to
    construct a fresh instance with the supplied children in their
    declared order.

    The expected order matches what :meth:`Visitable.get_visit_children`
    returns: callers obtain the original children via
    ``get_visit_children`` and pass replacements positionally to
    ``rebuild_with_visit_children``.
    """

    def rebuild_with_visit_children(
        self, new_children: Sequence[_ChildT]
    ) -> "RewritableMixin[_ChildT]":
        """Return a new node of the same kind with ``new_children`` substituted.

        Args:
            new_children: Replacement children in the same order as
                :meth:`Visitable.get_visit_children` returns them.

        Returns:
            A new instance of the same kind with ``new_children`` in
            place of the originals.

        Raises:
            NotImplementedError: When a subclass has children but does
                not override this method.

        """
        if not new_children:
            return self
        raise NotImplementedError(
            f"{type(self).__name__} has children but does not implement "
            "`rebuild_with_visit_children`."
        )
