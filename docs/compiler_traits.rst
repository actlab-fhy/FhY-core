Compiler Traits
===============

Traits in ``fhy_core.trait`` define reusable behavioral contracts for IR
objects and compiler data structures.

HasIdentifier
-------------

``HasIdentifier`` / ``HasIdentifierMixin`` establish stable object identity.

In compiler infrastructure, this enables deterministic naming, stable lookup keys,
and consistent cross-pass references.

.. code-block:: python

   from dataclasses import dataclass

   from fhy_core import HasIdentifierMixin, Identifier


   @dataclass
   class Op(HasIdentifierMixin):
       _identifier: Identifier

       @property
       def identifier(self) -> Identifier:
           return self._identifier


HasProvenance
-------------

``HasProvenance`` / ``HasProvenanceMixin`` attach source and transformation
history to objects.

In compiler infrastructure, this improves diagnostics and traceability across
lowering passes.

.. code-block:: python

   from dataclasses import dataclass

   from fhy_core import HasProvenanceMixin, Note, Provenance


   @dataclass
   class Node(HasProvenanceMixin):
       _provenance: Provenance

       @property
       def provenance(self) -> Provenance:
           return self._provenance

   tagged = Node(Provenance.unknown().add_note(Note("from parser")))


HasType
-------

``HasType`` / ``HasTypeMixin`` model typed compiler objects.

In compiler infrastructure, this supports type propagation, legality checks,
and typed lowering decisions.

.. code-block:: python

   from dataclasses import dataclass

   from fhy_core import HasTypeMixin


   @dataclass
   class TypedValue(HasTypeMixin[str]):
       _type: str

       @property
       def type(self) -> str:
           return self._type


HasOperands
-----------

``HasOperands`` / ``HasOperandsMixin`` represent operation-like nodes with
operand lists.

In compiler infrastructure, this unifies traversal, rewrite, and dataflow logic
across different op kinds.

.. code-block:: python

   from dataclasses import dataclass

   from fhy_core import HasOperandsMixin


   @dataclass
   class BinaryOp(HasOperandsMixin[int]):
       _operands: tuple[int, int]

       @property
       def operands(self) -> tuple[int, int]:
           return self._operands


HasResults
----------

``HasResults`` / ``HasResultsMixin`` represent operations that produce one or
more values.

In compiler infrastructure, this is useful for SSA-like IRs where ops define
result values consumed by later ops.

.. code-block:: python

   from dataclasses import dataclass

   from fhy_core import HasResultsMixin


   @dataclass
   class MultiResultOp(HasResultsMixin[str]):
       _results: tuple[str, ...]

       @property
       def results(self) -> tuple[str, ...]:
           return self._results


Frozen
------

``Frozen`` / ``FrozenMixin`` enforce immutability after construction.

In compiler infrastructure, immutable IR objects simplify pass reasoning,
analysis caching, and thread-safe reads.

.. code-block:: python

   from dataclasses import dataclass

   from fhy_core import FrozenMixin


   @dataclass
   class FrozenNode(FrozenMixin):
       value: int

       def __post_init__(self) -> None:
           self.freeze(deep=True)


Equality
--------

``PartialEqual`` / ``Equal`` plus their mixins formalize equality semantics.

In compiler infrastructure, this is useful for change detection in passes and
semantic checks in tests.

.. code-block:: python

   from dataclasses import dataclass

   from fhy_core import EqualMixin, PartialEqualMixin


   @dataclass(eq=True, frozen=True)
   class CanonicalName(EqualMixin):
       value: str

   @dataclass(eq=True)
   class ApproxName(PartialEqualMixin):
       value: str


Ordering
--------

``PartialOrderable`` / ``Orderable`` plus mixins define ordering contracts.

In compiler infrastructure, ordering helps canonicalize pass output and produce
deterministic symbol or diagnostic ordering.

.. code-block:: python

   from dataclasses import dataclass

   from fhy_core import OrderableMixin, PartialOrderableMixin


   @dataclass(order=True)
   class RankedNode(OrderableMixin):
       rank: int

   @dataclass(order=True)
   class PartialRank(PartialOrderableMixin):
       rank: int


Verifiable
----------

``Verifiable`` / ``VerifiableMixin`` define explicit invariant checks.

In compiler infrastructure, this is useful for fail-fast validation after
parser, lowering, or optimization stages.

.. code-block:: python

   from dataclasses import dataclass

   from fhy_core import VerifiableMixin, VerificationError


   @dataclass
   class CheckedNode(VerifiableMixin):
       valid: bool

       def verify(self) -> None:
           if not self.valid:
               raise VerificationError("Node invariant violated.")


Foldable
--------

``Foldable`` / ``FoldableMixin`` provide a constant-fold-like hook.

In compiler infrastructure, this makes local simplification uniformly callable
across many IR node types.

.. code-block:: python

   from dataclasses import dataclass

   from fhy_core import FoldableMixin


   @dataclass
   class ConstExpr(FoldableMixin[int]):
       folded_value: int | None

       def fold(self) -> int | None:
           return self.folded_value


Canonicalizable
---------------

``Canonicalizable`` / ``CanonicalizableMixin`` define local rewrite-to-normal-form
behavior.

In compiler infrastructure, this is useful before CSE, hashing, and equality-based
optimization steps.

.. code-block:: python

   from dataclasses import dataclass

   from fhy_core import CanonicalizableMixin


   @dataclass
   class SignedLiteral(CanonicalizableMixin):
       value: int

       def canonicalize(self) -> bool:
           if self.value < 0:
               self.value = -self.value
               return True
           return False


StructuralEquivalence
---------------------

``StructuralEquivalence`` / ``StructuralEquivalenceMixin`` compare shape/value
structure independent of object identity.

In compiler infrastructure, this is ideal for tree-matching, regression checks,
and rewrite correctness tests.

.. code-block:: python

   from dataclasses import dataclass

   from fhy_core import StructuralEquivalenceMixin


   @dataclass
   class Pattern(StructuralEquivalenceMixin):
       opcode: str
       operands: tuple[int, ...]

       def is_structurally_equivalent(self, other: object) -> bool:
           return (
               isinstance(other, Pattern)
               and self.opcode == other.opcode
               and self.operands == other.operands
           )


Visitable
---------

``Visitable`` / ``VisitableMixin`` integrate with visitor-style compiler passes.

In compiler infrastructure, this gives a standardized dispatch path for analyses
and transformations over IR trees.

.. code-block:: python

   from fhy_core import VisitableMixin, VisitablePass


   class ToyNode(VisitableMixin):
       pass


   class ToyNodePass(VisitablePass[ToyNode, int]):
       def get_noop_output(self, ir: ToyNode) -> int:
           _ = ir
           return 0

       def visit_toy_node(self, node: ToyNode) -> int:
           _ = node
           return 7

   assert ToyNode().accept(ToyNodePass()) == 7
