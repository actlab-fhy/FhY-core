Core Features
=============

Identifier
----------

``Identifier`` gives compiler objects a stable identity made of a unique integer ID
and a readable name hint.

In compiler infrastructure, this is useful for symbol keys, pass names, IR node
identity, and deterministic diagnostics.

.. code-block:: python

   from fhy_core import Identifier

   temp = Identifier("tmp")
   acc = Identifier("acc")

   print(temp.id)         # unique integer
   print(str(temp))       # "tmp"
   print(repr(temp))      # e.g. "tmp::42"
   print(temp == acc)     # False

.. code-block:: python

   # Identifiers are serializable and hashable, so they work well as map keys.
   from fhy_core import Identifier

   symbol = Identifier("x")
   table = {symbol: "i32"}


Error Registration
------------------

``register_error`` is a lightweight decorator for declaring project-specific
exception classes.

In compiler infrastructure, it provides a single place to register semantic,
validation, or lowering errors for consistent handling.

.. code-block:: python

   from fhy_core import register_error

   @register_error
   class LoweringError(RuntimeError):
       """Raised when AST-to-IR lowering fails."""

   raise LoweringError("unsupported pattern")

.. code-block:: python

   @register_error
   class TypeCheckError(ValueError):
       """Raised when type constraints are violated."""


Expression AST and Parsing
--------------------------

The expression subsystem models symbolic expressions as AST nodes and supports
parsing, rewriting, simplification, pretty-printing, and solver conversion.

In compiler infrastructure, this is useful for affine indexing, constraint solving,
constant folding, and symbolic analysis.

.. code-block:: python

   from fhy_core import parse_expression, pformat_expression

   expr = parse_expression("(x + 4) * 2")
   print(pformat_expression(expr))

.. code-block:: python

   from fhy_core import (
       Identifier,
       LiteralExpression,
       collect_identifiers,
       parse_expression,
       simplify_expression,
       substitute_identifiers,
   )

   x = Identifier("x")
   expr = parse_expression("x * 0 + 7")

   ids = collect_identifiers(expr)  # {x}
   simplified = simplify_expression(expr)
   substituted = substitute_identifiers(expr, {x: LiteralExpression(3)})


Constraint System
-----------------

Constraints represent logical restrictions over variables:
``EquationConstraint``, ``InSetConstraint``, and ``NotInSetConstraint``.

In compiler infrastructure, constraints are useful for legality checks,
parameter bounds, domain restrictions, and proving transformations.

.. code-block:: python

   from fhy_core import EquationConstraint, Identifier, parse_expression

   i = Identifier("i")
   constraint = EquationConstraint(i, parse_expression("i >= 0 && i < 16"))

   assert constraint.is_satisfied(3)
   assert not constraint.is_satisfied(20)

.. code-block:: python

   from fhy_core import Identifier, InSetConstraint, NotInSetConstraint

   mode = Identifier("mode")
   valid = InSetConstraint(mode, {"unroll", "tile", "vectorize"})
   blocked = NotInSetConstraint(mode, {"deprecated_pass"})


Parameters
----------

``Param`` types model compile-time tunables and domains: real, integer,
natural, ordinal, categorical, permutation, and bounded integer/natural forms.

In compiler infrastructure, parameters support autotuning spaces, schedule choices,
and backend constraints.

.. code-block:: python

   from fhy_core import IntParam, RealParam

   tile = IntParam.between(1, 64)
   alpha = RealParam.with_lower_bound(0.0)

   tile_value = tile.bind(16)
   alpha_value = alpha.bind(0.5)

.. code-block:: python

   from fhy_core import BoundIntParam, NatParam

   threads = BoundIntParam.with_lower_bound(1)
   threads = threads.add_upper_bound_constraint(1024)
   stages = NatParam(is_zero_included=False)

   assert threads.is_constraints_satisfied(256)
   assert stages.is_constraints_satisfied(3)


Type System
-----------

The type system includes primitive and template data types plus higher-level
``NumericalType``, ``IndexType``, and ``TupleType``.

In compiler infrastructure, this supports type checking, promotion, shape-aware IR,
and stable lowering rules.

.. code-block:: python

   from fhy_core import CoreDataType, NumericalType, PrimitiveDataType

   i32 = PrimitiveDataType(CoreDataType.INT32)
   tensor = NumericalType(i32)

.. code-block:: python

   from fhy_core import (
       CoreDataType,
       TypeQualifier,
       get_core_data_type_bit_width,
       promote_core_data_types,
       promote_type_qualifiers,
   )

   bw = get_core_data_type_bit_width(CoreDataType.FLOAT64)
   promoted_ty = promote_core_data_types(CoreDataType.INT32, CoreDataType.INT64)
   promoted_q = promote_type_qualifiers(TypeQualifier.PARAM, TypeQualifier.INPUT)


Symbol Table
------------

``SymbolTable`` manages nested namespaces and per-symbol frames
(import/variable/function).

In compiler infrastructure, symbol tables drive name resolution, scope analysis,
and interface checking across modules.

.. code-block:: python

   from fhy_core import Identifier, ImportSymbolTableFrame, SymbolTable

   global_ns = Identifier("global")
   x = Identifier("x")

   table = SymbolTable()
   table.add_namespace(global_ns)
   table.add_symbol(global_ns, x, ImportSymbolTableFrame(x))

.. code-block:: python

   from fhy_core import (
       CoreDataType,
       FunctionKeyword,
       FunctionSymbolTableFrame,
       Identifier,
       NumericalType,
       PrimitiveDataType,
       TypeQualifier,
       VariableSymbolTableFrame,
   )

   scalar_i32 = NumericalType(PrimitiveDataType(CoreDataType.INT32))

   frame = VariableSymbolTableFrame(Identifier("buf"), scalar_i32, TypeQualifier.STATE)
   fn = FunctionSymbolTableFrame(
       Identifier("matmul"),
       FunctionKeyword.PROCEDURE,
       signature=[(TypeQualifier.INPUT, scalar_i32), (TypeQualifier.OUTPUT, scalar_i32)],
   )


Serialization
-------------

``Serializable`` and ``WrappedFamilySerializable`` provide dict/JSON/binary
serialization, along with a registered type-ID dispatch system.

In compiler infrastructure, serialization is useful for IR snapshots,
cache artifacts, and pipeline interchange formats.

.. code-block:: python

   from dataclasses import dataclass

   from fhy_core import Serializable, SerializedDict, register_serializable


   @register_serializable(type_id="demo.span")
   @dataclass(frozen=True)
   class SpanLike(Serializable):
       lo: int
       hi: int

       def serialize_to_dict(self) -> SerializedDict:
           return {"lo": self.lo, "hi": self.hi}

       @classmethod
       def deserialize_from_dict(cls, data: SerializedDict) -> "SpanLike":
           return cls(int(data["lo"]), int(data["hi"]))

.. code-block:: python

   value = SpanLike(1, 8)
   as_dict = value.serialize_to_dict()
   as_json = value.to_json()
   as_bytes = value.to_bytes()

   assert SpanLike.deserialize_from_dict(as_dict) == value
   assert SpanLike.from_json(as_json) == value
   assert SpanLike.from_bytes(as_bytes) == value


Pass Infrastructure
-------------------

The pass subsystem provides reusable pass abstractions, diagnostics,
registration, analysis caching, and ordered/fixpoint pipelines.

In compiler infrastructure, this is the backbone for optimization, canonicalization,
validation, and lowering pipelines.

.. code-block:: python

   from fhy_core import CompilerPass, register_pass


   @register_pass("demo.increment", "Add 1 to an integer IR node.")
   class AddOnePass(CompilerPass[int, int]):
       def get_noop_output(self, ir: int) -> int:
           return ir

       def run_pass(self, ir: int) -> int:
           return ir + 1

   assert AddOnePass().execute(4).output == 5

.. code-block:: python

   from fhy_core import CompilerPass, Identifier, FixpointPassGroup, PassManager, register_pass

   pipeline = PassManager[int](name=Identifier("pipeline"))

   @register_pass("demo.decrement_to_zero", "Move integer toward zero.")
   class DecrementPass(CompilerPass[int, int]):
       def get_noop_output(self, ir: int) -> int:
           return ir

       def run_pass(self, ir: int) -> int:
           return max(ir - 1, 0)

   loop = FixpointPassGroup[int](name=Identifier("simplify"), max_iterations=8)
   loop.add_pass(DecrementPass())
   pipeline.add_fixpoint_group(loop)

   result = pipeline.run(3)
   assert result.output == 0
