General Utilities
=================

Provenance Primitives
---------------------

``Position``, ``Span``, ``Note``, and ``Provenance`` capture source mapping and
origin metadata.

In compiler infrastructure, provenance lets diagnostics point back to source
and helps trace transformations through pass pipelines.

.. code-block:: python

   from pathlib import Path

   from fhy_core import Note, Position, Provenance, Span

   span = Span(
       file_path=Path("kernel.fhy"),
       start_offset=12,
       end_offset=27,
       start_position=Position(2, 5),
       end_position=Position(2, 20),
   )

   prov = Provenance.unknown().with_span(span).add_note(Note("lowered from AST"))

.. code-block:: python

   from fhy_core import Provenance

   left = Provenance.unknown().add_source_id("ast::17")
   right = Provenance.unknown().add_source_id("ir::bb0")
   merged = left.merge(right)


Logging
-------

``get_logger``, ``configure_logging``, ``install_null_handler``, and
``add_file_handler`` support library-safe logging and explicit CLI setup.

In compiler infrastructure, this separates quiet library behavior from
application-level logging policy.

.. code-block:: python

   import logging

   from fhy_core import configure_logging, get_logger

   configure_logging(namespace="mycompiler", console_level=logging.INFO)
   log = get_logger("mycompiler.pipeline")
   log.info("starting optimization")

.. code-block:: python

   from pathlib import Path

   from fhy_core import add_file_handler, get_logger

   log = get_logger("mycompiler.pipeline")
   add_file_handler(log, Path("build.log"))


Python Enum Compatibility
-------------------------

``StrEnum`` and ``IntEnum`` are compatibility helpers for environments where the
standard-library behavior differs across Python versions.

In compiler infrastructure, enums are useful for stable opcode names, type
qualifiers, pass levels, and diagnostic categories.

.. code-block:: python

   from fhy_core import IntEnum, StrEnum


   class Phase(StrEnum):
       PARSE = "parse"
       LOWER = "lower"


   class Priority(IntEnum):
       LOW = 1
       HIGH = 2


Stack
-----

``Stack`` is a typed LIFO wrapper around ``deque``.

In compiler infrastructure, it is useful for parser state, traversal stacks,
and nested scope management.

.. code-block:: python

   from fhy_core import Stack

   stack = Stack[str]()
   stack.push("module")
   stack.push("function")

   assert stack.peek() == "function"
   assert stack.pop() == "function"

.. code-block:: python

   from fhy_core import Stack

   stack = Stack[str]()
   stack.push("block")
   stack.clear()
   assert len(stack) == 0


Partially Ordered Set (POSET)
-----------------------------

``PartiallyOrderedSet`` models partial orders using a DAG.

In compiler infrastructure, this helps express dependency relations between
passes, analyses, or type constraints.

.. code-block:: python

   from fhy_core import PartiallyOrderedSet

   order = PartiallyOrderedSet[str]()
   order.add_element("parse")
   order.add_element("typecheck")
   order.add_order("parse", "typecheck")

   assert order.is_less_than("parse", "typecheck")

.. code-block:: python

   from fhy_core import PartiallyOrderedSet

   order = PartiallyOrderedSet[str]()
   order.add_element("parse")
   order.add_element("typecheck")
   order.add_order("parse", "typecheck")
   ordered_phases = list(order)  # topological order


Lattice
-------

``Lattice`` builds on the POSET utility and adds meet/join operations.

In compiler infrastructure, lattices are useful for dataflow domains,
type promotion rules, and fixed-point analyses.

.. code-block:: python

   from fhy_core import Lattice

   level = Lattice[str]()
   for item in ("bottom", "known", "top"):
       level.add_element(item)

   level.add_order("bottom", "known")
   level.add_order("known", "top")

   assert level.get_meet("known", "top") == "known"
   assert level.get_join("bottom", "known") == "known"

.. code-block:: python

   from fhy_core import Lattice

   level = Lattice[str]()
   for item in ("bottom", "known", "top"):
       level.add_element(item)
   level.add_order("bottom", "known")
   level.add_order("known", "top")

   assert level.is_lattice()


Dictionary Helpers
------------------

``invert_dict`` and ``invert_frozen_dict`` reverse key/value maps.

In compiler infrastructure, this is handy for reverse symbol lookup and mapping
between user and lowered representations.

.. code-block:: python

   from frozendict import frozendict

   from fhy_core import invert_dict, invert_frozen_dict

   assert invert_dict({"x": 1, "y": 2}) == {1: "x", 2: "y"}
   assert invert_frozen_dict(frozendict({"a": 7})) == {7: "a"}


Array and String Helpers
------------------------

``get_array_size_in_bits`` and ``format_comma_separated_list`` are convenience
helpers used in diagnostics and shape-aware utilities.

In compiler infrastructure, these are useful for memory estimation and readable
error/report formatting.

.. code-block:: python

   from fhy_core import format_comma_separated_list, get_array_size_in_bits

   size_bits = get_array_size_in_bits([64, 64], element_size_in_bits=32)
   msg = format_comma_separated_list(["i32", "f32", "index"])

   print(size_bits)  # 131072
   print(msg)        # i32, f32, index
