API Reference
=============

This project intentionally re-exports many symbols from ``fhy_core.__init__`` for a
compact import surface. The module index below points to the primary namespaces used
in day-to-day compiler development.

Top-level API
-------------

- ``fhy_core``

Feature modules
---------------

- ``fhy_core.identifier``
- ``fhy_core.error``
- ``fhy_core.expression``
- ``fhy_core.constraint``
- ``fhy_core.param``
- ``fhy_core.types``
- ``fhy_core.symbol_table``
- ``fhy_core.serialization``
- ``fhy_core.pass_infrastructure``
- ``fhy_core.provenance``
- ``fhy_core.trait``
- ``fhy_core.logger``
- ``fhy_core.utils``

Quick introspection examples
----------------------------

.. code-block:: python

   import inspect

   import fhy_core
   from fhy_core import pass_infrastructure, trait

   print(fhy_core.__all__)  # exported public API
   print(inspect.getmembers(pass_infrastructure, inspect.isclass)[:5])
   print(inspect.getmembers(trait, inspect.isclass)[:5])
