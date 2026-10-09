"""Type checking of the expression IR against the core type system.

This subpackage joins the two halves of the language that are otherwise
independent: :mod:`fhy_core.symbolic.expression` owns expression syntax,
the entry registry, and the sort vocabulary, while :mod:`fhy_core.types`
owns the IR type system. Both are usable on their own; assigning IR types
to expressions is what needs them together, and that job lives here.

The subpackage exposes three layers:

- :mod:`~fhy_core.types.checking.sort_compatibility` translates a declared
  ``FunctionSort`` to and from concrete core data types.
- :mod:`~fhy_core.types.checking.type_checker` bidirectionally synthesizes
  and checks the type of an expression.
- :mod:`~fhy_core.types.checking.body_type_checker` checks that a
  registered function's body matches its declared result sort, one
  function at a time or across the whole registry at once.

Body checking is explicit: registering a function stores it, and
:func:`check_registered_function_body` is called by whoever wants the
answer. Because registration never rejects a body, a body is free to
call a function registered later; the whole-registry sweep,
:func:`check_all_registered_function_bodies`, is what holds such a body
to its declared result sort once every name it uses exists.
"""

__all__ = [
    "ExpressionTypeChecker",
    "RegisteredFunctionBodyTypeChecker",
    "check_all_registered_function_bodies",
    "check_expression_type",
    "check_registered_function_body",
    "get_core_data_type_from_literal_type",
    "get_result_core_data_type_for_sort",
    "is_core_data_type_compatible_with_sort",
    "synthesize_expression_type",
]

from .body_type_checker import (
    RegisteredFunctionBodyTypeChecker,
    check_all_registered_function_bodies,
    check_registered_function_body,
)
from .sort_compatibility import (
    get_result_core_data_type_for_sort,
    is_core_data_type_compatible_with_sort,
)
from .type_checker import (
    ExpressionTypeChecker,
    check_expression_type,
    get_core_data_type_from_literal_type,
    synthesize_expression_type,
)
