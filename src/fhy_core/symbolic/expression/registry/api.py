"""Public registration API for the expression registry.

Exposes ``register_function`` / ``register_native_function`` /
``register_native_constant``: the write-side surface that mutates the
process-wide registry. Read-side accessors and the underlying storage
live in :mod:`fhy_core.symbolic.expression.registry.storage`.

Registration records an entry; it does not type-check it. Checking a
function body against its declared sorts needs the IR type system, which
sits above this package in the dependency graph, so callers who want that
answer ask the type-checking layer for it explicitly. A body is therefore
free to call a function registered later: nothing here inspects the call
target, and the sweep in the type-checking layer runs once registration
is complete.
"""

__all__ = [
    "register_function",
    "register_native_constant",
    "register_native_function",
]

from collections.abc import Callable, Sequence

from fhy_core.identifier import Identifier

from ..core import Expression
from ..errors import EntryRegistrationError
from ..sort import FunctionSort
from .entries import NativeConstant, NativeFunction, RegisteredFunction
from .storage import _insert_unique_entry


def register_function(
    name: str,
    parameters: Sequence[Identifier],
    parameter_sorts: Sequence[FunctionSort],
    result_sort: FunctionSort,
    body: Expression,
) -> RegisteredFunction:
    """Register a pure expression-bodied function.

    The body is stored as given. Whether it synthesizes a type
    compatible with ``result_sort`` is a separate question, answered on
    demand by the body checker in the type-checking layer above. A body
    that calls a function not yet registered is accepted unconditionally
    here, on the strength of the eventual callee's declared sorts alone.

    Args:
        name: Unique registry key.
        parameters: Positional parameter identifiers, in declared order.
        parameter_sorts: Per-parameter declared sort, in the same order
            as ``parameters``.
        result_sort: Declared result sort.
        body: Body expression. Free identifiers must be a subset of
            ``parameters`` plus any identifiers whose name matches a
            registered :class:`NativeConstant`.

    Returns:
        The newly stored ``RegisteredFunction``.

    Raises:
        EntryRegistrationError: On duplicate name, sort-arity mismatch,
            or captured free identifier.

    """
    try:
        registered = RegisteredFunction(
            name=name,
            parameters=tuple(parameters),
            parameter_sorts=tuple(parameter_sorts),
            result_sort=result_sort,
            body=body,
        )
    except ValueError as exc:
        raise EntryRegistrationError(str(exc)) from exc
    _insert_unique_entry(name, registered)
    return registered


def register_native_function(
    name: str,
    parameter_sorts: Sequence[FunctionSort],
    result_sort: FunctionSort,
    implementation: Callable[..., bool | int | float],
) -> NativeFunction:
    """Register a native (Python-backed) function.

    Native functions are evaluated by :func:`evaluate_expression` when
    their call site has all-literal arguments. They are passed through
    the inliner untouched. Type checking uses the declared
    ``parameter_sorts`` / ``result_sort``; the implementation is not
    consulted at type-check time.

    Args:
        name: Unique registry key.
        parameter_sorts: Per-parameter declared sort. Arity is
            ``len(parameter_sorts)``.
        result_sort: Declared result sort.
        implementation: Python callable bound to the registry entry.

    Returns:
        The newly stored ``NativeFunction``.

    Raises:
        EntryRegistrationError: On duplicate name, empty name, or when
            the implementation's inspectable signature cannot accept
            ``len(parameter_sorts)`` positional arguments. Some
            C-implemented callables (e.g. some ``math`` builtins) do
            not expose an inspectable signature; arity is not checked
            in that case.

    Notes:
        Standard-library ``math`` implementations are backed by the
        platform's C math library; results may differ in their final
        bits across operating systems and CPU families.

    """
    try:
        registered = NativeFunction(
            name=name,
            parameter_sorts=tuple(parameter_sorts),
            result_sort=result_sort,
            implementation=implementation,
        )
    except ValueError as exc:
        raise EntryRegistrationError(str(exc)) from exc
    _insert_unique_entry(name, registered)
    return registered


def register_native_constant(
    name: str,
    sort: FunctionSort,
    value: bool | int | float,
) -> NativeConstant:
    """Register a named constant in the registry.

    Args:
        name: Unique registry key. An ``IdentifierExpression`` whose
            identifier name matches ``name`` is treated as a reference
            to this constant by the type checker and the evaluator.
        sort: Declared sort of the constant.
        value: Literal Python value. Must satisfy
            :func:`is_python_value_compatible_with_sort`.

    Returns:
        The newly stored ``NativeConstant``.

    Raises:
        EntryRegistrationError: On duplicate name, empty name, or
            sort-value incompatibility.

    Notes:
        Constants seeded from ``math`` carry the same platform-bit
        caveat as native function results.

    """
    try:
        registered = NativeConstant(name=name, sort=sort, value=value)
    except ValueError as exc:
        raise EntryRegistrationError(str(exc)) from exc
    _insert_unique_entry(name, registered)
    return registered
