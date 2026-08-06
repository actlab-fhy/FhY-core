"""Public registration API for the expression registry.

Exposes ``register_function`` / ``register_native_function`` /
``register_native_constant``: the write-side surface that mutates the
process-wide registry. Read-side accessors and the underlying storage
live in :mod:`fhy_core.symbolic.expression.registry.storage`.

A function body may forward-reference a name that is not yet
registered; :func:`register_function` accepts this and defers the
body's result-sort check. Registering a function or native function
re-runs the deferred check for every entry waiting on that name, so an
incompatible forward-referenced body is eventually rejected rather
than silently tolerated forever.
"""

__all__ = [
    "register_function",
    "register_native_constant",
    "register_native_function",
]

from collections.abc import Callable, Sequence

from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import PassExecutionError

from ..core import Expression
from ..errors import EntryRegistrationError
from ..passes.body_type_checker import check_registered_function_body
from ..sort import FunctionSort
from .entries import NativeConstant, NativeFunction, RegisteredFunction
from .storage import (
    _clear_deferred_body_check,
    _insert_unique_entry,
    _pop_entries_deferred_on,
    _record_deferred_body_check,
    _remove_entry,
    get_registered_entry,
)


def _unwrap_body_check_failure(exc: PassExecutionError) -> EntryRegistrationError:
    """Return the ``EntryRegistrationError`` cause of a body-check failure.

    Args:
        exc: The ``PassExecutionError`` raised by
            :func:`check_registered_function_body`.

    Returns:
        The underlying ``EntryRegistrationError``.

    Raises:
        PassExecutionError: ``exc`` itself, unchanged, if its cause is
            not an ``EntryRegistrationError`` (an unexpected internal
            failure that callers should not try to reinterpret).

    """
    body_check_error = exc.__cause__
    if isinstance(body_check_error, EntryRegistrationError):
        return body_check_error
    raise exc


def _revalidate_entries_deferred_on(missing_name: str) -> None:
    """Re-run the deferred body check for every entry waiting on ``missing_name``.

    Called after ``missing_name`` is successfully registered. Recurses
    when a revalidated entry itself becomes fully checked, so a chain
    of forward references resolves within a single registration call.
    Every entry deferred on ``missing_name`` is revalidated even if an
    earlier one in the batch is incompatible, so one bad entry cannot
    hide a sibling's incompatibility from ever being checked again;
    the first failure encountered is the one raised.

    Raises:
        EntryRegistrationError: If a deferred entry's body is
            incompatible with its declared result sort now that
            ``missing_name`` is available. The message names that
            entry, not ``missing_name``.

    """
    first_error: EntryRegistrationError | None = None
    for entry_name in _pop_entries_deferred_on(missing_name):
        try:
            _revalidate_deferred_entry(entry_name, triggered_by=missing_name)
        except EntryRegistrationError as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _revalidate_deferred_entry(entry_name: str, *, triggered_by: str) -> None:
    """Re-run the body check for one entry previously deferred on ``triggered_by``.

    On failure, ``entry_name`` is removed from the registry: its body
    never validly type-checked, so the earlier registration that
    tolerated the forward reference is undone.

    Raises:
        EntryRegistrationError: If the body is incompatible with its
            declared result sort.

    """
    entry = get_registered_entry(entry_name)
    if not isinstance(entry, RegisteredFunction):
        raise RuntimeError(  # pragma: no cover
            "Internal invariant violated: only RegisteredFunction entries "
            f"can have a deferred body check, but {entry_name!r} is a "
            f"{type(entry).__name__}."
        )
    try:
        missing_name = check_registered_function_body(
            name=entry.name,
            parameters=entry.parameters,
            parameter_sorts=entry.parameter_sorts,
            result_sort=entry.result_sort,
            body=entry.body,
            resolve_call_target=get_registered_entry,
        )
    except PassExecutionError as exc:
        _remove_entry(entry_name)
        body_check_error = _unwrap_body_check_failure(exc)
        raise EntryRegistrationError(
            f"Function {entry_name!r} deferred its body check pending "
            f"{triggered_by!r}; now that {triggered_by!r} is registered, "
            f"{entry_name!r}'s body fails to type-check: {body_check_error}"
        ) from body_check_error
    if missing_name is not None:
        _record_deferred_body_check(entry_name, missing_name)
    else:
        _revalidate_entries_deferred_on(entry_name)


def register_function(
    name: str,
    parameters: Sequence[Identifier],
    parameter_sorts: Sequence[FunctionSort],
    result_sort: FunctionSort,
    body: Expression,
) -> RegisteredFunction:
    """Register a pure expression-bodied function.

    The function is inserted under a placeholder entry before the body
    is type-checked, so a self-recursive body can resolve its own call
    site against the declared sorts. If the body type-check fails, the
    placeholder is removed.

    A body that calls a function not yet registered defers its result-
    sort check rather than failing: the call is accepted on the
    strength of the eventual callee's declared sorts alone. Registering
    ``name`` also re-runs the deferred check for every entry that was
    itself waiting on ``name``; if such an entry's body turns out to be
    incompatible, this call raises ``EntryRegistrationError`` naming
    that entry, even though ``name`` registers successfully.

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
        EntryRegistrationError: On duplicate name, sort-arity
            mismatch, captured free identifier, or body-result sort
            mismatch. Also raised when registering ``name`` unblocks a
            previously-deferred function whose body turns out to be
            incompatible; the message names that deferred function,
            not ``name``.

    """
    parameter_tuple = tuple(parameters)
    parameter_sort_tuple = tuple(parameter_sorts)
    try:
        registered = RegisteredFunction(
            name=name,
            parameters=parameter_tuple,
            parameter_sorts=parameter_sort_tuple,
            result_sort=result_sort,
            body=body,
        )
    except ValueError as exc:
        raise EntryRegistrationError(str(exc)) from exc
    _insert_unique_entry(name, registered)
    try:
        missing_name = check_registered_function_body(
            name=name,
            parameters=parameter_tuple,
            parameter_sorts=parameter_sort_tuple,
            result_sort=result_sort,
            body=body,
            resolve_call_target=get_registered_entry,
        )
    except PassExecutionError as exc:
        _remove_entry(name)
        _clear_deferred_body_check(name)
        body_check_error = _unwrap_body_check_failure(exc)
        raise body_check_error from body_check_error.__cause__
    if missing_name is not None:
        _record_deferred_body_check(name, missing_name)
    _revalidate_entries_deferred_on(name)
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

    Registering ``name`` also re-runs the deferred body check for
    every entry that was waiting on ``name``; see
    :func:`register_function` for that contract.

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
            in that case. Also raised when registering ``name``
            unblocks a previously-deferred function whose body turns
            out to be incompatible; the message names that deferred
            function, not ``name``.

    Notes:
        Standard-library ``math`` implementations are backed by the
        platform's C math library; results may differ in their final
        bits across operating systems and CPU families.

    """
    parameter_sort_tuple = tuple(parameter_sorts)
    try:
        registered = NativeFunction(
            name=name,
            parameter_sorts=parameter_sort_tuple,
            result_sort=result_sort,
            implementation=implementation,
        )
    except ValueError as exc:
        raise EntryRegistrationError(str(exc)) from exc
    _insert_unique_entry(name, registered)
    _revalidate_entries_deferred_on(name)
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
        caveat as native function results. Unlike a forward-referenced
        call, a body identifier that names an unregistered constant is
        never deferred: it is indistinguishable from an unbound
        identifier until the constant is registered, so it fails
        immediately rather than waiting.

    """
    try:
        registered = NativeConstant(name=name, sort=sort, value=value)
    except ValueError as exc:
        raise EntryRegistrationError(str(exc)) from exc
    _insert_unique_entry(name, registered)
    return registered
