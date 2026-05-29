"""Public registration API for the expression registry.

Exposes ``register_function`` / ``register_native_function`` /
``register_native_constant``: the write-side surface that mutates the
process-wide registry. Read-side accessors and the underlying storage
live in :mod:`fhy_core.expression.registry.storage`.
"""

__all__ = [
    "register_function",
    "register_native_constant",
    "register_native_function",
]

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import PassExecutionError

from ..core import Expression
from ..errors import EntryRegistrationError
from ..passes.basic import collect_identifiers
from ..passes.body_type_checker import check_registered_function_body
from ..sort import FunctionSort
from .entries import NativeConstant, NativeFunction, RegisteredFunction
from .storage import (
    _insert_unique_entry,
    _registered_constant_names,
    _remove_entry,
    get_registered_entry,
)

_POSITIONAL_PARAMETER_KINDS = frozenset(
    {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
)


@dataclass(frozen=True)
class _PositionalArityRange:
    """Inferred positional-argument arity range of a native implementation."""

    minimum: int
    maximum: int
    accepts_unbounded: bool

    def admits(self, count: int) -> bool:
        if self.accepts_unbounded:
            return count >= self.minimum
        return self.minimum <= count <= self.maximum


def _infer_positional_arity_range(
    signature: inspect.Signature,
) -> _PositionalArityRange:
    """Return the positional-argument arity range admitted by ``signature``."""
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in _POSITIONAL_PARAMETER_KINDS
    ]
    required_count = sum(
        1 for parameter in positional if parameter.default is inspect.Parameter.empty
    )
    accepts_unbounded = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    return _PositionalArityRange(
        minimum=required_count,
        maximum=len(positional),
        accepts_unbounded=accepts_unbounded,
    )


def _check_native_implementation_arity(
    name: str,
    parameter_sort_count: int,
    implementation: Callable[..., bool | int | float],
) -> None:
    """Best-effort check that ``implementation`` accepts the declared arity.

    Uses :func:`inspect.signature` to count positional parameters. When
    the implementation is a C builtin that does not expose an
    inspectable signature, no check is performed.
    """
    try:
        signature = inspect.signature(implementation)
    except (ValueError, TypeError):
        return
    arity = _infer_positional_arity_range(signature)
    if arity.admits(parameter_sort_count):
        return
    if arity.accepts_unbounded:
        raise EntryRegistrationError(
            f"NativeFunction {name!r}: implementation requires at least "
            f"{arity.minimum} positional argument(s), but parameter_sorts "
            f"has {parameter_sort_count}."
        )
    raise EntryRegistrationError(
        f"NativeFunction {name!r}: parameter_sorts arity "
        f"{parameter_sort_count} does not match the implementation's "
        f"accepted positional-argument range "
        f"[{arity.minimum}, {arity.maximum}]."
    )


def _reject_captured_free_identifiers(
    name: str,
    parameters: tuple[Identifier, ...],
    body: Expression,
) -> None:
    declared = set(parameters)
    captured = collect_identifiers(body) - declared
    if not captured:
        return
    # Any captured identifier whose name matches a registered constant
    # is permitted: it resolves to the constant at type-check / eval time.
    constant_names = _registered_constant_names()
    truly_captured = {
        identifier
        for identifier in captured
        if identifier.name_hint not in constant_names
    }
    if not truly_captured:
        return
    captured_names = ", ".join(
        sorted(identifier.name_hint for identifier in truly_captured)
    )
    raise EntryRegistrationError(
        f"Function {name!r} body references identifiers not in its "
        f"parameters: {captured_names}."
    )


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
            mismatch.

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
    _reject_captured_free_identifiers(name, parameter_tuple, body)
    _insert_unique_entry(name, registered)
    try:
        check_registered_function_body(
            name=name,
            parameters=parameter_tuple,
            parameter_sorts=parameter_sort_tuple,
            result_sort=result_sort,
            body=body,
            resolve_call_target=get_registered_entry,
        )
    except PassExecutionError as exc:
        _remove_entry(name)
        body_check_error = exc.__cause__
        if isinstance(body_check_error, EntryRegistrationError):
            # pylint: disable-next=raising-non-exception
            raise body_check_error from body_check_error.__cause__
        raise
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
    parameter_sort_tuple = tuple(parameter_sorts)
    _check_native_implementation_arity(name, len(parameter_sort_tuple), implementation)
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
