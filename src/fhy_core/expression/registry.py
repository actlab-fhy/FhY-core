"""Process-wide registry of expression-IR entries.

The registry holds three kinds of entries:

- :class:`RegisteredFunction`: a pure function whose body is an
  expression in the IR. Inlining substitutes the parameter identifiers
  with the call's argument expressions.
- :class:`NativeFunction`: a pure function whose body is a Python
  callable. The evaluator folds literal-argument calls; the inliner
  passes them through.
- :class:`NativeConstant`: a named literal value. The evaluator
  substitutes identifier references whose name matches the constant.

All three kinds expose declared sorts that drive the call-site type
checker, decoupling type inference from body inspection.

The registry is process-wide. Tests that need isolation should request
the ``function_registry_snapshot`` fixture defined in
``tests/expression/conftest.py`` rather than mutating the registry
directly.
"""

__all__ = [
    "FunctionLookupError",
    "FunctionRegistrationError",
    "NativeConstant",
    "NativeFunction",
    "RegisteredEntry",
    "RegisteredFunction",
    "get_registered_function",
    "get_registered_functions",
    "is_function_registered",
    "register_function",
    "register_native_constant",
    "register_native_function",
    "set_registry_state_for_tests",
]

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import TypeAlias

from frozendict import frozendict

from fhy_core.identifier import Identifier

from .core import Expression
from .errors import FunctionLookupError, FunctionRegistrationError
from .passes.basic import collect_identifiers
from .passes.body_type_checker import RegisteredFunctionBodyTypeChecker
from .sort import FunctionSort, is_python_value_compatible_with_sort


@dataclass(frozen=True)
class RegisteredFunction:
    """A named pure function over the expression IR.

    Attributes:
        name: Registry key. Used at call sites and in error messages.
        parameters: Ordered formal-parameter identifiers. Inlining
            substitutes these with the call's argument expressions.
        parameter_sorts: Per-parameter declared sort. Has the same
            length as ``parameters``.
        result_sort: Declared result sort. The call-site type checker
            uses this directly, without re-walking the body.
        body: Expression tree using the parameter identifiers. Free
            identifiers are a subset of ``parameters`` plus any
            identifiers whose name matches a registered
            ``NativeConstant``.

    """

    name: str
    parameters: tuple[Identifier, ...]
    parameter_sorts: tuple[FunctionSort, ...]
    result_sort: FunctionSort
    body: Expression


@dataclass(frozen=True)
class NativeFunction:
    """A function whose body is a Python callable.

    Native functions cannot be inlined: they have no expression body.
    They are folded to a :class:`LiteralExpression` by
    :func:`evaluate_expression` when every argument is a literal;
    otherwise they remain as :class:`CallExpression` nodes in the tree
    and pass through the inliner untouched.

    Attributes:
        name: Registry key.
        parameter_sorts: Per-parameter declared sort. Arity is
            ``len(parameter_sorts)``.
        result_sort: Declared result sort.
        implementation: Python callable. Receives positional Python
            values (``bool``, ``int``, ``float``, ``complex``) coerced
            from the literal arguments and returns a Python ``bool``,
            ``int``, ``float``, or ``complex`` whose runtime type is
            compatible with ``result_sort``.

    Notes:
        Numerical results from ``math``-backed implementations follow
        the platform's C math library and may differ in their final
        bits across operating systems and CPU families. Callers
        requiring exact cross-platform reproducibility must not rely
        on the low-order bits of native results.
    """

    name: str
    parameter_sorts: tuple[FunctionSort, ...]
    result_sort: FunctionSort
    implementation: Callable[..., bool | int | float | complex]


@dataclass(frozen=True)
class NativeConstant:
    """A named constant whose value is a Python literal.

    Constants are referenced in an expression tree as an
    :class:`IdentifierExpression` whose identifier name matches
    ``name``. :func:`evaluate_expression` substitutes such references
    with ``LiteralExpression(value)``; the type checker resolves them
    via the registry lookup when the identifier is not bound locally.

    Attributes:
        name: Registry key.
        sort: Declared sort.
        value: Literal Python value, compatible with ``sort`` per
            :func:`is_python_value_compatible_with_sort`.

    Notes:
        Constants seeded from ``math`` (``math.pi``, ``math.e``,
        ``math.inf``, ``math.nan``) carry the same platform-bit caveat
        as native function results.
    """

    name: str
    sort: FunctionSort
    value: bool | int | float | complex


RegisteredEntry: TypeAlias = RegisteredFunction | NativeFunction | NativeConstant


_REGISTRY: dict[str, RegisteredEntry] = {}
_REGISTRY_LOCK = Lock()


def _insert_unique_entry(name: str, entry: RegisteredEntry) -> None:
    """Insert ``entry`` under ``name`` if the name is free; otherwise raise."""
    with _REGISTRY_LOCK:
        if name in _REGISTRY:
            raise FunctionRegistrationError(f"A name is already registered: {name!r}.")
        _REGISTRY[name] = entry


def _remove_entry(name: str) -> None:
    """Remove ``name`` from the registry if present."""
    with _REGISTRY_LOCK:
        _REGISTRY.pop(name, None)


def _check_parameter_sort_arity(
    name: str,
    parameters: tuple[Identifier, ...],
    parameter_sorts: tuple[FunctionSort, ...],
) -> None:
    if len(parameters) != len(parameter_sorts):
        raise FunctionRegistrationError(
            f"Function {name!r}: parameter_sorts length "
            f"({len(parameter_sorts)}) does not match parameters length "
            f"({len(parameters)})."
        )


def _registered_constant_names() -> set[str]:
    """Return the set of names that resolve to a :class:`NativeConstant`."""
    with _REGISTRY_LOCK:
        return {
            name
            for name, entry in _REGISTRY.items()
            if isinstance(entry, NativeConstant)
        }


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
    raise FunctionRegistrationError(
        f"Function {name!r} body references identifiers not in its "
        f"parameters: {captured_names}."
    )


def _check_body_against_result_sort(
    name: str,
    parameters: tuple[Identifier, ...],
    parameter_sorts: tuple[FunctionSort, ...],
    result_sort: FunctionSort,
    body: Expression,
) -> None:
    """Run ``RegisteredFunctionBodyTypeChecker`` against ``body``."""
    RegisteredFunctionBodyTypeChecker(
        name=name,
        parameters=parameters,
        parameter_sorts=parameter_sorts,
        result_sort=result_sort,
    ).check(body)


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
        FunctionRegistrationError: On duplicate name, sort-arity
            mismatch, captured free identifier, or body-result sort
            mismatch.

    """
    parameter_tuple = tuple(parameters)
    parameter_sort_tuple = tuple(parameter_sorts)
    _check_parameter_sort_arity(name, parameter_tuple, parameter_sort_tuple)
    _reject_captured_free_identifiers(name, parameter_tuple, body)

    registered = RegisteredFunction(
        name=name,
        parameters=parameter_tuple,
        parameter_sorts=parameter_sort_tuple,
        result_sort=result_sort,
        body=body,
    )
    _insert_unique_entry(name, registered)
    try:
        _check_body_against_result_sort(
            name, parameter_tuple, parameter_sort_tuple, result_sort, body
        )
    except FunctionRegistrationError:
        _remove_entry(name)
        raise
    return registered


def register_native_function(
    name: str,
    parameter_sorts: Sequence[FunctionSort],
    result_sort: FunctionSort,
    implementation: Callable[..., bool | int | float | complex],
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

    Notes:
        Standard-library ``math`` implementations are backed by the
        platform's C math library; results may differ in their final
        bits across operating systems and CPU families.

    Raises:
        FunctionRegistrationError: On duplicate name.

    """
    registered = NativeFunction(
        name=name,
        parameter_sorts=tuple(parameter_sorts),
        result_sort=result_sort,
        implementation=implementation,
    )
    _insert_unique_entry(name, registered)
    return registered


def register_native_constant(
    name: str,
    sort: FunctionSort,
    value: bool | int | float | complex,
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

    Notes:
        Constants seeded from ``math`` carry the same platform-bit
        caveat as native function results.

    Raises:
        FunctionRegistrationError: On duplicate name or sort-value
            incompatibility.

    """
    if not is_python_value_compatible_with_sort(value, sort):
        raise FunctionRegistrationError(
            f"Constant {name!r}: value {value!r} is not compatible with "
            f"the declared sort {sort}."
        )
    registered = NativeConstant(name=name, sort=sort, value=value)
    _insert_unique_entry(name, registered)
    return registered


def get_registered_function(name: str) -> RegisteredEntry:
    """Return the entry registered under ``name``.

    The return type widens to ``RegisteredEntry`` (the union of
    :class:`RegisteredFunction`, :class:`NativeFunction`, and
    :class:`NativeConstant`). Callers that need to distinguish kinds
    use ``isinstance``.

    Args:
        name: Registry key.

    Returns:
        The stored entry.

    Raises:
        FunctionLookupError: If no entry is registered under ``name``.

    """
    with _REGISTRY_LOCK:
        registered = _REGISTRY.get(name)
    if registered is None:
        raise FunctionLookupError(f"No entry is registered under the name {name!r}.")
    return registered


def get_registered_functions() -> Mapping[str, RegisteredEntry]:
    """Return an immutable snapshot of the current registry."""
    with _REGISTRY_LOCK:
        return frozendict(_REGISTRY)


def is_function_registered(name: str) -> bool:
    """Return whether any entry is registered under ``name``."""
    with _REGISTRY_LOCK:
        return name in _REGISTRY


def set_registry_state_for_tests(
    state: Mapping[str, RegisteredEntry],
) -> None:
    """Replace the registry contents with ``state``.

    Intended for the test-isolation fixture in
    ``tests/expression/conftest.py``: the fixture snapshots the
    registry before each test and calls this hook to restore the
    snapshot after the test runs. The leading underscore marks this
    as private; it should not be used outside test infrastructure.
    """
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
        _REGISTRY.update(state)
