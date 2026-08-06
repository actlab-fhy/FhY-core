"""Value types for entries held by the expression registry.

This module defines the immutable dataclasses that represent the three
kinds of things the registry can hold:

- :class:`RegisteredFunction`: a pure function whose body is an
  expression tree.
- :class:`NativeFunction`: a pure function whose body is a Python
  callable.
- :class:`NativeConstant`: a named literal value.

"""

__all__ = [
    "CallTargetResolver",
    "NativeConstant",
    "NativeFunction",
    "RegisteredEntry",
    "RegisteredFunction",
]

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias

from fhy_core.identifier import Identifier
from fhy_core.term import (
    DerivedEquivalenceMixin,
    compared_as_binder,
    excluded_from_equivalence,
)

from ..core import Expression
from ..sort import FunctionSort, is_python_value_compatible_with_sort


def _reject_captured_free_identifiers(
    name: str,
    parameters: tuple[Identifier, ...],
    body: Expression,
) -> None:
    """Raise if ``body`` references a free identifier outside ``parameters``.

    An identifier whose name matches a registered ``NativeConstant`` is
    exempt: it resolves to the constant at type-check / evaluation time
    rather than being treated as captured.

    Raises:
        ValueError: If ``body`` references a free identifier that is
            neither a declared parameter nor a registered constant
            name.

    """
    # Deferred import: `storage` imports this module for its entry types,
    # so importing `storage` at module scope here would form a cycle.
    from .storage import _registered_constant_names  # noqa: PLC0415

    declared = set(parameters)
    captured = body.get_free_identifiers() - declared
    if not captured:
        return
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
    raise ValueError(
        f"RegisteredFunction {name!r}: body references identifiers not in "
        f"its parameters: {captured_names}."
    )


@dataclass(frozen=True)
class RegisteredFunction(DerivedEquivalenceMixin):
    """A named pure function over the expression IR.

    Structural and alpha equivalence are derived from the fields: ``name``
    is registry identity and is excluded; ``parameters`` is a binder whose
    bound identifiers scope over ``body`` (so two functions identical up to
    a consistent parameter rename are alpha-equivalent); ``parameter_sorts``
    and ``result_sort`` compare by value; ``body`` recurses.

    A call to another function is a reference by name
    (``CallExpression.function_name`` is a plain string, not an
    ``Identifier``), so a self-recursive or mutually-recursive body
    never appears in its own free identifiers and never trips the
    closure check below.

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

    Raises:
        ValueError: If ``name`` is empty; if ``parameter_sorts`` and
            ``parameters`` differ in length; or if ``body`` references
            a free identifier that is neither a declared parameter nor
            a registered constant name.

    """

    name: str = field(metadata=excluded_from_equivalence())
    parameters: tuple[Identifier, ...] = field(
        metadata=compared_as_binder(scopes_over=("body",))
    )
    parameter_sorts: tuple[FunctionSort, ...]
    result_sort: FunctionSort
    body: Expression

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RegisteredFunction.name must be non-empty.")
        if len(self.parameters) != len(self.parameter_sorts):
            raise ValueError(
                f"RegisteredFunction {self.name!r}: parameter_sorts length "
                f"({len(self.parameter_sorts)}) does not match parameters "
                f"length ({len(self.parameters)})."
            )
        _reject_captured_free_identifiers(self.name, self.parameters, self.body)


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
    """Raise if ``implementation`` cannot accept the declared arity.

    Uses :func:`inspect.signature` to count positional parameters. When
    the implementation is a C builtin that does not expose an
    inspectable signature, no check is performed.

    Raises:
        ValueError: If the implementation's inspectable signature
            cannot accept ``parameter_sort_count`` positional
            arguments.

    """
    try:
        signature = inspect.signature(implementation)
    except (ValueError, TypeError):
        return
    arity = _infer_positional_arity_range(signature)
    if arity.admits(parameter_sort_count):
        return
    if arity.accepts_unbounded:
        raise ValueError(
            f"NativeFunction {name!r}: implementation requires at least "
            f"{arity.minimum} positional argument(s), but parameter_sorts "
            f"has {parameter_sort_count}."
        )
    raise ValueError(
        f"NativeFunction {name!r}: parameter_sorts arity "
        f"{parameter_sort_count} does not match the implementation's "
        f"accepted positional-argument range "
        f"[{arity.minimum}, {arity.maximum}]."
    )


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
            values (``bool``, ``int``, ``float``) coerced from the
            literal arguments and returns a Python ``bool``, ``int``,
            or ``float`` whose runtime type is compatible with
            ``result_sort``.

    Raises:
        ValueError: If ``name`` is empty, or if ``implementation``'s
            inspectable signature cannot accept
            ``len(parameter_sorts)`` positional arguments. Some
            C-implemented callables (e.g. some ``math`` builtins) do
            not expose an inspectable signature; arity is not checked
            in that case.

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
    implementation: Callable[..., bool | int | float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("NativeFunction.name must be non-empty.")
        _check_native_implementation_arity(
            self.name, len(self.parameter_sorts), self.implementation
        )


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
    value: bool | int | float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("NativeConstant.name must be non-empty.")
        if not is_python_value_compatible_with_sort(self.value, self.sort):
            raise ValueError(
                f"NativeConstant {self.name!r}: value {self.value!r} is not "
                f"compatible with the declared sort {self.sort}."
            )


RegisteredEntry: TypeAlias = RegisteredFunction | NativeFunction | NativeConstant

CallTargetResolver: TypeAlias = Callable[[str], RegisteredEntry]
"""Resolve a call-site name to its registered entry, or raise ``EntryLookupError``."""
