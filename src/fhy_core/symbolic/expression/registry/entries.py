"""The entry classes of the expression registry.

The registry holds three kinds of entries:

- :class:`RegisteredFunction`: a pure function whose body is an
  expression tree.
- :class:`NativeFunction`: a pure function whose body is a Python
  callable.
- :class:`NativeConstant`: a named literal value.

The classes are backed by the Rust implementation (``fhy_core._rs``), over
the core's ``FunctionDefinition``, ``NativeFunction`` and
``NativeConstant``, or, for a built-in's entry, over its item of the core's
catalogue. Each keeps its field objects, so ``entry.body is body`` holds;
entries are frozen, and mutating one raises ``FrozenMutationError``. They
compare, hash and print by their fields, as the dataclasses they replace
did. A user entry pickles as a call of its class with its fields, a
built-in's entry as the built-in itself.

Building an entry checks only the entry itself. Which identifiers a
function's body may refer to depends on the registry, which checks them
when the function is registered.
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
from dataclasses import dataclass
from typing import TypeAlias

from fhy_core import _rs
from fhy_core.traits import FrozenMixin


class RegisteredFunction(_rs.RegisteredFunction):
    """A named pure function over the expression IR.

    ``RegisteredFunction(name, parameters, parameter_sorts, result_sort,
    body)`` builds an entry without registering it.

    Structural and alpha equivalence compare the functions as binder
    terms: ``name`` is excluded; ``parameters`` bind the identifiers of
    ``body``, so two functions identical up to a consistent parameter
    rename are alpha-equivalent; ``parameter_sorts`` and ``result_sort``
    compare by value. A pairing of the parameters that is not injective is
    no renaming. Structural equivalence requires the same parameters.

    A call to another function is a reference by name
    (``CallExpression.function_name``), so a recursive body is accepted
    here and refused only when it is inlined.

    Attributes:
        name: Registry key. Used at call sites and in error messages.
        parameters: Ordered formal-parameter identifiers. Inlining
            substitutes these with the call's argument expressions.
        parameter_sorts: Per-parameter declared sort, one per parameter.
        result_sort: Declared result sort. The call-site type checker
            uses this directly, without re-walking the body.
        body: Expression tree over the parameters.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``name`` is empty or a built-in function's name, if
            ``parameter_sorts`` and ``parameters`` differ in length, or if
            a parameter is repeated, with the core's text.

    """

    __slots__ = ()
    __match_args__ = ("name", "parameters", "parameter_sorts", "result_sort", "body")


FrozenMixin.register(RegisteredFunction)
RegisteredFunction._register_public_class()


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


class NativeFunction(_rs.NativeFunction):
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
        TypeError: If an argument has the wrong type, a non-callable
            ``implementation`` included.
        ValueError: If ``name`` is empty or a built-in function's name,
            or if ``implementation``'s inspectable signature cannot accept
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

    __slots__ = ()
    __match_args__ = ("name", "parameter_sorts", "result_sort", "implementation")


FrozenMixin.register(NativeFunction)
NativeFunction._register_public_class()


class NativeConstant(_rs.NativeConstant):
    """A named constant whose value is a Python literal.

    Registration mints one canonical :class:`Identifier` for the
    constant, retrievable with :func:`get_native_constant_identifier`; a
    built-in constant's identifier has a fixed reserved id (``pi`` 48,
    ``e`` 49, ``inf`` 50, ``nan`` 51), the same in every process. A
    constant is referenced in an expression tree as an
    :class:`IdentifierExpression` wrapping that identifier:
    :func:`evaluate_expression` substitutes such references with
    ``LiteralExpression(value)``, and the type checker resolves them from
    the registry when the identifier is not bound locally. Recognition is
    by identifier identity, so an identifier that merely shares ``name`` as
    its ``name_hint`` is an ordinary free variable that callers may bind to
    whatever they like.

    Attributes:
        name: Registry key.
        sort: Declared sort.
        value: Literal Python value, compatible with ``sort`` per
            :func:`is_python_value_compatible_with_sort`.

    Raises:
        TypeError: If an argument has the wrong type, a value that is not
            a ``bool``, an ``int`` or a ``float`` included.
        ValueError: If ``name`` is empty or a built-in function's name, or
            if ``value`` is not compatible with ``sort``, with the core's
            text.

    Notes:
        Constants seeded from ``math`` (``math.pi``, ``math.e``,
        ``math.inf``, ``math.nan``) carry the same platform-bit caveat
        as native function results.
    """

    __slots__ = ()
    __match_args__ = ("name", "sort", "value")


FrozenMixin.register(NativeConstant)
NativeConstant._register_public_class()


RegisteredEntry: TypeAlias = RegisteredFunction | NativeFunction | NativeConstant

CallTargetResolver: TypeAlias = Callable[[str], RegisteredEntry]
"""Resolve a call-site name to its registered entry, or raise ``EntryLookupError``."""
