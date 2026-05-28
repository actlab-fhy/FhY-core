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
    "NativeConstant",
    "NativeFunction",
    "RegisteredEntry",
    "RegisteredFunction",
]

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from fhy_core.identifier import Identifier
from fhy_core.trait import AlphaEquivalenceMixin, AlphaRenaming

from ..core import Expression
from ..sort import FunctionSort, is_python_value_compatible_with_sort


@dataclass(frozen=True)
class RegisteredFunction(AlphaEquivalenceMixin):
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

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RegisteredFunction.name must be non-empty.")
        if len(self.parameters) != len(self.parameter_sorts):
            raise ValueError(
                f"RegisteredFunction {self.name!r}: parameter_sorts length "
                f"({len(self.parameter_sorts)}) does not match parameters "
                f"length ({len(self.parameters)})."
            )

    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> bool:
        if not isinstance(other, RegisteredFunction):
            return False
        elif self.parameter_sorts != other.parameter_sorts:
            return False
        elif self.result_sort != other.result_sort:
            return False
        elif len(self.parameters) != len(other.parameters):
            return False
        else:
            extended = renaming.extend(dict(zip(self.parameters, other.parameters)))
            return self.body.is_alpha_equivalent_under(other.body, extended)


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
