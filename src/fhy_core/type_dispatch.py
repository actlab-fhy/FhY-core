"""Extensible dispatchers for the core type system.

This module provides the four extension points (six dispatchers) that the
core type system exposes for downstream packages:

- ``bind_template(pattern, actual, env)``: one-directional binding for
  template patterns against concrete types.
- ``substitute_template(t, env)``: replace template placeholders in a type
  with bound values.
- ``unify(expected, actual, env)``: bidirectional unification with
  placeholder binding allowed on either side.
- ``structural_eq(a, b)``: pure structural comparison.
- ``bind_data_template(pattern, actual, env)``: data-type-tier binding.
- ``substitute_data_template(dt, env)``: data-type-tier substitution.

All dispatchers are :func:`functools.singledispatch` functions keyed by the
concrete class of their first argument. Downstream packages defining new
``Type`` or ``DataType`` subclasses register handlers against these
dispatchers from wherever the class is defined; no modification to
``fhy_core`` is required.
"""

__all__ = [
    "TypeUnificationEnv",
    "bind_data_template",
    "bind_template",
    "structural_eq",
    "substitute_data_template",
    "substitute_template",
    "unify",
]

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import singledispatch
from types import EllipsisType
from typing import Any

from frozendict import frozendict

from .expression.core import (
    BinaryExpression,
    Expression,
    IdentifierExpression,
    UnaryExpression,
)
from .identifier import Identifier
from .trait import FrozenMixin, StructuralEquivalenceMixin, VerificationError
from .types import (
    DataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TupleType,
    Type,
)


@dataclass(frozen=True)
class TypeUnificationEnv(FrozenMixin, StructuralEquivalenceMixin):
    """Carrier for binding state used during type-system operations.

    Three independent binding tables:

    - ``data_type_bindings``: maps a ``TemplateDataType`` placeholder name
      (the placeholder's ``Identifier.name_hint``) to a concrete
      ``DataType``. Populated when a ``TemplateDataType`` in a pattern is
      matched against a concrete data type in an actual.
    - ``type_bindings``: maps a full-type-template placeholder name to a
      concrete ``Type``. Populated when a ``NumericalType`` whose data type
      is a ``TemplateDataType`` and whose shape is a wildcard (``[...]``)
      is matched against a concrete type — the entire actual type is
      captured, not just its data-type part.
    - ``expression_bindings``: maps an ``Identifier`` (a shape-variable
      placeholder used as the head of an ``IdentifierExpression``) to a
      concrete ``Expression``. Populated when shape elements are unified
      pairwise and one side is a placeholder.

    The env is frozen. Updates produce new envs via the ``with_*`` helpers;
    the dispatchers thread the env through recursive calls and return
    updated envs.

    Subclasses may add layer-specific extras (e.g. per-call-site state in
    a type-inferencer) without changing the dispatcher signatures.
    """

    data_type_bindings: "frozendict[str, DataType]" = field(default_factory=frozendict)
    type_bindings: "frozendict[str, Type]" = field(default_factory=frozendict)
    expression_bindings: "frozendict[Identifier, Expression]" = field(
        default_factory=frozendict
    )

    @classmethod
    def empty(cls) -> "TypeUnificationEnv":
        """Construct an empty env with no bindings."""
        return cls()

    def with_data_type_binding(
        self, name: str, value: DataType
    ) -> "TypeUnificationEnv":
        """Return a new env with an additional data-type binding."""
        new_bindings = frozendict({**self.data_type_bindings, name: value})
        return replace(self, data_type_bindings=new_bindings)

    def with_type_binding(self, name: str, value: Type) -> "TypeUnificationEnv":
        """Return a new env with an additional full-type binding."""
        new_bindings = frozendict({**self.type_bindings, name: value})
        return replace(self, type_bindings=new_bindings)

    def with_expression_binding(
        self, name: Identifier, value: Expression
    ) -> "TypeUnificationEnv":
        """Return a new env with an additional expression binding."""
        new_bindings = frozendict({**self.expression_bindings, name: value})
        return replace(self, expression_bindings=new_bindings)

    def get_data_type_binding(self, name: str) -> DataType | None:
        """Return the bound data type for a placeholder name, or ``None``."""
        return self.data_type_bindings.get(name)

    def get_type_binding(self, name: str) -> Type | None:
        """Return the bound full type for a placeholder name, or ``None``."""
        return self.type_bindings.get(name)

    def get_expression_binding(self, name: Identifier) -> Expression | None:
        """Return the bound expression for an identifier, or ``None``."""
        return self.expression_bindings.get(name)

    def is_structurally_equivalent(self, other: object) -> bool:
        if not isinstance(other, TypeUnificationEnv):
            return False
        if not _data_type_bindings_equivalent(
            self.data_type_bindings, other.data_type_bindings
        ):
            return False
        if not _type_bindings_equivalent(self.type_bindings, other.type_bindings):
            return False
        return _expression_bindings_equivalent(
            self.expression_bindings, other.expression_bindings
        )


def _data_type_bindings_equivalent(
    left: "frozendict[str, DataType]", right: "frozendict[str, DataType]"
) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for name, value in left.items():
        if not value.is_structurally_equivalent(right[name]):
            return False
    return True


def _type_bindings_equivalent(
    left: "frozendict[str, Type]", right: "frozendict[str, Type]"
) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for name, value in left.items():
        if not value.is_structurally_equivalent(right[name]):
            return False
    return True


def _expression_bindings_equivalent(
    left: "frozendict[Identifier, Expression]",
    right: "frozendict[Identifier, Expression]",
) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for identifier, value in left.items():
        if not value.is_structurally_equivalent(right[identifier]):
            return False
    return True


def _is_wildcard_shape(shape: Sequence[Expression | EllipsisType]) -> bool:
    """Return ``True`` for the single-element ``[...]`` wildcard shape."""
    return len(shape) == 1 and shape[0] is Ellipsis


def _resolve_expression(expression: Expression, env: TypeUnificationEnv) -> Expression:
    """Follow chains of expression bindings until reaching a non-bound node."""
    seen: set[Identifier] = set()
    current: Expression = expression
    while isinstance(current, IdentifierExpression):
        if current.identifier in seen:
            return current
        bound = env.get_expression_binding(current.identifier)
        if bound is None:
            return current
        seen.add(current.identifier)
        current = bound
    return current


def _identifier_occurs_in(identifier: Identifier, expression: Expression) -> bool:
    """Return whether ``identifier`` syntactically appears inside ``expression``."""
    if isinstance(expression, IdentifierExpression):
        return expression.identifier == identifier
    if isinstance(expression, BinaryExpression):
        return _identifier_occurs_in(
            identifier, expression.left
        ) or _identifier_occurs_in(identifier, expression.right)
    if isinstance(expression, UnaryExpression):
        return _identifier_occurs_in(identifier, expression.operand)
    return False


def _substitute_expression(
    expression: Expression,
    env: TypeUnificationEnv,
    visited: frozenset[Identifier] = frozenset(),
) -> Expression:
    """Recursively substitute identifier expressions in ``expression``."""
    if isinstance(expression, IdentifierExpression):
        if expression.identifier in visited:
            return expression
        bound = env.get_expression_binding(expression.identifier)
        if bound is None:
            return expression
        return _substitute_expression(bound, env, visited | {expression.identifier})
    if isinstance(expression, BinaryExpression):
        new_left = _substitute_expression(expression.left, env, visited)
        new_right = _substitute_expression(expression.right, env, visited)
        if new_left is expression.left and new_right is expression.right:
            return expression
        return BinaryExpression(expression.operation, new_left, new_right)
    if isinstance(expression, UnaryExpression):
        new_operand = _substitute_expression(expression.operand, env, visited)
        if new_operand is expression.operand:
            return expression
        return UnaryExpression(expression.operation, new_operand)
    return expression


def _bind_shape_dimension(
    pattern_dim: Expression,
    actual_dim: Expression,
    env: TypeUnificationEnv,
) -> TypeUnificationEnv:
    """Record an expression binding for a single shape dimension during binding."""
    if isinstance(pattern_dim, IdentifierExpression):
        existing = env.get_expression_binding(pattern_dim.identifier)
        if existing is None:
            return env.with_expression_binding(pattern_dim.identifier, actual_dim)
        if not existing.is_structurally_equivalent(actual_dim):
            raise VerificationError(
                f"Conflicting binding for shape variable "
                f"{pattern_dim.identifier!r}: {existing!r} vs {actual_dim!r}."
            )
        return env
    if not pattern_dim.is_structurally_equivalent(actual_dim):
        raise VerificationError(
            f"Shape dimension mismatch: {pattern_dim!r} vs {actual_dim!r}."
        )
    return env


def _bind_index_expression(
    pattern_expression: Expression,
    actual_expression: Expression,
    env: TypeUnificationEnv,
) -> TypeUnificationEnv:
    """Record an expression binding for an index-type bound during binding."""
    return _bind_shape_dimension(pattern_expression, actual_expression, env)


def _unify_expressions(
    expression_a: Expression,
    expression_b: Expression,
    env: TypeUnificationEnv,
) -> tuple[Expression, TypeUnificationEnv]:
    """Bidirectionally unify two expressions, with occurs-check."""
    resolved_a = _resolve_expression(expression_a, env)
    resolved_b = _resolve_expression(expression_b, env)

    if (
        isinstance(resolved_a, IdentifierExpression)
        and isinstance(resolved_b, IdentifierExpression)
        and resolved_a.identifier == resolved_b.identifier
    ):
        return resolved_a, env

    if isinstance(resolved_a, IdentifierExpression):
        if _identifier_occurs_in(resolved_a.identifier, resolved_b):
            raise VerificationError(
                f"Occurs check failed: identifier {resolved_a.identifier!r} "
                f"appears in {resolved_b!r}."
            )
        return resolved_b, env.with_expression_binding(
            resolved_a.identifier, resolved_b
        )

    if isinstance(resolved_b, IdentifierExpression):
        if _identifier_occurs_in(resolved_b.identifier, resolved_a):
            raise VerificationError(
                f"Occurs check failed: identifier {resolved_b.identifier!r} "
                f"appears in {resolved_a!r}."
            )
        return resolved_a, env.with_expression_binding(
            resolved_b.identifier, resolved_a
        )

    if resolved_a.is_structurally_equivalent(resolved_b):
        return resolved_a, env

    raise VerificationError(
        f"Cannot unify expressions {resolved_a!r} and {resolved_b!r}."
    )


def _unify_data_types(
    data_type_a: DataType,
    data_type_b: DataType,
    env: TypeUnificationEnv,
) -> tuple[DataType, TypeUnificationEnv]:
    """Bidirectionally unify two data types."""
    if isinstance(data_type_a, TemplateDataType) and isinstance(
        data_type_b, TemplateDataType
    ):
        return data_type_a, env
    if isinstance(data_type_a, TemplateDataType):
        new_env = bind_data_template(data_type_a, data_type_b, env)
        return data_type_b, new_env
    if isinstance(data_type_b, TemplateDataType):
        new_env = bind_data_template(data_type_b, data_type_a, env)
        return data_type_a, new_env
    if not structural_eq(data_type_a, data_type_b):
        raise VerificationError(
            f"Data type mismatch during unification: "
            f"{data_type_a!r} vs {data_type_b!r}."
        )
    return data_type_a, env


@singledispatch
def structural_eq(a: Any, b: Any) -> bool:
    """Pure structural comparison between two ``Type`` or ``DataType`` values.

    Default returns ``False`` — register a handler for any concrete subclass
    that should support structural comparison.
    """
    return False


@singledispatch
def bind_template(
    pattern: Any, actual: Any, env: TypeUnificationEnv
) -> TypeUnificationEnv:
    """One-directional binding of a template pattern against a concrete type.

    Default behavior: require structural equivalence; raise on mismatch.
    Register a handler for any concrete subclass that should participate in
    template binding.
    """
    if not structural_eq(pattern, actual):
        raise VerificationError(
            f"Cannot bind {pattern!r} against {actual!r}: structural mismatch."
        )
    return env


@singledispatch
def substitute_template(t: Any, env: TypeUnificationEnv) -> Type:
    """Substitute template placeholders within a type.

    Default behavior: return ``t`` unchanged.
    """
    if not isinstance(t, Type):
        raise VerificationError(
            f"substitute_template received {type(t).__name__}, which is not a Type."
        )
    return t


@singledispatch
def unify(
    expected: Any, actual: Any, env: TypeUnificationEnv
) -> tuple[Type, TypeUnificationEnv]:
    """Bidirectional unification of two types with placeholder binding.

    Default behavior: require structural equivalence; raise on mismatch.
    """
    if not structural_eq(expected, actual):
        raise VerificationError(
            f"Cannot unify {expected!r} with {actual!r}: structural mismatch."
        )
    return expected, env


@singledispatch
def bind_data_template(
    pattern: Any, actual: Any, env: TypeUnificationEnv
) -> TypeUnificationEnv:
    """One-directional binding of a data-type template against a concrete data type.

    Default behavior: require structural equivalence; raise on mismatch.
    """
    if not structural_eq(pattern, actual):
        raise VerificationError(
            f"Cannot bind data type {pattern!r} against {actual!r}: "
            f"structural mismatch."
        )
    return env


@singledispatch
def substitute_data_template(dt: Any, env: TypeUnificationEnv) -> DataType:
    """Substitute template placeholders within a data type.

    Default behavior: return ``dt`` unchanged.
    """
    if not isinstance(dt, DataType):
        raise VerificationError(
            f"substitute_data_template received {type(dt).__name__}, which is "
            f"not a DataType."
        )
    return dt


@structural_eq.register
def _(a: NumericalType, b: object) -> bool:
    if not isinstance(b, NumericalType):
        return False
    if not structural_eq(a.data_type, b.data_type):
        return False
    if len(a.shape) != len(b.shape):
        return False
    for left_dim, right_dim in zip(a.shape, b.shape, strict=True):
        if left_dim is Ellipsis or right_dim is Ellipsis:
            if left_dim is not Ellipsis or right_dim is not Ellipsis:
                return False
            continue
        if not left_dim.is_structurally_equivalent(right_dim):
            return False
    return True


@structural_eq.register
def _(a: IndexType, b: object) -> bool:
    if not isinstance(b, IndexType):
        return False
    return (
        a.lower_bound.is_structurally_equivalent(b.lower_bound)
        and a.upper_bound.is_structurally_equivalent(b.upper_bound)
        and a.stride.is_structurally_equivalent(b.stride)
    )


@structural_eq.register
def _(a: TupleType, b: object) -> bool:
    if not isinstance(b, TupleType):
        return False
    if len(a.types) != len(b.types):
        return False
    return all(
        structural_eq(left, right) for left, right in zip(a.types, b.types, strict=True)
    )


@structural_eq.register
def _(a: PrimitiveDataType, b: object) -> bool:
    return isinstance(b, PrimitiveDataType) and a.core_data_type == b.core_data_type


@structural_eq.register
def _(a: TemplateDataType, b: object) -> bool:
    return (
        isinstance(b, TemplateDataType)
        and a.data_type == b.data_type
        and a.widths == b.widths
    )


@bind_template.register
def _(
    pattern: NumericalType, actual: Type, env: TypeUnificationEnv
) -> TypeUnificationEnv:
    if not isinstance(actual, NumericalType):
        raise VerificationError(
            f"Cannot bind NumericalType pattern against {type(actual).__name__}."
        )

    new_env = bind_data_template(pattern.data_type, actual.data_type, env)

    if isinstance(pattern.data_type, TemplateDataType) and _is_wildcard_shape(
        pattern.shape
    ):
        template_name = pattern.data_type.data_type.name_hint
        existing = new_env.get_type_binding(template_name)
        if existing is not None and not structural_eq(existing, actual):
            raise VerificationError(
                f"Conflicting full-type binding for {template_name!r}: "
                f"{existing!r} vs {actual!r}."
            )
        return new_env.with_type_binding(template_name, actual)

    if _is_wildcard_shape(pattern.shape):
        return new_env

    if len(pattern.shape) != len(actual.shape):
        raise VerificationError(
            f"Shape rank mismatch: pattern has {len(pattern.shape)} "
            f"dimensions, actual has {len(actual.shape)}."
        )

    for pattern_dim, actual_dim in zip(pattern.shape, actual.shape, strict=True):
        if pattern_dim is Ellipsis:
            continue
        if actual_dim is Ellipsis:
            raise VerificationError(
                "Wildcard `...` cannot appear in `actual` during template binding."
            )
        new_env = _bind_shape_dimension(pattern_dim, actual_dim, new_env)

    return new_env


@bind_template.register
def _(pattern: TupleType, actual: Type, env: TypeUnificationEnv) -> TypeUnificationEnv:
    if not isinstance(actual, TupleType):
        raise VerificationError(
            f"Cannot bind TupleType pattern against {type(actual).__name__}."
        )
    if len(pattern.types) != len(actual.types):
        raise VerificationError(
            f"Tuple arity mismatch: pattern has {len(pattern.types)}, "
            f"actual has {len(actual.types)}."
        )
    new_env = env
    for pattern_element, actual_element in zip(
        pattern.types, actual.types, strict=True
    ):
        new_env = bind_template(pattern_element, actual_element, new_env)
    return new_env


@bind_template.register
def _(pattern: IndexType, actual: Type, env: TypeUnificationEnv) -> TypeUnificationEnv:
    if not isinstance(actual, IndexType):
        raise VerificationError(
            f"Cannot bind IndexType pattern against {type(actual).__name__}."
        )
    new_env = _bind_index_expression(pattern.lower_bound, actual.lower_bound, env)
    new_env = _bind_index_expression(pattern.upper_bound, actual.upper_bound, new_env)
    new_env = _bind_index_expression(pattern.stride, actual.stride, new_env)
    return new_env


@bind_data_template.register
def _(
    pattern: TemplateDataType, actual: DataType, env: TypeUnificationEnv
) -> TypeUnificationEnv:
    template_name = pattern.data_type.name_hint
    existing = env.get_data_type_binding(template_name)
    if existing is None:
        return env.with_data_type_binding(template_name, actual)
    if not structural_eq(existing, actual):
        raise VerificationError(
            f"Conflicting data-type binding for {template_name!r}: "
            f"{existing!r} vs {actual!r}."
        )
    return env


@bind_data_template.register
def _(
    pattern: PrimitiveDataType, actual: DataType, env: TypeUnificationEnv
) -> TypeUnificationEnv:
    if not isinstance(actual, PrimitiveDataType):
        raise VerificationError(
            f"Cannot bind PrimitiveDataType pattern against {type(actual).__name__}."
        )
    if pattern.core_data_type != actual.core_data_type:
        raise VerificationError(
            f"Core data type mismatch: {pattern.core_data_type} vs "
            f"{actual.core_data_type}."
        )
    return env


@substitute_template.register
def _(t: NumericalType, env: TypeUnificationEnv) -> Type:
    if isinstance(t.data_type, TemplateDataType) and _is_wildcard_shape(t.shape):
        bound = env.get_type_binding(t.data_type.data_type.name_hint)
        if bound is not None:
            return bound

    new_data_type = substitute_data_template(t.data_type, env)
    new_shape: list[Expression | EllipsisType] = []
    for dim in t.shape:
        if dim is Ellipsis:
            new_shape.append(dim)
        else:
            new_shape.append(_substitute_expression(dim, env))
    return NumericalType(new_data_type, new_shape)


@substitute_template.register
def _(t: TupleType, env: TypeUnificationEnv) -> Type:
    return TupleType([substitute_template(element, env) for element in t.types])


@substitute_template.register
def _(t: IndexType, env: TypeUnificationEnv) -> Type:
    return IndexType(
        _substitute_expression(t.lower_bound, env),
        _substitute_expression(t.upper_bound, env),
        _substitute_expression(t.stride, env),
    )


@substitute_data_template.register
def _(dt: TemplateDataType, env: TypeUnificationEnv) -> DataType:
    bound = env.get_data_type_binding(dt.data_type.name_hint)
    if bound is None:
        return dt
    return bound


@substitute_data_template.register
def _(dt: PrimitiveDataType, env: TypeUnificationEnv) -> DataType:
    return dt


@unify.register
def _(
    expected: NumericalType, actual: Type, env: TypeUnificationEnv
) -> tuple[Type, TypeUnificationEnv]:
    if not isinstance(actual, NumericalType):
        raise VerificationError(
            f"Cannot unify NumericalType with {type(actual).__name__}."
        )
    unified_data_type, new_env = _unify_data_types(
        expected.data_type, actual.data_type, env
    )
    if len(expected.shape) != len(actual.shape):
        raise VerificationError(
            f"Shape rank mismatch during unification: "
            f"{len(expected.shape)} vs {len(actual.shape)}."
        )
    unified_shape: list[Expression | EllipsisType] = []
    for expected_dim, actual_dim in zip(expected.shape, actual.shape, strict=True):
        if expected_dim is Ellipsis or actual_dim is Ellipsis:
            raise VerificationError(
                "Wildcard `...` is not supported during unification."
            )
        unified_dim, new_env = _unify_expressions(expected_dim, actual_dim, new_env)
        unified_shape.append(unified_dim)
    return NumericalType(unified_data_type, unified_shape), new_env


@unify.register
def _(
    expected: TupleType, actual: Type, env: TypeUnificationEnv
) -> tuple[Type, TypeUnificationEnv]:
    if not isinstance(actual, TupleType):
        raise VerificationError(f"Cannot unify TupleType with {type(actual).__name__}.")
    if len(expected.types) != len(actual.types):
        raise VerificationError(
            f"Tuple arity mismatch during unification: "
            f"{len(expected.types)} vs {len(actual.types)}."
        )
    new_env = env
    unified_types: list[Type] = []
    for expected_element, actual_element in zip(
        expected.types, actual.types, strict=True
    ):
        unified_element, new_env = unify(expected_element, actual_element, new_env)
        unified_types.append(unified_element)
    return TupleType(unified_types), new_env


@unify.register
def _(
    expected: IndexType, actual: Type, env: TypeUnificationEnv
) -> tuple[Type, TypeUnificationEnv]:
    if not isinstance(actual, IndexType):
        raise VerificationError(f"Cannot unify IndexType with {type(actual).__name__}.")
    new_env = env
    unified_lower, new_env = _unify_expressions(
        expected.lower_bound, actual.lower_bound, new_env
    )
    unified_upper, new_env = _unify_expressions(
        expected.upper_bound, actual.upper_bound, new_env
    )
    unified_stride, new_env = _unify_expressions(
        expected.stride, actual.stride, new_env
    )
    return IndexType(unified_lower, unified_upper, unified_stride), new_env
