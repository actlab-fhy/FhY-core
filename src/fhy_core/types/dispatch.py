"""Extensible dispatchers for the core type system.

This module provides the four extension points (six dispatchers) that the
core type system exposes for downstream packages:

- ``bind_template(pattern, actual, environment)``: one-directional binding
  for template patterns against concrete types.
- ``substitute_template(type_, environment)``: replace template placeholders
  in a type with bound values.
- ``unify(expected, actual, environment)``: bidirectional unification with
  placeholder binding allowed on either side.
- ``structural_eq(left, right)``: pure structural comparison.
- ``bind_data_template(pattern, actual, environment)``: data-type-tier
  binding.
- ``substitute_data_template(data_type, environment)``: data-type-tier
  substitution.

All dispatchers are :func:`functools.singledispatch` functions keyed by the
concrete class of their first argument. Downstream packages defining new
``Type`` or ``DataType`` subclasses register handlers against these
dispatchers from wherever the class is defined; no modification to
``fhy_core`` is required.
"""

__all__ = [
    "TypeUnificationEnvironment",
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

from ..expression.core import (
    BinaryExpression,
    Expression,
    IdentifierExpression,
    UnaryExpression,
)
from ..identifier import Identifier
from ..trait import FrozenMixin, StructuralEquivalenceMixin, VerificationError
from .core import (
    DataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TupleType,
    Type,
)


@dataclass(frozen=True)
class TypeUnificationEnvironment(FrozenMixin, StructuralEquivalenceMixin):
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

    The environment is frozen. Updates produce new environments via the
    ``with_*`` helpers; the dispatchers thread the environment through
    recursive calls and return updated environments.

    Subclasses may add layer-specific extras (e.g. per-call-site state in
    a type-inferencer) without changing the dispatcher signatures.
    """

    data_type_bindings: "frozendict[str, DataType]" = field(default_factory=frozendict)
    type_bindings: "frozendict[str, Type]" = field(default_factory=frozendict)
    expression_bindings: "frozendict[Identifier, Expression]" = field(
        default_factory=frozendict
    )

    @classmethod
    def empty(cls) -> "TypeUnificationEnvironment":
        """Construct an empty environment with no bindings."""
        return cls()

    def with_data_type_binding(
        self, name: str, value: DataType
    ) -> "TypeUnificationEnvironment":
        """Return a new environment with an additional data-type binding."""
        new_bindings = frozendict({**self.data_type_bindings, name: value})
        return replace(self, data_type_bindings=new_bindings)

    def with_type_binding(self, name: str, value: Type) -> "TypeUnificationEnvironment":
        """Return a new environment with an additional full-type binding."""
        new_bindings = frozendict({**self.type_bindings, name: value})
        return replace(self, type_bindings=new_bindings)

    def with_expression_binding(
        self, name: Identifier, value: Expression
    ) -> "TypeUnificationEnvironment":
        """Return a new environment with an additional expression binding."""
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
        if not isinstance(other, TypeUnificationEnvironment):
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


def _resolve_expression(
    expression: Expression, environment: TypeUnificationEnvironment
) -> Expression:
    """Follow chains of expression bindings until reaching a non-bound node."""
    visited_identifiers: set[Identifier] = set()
    current_expression: Expression = expression
    while isinstance(current_expression, IdentifierExpression):
        if current_expression.identifier in visited_identifiers:
            return current_expression
        bound_expression = environment.get_expression_binding(
            current_expression.identifier
        )
        if bound_expression is None:
            return current_expression
        visited_identifiers.add(current_expression.identifier)
        current_expression = bound_expression
    return current_expression


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
    environment: TypeUnificationEnvironment,
    visited_identifiers: frozenset[Identifier] = frozenset(),
) -> Expression:
    """Recursively substitute identifier expressions in ``expression``."""
    if isinstance(expression, IdentifierExpression):
        if expression.identifier in visited_identifiers:
            return expression
        bound_expression = environment.get_expression_binding(expression.identifier)
        if bound_expression is None:
            return expression
        return _substitute_expression(
            bound_expression,
            environment,
            visited_identifiers | {expression.identifier},
        )
    if isinstance(expression, BinaryExpression):
        new_left = _substitute_expression(
            expression.left, environment, visited_identifiers
        )
        new_right = _substitute_expression(
            expression.right, environment, visited_identifiers
        )
        if new_left is expression.left and new_right is expression.right:
            return expression
        return BinaryExpression(expression.operation, new_left, new_right)
    if isinstance(expression, UnaryExpression):
        new_operand = _substitute_expression(
            expression.operand, environment, visited_identifiers
        )
        if new_operand is expression.operand:
            return expression
        return UnaryExpression(expression.operation, new_operand)
    return expression


def _bind_shape_dimension(
    pattern_dimension: Expression,
    actual_dimension: Expression,
    environment: TypeUnificationEnvironment,
) -> TypeUnificationEnvironment:
    """Record an expression binding for a single shape dimension during binding."""
    if isinstance(pattern_dimension, IdentifierExpression):
        existing = environment.get_expression_binding(pattern_dimension.identifier)
        if existing is None:
            return environment.with_expression_binding(
                pattern_dimension.identifier, actual_dimension
            )
        if not existing.is_structurally_equivalent(actual_dimension):
            raise VerificationError(
                f"Conflicting binding for shape variable "
                f"{pattern_dimension.identifier!r}: {existing!r} vs "
                f"{actual_dimension!r}."
            )
        return environment
    if not pattern_dimension.is_structurally_equivalent(actual_dimension):
        raise VerificationError(
            f"Shape dimension mismatch: {pattern_dimension!r} vs {actual_dimension!r}."
        )
    return environment


def _bind_index_expression(
    pattern_expression: Expression,
    actual_expression: Expression,
    environment: TypeUnificationEnvironment,
) -> TypeUnificationEnvironment:
    """Record an expression binding for an index-type bound during binding."""
    return _bind_shape_dimension(pattern_expression, actual_expression, environment)


def _unify_expressions(
    left_expression: Expression,
    right_expression: Expression,
    environment: TypeUnificationEnvironment,
) -> tuple[Expression, TypeUnificationEnvironment]:
    """Bidirectionally unify two expressions, with occurs-check."""
    resolved_left = _resolve_expression(left_expression, environment)
    resolved_right = _resolve_expression(right_expression, environment)

    if (
        isinstance(resolved_left, IdentifierExpression)
        and isinstance(resolved_right, IdentifierExpression)
        and resolved_left.identifier == resolved_right.identifier
    ):
        return resolved_left, environment

    if isinstance(resolved_left, IdentifierExpression):
        if _identifier_occurs_in(resolved_left.identifier, resolved_right):
            raise VerificationError(
                f"Occurs check failed: identifier "
                f"{resolved_left.identifier!r} appears in {resolved_right!r}."
            )
        return resolved_right, environment.with_expression_binding(
            resolved_left.identifier, resolved_right
        )

    if isinstance(resolved_right, IdentifierExpression):
        if _identifier_occurs_in(resolved_right.identifier, resolved_left):
            raise VerificationError(
                f"Occurs check failed: identifier "
                f"{resolved_right.identifier!r} appears in {resolved_left!r}."
            )
        return resolved_left, environment.with_expression_binding(
            resolved_right.identifier, resolved_left
        )

    if resolved_left.is_structurally_equivalent(resolved_right):
        return resolved_left, environment

    raise VerificationError(
        f"Cannot unify expressions {resolved_left!r} and {resolved_right!r}."
    )


def _unify_data_types(
    left_data_type: DataType,
    right_data_type: DataType,
    environment: TypeUnificationEnvironment,
) -> tuple[DataType, TypeUnificationEnvironment]:
    """Bidirectionally unify two data types."""
    if isinstance(left_data_type, TemplateDataType) and isinstance(
        right_data_type, TemplateDataType
    ):
        return left_data_type, environment
    if isinstance(left_data_type, TemplateDataType):
        next_environment = bind_data_template(
            left_data_type, right_data_type, environment
        )
        return right_data_type, next_environment
    if isinstance(right_data_type, TemplateDataType):
        next_environment = bind_data_template(
            right_data_type, left_data_type, environment
        )
        return left_data_type, next_environment
    if not structural_eq(left_data_type, right_data_type):
        raise VerificationError(
            f"Data type mismatch during unification: "
            f"{left_data_type!r} vs {right_data_type!r}."
        )
    return left_data_type, environment


@singledispatch
def structural_eq(left: Any, right: Any) -> bool:
    """Pure structural comparison between two ``Type`` or ``DataType`` values.

    Default returns ``False`` — register a handler for any concrete subclass
    that should support structural comparison.
    """
    return False


@singledispatch
def bind_template(
    pattern: Any, actual: Any, environment: TypeUnificationEnvironment
) -> TypeUnificationEnvironment:
    """One-directional binding of a template pattern against a concrete type.

    Default behavior: require structural equivalence; raise on mismatch.
    Register a handler for any concrete subclass that should participate in
    template binding.
    """
    if not structural_eq(pattern, actual):
        raise VerificationError(
            f"Cannot bind {pattern!r} against {actual!r}: structural mismatch."
        )
    return environment


@singledispatch
def substitute_template(type_: Any, environment: TypeUnificationEnvironment) -> Type:
    """Substitute template placeholders within a type.

    Default behavior: return ``type_`` unchanged.
    """
    if not isinstance(type_, Type):
        raise VerificationError(
            f"substitute_template received {type(type_).__name__}, which is not a Type."
        )
    return type_


@singledispatch
def unify(
    expected: Any, actual: Any, environment: TypeUnificationEnvironment
) -> tuple[Type, TypeUnificationEnvironment]:
    """Bidirectional unification of two types with placeholder binding.

    Default behavior: require structural equivalence; raise on mismatch.
    """
    if not structural_eq(expected, actual):
        raise VerificationError(
            f"Cannot unify {expected!r} with {actual!r}: structural mismatch."
        )
    return expected, environment


@singledispatch
def bind_data_template(
    pattern: Any, actual: Any, environment: TypeUnificationEnvironment
) -> TypeUnificationEnvironment:
    """One-directional binding of a data-type template against a concrete data type.

    Default behavior: require structural equivalence; raise on mismatch.
    """
    if not structural_eq(pattern, actual):
        raise VerificationError(
            f"Cannot bind data type {pattern!r} against {actual!r}: "
            f"structural mismatch."
        )
    return environment


@singledispatch
def substitute_data_template(
    data_type: Any, environment: TypeUnificationEnvironment
) -> DataType:
    """Substitute template placeholders within a data type.

    Default behavior: return ``data_type`` unchanged.
    """
    if not isinstance(data_type, DataType):
        raise VerificationError(
            f"substitute_data_template received {type(data_type).__name__}, "
            f"which is not a DataType."
        )
    return data_type


@structural_eq.register
def _(left: NumericalType, right: object) -> bool:
    if not isinstance(right, NumericalType):
        return False
    if not structural_eq(left.data_type, right.data_type):
        return False
    if len(left.shape) != len(right.shape):
        return False
    for left_dimension, right_dimension in zip(left.shape, right.shape, strict=True):
        if left_dimension is Ellipsis or right_dimension is Ellipsis:
            if left_dimension is not Ellipsis or right_dimension is not Ellipsis:
                return False
            continue
        if not left_dimension.is_structurally_equivalent(right_dimension):
            return False
    return True


@structural_eq.register
def _(left: IndexType, right: object) -> bool:
    if not isinstance(right, IndexType):
        return False
    return (
        left.lower_bound.is_structurally_equivalent(right.lower_bound)
        and left.upper_bound.is_structurally_equivalent(right.upper_bound)
        and left.stride.is_structurally_equivalent(right.stride)
    )


@structural_eq.register
def _(left: TupleType, right: object) -> bool:
    if not isinstance(right, TupleType):
        return False
    if len(left.types) != len(right.types):
        return False
    return all(
        structural_eq(left_element, right_element)
        for left_element, right_element in zip(left.types, right.types, strict=True)
    )


@structural_eq.register
def _(left: PrimitiveDataType, right: object) -> bool:
    return (
        isinstance(right, PrimitiveDataType)
        and left.core_data_type == right.core_data_type
    )


@structural_eq.register
def _(left: TemplateDataType, right: object) -> bool:
    return (
        isinstance(right, TemplateDataType)
        and left.data_type == right.data_type
        and left.widths == right.widths
    )


@bind_template.register
def _(
    pattern: NumericalType,
    actual: Type,
    environment: TypeUnificationEnvironment,
) -> TypeUnificationEnvironment:
    if not isinstance(actual, NumericalType):
        raise VerificationError(
            f"Cannot bind NumericalType pattern against {type(actual).__name__}."
        )

    next_environment = bind_data_template(
        pattern.data_type, actual.data_type, environment
    )

    if isinstance(pattern.data_type, TemplateDataType) and _is_wildcard_shape(
        pattern.shape
    ):
        template_name = pattern.data_type.data_type.name_hint
        existing = next_environment.get_type_binding(template_name)
        if existing is not None and not structural_eq(existing, actual):
            raise VerificationError(
                f"Conflicting full-type binding for {template_name!r}: "
                f"{existing!r} vs {actual!r}."
            )
        return next_environment.with_type_binding(template_name, actual)

    if _is_wildcard_shape(pattern.shape):
        return next_environment

    if len(pattern.shape) != len(actual.shape):
        raise VerificationError(
            f"Shape rank mismatch: pattern has {len(pattern.shape)} "
            f"dimensions, actual has {len(actual.shape)}."
        )

    for pattern_dimension, actual_dimension in zip(
        pattern.shape, actual.shape, strict=True
    ):
        if pattern_dimension is Ellipsis:
            continue
        if actual_dimension is Ellipsis:
            raise VerificationError(
                "Wildcard `...` cannot appear in `actual` during template binding."
            )
        next_environment = _bind_shape_dimension(
            pattern_dimension, actual_dimension, next_environment
        )

    return next_environment


@bind_template.register
def _(
    pattern: TupleType,
    actual: Type,
    environment: TypeUnificationEnvironment,
) -> TypeUnificationEnvironment:
    if not isinstance(actual, TupleType):
        raise VerificationError(
            f"Cannot bind TupleType pattern against {type(actual).__name__}."
        )
    if len(pattern.types) != len(actual.types):
        raise VerificationError(
            f"Tuple arity mismatch: pattern has {len(pattern.types)}, "
            f"actual has {len(actual.types)}."
        )
    next_environment = environment
    for pattern_element, actual_element in zip(
        pattern.types, actual.types, strict=True
    ):
        next_environment = bind_template(
            pattern_element, actual_element, next_environment
        )
    return next_environment


@bind_template.register
def _(
    pattern: IndexType,
    actual: Type,
    environment: TypeUnificationEnvironment,
) -> TypeUnificationEnvironment:
    if not isinstance(actual, IndexType):
        raise VerificationError(
            f"Cannot bind IndexType pattern against {type(actual).__name__}."
        )
    next_environment = _bind_index_expression(
        pattern.lower_bound, actual.lower_bound, environment
    )
    next_environment = _bind_index_expression(
        pattern.upper_bound, actual.upper_bound, next_environment
    )
    next_environment = _bind_index_expression(
        pattern.stride, actual.stride, next_environment
    )
    return next_environment


@bind_data_template.register
def _(
    pattern: TemplateDataType,
    actual: DataType,
    environment: TypeUnificationEnvironment,
) -> TypeUnificationEnvironment:
    template_name = pattern.data_type.name_hint
    existing = environment.get_data_type_binding(template_name)
    if existing is None:
        return environment.with_data_type_binding(template_name, actual)
    if not structural_eq(existing, actual):
        raise VerificationError(
            f"Conflicting data-type binding for {template_name!r}: "
            f"{existing!r} vs {actual!r}."
        )
    return environment


@bind_data_template.register
def _(
    pattern: PrimitiveDataType,
    actual: DataType,
    environment: TypeUnificationEnvironment,
) -> TypeUnificationEnvironment:
    if not isinstance(actual, PrimitiveDataType):
        raise VerificationError(
            f"Cannot bind PrimitiveDataType pattern against {type(actual).__name__}."
        )
    if pattern.core_data_type != actual.core_data_type:
        raise VerificationError(
            f"Core data type mismatch: {pattern.core_data_type} vs "
            f"{actual.core_data_type}."
        )
    return environment


@substitute_template.register
def _(type_: NumericalType, environment: TypeUnificationEnvironment) -> Type:
    if isinstance(type_.data_type, TemplateDataType) and _is_wildcard_shape(
        type_.shape
    ):
        bound_type = environment.get_type_binding(type_.data_type.data_type.name_hint)
        if bound_type is not None:
            return bound_type

    new_data_type = substitute_data_template(type_.data_type, environment)
    new_shape: list[Expression | EllipsisType] = []
    for dimension in type_.shape:
        if dimension is Ellipsis:
            new_shape.append(dimension)
        else:
            new_shape.append(_substitute_expression(dimension, environment))
    return NumericalType(new_data_type, new_shape)


@substitute_template.register
def _(type_: TupleType, environment: TypeUnificationEnvironment) -> Type:
    return TupleType(
        [substitute_template(element, environment) for element in type_.types]
    )


@substitute_template.register
def _(type_: IndexType, environment: TypeUnificationEnvironment) -> Type:
    return IndexType(
        _substitute_expression(type_.lower_bound, environment),
        _substitute_expression(type_.upper_bound, environment),
        _substitute_expression(type_.stride, environment),
    )


@substitute_data_template.register
def _(data_type: TemplateDataType, environment: TypeUnificationEnvironment) -> DataType:
    bound_data_type = environment.get_data_type_binding(data_type.data_type.name_hint)
    if bound_data_type is None:
        return data_type
    return bound_data_type


@substitute_data_template.register
def _(
    data_type: PrimitiveDataType, environment: TypeUnificationEnvironment
) -> DataType:
    return data_type


@unify.register
def _(
    expected: NumericalType,
    actual: Type,
    environment: TypeUnificationEnvironment,
) -> tuple[Type, TypeUnificationEnvironment]:
    if not isinstance(actual, NumericalType):
        raise VerificationError(
            f"Cannot unify NumericalType with {type(actual).__name__}."
        )
    unified_data_type, next_environment = _unify_data_types(
        expected.data_type, actual.data_type, environment
    )
    if len(expected.shape) != len(actual.shape):
        raise VerificationError(
            f"Shape rank mismatch during unification: "
            f"{len(expected.shape)} vs {len(actual.shape)}."
        )
    unified_shape: list[Expression | EllipsisType] = []
    for expected_dimension, actual_dimension in zip(
        expected.shape, actual.shape, strict=True
    ):
        if expected_dimension is Ellipsis or actual_dimension is Ellipsis:
            raise VerificationError(
                "Wildcard `...` is not supported during unification."
            )
        unified_dimension, next_environment = _unify_expressions(
            expected_dimension, actual_dimension, next_environment
        )
        unified_shape.append(unified_dimension)
    return NumericalType(unified_data_type, unified_shape), next_environment


@unify.register
def _(
    expected: TupleType,
    actual: Type,
    environment: TypeUnificationEnvironment,
) -> tuple[Type, TypeUnificationEnvironment]:
    if not isinstance(actual, TupleType):
        raise VerificationError(f"Cannot unify TupleType with {type(actual).__name__}.")
    if len(expected.types) != len(actual.types):
        raise VerificationError(
            f"Tuple arity mismatch during unification: "
            f"{len(expected.types)} vs {len(actual.types)}."
        )
    next_environment = environment
    unified_types: list[Type] = []
    for expected_element, actual_element in zip(
        expected.types, actual.types, strict=True
    ):
        unified_element, next_environment = unify(
            expected_element, actual_element, next_environment
        )
        unified_types.append(unified_element)
    return TupleType(unified_types), next_environment


@unify.register
def _(
    expected: IndexType,
    actual: Type,
    environment: TypeUnificationEnvironment,
) -> tuple[Type, TypeUnificationEnvironment]:
    if not isinstance(actual, IndexType):
        raise VerificationError(f"Cannot unify IndexType with {type(actual).__name__}.")
    next_environment = environment
    unified_lower_bound, next_environment = _unify_expressions(
        expected.lower_bound, actual.lower_bound, next_environment
    )
    unified_upper_bound, next_environment = _unify_expressions(
        expected.upper_bound, actual.upper_bound, next_environment
    )
    unified_stride, next_environment = _unify_expressions(
        expected.stride, actual.stride, next_environment
    )
    return (
        IndexType(unified_lower_bound, unified_upper_bound, unified_stride),
        next_environment,
    )
