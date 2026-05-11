"""Extensible dispatchers for the core type system.

This module exposes four conceptual extension points --- *binding*,
*substitution*, *unification*, and *structural equivalence* --- as six
:func:`functools.singledispatch` functions. Binding and substitution are
split across a ``Type`` tier and a parallel ``DataType`` tier;
``is_structurally_equivalent`` is registered against both tiers and accepts
either argument type; ``unify`` operates on ``Type`` arguments only and
handles data-type unification internally.

- ``bind_template(pattern, actual, environment)``: one-directional binding
  for template patterns against concrete types.
- ``substitute_template(type_, environment)``: replace template placeholders
  in a type with bound values.
- ``unify(expected, actual, environment)``: bidirectional unification with
  placeholder binding allowed on either side.
- ``is_structurally_equivalent(left, right)``: pure structural comparison.
- ``bind_data_template(pattern, actual, environment)``: data-type-tier
  binding.
- ``substitute_data_template(data_type, environment)``: data-type-tier
  substitution.

Each dispatcher is keyed by the concrete class of its first argument.
Downstream packages defining new ``Type`` or ``DataType`` subclasses
register handlers against these dispatchers from wherever the class is
defined; no modification to ``fhy_core`` is required.

Template placeholders are scoped by their underlying ``Identifier`` --- the
same id-based equality used everywhere else in ``fhy_core``. Two
``TemplateDataType`` values backed by ``Identifier`` instances that share a
``name_hint`` but differ in id are *distinct* placeholders, and unification
between them raises rather than silently merging them.
"""

from __future__ import annotations

__all__ = [
    "TypeUnificationEnvironment",
    "bind_data_template",
    "bind_template",
    "is_structurally_equivalent",
    "substitute_data_template",
    "substitute_template",
    "unify",
    "unify_expression",
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
    Type,
    get_core_data_type_bit_width,
)


def _is_bindings_equivalent(
    left_bindings: frozendict[Identifier, Any],
    right_bindings: frozendict[Identifier, Any],
) -> bool:
    if set(left_bindings.keys()) != set(right_bindings.keys()):
        return False
    for binding_identifier, left_value in left_bindings.items():
        if not left_value.is_structurally_equivalent(
            right_bindings[binding_identifier]
        ):
            return False
    return True


def _is_wildcard_shape(shape: Sequence[Expression | EllipsisType]) -> bool:
    return len(shape) == 1 and shape[0] is Ellipsis


def _is_identifier_in_expression(
    identifier: Identifier, expression: Expression
) -> bool:
    if isinstance(expression, IdentifierExpression):
        return expression.identifier == identifier
    elif isinstance(expression, BinaryExpression):
        return _is_identifier_in_expression(
            identifier, expression.left
        ) or _is_identifier_in_expression(identifier, expression.right)
    elif isinstance(expression, UnaryExpression):
        return _is_identifier_in_expression(identifier, expression.operand)
    else:
        return False


def _resolve_expression(
    expression: Expression, environment: TypeUnificationEnvironment
) -> Expression:
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


def _substitute_expression(
    expression: Expression,
    environment: TypeUnificationEnvironment,
    visited_identifiers: frozenset[Identifier] = frozenset(),
) -> Expression:
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
    elif isinstance(expression, BinaryExpression):
        substituted_left = _substitute_expression(
            expression.left, environment, visited_identifiers
        )
        substituted_right = _substitute_expression(
            expression.right, environment, visited_identifiers
        )
        if (
            substituted_left is expression.left
            and substituted_right is expression.right
        ):
            return expression
        return BinaryExpression(
            expression.operation, substituted_left, substituted_right
        )
    elif isinstance(expression, UnaryExpression):
        substituted_operand = _substitute_expression(
            expression.operand, environment, visited_identifiers
        )
        if substituted_operand is expression.operand:
            return expression
        return UnaryExpression(expression.operation, substituted_operand)
    else:
        return expression


def _bind_shape_dimension(
    pattern_dimension: Expression,
    actual_dimension: Expression,
    environment: TypeUnificationEnvironment,
) -> TypeUnificationEnvironment:
    if isinstance(pattern_dimension, IdentifierExpression):
        existing_binding = environment.get_expression_binding(
            pattern_dimension.identifier
        )
        if existing_binding is None:
            return environment.with_expression_binding(
                pattern_dimension.identifier, actual_dimension
            )
        elif not existing_binding.is_structurally_equivalent(actual_dimension):
            raise VerificationError(
                f"Conflicting binding for shape variable "
                f"{pattern_dimension.identifier!r}: {existing_binding!r} vs "
                f"{actual_dimension!r}."
            )
        else:
            return environment
    elif not pattern_dimension.is_structurally_equivalent(actual_dimension):
        raise VerificationError(
            f"Shape dimension mismatch: {pattern_dimension!r} vs {actual_dimension!r}."
        )
    else:
        return environment


def _bind_index_expression(
    pattern_expression: Expression,
    actual_expression: Expression,
    environment: TypeUnificationEnvironment,
) -> TypeUnificationEnvironment:
    return _bind_shape_dimension(pattern_expression, actual_expression, environment)


def _unify_expressions(
    left_expression: Expression,
    right_expression: Expression,
    environment: TypeUnificationEnvironment,
) -> tuple[Expression, TypeUnificationEnvironment]:
    resolved_left = _resolve_expression(left_expression, environment)
    resolved_right = _resolve_expression(right_expression, environment)

    if (
        isinstance(resolved_left, IdentifierExpression)
        and isinstance(resolved_right, IdentifierExpression)
        and resolved_left.identifier == resolved_right.identifier
    ):
        return resolved_left, environment
    elif isinstance(resolved_left, IdentifierExpression):
        substituted_right = _substitute_expression(resolved_right, environment)
        if _is_identifier_in_expression(resolved_left.identifier, substituted_right):
            raise VerificationError(
                f"Occurs check failed: identifier "
                f"{resolved_left.identifier!r} appears in {resolved_right!r} "
                f"after substitution through existing bindings "
                f"({substituted_right!r})."
            )
        return resolved_right, environment.with_expression_binding(
            resolved_left.identifier, resolved_right
        )
    elif isinstance(resolved_right, IdentifierExpression):
        substituted_left = _substitute_expression(resolved_left, environment)
        if _is_identifier_in_expression(resolved_right.identifier, substituted_left):
            raise VerificationError(
                f"Occurs check failed: identifier "
                f"{resolved_right.identifier!r} appears in {resolved_left!r} "
                f"after substitution through existing bindings "
                f"({substituted_left!r})."
            )
        return resolved_left, environment.with_expression_binding(
            resolved_right.identifier, resolved_left
        )
    elif resolved_left.is_structurally_equivalent(resolved_right):
        return resolved_left, environment
    else:
        raise VerificationError(
            f"Cannot unify expressions {resolved_left!r} and {resolved_right!r}."
        )


def _unify_data_types(
    left_data_type: DataType,
    right_data_type: DataType,
    environment: TypeUnificationEnvironment,
) -> tuple[DataType, TypeUnificationEnvironment]:
    if isinstance(left_data_type, TemplateDataType) and isinstance(
        right_data_type, TemplateDataType
    ):
        # TODO: chain-binding for two-template unify
        if not is_structurally_equivalent(left_data_type, right_data_type):
            raise VerificationError(
                f"Cannot unify distinct template data types: "
                f"{left_data_type!r} vs {right_data_type!r}."
            )
        return left_data_type, environment
    elif isinstance(left_data_type, TemplateDataType):
        next_environment = bind_data_template(
            left_data_type, right_data_type, environment
        )
        return right_data_type, next_environment
    elif isinstance(right_data_type, TemplateDataType):
        next_environment = bind_data_template(
            right_data_type, left_data_type, environment
        )
        return left_data_type, next_environment
    elif not is_structurally_equivalent(left_data_type, right_data_type):
        raise VerificationError(
            f"Data type mismatch during unification: "
            f"{left_data_type!r} vs {right_data_type!r}."
        )
    else:
        return left_data_type, environment


@dataclass(frozen=True)
class TypeUnificationEnvironment(FrozenMixin, StructuralEquivalenceMixin):
    """Carrier for binding state used during type-system operations.

    Three independent binding tables hold placeholder resolutions accumulated
    while binding, substituting, or unifying types. All three are keyed by
    ``Identifier`` so that placeholder identity follows the codebase-wide
    id-based equality on ``Identifier`` rather than any name string:

    - ``data_type_bindings`` maps a ``TemplateDataType``'s ``Identifier`` to
      a concrete ``DataType``. Populated when a ``TemplateDataType`` in a
      pattern is matched against a concrete data type in an actual.
    - ``type_bindings`` maps a full-type-template ``Identifier`` to a
      concrete ``Type``. Populated when a ``NumericalType`` whose data type
      is a ``TemplateDataType`` and whose shape is a wildcard (``[...]``) is
      matched against a concrete type --- the entire actual type is captured,
      not just its data-type part.
    - ``expression_bindings`` maps a shape-variable placeholder ``Identifier``
      (the head of an ``IdentifierExpression``) to a concrete ``Expression``.
      Populated when shape elements are unified pairwise and one side is a
      placeholder.

    The environment is frozen. Updates produce new environments via the
    ``with_*`` helpers; the dispatchers thread the environment through
    recursive calls and return updated environments.

    Subclasses may add layer-specific extras (e.g. per-call-site state in a
    type-inferencer) without changing the dispatcher signatures.

    Attributes:
        data_type_bindings: Bindings for ``TemplateDataType`` placeholders.
        type_bindings: Bindings for full-type wildcard placeholders.
        expression_bindings: Bindings for shape-variable placeholders.
    """

    data_type_bindings: frozendict[Identifier, DataType] = field(
        default_factory=frozendict
    )
    type_bindings: frozendict[Identifier, Type] = field(default_factory=frozendict)
    expression_bindings: frozendict[Identifier, Expression] = field(
        default_factory=frozendict
    )

    @classmethod
    def empty(cls) -> TypeUnificationEnvironment:
        """Construct an empty environment with no bindings.

        Returns:
            A fresh environment with all three binding tables empty.
        """
        return cls()

    def with_data_type_binding(
        self, name: Identifier, value: DataType
    ) -> TypeUnificationEnvironment:
        """Return a new environment with an additional data-type binding.

        Args:
            name: Placeholder identifier to bind.
            value: Concrete data type to bind ``name`` to.

        Returns:
            A new environment that extends ``self`` with the new binding;
            ``self`` is left unchanged.
        """
        return replace(
            self, data_type_bindings=self.data_type_bindings.set(name, value)
        )

    def with_type_binding(
        self, name: Identifier, value: Type
    ) -> TypeUnificationEnvironment:
        """Return a new environment with an additional full-type binding.

        Args:
            name: Placeholder identifier to bind.
            value: Concrete type to bind ``name`` to.

        Returns:
            A new environment that extends ``self`` with the new binding;
            ``self`` is left unchanged.
        """
        return replace(self, type_bindings=self.type_bindings.set(name, value))

    def with_expression_binding(
        self, name: Identifier, value: Expression
    ) -> TypeUnificationEnvironment:
        """Return a new environment with an additional expression binding.

        Args:
            name: Placeholder identifier to bind.
            value: Concrete expression to bind ``name`` to.

        Returns:
            A new environment that extends ``self`` with the new binding;
            ``self`` is left unchanged.
        """
        return replace(
            self, expression_bindings=self.expression_bindings.set(name, value)
        )

    def get_data_type_binding(self, name: Identifier) -> DataType | None:
        """Return the bound data type for a placeholder identifier.

        Args:
            name: Placeholder identifier to look up.

        Returns:
            The bound data type, or ``None`` if ``name`` is unbound.
        """
        return self.data_type_bindings.get(name)

    def get_type_binding(self, name: Identifier) -> Type | None:
        """Return the bound full type for a placeholder identifier.

        Args:
            name: Placeholder identifier to look up.

        Returns:
            The bound type, or ``None`` if ``name`` is unbound.
        """
        return self.type_bindings.get(name)

    def get_expression_binding(self, name: Identifier) -> Expression | None:
        """Return the bound expression for a placeholder identifier.

        Args:
            name: Placeholder identifier to look up.

        Returns:
            The bound expression, or ``None`` if ``name`` is unbound.
        """
        return self.expression_bindings.get(name)

    def is_structurally_equivalent(self, other: object) -> bool:
        if not isinstance(other, TypeUnificationEnvironment):
            return False
        return (
            _is_bindings_equivalent(self.data_type_bindings, other.data_type_bindings)
            and _is_bindings_equivalent(self.type_bindings, other.type_bindings)
            and _is_bindings_equivalent(
                self.expression_bindings, other.expression_bindings
            )
        )


@singledispatch
def is_structurally_equivalent(left: Any, right: Any) -> bool:
    """Compare two ``Type`` or ``DataType`` values for structural equivalence.

    The default returns ``False`` --- register a handler for any concrete
    subclass that should support structural comparison.

    Args:
        left: First value to compare.
        right: Second value to compare.

    Returns:
        ``True`` when the two values are structurally equivalent under the
        registered handlers, ``False`` otherwise.
    """
    return False


@singledispatch
def bind_template(
    pattern: Any, actual: Any, environment: TypeUnificationEnvironment
) -> TypeUnificationEnvironment:
    """Bind a template pattern against a concrete type.

    The default behavior requires the two values to be structurally
    equivalent. Register a handler for any concrete subclass that should
    participate in template binding.

    Args:
        pattern: Pattern that may contain template placeholders.
        actual: Concrete value to bind ``pattern`` against.
        environment: Current binding environment.

    Returns:
        An updated environment that records all bindings learned while
        matching ``pattern`` against ``actual``.

    Raises:
        VerificationError: If ``pattern`` and ``actual`` cannot be matched
            (structural mismatch, conflicting bindings, or shape rank
            mismatch).
    """
    if not is_structurally_equivalent(pattern, actual):
        raise VerificationError(
            f"Cannot bind {pattern!r} against {actual!r}: structural mismatch."
        )
    return environment


@singledispatch
def substitute_template(type_: Any, environment: TypeUnificationEnvironment) -> Type:
    """Substitute template placeholders inside a type.

    The default returns ``type_`` unchanged. Register a handler for any
    concrete subclass whose internals may contain template placeholders.

    Args:
        type_: Type that may contain template placeholders.
        environment: Binding environment whose ``type_bindings``,
            ``data_type_bindings``, and ``expression_bindings`` resolve
            placeholders.

    Returns:
        A new type with bound placeholders replaced; unbound placeholders
        are left as-is (partial substitution is allowed).

    Raises:
        TypeError: If ``type_`` is not a ``Type`` and no handler is
            registered for its concrete class. (Reaching this default with
            a non-``Type`` is a programmer/registration bug, not a
            verification failure.)
    """
    if not isinstance(type_, Type):
        raise TypeError(
            f"substitute_template received {type(type_).__name__}, which is not a Type."
        )
    return type_


@singledispatch
def unify(
    expected: Any, actual: Any, environment: TypeUnificationEnvironment
) -> tuple[Type, TypeUnificationEnvironment]:
    """Bidirectionally unify two types, allowing placeholders on either side.

    The default behavior requires structural equivalence. Register a handler
    for any concrete subclass that should support unification.

    Args:
        expected: Expected type, possibly containing placeholders.
        actual: Actual type, possibly containing placeholders.
        environment: Current binding environment.

    Returns:
        A tuple ``(unified_type, new_environment)`` where ``unified_type``
        is the more-specific reconciliation of the two inputs and
        ``new_environment`` carries every binding learned during
        unification.

    Raises:
        VerificationError: If the two types are incompatible, an occurs
            check fails, or a binding conflict is detected.
    """
    if not is_structurally_equivalent(expected, actual):
        raise VerificationError(
            f"Cannot unify {expected!r} with {actual!r}: structural mismatch."
        )
    return expected, environment


def unify_expression(
    left_expression: Expression,
    right_expression: Expression,
    environment: TypeUnificationEnvironment,
) -> tuple[Expression, TypeUnificationEnvironment]:
    """Bidirectionally unify two expressions, allowing placeholders on either side.

    Resolves any chained ``IdentifierExpression`` placeholders against the
    current ``expression_bindings``, then unifies the resolved forms: a
    placeholder on either side is bound to the other expression (subject to
    the occurs check), and two concrete expressions must be structurally
    equivalent.

    Args:
        left_expression: First expression, possibly containing placeholders.
        right_expression: Second expression, possibly containing placeholders.
        environment: Current binding environment.

    Returns:
        A tuple ``(unified_expression, new_environment)`` where
        ``unified_expression`` is the more-specific reconciliation of the two
        inputs and ``new_environment`` carries every binding learned during
        unification.

    Raises:
        VerificationError: If the two expressions are incompatible or an
            occurs check fails.
    """
    return _unify_expressions(left_expression, right_expression, environment)


@singledispatch
def bind_data_template(
    pattern: Any, actual: Any, environment: TypeUnificationEnvironment
) -> TypeUnificationEnvironment:
    """Bind a data-type template against a concrete data type.

    The default behavior requires structural equivalence. Register a handler
    for any concrete ``DataType`` subclass that should participate in
    data-type-tier template binding.

    Args:
        pattern: Data-type pattern that may be a ``TemplateDataType``.
        actual: Concrete data type to bind ``pattern`` against.
        environment: Current binding environment.

    Returns:
        An updated environment that records the new data-type binding (or
        ``environment`` unchanged when the binding was already present and
        consistent).

    Raises:
        VerificationError: If ``pattern`` and ``actual`` are incompatible
            or if a binding conflict is detected.
    """
    if not is_structurally_equivalent(pattern, actual):
        raise VerificationError(
            f"Cannot bind data type {pattern!r} against {actual!r}: "
            f"structural mismatch."
        )
    return environment


@singledispatch
def substitute_data_template(
    data_type: Any, environment: TypeUnificationEnvironment
) -> DataType:
    """Substitute template placeholders inside a data type.

    The default returns ``data_type`` unchanged. Register a handler for any
    concrete ``DataType`` subclass that may contain template placeholders.

    Args:
        data_type: Data type that may contain template placeholders.
        environment: Binding environment whose ``data_type_bindings``
            resolve ``TemplateDataType`` placeholders.

    Returns:
        A new data type with bound placeholders replaced; unbound
        placeholders are left as-is.

    Raises:
        TypeError: If ``data_type`` is not a ``DataType`` and no handler
            is registered for its concrete class. (Reaching this default
            with a non-``DataType`` is a programmer/registration bug, not
            a verification failure.)
    """
    if not isinstance(data_type, DataType):
        raise TypeError(
            f"substitute_data_template received {type(data_type).__name__}, "
            f"which is not a DataType."
        )
    return data_type


@is_structurally_equivalent.register
def _(left: NumericalType, right: object) -> bool:
    if not isinstance(right, NumericalType):
        return False
    elif not is_structurally_equivalent(left.data_type, right.data_type):
        return False
    elif len(left.shape) != len(right.shape):
        return False
    else:
        for left_dimension, right_dimension in zip(
            left.shape, right.shape, strict=True
        ):
            if left_dimension is Ellipsis or right_dimension is Ellipsis:
                if left_dimension is not Ellipsis or right_dimension is not Ellipsis:
                    return False
                continue
            if not left_dimension.is_structurally_equivalent(right_dimension):
                return False
        return True


@is_structurally_equivalent.register
def _(left: IndexType, right: object) -> bool:
    if not isinstance(right, IndexType):
        return False
    else:
        return (
            left.lower_bound.is_structurally_equivalent(right.lower_bound)
            and left.upper_bound.is_structurally_equivalent(right.upper_bound)
            and left.stride.is_structurally_equivalent(right.stride)
        )


@is_structurally_equivalent.register
def _(left: PrimitiveDataType, right: object) -> bool:
    return (
        isinstance(right, PrimitiveDataType)
        and left.core_data_type == right.core_data_type
    )


@is_structurally_equivalent.register
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
        template_identifier = pattern.data_type.data_type
        existing_full_type = next_environment.get_type_binding(template_identifier)
        if existing_full_type is not None and not is_structurally_equivalent(
            existing_full_type, actual
        ):
            raise VerificationError(
                f"Conflicting full-type binding for {template_identifier!r}: "
                f"{existing_full_type!r} vs {actual!r}."
            )
        return next_environment.with_type_binding(template_identifier, actual)
    elif _is_wildcard_shape(pattern.shape):
        return next_environment
    elif len(pattern.shape) != len(actual.shape):
        raise VerificationError(
            f"Shape rank mismatch: pattern has {len(pattern.shape)} "
            f"dimensions, actual has {len(actual.shape)}."
        )
    else:
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


def _check_template_width_constraint(
    pattern: TemplateDataType, actual: DataType
) -> None:
    """Raise ``VerificationError`` when ``actual`` violates ``pattern.widths``.

    A ``TemplateDataType`` whose ``widths`` is non-``None`` matches only
    actuals whose bit width appears in the list. Weak-literal types (bit
    width ``None``) never satisfy a non-``None`` constraint, even when the
    constraint is empty. ``widths=None`` is unconstrained.
    """
    pattern_widths = pattern.widths
    if pattern_widths is None:
        return
    if not isinstance(actual, PrimitiveDataType):
        raise VerificationError(
            f"Cannot satisfy width constraint {pattern_widths!r} on "
            f"{pattern!r} with non-primitive actual {actual!r}."
        )
    actual_width = get_core_data_type_bit_width(actual.core_data_type)
    if actual_width is None or actual_width not in pattern_widths:
        raise VerificationError(
            f"Width mismatch for template {pattern!r}: actual {actual!r} "
            f"has width {actual_width!r}, not in {pattern_widths!r}."
        )


@bind_data_template.register
def _(
    pattern: TemplateDataType,
    actual: DataType,
    environment: TypeUnificationEnvironment,
) -> TypeUnificationEnvironment:
    if isinstance(actual, TemplateDataType):
        # Mirrors ``_unify_data_types``: distinct template placeholders cannot
        # be silently aliased; the same placeholder is a no-op.
        if not is_structurally_equivalent(pattern, actual):
            raise VerificationError(
                f"Cannot bind distinct template data types: {pattern!r} vs {actual!r}."
            )
        return environment
    _check_template_width_constraint(pattern, actual)
    template_identifier = pattern.data_type
    existing_binding = environment.get_data_type_binding(template_identifier)
    if existing_binding is None:
        return environment.with_data_type_binding(template_identifier, actual)
    elif not is_structurally_equivalent(existing_binding, actual):
        raise VerificationError(
            f"Conflicting data-type binding for {template_identifier!r}: "
            f"{existing_binding!r} vs {actual!r}."
        )
    else:
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
    elif pattern.core_data_type != actual.core_data_type:
        raise VerificationError(
            f"Core data type mismatch: {pattern.core_data_type} vs "
            f"{actual.core_data_type}."
        )
    else:
        return environment


@substitute_template.register
def _(type_: NumericalType, environment: TypeUnificationEnvironment) -> Type:
    if isinstance(type_.data_type, TemplateDataType) and _is_wildcard_shape(
        type_.shape
    ):
        bound_full_type = environment.get_type_binding(type_.data_type.data_type)
        if bound_full_type is not None:
            return bound_full_type

    substituted_data_type = substitute_data_template(type_.data_type, environment)
    substituted_shape: list[Expression | EllipsisType] = []
    for dimension in type_.shape:
        if dimension is Ellipsis:
            substituted_shape.append(dimension)
        else:
            substituted_shape.append(_substitute_expression(dimension, environment))
    return NumericalType(substituted_data_type, substituted_shape)


@substitute_template.register
def _(type_: IndexType, environment: TypeUnificationEnvironment) -> Type:
    return IndexType(
        _substitute_expression(type_.lower_bound, environment),
        _substitute_expression(type_.upper_bound, environment),
        _substitute_expression(type_.stride, environment),
    )


@substitute_data_template.register
def _(data_type: TemplateDataType, environment: TypeUnificationEnvironment) -> DataType:
    bound_data_type = environment.get_data_type_binding(data_type.data_type)
    if bound_data_type is None:
        return data_type
    else:
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
    elif len(expected.shape) != len(actual.shape):
        raise VerificationError(
            f"Shape rank mismatch during unification: "
            f"{len(expected.shape)} vs {len(actual.shape)}."
        )
    else:
        unified_data_type, next_environment = _unify_data_types(
            expected.data_type, actual.data_type, environment
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
