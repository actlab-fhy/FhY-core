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

The rules of the built-in classes, and each dispatcher's default, run in
the Rust core (S11a of ``docs/design/python-switch.md``). When the core
meets a Python-defined type or data type inside a built-in one, or as an
argument, it calls the handler registered for its class, once per such
node; a class without a handler gets the default rule, without a call into
Python. A handler registered again for a built-in class serves direct
calls of the dispatcher on that class; the core handles the built-in nodes
it meets itself.

Template placeholders are scoped by their underlying ``Identifier`` --- the
same id-based equality used everywhere else in ``fhy_core``. Two
``TemplateDataType`` values backed by ``Identifier`` instances that share a
``name_hint`` but differ in id are *distinct* placeholders, and unification
between them raises rather than silently merging them. Substitution and the
occurs check see every expression node, calls and piecewise nodes
included.
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

from functools import singledispatch
from typing import Any, ClassVar

from fhy_core import _rs
from fhy_core.traits.frozen import (
    _CONSTRUCTION_DEPTH_FLAG,
    FrozenValidationError,
    _install_init_wrap,
)
from fhy_core.utils.override import override

from ..symbolic.expression.core import Expression
from ..traits import FrozenMixin
from .core import (
    DataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    Type,
)


class TypeUnificationEnvironment(_rs.TypeUnificationEnvironment):
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

    The environment is frozen, and backed by the Rust core: the constructor
    takes the three tables as mappings, and ``empty()``, the ``with_*``
    helpers and the dispatchers produce new environments. ``==`` and
    ``hash`` compare the tables, and it pickles as a call of its class.

    Subclasses may add layer-specific extras (e.g. per-call-site state in a
    type-inferencer) without changing the dispatcher signatures: every
    environment derived from one is an instance of its class, with a copy of
    its instance attributes. A subclass sets its extras in its own
    ``__init__``, after which the instance is frozen, as a ``FrozenMixin``
    subclass is.

    Attributes:
        data_type_bindings: Bindings for ``TemplateDataType`` placeholders.
        type_bindings: Bindings for full-type wildcard placeholders.
        expression_bindings: Bindings for shape-variable placeholders.
    """

    _FREEZE_ON_INIT: ClassVar[bool] = True

    @override
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__init__" in cls.__dict__:
            _install_init_wrap(cls)

    @property
    def is_frozen(self) -> bool:
        """Whether it is frozen: always, except inside a subclass's ``__init__``."""
        return not vars(self).get(_CONSTRUCTION_DEPTH_FLAG, 0)

    def freeze(self) -> None:
        """Do nothing: an environment is frozen once constructed."""

    def assert_frozen(self) -> None:
        """Raise ``FrozenValidationError`` inside a subclass's ``__init__``."""
        if not self.is_frozen:
            raise FrozenValidationError(f"{type(self).__name__} is not frozen.")

    __setattr__ = FrozenMixin.__setattr__
    __delattr__ = FrozenMixin.__delattr__


FrozenMixin.register(TypeUnificationEnvironment)
TypeUnificationEnvironment._register_public_class()


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
    return _rs.types_is_structurally_equivalent(left, right)


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
        matching ``pattern`` against ``actual``; ``environment`` itself when
        nothing new was learned.

    Raises:
        VerificationError: If ``pattern`` and ``actual`` cannot be matched
            (structural mismatch, conflicting bindings, or shape rank
            mismatch).
    """
    bound: TypeUnificationEnvironment = _rs.types_bind_template(
        pattern, actual, environment
    )
    return bound


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
    substituted: Type = _rs.types_substitute_template(type_, environment)
    return substituted


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
    return _rs.types_unify(expected, actual, environment)


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
    return _rs.types_unify_expression(left_expression, right_expression, environment)


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
    bound: TypeUnificationEnvironment = _rs.types_bind_data_template(
        pattern, actual, environment
    )
    return bound


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
    substituted: DataType = _rs.types_substitute_data_template(data_type, environment)
    return substituted


# The built-in classes are registered explicitly, so a handler registered for
# `Type` or `DataType` does not capture them.
for _built_in in (NumericalType, IndexType, PrimitiveDataType, TemplateDataType):
    is_structurally_equivalent.register(_built_in, _rs.types_is_structurally_equivalent)
for _built_in in (NumericalType, IndexType):
    bind_template.register(_built_in, _rs.types_bind_template)
    substitute_template.register(_built_in, _rs.types_substitute_template)
    unify.register(_built_in, _rs.types_unify)
for _built_in in (PrimitiveDataType, TemplateDataType):
    bind_data_template.register(_built_in, _rs.types_bind_data_template)
    substitute_data_template.register(_built_in, _rs.types_substitute_data_template)
del _built_in
