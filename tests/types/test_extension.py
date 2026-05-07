"""Out-of-tree extension tests for the type-system dispatchers.

Demonstrates that a downstream package can register a brand-new ``Type``
subclass against the dispatchers without modifying ``fhy_core``.
"""

import pytest

from fhy_core.expression import IdentifierExpression, LiteralExpression
from fhy_core.identifier import Identifier
from fhy_core.serialization import SerializedDict
from fhy_core.trait import VerificationError
from fhy_core.types import (
    CoreDataType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    Type,
    TypeUnificationEnvironment,
    bind_template,
    is_structurally_equivalent,
    substitute_template,
    unify,
)


class SyntheticTaggedType(Type):
    """Test-only Type subclass that wraps a ``Type`` and a string tag."""

    _tag: str
    _inner: Type

    def __init__(self, tag: str, inner: Type) -> None:
        super().__init__()
        self._tag = tag
        self._inner = inner
        self.freeze(deep=True)

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def inner(self) -> Type:
        return self._inner

    def serialize_data_to_dict(self) -> SerializedDict:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def deserialize_data_from_dict(  # pragma: no cover
        cls, data: SerializedDict
    ) -> "SyntheticTaggedType":
        raise NotImplementedError


@is_structurally_equivalent.register
def _(left: SyntheticTaggedType, right: object) -> bool:
    return (
        isinstance(right, SyntheticTaggedType)
        and left.tag == right.tag
        and is_structurally_equivalent(left.inner, right.inner)
    )


@bind_template.register
def _(
    pattern: SyntheticTaggedType,
    actual: Type,
    environment: TypeUnificationEnvironment,
) -> TypeUnificationEnvironment:
    if not isinstance(actual, SyntheticTaggedType):
        raise VerificationError(
            f"Cannot bind SyntheticTaggedType against {type(actual).__name__}."
        )
    elif pattern.tag != actual.tag:
        raise VerificationError(f"Tag mismatch: {pattern.tag!r} vs {actual.tag!r}.")
    else:
        return bind_template(pattern.inner, actual.inner, environment)


@substitute_template.register
def _(type_: SyntheticTaggedType, environment: TypeUnificationEnvironment) -> Type:
    return SyntheticTaggedType(type_.tag, substitute_template(type_.inner, environment))


@unify.register
def _(
    expected: SyntheticTaggedType,
    actual: Type,
    environment: TypeUnificationEnvironment,
) -> tuple[Type, TypeUnificationEnvironment]:
    if not isinstance(actual, SyntheticTaggedType):
        raise VerificationError(
            f"Cannot unify SyntheticTaggedType with {type(actual).__name__}."
        )
    elif expected.tag != actual.tag:
        raise VerificationError(f"Tag mismatch: {expected.tag!r} vs {actual.tag!r}.")
    else:
        unified_inner, next_environment = unify(
            expected.inner, actual.inner, environment
        )
        return (
            SyntheticTaggedType(expected.tag, unified_inner),
            next_environment,
        )


def test_out_of_tree_class_supports_structural_equivalence_dispatch(
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test the registered out-of-tree class participates in structural equivalence."""
    inner_first = NumericalType(int32_data_type, [LiteralExpression(2)])
    inner_first_duplicate = NumericalType(int32_data_type, [LiteralExpression(2)])
    inner_second = NumericalType(int32_data_type, [LiteralExpression(3)])

    dense_first = SyntheticTaggedType("dense", inner_first)
    dense_first_duplicate = SyntheticTaggedType("dense", inner_first_duplicate)
    sparse_first = SyntheticTaggedType("sparse", inner_first)
    dense_second = SyntheticTaggedType("dense", inner_second)
    plain_numerical_type = NumericalType(int32_data_type, [LiteralExpression(2)])

    assert is_structurally_equivalent(dense_first, dense_first_duplicate)
    assert dense_first.is_structurally_equivalent(dense_first_duplicate)
    assert not is_structurally_equivalent(dense_first, sparse_first)
    assert not is_structurally_equivalent(dense_first, dense_second)
    assert not is_structurally_equivalent(dense_first, plain_numerical_type)


def test_out_of_tree_class_supports_bind_then_substitute_round_trip(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test the registered out-of-tree class participates in bind and substitute."""
    template_data_type = TemplateDataType(Identifier("T"))
    n_identifier = Identifier("N")
    pattern = SyntheticTaggedType(
        "dense",
        NumericalType(template_data_type, [IdentifierExpression(n_identifier)]),
    )
    actual = SyntheticTaggedType(
        "dense", NumericalType(int32_data_type, [LiteralExpression(8)])
    )

    environment = bind_template(pattern, actual, empty_environment)
    substituted = substitute_template(pattern, environment)
    assert isinstance(substituted, SyntheticTaggedType)
    assert is_structurally_equivalent(substituted, actual)


def test_out_of_tree_class_propagates_inner_bindings_during_unification(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test unification through an out-of-tree wrapper records bindings on its inner."""
    t_identifier = Identifier("T")
    template_data_type = TemplateDataType(t_identifier)
    n_identifier = Identifier("N")
    expected = SyntheticTaggedType(
        "dense",
        NumericalType(template_data_type, [IdentifierExpression(n_identifier)]),
    )
    actual = SyntheticTaggedType(
        "dense", NumericalType(int32_data_type, [LiteralExpression(8)])
    )
    unified, environment = unify(expected, actual, empty_environment)
    assert isinstance(unified, SyntheticTaggedType)
    assert is_structurally_equivalent(unified, actual)
    assert is_structurally_equivalent(
        environment.get_data_type_binding(t_identifier), int32_data_type
    )
    n_binding = environment.get_expression_binding(n_identifier)
    assert n_binding is not None and n_binding.is_structurally_equivalent(
        LiteralExpression(8)
    )


def test_out_of_tree_class_raises_on_tag_mismatch(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test bind and unify both raise when out-of-tree wrapper tags differ."""
    inner = NumericalType(int32_data_type)
    expected = SyntheticTaggedType("dense", inner)
    actual = SyntheticTaggedType("sparse", inner)
    with pytest.raises(VerificationError):
        bind_template(expected, actual, empty_environment)
    with pytest.raises(VerificationError):
        unify(expected, actual, empty_environment)


def test_out_of_tree_class_propagates_width_constraint_violation_through_inner(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test a downstream wrapper surfaces a `widths` violation from its inner type."""
    t_identifier = Identifier("T")
    constrained_template = TemplateDataType(t_identifier, widths=[8])
    pattern = SyntheticTaggedType(
        "dense", NumericalType(constrained_template, [LiteralExpression(4)])
    )
    actual = SyntheticTaggedType(
        "dense",
        NumericalType(PrimitiveDataType(CoreDataType.INT32), [LiteralExpression(4)]),
    )
    with pytest.raises(VerificationError, match="width"):
        bind_template(pattern, actual, empty_environment)
    with pytest.raises(VerificationError, match="width"):
        unify(pattern, actual, empty_environment)
