"""Tests for the type-system dispatchers and `TypeUnificationEnvironment`."""

import pytest

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.identifier import Identifier
from fhy_core.serialization import SerializedDict
from fhy_core.trait import VerificationError
from fhy_core.types import (
    CoreDataType,
    DataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    Type,
    TypeUnificationEnvironment,
    bind_data_template,
    bind_template,
    is_structurally_equivalent,
    substitute_data_template,
    substitute_template,
    unify,
    unify_expression,
)

# =============================================================================
# `TypeUnificationEnvironment` construction and helpers
# =============================================================================


def test_empty_environment_has_no_bindings(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test ``empty()`` returns an environment whose binding tables are empty."""
    assert empty_environment.get_data_type_binding(Identifier("T")) is None
    assert empty_environment.get_type_binding(Identifier("T")) is None
    assert empty_environment.get_expression_binding(Identifier("N")) is None


def test_with_helpers_return_new_environments_without_mutating_original(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test ``with_*`` helpers return new environments and leave the original alone."""
    t_identifier = Identifier("T")
    u_identifier = Identifier("U")
    n_identifier = Identifier("N")

    environment_with_data_type_binding = empty_environment.with_data_type_binding(
        t_identifier, int32_data_type
    )
    environment_with_type_binding = empty_environment.with_type_binding(
        u_identifier, NumericalType(int32_data_type)
    )
    environment_with_expression_binding = empty_environment.with_expression_binding(
        n_identifier, LiteralExpression(4)
    )

    assert empty_environment.get_data_type_binding(t_identifier) is None
    assert empty_environment.get_type_binding(u_identifier) is None
    assert empty_environment.get_expression_binding(n_identifier) is None

    assert isinstance(
        environment_with_data_type_binding.get_data_type_binding(t_identifier),
        PrimitiveDataType,
    )
    assert isinstance(
        environment_with_type_binding.get_type_binding(u_identifier), NumericalType
    )
    assert (
        environment_with_expression_binding.get_expression_binding(n_identifier)
        is not None
    )

    assert not empty_environment.is_structurally_equivalent(
        environment_with_data_type_binding
    )
    assert not empty_environment.is_structurally_equivalent(
        environment_with_type_binding
    )
    assert not empty_environment.is_structurally_equivalent(
        environment_with_expression_binding
    )


def test_environment_structural_equivalence_compares_bindings_by_value(
    int32_data_type: PrimitiveDataType,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test environment structural equivalence compares bindings by structural value."""
    t_identifier = Identifier("T")
    n_identifier = Identifier("N")
    environment_with_int = TypeUnificationEnvironment.empty().with_data_type_binding(
        t_identifier, int32_data_type
    )
    environment_with_int_duplicate = (
        TypeUnificationEnvironment.empty().with_data_type_binding(
            t_identifier, int32_data_type
        )
    )
    environment_with_float = TypeUnificationEnvironment.empty().with_data_type_binding(
        t_identifier, float32_data_type
    )
    environment_with_expression = (
        TypeUnificationEnvironment.empty().with_expression_binding(
            n_identifier, LiteralExpression(4)
        )
    )

    assert environment_with_int.is_structurally_equivalent(
        environment_with_int_duplicate
    )
    assert not environment_with_int.is_structurally_equivalent(environment_with_float)
    assert not environment_with_int.is_structurally_equivalent(
        environment_with_expression
    )
    assert not environment_with_int.is_structurally_equivalent("not an environment")


def test_environment_structural_equivalence_distinguishes_type_bindings(
    int32_data_type: PrimitiveDataType,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test type-binding tables are part of environment structural equivalence."""
    u_identifier = Identifier("U")
    environment_with_int_type = TypeUnificationEnvironment.empty().with_type_binding(
        u_identifier, NumericalType(int32_data_type)
    )
    environment_with_float_type = TypeUnificationEnvironment.empty().with_type_binding(
        u_identifier, NumericalType(float32_data_type)
    )

    assert not environment_with_int_type.is_structurally_equivalent(
        environment_with_float_type
    )


def test_chained_with_helpers_produce_pairwise_distinct_environments(
    int32_data_type: PrimitiveDataType,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test successive ``with_*`` calls yield pairwise structurally distinct envs."""
    t_identifier = Identifier("T")
    u_identifier = Identifier("U")
    n_identifier = Identifier("N")
    environment_0 = TypeUnificationEnvironment.empty()
    environment_1 = environment_0.with_data_type_binding(t_identifier, int32_data_type)
    environment_2 = environment_1.with_expression_binding(
        n_identifier, LiteralExpression(7)
    )
    environment_3 = environment_2.with_type_binding(
        u_identifier, NumericalType(float32_data_type)
    )

    environments = [environment_0, environment_1, environment_2, environment_3]
    for left_index, left_environment in enumerate(environments):
        for right_environment in environments[left_index + 1 :]:
            assert not left_environment.is_structurally_equivalent(right_environment)
    assert environment_0.get_data_type_binding(t_identifier) is None
    assert environment_0.get_expression_binding(n_identifier) is None
    assert environment_0.get_type_binding(u_identifier) is None


# =============================================================================
# `is_structurally_equivalent` dispatcher
# =============================================================================


def test_is_structurally_equivalent_dispatches_for_numerical_type(
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test dispatcher and `NumericalType.is_structurally_equivalent` agree."""
    left = NumericalType(int32_data_type, [LiteralExpression(4), LiteralExpression(8)])
    right = NumericalType(int32_data_type, [LiteralExpression(4), LiteralExpression(8)])
    different = NumericalType(
        int32_data_type, [LiteralExpression(4), LiteralExpression(9)]
    )

    assert is_structurally_equivalent(left, right)
    assert left.is_structurally_equivalent(right)
    assert not is_structurally_equivalent(left, different)
    assert not left.is_structurally_equivalent(different)


def test_is_structurally_equivalent_dispatches_for_data_type(
    int32_data_type: PrimitiveDataType,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test dispatcher and `DataType.is_structurally_equivalent` agree on primitives."""
    int32_duplicate = PrimitiveDataType(CoreDataType.INT32)

    assert is_structurally_equivalent(int32_data_type, int32_duplicate)
    assert int32_data_type.is_structurally_equivalent(int32_duplicate)
    assert not is_structurally_equivalent(int32_data_type, float32_data_type)
    assert not int32_data_type.is_structurally_equivalent(float32_data_type)


def test_is_structurally_equivalent_returns_false_for_unrelated_concrete_classes(
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test the dispatcher returns ``False`` for unrelated concrete classes."""
    numerical_type = NumericalType(int32_data_type)
    index_type = IndexType(LiteralExpression(0), LiteralExpression(10))
    assert not is_structurally_equivalent(numerical_type, index_type)


def test_template_data_type_equivalence_is_identifier_based() -> None:
    """Test ``is_structurally_equivalent`` compares ``TemplateDataType`` by id."""
    t_identifier = Identifier("T")
    left = TemplateDataType(t_identifier)
    same_identifier = TemplateDataType(t_identifier)
    same_name_distinct_identifier = TemplateDataType(Identifier("T"))
    different_name = TemplateDataType(Identifier("U"))

    assert is_structurally_equivalent(left, same_identifier)
    assert left.is_structurally_equivalent(same_identifier)
    assert not is_structurally_equivalent(left, same_name_distinct_identifier)
    assert not is_structurally_equivalent(left, different_name)


def test_template_data_type_equivalence_respects_widths() -> None:
    """Test same-identifier templates with differing widths are not equivalent."""
    t_identifier = Identifier("T")
    eight_bit = TemplateDataType(t_identifier, widths=[8])
    sixteen_bit = TemplateDataType(t_identifier, widths=[16])
    assert not is_structurally_equivalent(eight_bit, sixteen_bit)


# =============================================================================
# Template binding (NumericalType / IndexType / TemplateDataType / PrimitiveDataType)
# =============================================================================


def test_bind_template_then_substitute_round_trips_for_numerical_type(
    empty_environment: TypeUnificationEnvironment,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test a `NumericalType` bind/substitute cycle reproduces the actual type."""
    t_identifier = Identifier("T")
    template_data_type = TemplateDataType(t_identifier)
    n_identifier = Identifier("N")
    m_identifier = Identifier("M")
    pattern = NumericalType(
        template_data_type,
        [
            IdentifierExpression(n_identifier),
            IdentifierExpression(m_identifier),
        ],
    )
    actual = NumericalType(
        float32_data_type, [LiteralExpression(10), LiteralExpression(20)]
    )

    environment = bind_template(pattern, actual, empty_environment)
    assert is_structurally_equivalent(
        environment.get_data_type_binding(t_identifier), float32_data_type
    )
    n_binding = environment.get_expression_binding(n_identifier)
    m_binding = environment.get_expression_binding(m_identifier)
    assert n_binding is not None and n_binding.is_structurally_equivalent(
        LiteralExpression(10)
    )
    assert m_binding is not None and m_binding.is_structurally_equivalent(
        LiteralExpression(20)
    )

    substituted = substitute_template(pattern, environment)
    assert is_structurally_equivalent(substituted, actual)


def test_bind_template_then_substitute_round_trips_for_index_type(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test an `IndexType` bind/substitute cycle reproduces the actual type."""
    n_identifier = Identifier("N")
    pattern = IndexType(
        LiteralExpression(0),
        IdentifierExpression(n_identifier),
        LiteralExpression(1),
    )
    actual = IndexType(
        LiteralExpression(0), LiteralExpression(64), LiteralExpression(1)
    )

    environment = bind_template(pattern, actual, empty_environment)
    binding = environment.get_expression_binding(n_identifier)
    assert binding is not None and binding.is_structurally_equivalent(
        LiteralExpression(64)
    )

    substituted = substitute_template(pattern, environment)
    assert is_structurally_equivalent(substituted, actual)


def test_bind_template_full_type_wildcard_records_entire_actual(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test `[T, ...]` against a concrete type binds the entire actual to ``T``."""
    t_identifier = Identifier("T")
    template_data_type = TemplateDataType(t_identifier)
    pattern = NumericalType(template_data_type, [...])
    actual = NumericalType(
        int32_data_type,
        [LiteralExpression(4), LiteralExpression(5), LiteralExpression(6)],
    )

    environment = bind_template(pattern, actual, empty_environment)
    bound_full_type = environment.get_type_binding(t_identifier)
    assert bound_full_type is not None and is_structurally_equivalent(
        bound_full_type, actual
    )
    assert is_structurally_equivalent(
        environment.get_data_type_binding(t_identifier), int32_data_type
    )

    substituted = substitute_template(pattern, environment)
    assert is_structurally_equivalent(substituted, actual)


def test_bind_template_wildcard_with_concrete_data_type_accepts_any_shape(
    empty_environment: TypeUnificationEnvironment,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test `[concrete, ...]` accepts any actual shape without recording bindings."""
    pattern = NumericalType(float32_data_type, [...])
    actual = NumericalType(
        float32_data_type, [LiteralExpression(1), LiteralExpression(2)]
    )

    environment = bind_template(pattern, actual, empty_environment)
    assert environment.get_data_type_binding(Identifier("anything")) is None
    assert environment.get_type_binding(Identifier("anything")) is None


def test_bind_template_raises_on_shape_rank_mismatch(
    empty_environment: TypeUnificationEnvironment,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test bind raises `VerificationError` when the pattern and actual ranks differ."""
    pattern = NumericalType(float32_data_type, [LiteralExpression(4)])
    actual = NumericalType(
        float32_data_type, [LiteralExpression(4), LiteralExpression(5)]
    )
    with pytest.raises(VerificationError):
        bind_template(pattern, actual, empty_environment)


def test_bind_template_raises_on_concrete_class_mismatch(
    empty_environment: TypeUnificationEnvironment,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test bind raises `VerificationError` when pattern and actual classes differ."""
    pattern = NumericalType(float32_data_type)
    actual = IndexType(LiteralExpression(0), LiteralExpression(10))
    with pytest.raises(VerificationError):
        bind_template(pattern, actual, empty_environment)


def test_bind_template_raises_on_conflicting_shape_binding(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test bind raises when one shape variable binds to two different dimensions."""
    template_data_type = TemplateDataType(Identifier("T"))
    n_identifier = Identifier("N")
    pattern = NumericalType(
        template_data_type,
        [
            IdentifierExpression(n_identifier),
            IdentifierExpression(n_identifier),
        ],
    )
    actual = NumericalType(
        int32_data_type, [LiteralExpression(4), LiteralExpression(5)]
    )
    with pytest.raises(VerificationError):
        bind_template(pattern, actual, empty_environment)


def test_bind_data_template_raises_on_conflicting_binding(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test `bind_data_template` raises on a conflicting second binding."""
    template_data_type = TemplateDataType(Identifier("T"))
    environment = bind_data_template(
        template_data_type, int32_data_type, empty_environment
    )
    with pytest.raises(VerificationError):
        bind_data_template(template_data_type, float32_data_type, environment)


def test_bind_data_template_repeated_consistent_binding_is_idempotent(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test rebinding a template name to the same value leaves the env unchanged."""
    template_data_type = TemplateDataType(Identifier("T"))
    environment = bind_data_template(
        template_data_type, int32_data_type, empty_environment
    )
    same_environment = bind_data_template(
        template_data_type, int32_data_type, environment
    )
    assert environment.is_structurally_equivalent(same_environment)


# =============================================================================
# Width enforcement at bind / unify
# =============================================================================


def test_bind_data_template_accepts_actual_with_matching_width(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test a width-constrained template binds to an actual with an in-list width."""
    template = TemplateDataType(Identifier("T"), widths=[8, 16])
    environment = bind_data_template(
        template, PrimitiveDataType(CoreDataType.INT16), empty_environment
    )
    assert isinstance(
        environment.get_data_type_binding(template.data_type), PrimitiveDataType
    )


def test_bind_data_template_raises_for_actual_with_out_of_list_width(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test a width-constrained template rejects an actual with an out-of-list width."""
    template = TemplateDataType(Identifier("T"), widths=[8, 16])
    with pytest.raises(VerificationError, match="width"):
        bind_data_template(
            template, PrimitiveDataType(CoreDataType.INT32), empty_environment
        )


def test_bind_data_template_raises_for_weak_literal_actual_against_constrained_template(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test a width-constrained template rejects a weak-literal (None width) actual."""
    template = TemplateDataType(Identifier("T"), widths=[8])
    with pytest.raises(VerificationError, match="width"):
        bind_data_template(
            template, PrimitiveDataType(CoreDataType.INT), empty_environment
        )


def test_bind_data_template_empty_widths_rejects_every_concrete_actual(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test ``widths=[]`` causes every concrete bind to raise."""
    template = TemplateDataType(Identifier("T"), widths=[])
    with pytest.raises(VerificationError, match="width"):
        bind_data_template(
            template, PrimitiveDataType(CoreDataType.INT8), empty_environment
        )


def test_bind_data_template_unconstrained_template_accepts_any_width(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test ``widths=None`` (unconstrained) accepts any concrete actual width."""
    template = TemplateDataType(Identifier("T"))
    environment = bind_data_template(
        template, PrimitiveDataType(CoreDataType.FLOAT64), empty_environment
    )
    assert environment.get_data_type_binding(template.data_type) is not None


def test_bind_data_template_raises_on_distinct_template_actual(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test `bind_data_template` rejects a distinct `TemplateDataType` actual."""
    pattern = TemplateDataType(Identifier("T"))
    actual = TemplateDataType(Identifier("U"))
    with pytest.raises(VerificationError, match="template"):
        bind_data_template(pattern, actual, empty_environment)


def test_bind_data_template_accepts_same_template_actual_as_no_op(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test `bind_data_template` no-ops when pattern and actual share an identifier."""
    t_identifier = Identifier("T")
    pattern = TemplateDataType(t_identifier)
    actual = TemplateDataType(t_identifier)
    environment = bind_data_template(pattern, actual, empty_environment)
    assert environment.is_structurally_equivalent(empty_environment)


def test_bind_data_template_widths_are_set_membership_not_order_sensitive(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test ``widths`` uses set-membership semantics (order/duplicates irrelevant)."""
    template = TemplateDataType(Identifier("T"), widths=[16, 8, 8, 16])
    environment = bind_data_template(
        template, PrimitiveDataType(CoreDataType.INT8), empty_environment
    )
    assert environment.get_data_type_binding(template.data_type) is not None


def test_unify_enforces_width_constraint_on_template(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test unification respects the template-data-type width constraint."""
    t_identifier = Identifier("T")
    template = TemplateDataType(t_identifier, widths=[8])
    expected = NumericalType(template, [LiteralExpression(4)])
    actual = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), [LiteralExpression(4)]
    )
    with pytest.raises(VerificationError, match="width"):
        unify(expected, actual, empty_environment)


def test_bind_template_through_numerical_pattern_enforces_width_constraint(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test `bind_template` for `NumericalType` propagates the width constraint."""
    t_identifier = Identifier("T")
    template = TemplateDataType(t_identifier, widths=[8])
    pattern = NumericalType(template, [LiteralExpression(4)])
    actual = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), [LiteralExpression(4)]
    )
    with pytest.raises(VerificationError, match="width"):
        bind_template(pattern, actual, empty_environment)


# =============================================================================
# Template substitution
# =============================================================================


def test_substitute_template_leaves_unbound_placeholders_alone(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test substitution leaves unbound placeholders in the input unchanged."""
    template_data_type = TemplateDataType(Identifier("T"))
    n_identifier = Identifier("N")
    pattern = NumericalType(template_data_type, [IdentifierExpression(n_identifier)])
    substituted = substitute_template(pattern, empty_environment)
    assert is_structurally_equivalent(substituted, pattern)


def test_substitute_template_walks_compound_shape_expressions(
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test substitution recurses into binary shape expressions for placeholders."""
    n_identifier = Identifier("N")
    pattern = NumericalType(
        float32_data_type,
        [
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(n_identifier),
                LiteralExpression(1),
            )
        ],
    )
    environment = TypeUnificationEnvironment.empty().with_expression_binding(
        n_identifier, LiteralExpression(8)
    )
    substituted = substitute_template(pattern, environment)
    assert isinstance(substituted, NumericalType)
    expected_dimension = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(8), LiteralExpression(1)
    )
    dimension = substituted.shape[0]
    assert isinstance(dimension, BinaryExpression)
    assert dimension.is_structurally_equivalent(expected_dimension)


def test_substitute_data_template_resolves_a_bound_template(
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test data-type substitution returns the bound concrete type for a placeholder."""
    t_identifier = Identifier("T")
    template_data_type = TemplateDataType(t_identifier)
    environment = TypeUnificationEnvironment.empty().with_data_type_binding(
        t_identifier, int32_data_type
    )
    substituted = substitute_data_template(template_data_type, environment)
    assert is_structurally_equivalent(substituted, int32_data_type)


def test_substitute_data_template_leaves_an_unbound_template_alone(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test data-type substitution returns the placeholder unchanged when unbound."""
    template_data_type = TemplateDataType(Identifier("T"))
    substituted = substitute_data_template(template_data_type, empty_environment)
    assert isinstance(substituted, TemplateDataType)
    assert substituted.data_type.name_hint == "T"


# =============================================================================
# Unification
# =============================================================================


def test_unify_binds_a_placeholder_when_appearing_on_either_side(
    empty_environment: TypeUnificationEnvironment,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test unification binds placeholders regardless of which side carries them."""
    n_identifier = Identifier("N")
    m_identifier = Identifier("M")
    expected = NumericalType(
        float32_data_type,
        [
            IdentifierExpression(n_identifier),
            IdentifierExpression(m_identifier),
        ],
    )
    actual = NumericalType(
        float32_data_type,
        [LiteralExpression(10), IdentifierExpression(m_identifier)],
    )

    unified, environment = unify(expected, actual, empty_environment)

    assert isinstance(unified, NumericalType)
    expected_unified = NumericalType(
        float32_data_type,
        [LiteralExpression(10), IdentifierExpression(m_identifier)],
    )
    assert is_structurally_equivalent(unified, expected_unified)

    n_binding = environment.get_expression_binding(n_identifier)
    assert n_binding is not None and n_binding.is_structurally_equivalent(
        LiteralExpression(10)
    )
    assert environment.get_expression_binding(m_identifier) is None


def test_unify_raises_when_occurs_check_fails(
    empty_environment: TypeUnificationEnvironment,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test unification raises when binding a placeholder would create a cycle."""
    n_identifier = Identifier("N")
    expected = NumericalType(float32_data_type, [IdentifierExpression(n_identifier)])
    actual = NumericalType(
        float32_data_type,
        [
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(n_identifier),
                LiteralExpression(1),
            )
        ],
    )
    with pytest.raises(VerificationError):
        unify(expected, actual, empty_environment)


def test_unify_returns_index_types_unchanged_when_already_structurally_equal(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test unifying two structurally equal `IndexType`s returns them unchanged."""
    expected = IndexType(
        LiteralExpression(0), LiteralExpression(10), LiteralExpression(1)
    )
    actual = IndexType(
        LiteralExpression(0), LiteralExpression(10), LiteralExpression(1)
    )
    unified, environment = unify(expected, actual, empty_environment)
    assert is_structurally_equivalent(unified, expected)
    assert environment.is_structurally_equivalent(empty_environment)


def test_unify_raises_on_mismatched_concrete_types(
    empty_environment: TypeUnificationEnvironment,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test unification raises when the two concrete type classes are incompatible."""
    expected = NumericalType(float32_data_type)
    actual = IndexType(LiteralExpression(0), LiteralExpression(10))
    with pytest.raises(VerificationError):
        unify(expected, actual, empty_environment)


def test_unify_binds_data_type_template_appearing_on_either_side(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test unification binds a `TemplateDataType` when it appears on either side."""
    t_identifier = Identifier("T")
    template_data_type = TemplateDataType(t_identifier)
    expected = NumericalType(template_data_type, [LiteralExpression(4)])
    actual = NumericalType(int32_data_type, [LiteralExpression(4)])
    unified, environment = unify(expected, actual, empty_environment)
    assert is_structurally_equivalent(unified, actual)
    assert is_structurally_equivalent(
        environment.get_data_type_binding(t_identifier), int32_data_type
    )


def test_unify_raises_on_distinct_template_data_types(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test unifying two distinct `TemplateDataType` placeholders raises."""
    left_template = TemplateDataType(Identifier("T"))
    right_template = TemplateDataType(Identifier("U"))
    expected = NumericalType(left_template, [LiteralExpression(4)])
    actual = NumericalType(right_template, [LiteralExpression(4)])
    with pytest.raises(VerificationError):
        unify(expected, actual, empty_environment)


def test_unify_treats_same_name_distinct_identifiers_as_distinct_templates(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test two ``TemplateDataType`` values backed by distinct ``Identifier``s raise."""
    expected = NumericalType(TemplateDataType(Identifier("T")), [LiteralExpression(4)])
    actual = NumericalType(TemplateDataType(Identifier("T")), [LiteralExpression(4)])
    with pytest.raises(VerificationError):
        unify(expected, actual, empty_environment)


def test_unify_accepts_a_template_with_itself(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test unifying two `TemplateDataType`s sharing one identifier is a no-op."""
    t_identifier = Identifier("T")
    expected = NumericalType(TemplateDataType(t_identifier), [LiteralExpression(4)])
    actual = NumericalType(TemplateDataType(t_identifier), [LiteralExpression(4)])
    unified, environment = unify(expected, actual, empty_environment)
    assert is_structurally_equivalent(unified, expected)
    assert environment.is_structurally_equivalent(empty_environment)


def test_unify_raises_on_index_type_with_mismatched_literal_bound(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test unifying two `IndexType`s with different literal bounds raises."""
    expected = IndexType(LiteralExpression(0), LiteralExpression(10))
    actual = IndexType(LiteralExpression(0), LiteralExpression(11))
    with pytest.raises(VerificationError):
        unify(expected, actual, empty_environment)


# =============================================================================
# `unify_expression`
# =============================================================================


def test_unify_expression_returns_identical_concrete_expressions_unchanged(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test unifying two identical concrete expressions records no new bindings."""
    left = LiteralExpression(7)
    right = LiteralExpression(7)
    unified, environment = unify_expression(left, right, empty_environment)
    assert unified.is_structurally_equivalent(LiteralExpression(7))
    assert environment.is_structurally_equivalent(empty_environment)


def test_unify_expression_binds_left_placeholder_to_right_concrete(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test a left-side placeholder binds to the right's concrete expression."""
    n_identifier = Identifier("N")
    left = IdentifierExpression(n_identifier)
    right = LiteralExpression(10)
    unified, environment = unify_expression(left, right, empty_environment)
    assert unified.is_structurally_equivalent(LiteralExpression(10))
    binding = environment.get_expression_binding(n_identifier)
    assert binding is not None and binding.is_structurally_equivalent(
        LiteralExpression(10)
    )


def test_unify_expression_binds_right_placeholder_to_left_concrete(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test a right-side placeholder binds to the left's concrete expression."""
    n_identifier = Identifier("N")
    left = LiteralExpression(10)
    right = IdentifierExpression(n_identifier)
    unified, environment = unify_expression(left, right, empty_environment)
    assert unified.is_structurally_equivalent(LiteralExpression(10))
    binding = environment.get_expression_binding(n_identifier)
    assert binding is not None and binding.is_structurally_equivalent(
        LiteralExpression(10)
    )


def test_unify_expression_resolves_through_binary_expression_with_placeholder(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test a placeholder pre-bound to a binary expression resolves during unify."""
    n_identifier = Identifier("N")
    pre_bound_environment = empty_environment.with_expression_binding(
        n_identifier,
        BinaryExpression(
            BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
        ),
    )
    left = IdentifierExpression(n_identifier)
    right = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )
    unified, environment = unify_expression(left, right, pre_bound_environment)
    assert unified.is_structurally_equivalent(right)
    assert environment.is_structurally_equivalent(pre_bound_environment)


def test_unify_expression_raises_on_distinct_concrete_expressions(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test unifying two distinct concrete expressions raises `VerificationError`."""
    left = LiteralExpression(1)
    right = LiteralExpression(2)
    with pytest.raises(VerificationError):
        unify_expression(left, right, empty_environment)


def test_unify_expression_raises_when_occurs_check_fails(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test binding a placeholder into an expression that contains it raises."""
    n_identifier = Identifier("N")
    left = IdentifierExpression(n_identifier)
    right = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(n_identifier),
        LiteralExpression(1),
    )
    with pytest.raises(VerificationError):
        unify_expression(left, right, empty_environment)


def test_unify_expression_raises_when_occurs_check_fails_indirectly_via_binding(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test the occurs check follows existing bindings when detecting cycles."""
    m_identifier = Identifier("M")
    n_identifier = Identifier("N")
    pre_bound_environment = empty_environment.with_expression_binding(
        m_identifier, IdentifierExpression(n_identifier)
    )
    left = IdentifierExpression(n_identifier)
    right = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(m_identifier),
        LiteralExpression(1),
    )
    with pytest.raises(VerificationError):
        unify_expression(left, right, pre_bound_environment)


def test_unify_expression_raises_when_occurs_check_fails_indirectly_on_right(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test the indirect occurs check fires for right-side placeholders too."""
    m_identifier = Identifier("M")
    n_identifier = Identifier("N")
    pre_bound_environment = empty_environment.with_expression_binding(
        m_identifier, IdentifierExpression(n_identifier)
    )
    left = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(m_identifier),
        LiteralExpression(1),
    )
    right = IdentifierExpression(n_identifier)
    with pytest.raises(VerificationError):
        unify_expression(left, right, pre_bound_environment)


def test_unify_expression_does_not_treat_concrete_binding_as_indirect_cycle(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test the indirect occurs check does not false-positive on concrete bindings."""
    m_identifier = Identifier("M")
    n_identifier = Identifier("N")
    pre_bound_environment = empty_environment.with_expression_binding(
        m_identifier, LiteralExpression(5)
    )
    left = IdentifierExpression(n_identifier)
    right = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(m_identifier),
        LiteralExpression(1),
    )
    unified, environment = unify_expression(left, right, pre_bound_environment)
    assert unified.is_structurally_equivalent(right)
    n_binding = environment.get_expression_binding(n_identifier)
    assert n_binding is not None and n_binding.is_structurally_equivalent(right)


def test_unify_expression_chains_through_existing_placeholder_binding(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test unifying ``X`` with ``Z`` when ``X -> Y`` extends to a ``Y -> Z`` chain."""
    x_identifier = Identifier("X")
    y_identifier = Identifier("Y")
    z_identifier = Identifier("Z")
    pre_bound_environment = empty_environment.with_expression_binding(
        x_identifier, IdentifierExpression(y_identifier)
    )
    left = IdentifierExpression(x_identifier)
    right = IdentifierExpression(z_identifier)
    _, environment = unify_expression(left, right, pre_bound_environment)
    y_binding = environment.get_expression_binding(y_identifier)
    assert y_binding is not None and y_binding.is_structurally_equivalent(
        IdentifierExpression(z_identifier)
    )
    x_binding = environment.get_expression_binding(x_identifier)
    assert x_binding is not None and x_binding.is_structurally_equivalent(
        IdentifierExpression(y_identifier)
    )


# =============================================================================
# Dispatcher defaults
# =============================================================================


class _UnregisteredType(Type):
    """`Type` subclass with no dispatcher handlers registered.

    Exists to drive the singledispatch fallbacks in ``bind_template``,
    ``substitute_template``, and ``unify``.
    """

    _tag: str

    def __init__(self, tag: str) -> None:
        super().__init__()
        self._tag = tag
        self.freeze(deep=True)

    @property
    def tag(self) -> str:
        return self._tag

    def serialize_data_to_dict(self) -> SerializedDict:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def deserialize_data_from_dict(  # pragma: no cover
        cls, data: SerializedDict
    ) -> "_UnregisteredType":
        raise NotImplementedError


class _UnregisteredDataType(DataType):
    """`DataType` subclass with no dispatcher handlers registered.

    Exists to drive the singledispatch fallback in ``bind_data_template``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.freeze(deep=True)

    def serialize_data_to_dict(self) -> SerializedDict:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def deserialize_data_from_dict(  # pragma: no cover
        cls, data: SerializedDict
    ) -> "_UnregisteredDataType":
        raise NotImplementedError


def test_substitute_template_default_raises_type_error_for_non_type(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test the `substitute_template` default raises `TypeError` on a non-`Type`."""
    with pytest.raises(TypeError, match="Type"):
        substitute_template("not-a-type", empty_environment)


def test_substitute_data_template_default_raises_type_error_for_non_data_type(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test `substitute_data_template` default raises `TypeError` on non-`DataType`."""
    with pytest.raises(TypeError, match="DataType"):
        substitute_data_template("not-a-data-type", empty_environment)


def test_bind_template_default_raises_for_unregistered_class_pair(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test the `bind_template` default raises when no handler is registered."""
    pattern = _UnregisteredType("a")
    actual = _UnregisteredType("b")
    with pytest.raises(VerificationError):
        bind_template(pattern, actual, empty_environment)


def test_unify_default_raises_for_unregistered_class_pair(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test the `unify` default raises when no handler is registered."""
    expected = _UnregisteredType("a")
    actual = _UnregisteredType("b")
    with pytest.raises(VerificationError):
        unify(expected, actual, empty_environment)


def test_bind_data_template_default_raises_for_unregistered_class_pair(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test the `bind_data_template` default raises when no handler is registered."""
    pattern = _UnregisteredDataType()
    actual = _UnregisteredDataType()
    with pytest.raises(VerificationError):
        bind_data_template(pattern, actual, empty_environment)
