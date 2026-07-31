"""Tests for the `Param` container and shared base behavior."""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.error import _COMPILER_ERRORS
from fhy_core.symbolic.constraint import EquationConstraint, InSetConstraint
from fhy_core.symbolic.param import (
    CategoricalDomain,
    IntegerDomain,
    OrdinalDomain,
    Param,
    ParamError,
    PermutationDomain,
    create_categorical_param,
    create_integer_param,
    create_integer_param_between,
    create_integer_param_with_lower_bound,
    create_natural_param,
    create_ordinal_param,
    create_permutation_param,
    create_real_param,
    create_real_param_with_lower_bound,
)
from fhy_core.symbolic.param.domains import build_ordinal_domain
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.traits import FrozenMutationError, StructuralEquivalence

from .conftest import assert_all_satisfied, assert_none_satisfied, mock_identifier

# =============================================================================
# Error registration
# =============================================================================


def test_param_error_is_value_error_subclass() -> None:
    """Test `ParamError` subclasses `ValueError`."""
    assert issubclass(ParamError, ValueError)


def test_param_error_is_registered_in_compiler_error_registry() -> None:
    """Test `ParamError` is registered via the `@register_error` decorator."""
    assert ParamError in _COMPILER_ERRORS


# =============================================================================
# Structural equivalence
# =============================================================================


def test_param_implementation_satisfies_structural_equivalence_protocol() -> None:
    """Test `Param` instances satisfy the `StructuralEquivalence` protocol."""
    assert isinstance(create_integer_param(), StructuralEquivalence)


def test_structural_equivalence_is_true_for_constraint_reordering() -> None:
    """Test params compare structurally when equivalent constraints are reordered."""
    shared_name = mock_identifier("x", 1)
    # A separately-constructed identifier with the same id is equal but distinct,
    # standing in for a serialize/deserialize round-tripped copy.
    shared_name_copy = mock_identifier("x", 1)

    left_base = create_integer_param(name=shared_name)
    right_base = create_integer_param(name=shared_name_copy)

    left = left_base.add_constraints(
        [
            EquationConstraint(left_base.variable, left_base.variable_expression >= 0),
            EquationConstraint(left_base.variable, left_base.variable_expression <= 10),
        ]
    )
    right = right_base.add_constraints(
        [
            EquationConstraint(
                right_base.variable, right_base.variable_expression <= 10
            ),
            EquationConstraint(
                right_base.variable, right_base.variable_expression >= 0
            ),
        ]
    )

    assert left.is_structurally_equivalent(right)


def test_structural_equivalence_is_false_when_constraints_differ() -> None:
    """Test params are not structurally equivalent when constraints differ."""
    shared_name = mock_identifier("x", 1)
    # A separately-constructed identifier with the same id is equal but distinct,
    # standing in for a serialize/deserialize round-tripped copy.
    shared_name_copy = mock_identifier("x", 1)

    left_base = create_integer_param(name=shared_name)
    right_base = create_integer_param(name=shared_name_copy)

    left = left_base.add_constraints(
        [
            EquationConstraint(left_base.variable, left_base.variable_expression >= 0),
            EquationConstraint(left_base.variable, left_base.variable_expression <= 10),
        ]
    )
    right = right_base.add_constraints(
        [
            EquationConstraint(
                right_base.variable, right_base.variable_expression >= 1
            ),
            EquationConstraint(
                right_base.variable, right_base.variable_expression <= 10
            ),
        ]
    )

    assert not left.is_structurally_equivalent(right)


@pytest.mark.parametrize(
    "other",
    [pytest.param(object(), id="object"), pytest.param("not a param", id="string")],
)
def test_structural_equivalence_is_false_for_non_param_other(other: object) -> None:
    """Test `is_structurally_equivalent` returns ``False`` for a non-`Param` other."""
    param = create_integer_param()

    assert not param.is_structurally_equivalent(other)


def test_structural_equivalence_is_false_when_variable_identifiers_differ() -> None:
    """Test params with different variable identifiers are not structurally equivalent.

    Both params have no constraints, so the variable is the only discriminator.
    """
    left = create_integer_param(name=mock_identifier("x", 1))
    right = create_integer_param(name=mock_identifier("y", 2))

    assert not left.is_structurally_equivalent(right)


def test_structural_equivalence_is_false_when_domains_differ() -> None:
    """Test params with the same variable but different domains are not equivalent.

    Both params share a variable and have no constraints, so the only
    discriminator is the value domain. An integer parameter and a real
    parameter occupy different value spaces and must not compare structurally.
    """
    shared_name = mock_identifier("x", 1)
    shared_name_copy = mock_identifier("x", 1)
    integer = create_integer_param(name=shared_name)
    real = create_real_param(name=shared_name_copy)

    assert not integer.is_structurally_equivalent(real)


@pytest.mark.parametrize(
    "self_count, other_count",
    [
        pytest.param(2, 1, id="self-has-more"),
        pytest.param(1, 2, id="other-has-more"),
    ],
)
def test_structural_equivalence_is_false_when_constraint_counts_differ(
    self_count: int, other_count: int
) -> None:
    """Test params with different constraint counts are not structurally equivalent."""
    shared_name = mock_identifier("x", 1)
    # A separately-constructed identifier with the same id is equal but distinct,
    # standing in for a serialize/deserialize round-tripped copy.
    shared_name_copy = mock_identifier("x", 1)
    bound_builders: list[Callable[[Param[int]], EquationConstraint]] = [
        lambda p: EquationConstraint(p.variable, p.variable_expression >= 0),
        lambda p: EquationConstraint(p.variable, p.variable_expression <= 10),
    ]
    left_base = create_integer_param(name=shared_name)
    right_base = create_integer_param(name=shared_name_copy)
    left = left_base.add_constraints(
        [b(left_base) for b in bound_builders[:self_count]]
    )
    right = right_base.add_constraints(
        [b(right_base) for b in bound_builders[:other_count]]
    )

    assert not left.is_structurally_equivalent(right)


# =============================================================================
# is_value_valid (admissibility + constraint satisfaction)
# =============================================================================


@pytest.mark.parametrize(
    ("param", "valid_value", "invalid_type_or_shape_value", "constraint_fail_value"),
    [
        pytest.param(
            create_real_param_with_lower_bound(0.0),
            1.5,
            [],
            -1.0,
            id="real-lower-bounded",
        ),
        pytest.param(
            create_integer_param_with_lower_bound(0), 3, 1.2, -1, id="int-lower-bounded"
        ),
        pytest.param(create_ordinal_param([1, 2, 3]), 2, "2", 4, id="ordinal-1-2-3"),
        pytest.param(
            create_categorical_param({"a", "b"}), "a", 3, "z", id="categorical-a-b"
        ),
        pytest.param(
            create_permutation_param(["n", "c", "h", "w"]),
            ("n", "c", "h", "w"),
            7,
            ("n", "c"),
            id="perm-nchw",
        ),
    ],
)
def test_is_value_valid_combines_admissibility_and_constraint_check(
    param: Param[Any],
    valid_value: object,
    invalid_type_or_shape_value: object,
    constraint_fail_value: object,
) -> None:
    """Test `is_value_valid` returns true only when admissible and satisfying."""
    assert param.is_value_valid(valid_value)
    assert not param.is_value_valid(invalid_type_or_shape_value)
    assert not param.is_value_valid(constraint_fail_value)


# =============================================================================
# Constraint addition (returns new param)
# =============================================================================


def test_add_constraint_returns_new_param_without_mutating_original() -> None:
    """Test `add_constraint` returns a new parameter without mutating the original."""
    param = create_ordinal_param([1, 2, 3])

    updated = param.add_constraint(InSetConstraint(param.variable, {1, 2}))

    assert_all_satisfied(updated, [1, 2])
    assert_none_satisfied(updated, [3])
    assert_all_satisfied(param, [1, 2, 3])


def test_add_constraints_applies_multiple_constraints_in_one_call() -> None:
    """Test `add_constraints` adds multiple constraints in a single call."""
    param = create_real_param()
    constraints = [
        EquationConstraint(param.variable, param.variable_expression >= 1.0),
        EquationConstraint(param.variable, param.variable_expression <= 3.0),
    ]

    param = param.add_constraints(constraints)

    assert_all_satisfied(param, [1.0, 2.0, 3.0])
    assert_none_satisfied(param, [0.5, 3.5])


@pytest.mark.parametrize(
    "empty",
    [
        pytest.param([], id="empty-list"),
        pytest.param((), id="empty-tuple"),
        pytest.param(set(), id="empty-set"),
    ],
)
def test_add_constraints_returns_self_for_empty_collection(empty: object) -> None:
    """Test `add_constraints` returns ``self`` when no new constraints are added.

    Calling ``add_constraints`` with an empty collection (or any input that
    contains only constraints already present) does no work and yields the
    original instance unchanged. Callers needing a fresh instance for
    identity-sensitive purposes must build one explicitly.
    """
    original = create_integer_param_with_lower_bound(0)

    result = original.add_constraints(empty)  # type: ignore[arg-type]  # test: empty variants

    assert result is original
    assert result.is_structurally_equivalent(original)


def test_add_constraints_validates_each_domain_constraint_rule() -> None:
    """Test `add_constraints` enforces domain-specific constraint validation."""
    param = create_ordinal_param([1, 2, 3])
    with pytest.raises(ParamError):
        param.add_constraints(
            [
                InSetConstraint(param.variable, {1, 2}),
                EquationConstraint(param.variable, param.variable_expression > 1),
            ]
        )


def test_add_constraint_rejects_constraint_with_mismatched_variable() -> None:
    """Test `add_constraint` rejects a constraint built on a different variable."""
    param = create_integer_param(name=mock_identifier("x", 1))
    other = mock_identifier("y", 2)
    with pytest.raises(ParamError):
        param.add_constraint(EquationConstraint(other, param.variable_expression >= 0))


def test_add_constraint_deduplicates_structurally_equivalent_constraints() -> None:
    """Test `add_constraint` skips a structurally-equivalent existing constraint.

    Adding the same logical constraint twice should not pollute the internal
    constraint tuple. Returning ``self`` is the explicit "no-op" signal.
    """
    param = create_integer_param(name=mock_identifier("x", 1))
    constraint = EquationConstraint(param.variable, param.variable_expression >= 0)

    once = param.add_constraint(constraint)
    twice = once.add_constraint(constraint)

    assert len(once.constraints) == 1
    assert len(twice.constraints) == 1
    assert twice is once


def test_add_constraints_deduplicates_within_input_and_against_existing() -> None:
    """Test `add_constraints` dedups within the new list and against existing.

    A list containing three structurally-identical constraints results in
    a single added constraint. If the same logical constraint is already
    present, no new entry is added.
    """
    param = create_integer_param(name=mock_identifier("x", 1))
    constraint = EquationConstraint(param.variable, param.variable_expression >= 0)

    bulk = param.add_constraints([constraint, constraint, constraint])
    assert len(bulk.constraints) == 1

    again = bulk.add_constraints([constraint])
    assert len(again.constraints) == 1
    assert again is bulk


def test_constructor_constraints_kwarg_stores_validated_constraints() -> None:
    """Test `Param.__init__` accepts and stores a ``constraints`` sequence."""
    variable = mock_identifier("x", 1)
    lower = EquationConstraint(
        variable, create_integer_param(name=variable).variable_expression >= 0
    )
    upper = EquationConstraint(
        variable, create_integer_param(name=variable).variable_expression <= 9
    )

    param = create_integer_param(name=variable, constraints=(lower, upper))

    assert len(param.constraints) == 2


def test_constructor_constraints_kwarg_rejects_mismatched_variable() -> None:
    """Test `Param.__init__` validates each constraint against its variable."""
    variable = mock_identifier("x", 1)
    other = mock_identifier("y", 2)
    bad = EquationConstraint(
        other, create_integer_param(name=variable).variable_expression >= 0
    )

    with pytest.raises(ParamError):
        create_integer_param(name=variable, constraints=(bad,))


def test_constructor_constraints_kwarg_deduplicates_structurally_equivalent() -> None:
    """Test `Param.__init__` dedups structurally-equivalent incoming constraints."""
    variable = mock_identifier("x", 1)
    constraint = EquationConstraint(
        variable, create_integer_param(name=variable).variable_expression >= 0
    )

    param = create_integer_param(name=variable, constraints=(constraint, constraint))

    assert len(param.constraints) == 1


def test_add_constraint_returns_frozen_immutable_param() -> None:
    """Test `add_constraint` returns a frozen parameter that rejects mutation."""
    param = create_integer_param(name=mock_identifier("x", 1))
    constraint = EquationConstraint(param.variable, param.variable_expression >= 0)

    updated = param.add_constraint(constraint)

    assert updated.is_frozen is True
    with pytest.raises(FrozenMutationError):
        updated._variable = mock_identifier("hacked", 99)


def test_constraint_builder_factories_return_frozen_params() -> None:
    """Test the public constraint-builder factories return frozen parameters."""
    assert create_integer_param_with_lower_bound(0).is_frozen is True
    assert create_integer_param_between(0, 10).is_frozen is True


def test_constraints_property_returns_constraint_tuple() -> None:
    """Test the public ``constraints`` property exposes the constraint tuple."""
    variable = mock_identifier("x", 1)
    constraint = EquationConstraint(
        variable, create_integer_param(name=variable).variable_expression >= 0
    )

    param = create_integer_param(name=variable, constraints=(constraint,))

    assert param.constraints == (constraint,)
    assert isinstance(param.constraints, tuple)


def test_replace_constraints_replaces_constraints_keeping_definition() -> None:
    """Test ``replace_constraints`` replaces constraints without changing shape."""
    param = create_ordinal_param([3, 1, 2])
    first = InSetConstraint(param.variable, {1, 2})
    second = InSetConstraint(param.variable, {2, 3})
    with_first = param.add_constraint(first)

    replaced = with_first.replace_constraints((second,))

    assert replaced.constraints == (second,)
    assert isinstance(replaced.domain, OrdinalDomain)
    assert replaced.domain.sorted_values == (1, 2, 3)
    assert replaced.is_frozen is True
    assert with_first.constraints == (first,)


def test_replace_constraints_deduplicates_structurally_equivalent_constraints() -> None:
    """Test ``replace_constraints`` drops structurally-equivalent duplicates."""
    param = create_integer_param(name=mock_identifier("x", 1))
    constraint = EquationConstraint(param.variable, param.variable_expression >= 0)

    deduped = param.replace_constraints((constraint, constraint))

    assert len(deduped.constraints) == 1


def test_replace_constraints_rejects_constraint_with_mismatched_variable() -> None:
    """Test ``replace_constraints`` validates each constraint against the variable."""
    param = create_integer_param(name=mock_identifier("x", 1))
    mismatched = EquationConstraint(
        mock_identifier("y", 2), param.variable_expression >= 0
    )

    with pytest.raises(ParamError):
        param.replace_constraints((mismatched,))


def test_add_constraint_preserves_ordinal_values() -> None:
    """Test ``add_constraint`` preserves an ordinal domain's sorted values."""
    param = create_ordinal_param([3, 1, 2])

    updated = param.add_constraint(InSetConstraint(param.variable, {1, 2}))

    assert isinstance(updated.domain, OrdinalDomain)
    assert updated.domain.sorted_values == (1, 2, 3)
    assert updated.is_frozen is True
    assert len(updated.constraints) == 1
    assert param.constraints == ()


def test_add_constraint_preserves_categorical_categories() -> None:
    """Test ``add_constraint`` preserves a categorical domain's categories."""
    param = create_categorical_param({"a", "b"})

    updated = param.add_constraint(InSetConstraint(param.variable, {"a"}))

    assert isinstance(updated.domain, CategoricalDomain)
    assert set(updated.domain.categories) == {"a", "b"}
    assert updated.is_frozen is True


def test_add_constraint_preserves_perm_members() -> None:
    """Test ``add_constraint`` preserves a permutation domain's ordered members."""
    param = create_permutation_param(["x", "y", "z"])

    updated = param.add_constraint(InSetConstraint(param.variable, {("x", "y", "z")}))

    assert isinstance(updated.domain, PermutationDomain)
    assert updated.domain.ordered_members == ("x", "y", "z")
    assert updated.is_frozen is True


def test_nat_param_reconstructed_via_constraints_kwarg_does_not_double_basic() -> None:
    """Test constructing a natural param with its own constraints keeps one basic.

    The auto-added ``var >= 0`` basic constraint dedups against an incoming
    copy of itself, so passing a natural parameter's own constraints back
    through the ``constraints`` argument does not accumulate duplicates.
    """
    base = create_natural_param()

    rebuilt = create_natural_param(name=base.variable, constraints=base.constraints)

    assert len(base.constraints) == 1
    assert len(rebuilt.constraints) == 1


def test_nat_param_explicit_lower_bound_zero_does_not_duplicate_basic_constraint() -> (
    None
):
    """Test ``create_natural_param(zero_included=False)`` dedups explicit ``var > 0``.

    The basic constraint ``var > 0`` is inserted during ``__init__``. An
    explicit ``add_lower_bound_constraint(0, is_inclusive=False)`` builds
    the same constraint, so structural-equivalence dedup leaves the
    result with a single constraint.
    """
    base = create_natural_param(zero_included=False)

    extended = base.add_lower_bound_constraint(0, is_inclusive=False)

    assert len(base.constraints) == 1
    assert len(extended.constraints) == 1
    assert extended is base


def test_param_is_structurally_equivalent_returns_false_for_non_param() -> None:
    """Test ``Param.is_structurally_equivalent`` is ``False`` for a non-`Param`."""
    param = create_integer_param()

    assert not param.is_structurally_equivalent("not a param")


# =============================================================================
# Symbol types
# =============================================================================


def test_real_param_symbol_type_is_real() -> None:
    """Test a real parameter's ``symbol_type`` is `SymbolType.REAL`."""
    assert create_real_param().symbol_type == SymbolType.REAL


def test_int_param_symbol_type_is_int() -> None:
    """Test an integer parameter's ``symbol_type`` is `SymbolType.INT`."""
    assert create_integer_param().symbol_type == SymbolType.INT


# =============================================================================
# Numeric vs non-numeric value space (symbol_type presence)
# =============================================================================


def test_real_param_is_numeric() -> None:
    """Test a real parameter occupies a numeric value space (``symbol_type``)."""
    assert create_real_param().symbol_type is not None


def test_int_param_is_numeric() -> None:
    """Test an integer parameter occupies a numeric value space (``symbol_type``)."""
    assert create_integer_param().symbol_type is not None


def test_ordinal_param_is_not_numeric() -> None:
    """Test an ordinal parameter has no numeric value space (``symbol_type`` None)."""
    assert create_ordinal_param([1, 2, 3]).symbol_type is None


def test_categorical_param_is_not_numeric() -> None:
    """Test a categorical parameter is non-numeric (``symbol_type`` is None)."""
    assert create_categorical_param([1, 2]).symbol_type is None


def test_perm_param_is_not_numeric() -> None:
    """Test a permutation parameter is non-numeric (``symbol_type`` is None)."""
    assert create_permutation_param([1, 2]).symbol_type is None


# =============================================================================
# repr
# =============================================================================


def test_repr_starts_with_param_marker() -> None:
    """Test `__repr__` renders as ``Param(...)`` after the inheritance removal."""
    assert repr(create_integer_param(name=mock_identifier("x", 1))).startswith("Param(")


def test_repr_omits_param_set_separator_when_param_set_repr_is_empty() -> None:
    """Test `__repr__` does not insert a stray ``", "`` for an empty param-set repr."""
    text = repr(create_integer_param(name=mock_identifier("x", 1)))

    assert ", , " not in text
    assert "constraints=" in text


def test_repr_inserts_param_set_separator_when_param_set_repr_is_non_empty() -> None:
    """Test `__repr__` inserts ``", "`` after a non-empty param-set repr."""
    text = repr(create_ordinal_param([1, 2, 3], name=mock_identifier("x", 1)))

    assert "}, constraints=" in text


# =============================================================================
# Misc - serialize_to_dict shape sanity
# =============================================================================


def test_serialize_to_dict_includes_domain_variable_and_constraints_keys() -> None:
    """Test `Param.serialize_to_dict` exposes the derived top-level keys."""
    param = create_integer_param_with_lower_bound(0)

    data = param.serialize_to_dict()

    assert "domain" in data
    assert "variable" in data
    assert "constraints" in data


# =============================================================================
# Direct construction with a domain (no factory)
# =============================================================================


def test_param_constructed_directly_with_domain_matches_factory() -> None:
    """Test constructing a `Param` directly with a domain matches its factory."""
    direct: Param[int] = Param(IntegerDomain(), variable=mock_identifier("x", 1))
    via_factory = create_integer_param(name=mock_identifier("x", 1))

    assert direct.is_structurally_equivalent(via_factory)
    assert direct.is_value_admissible(5)
    assert not direct.is_value_admissible("not an int")


def test_param_constructed_directly_with_finite_domain_admits_only_members() -> None:
    """Test a directly-constructed ordinal `Param` admits exactly its members."""
    param: Param[int] = Param(
        build_ordinal_domain((1, 2, 3)), variable=mock_identifier("x", 1)
    )

    assert param.is_value_admissible(2)
    assert not param.is_value_admissible(4)


def test_validate_value_accepts_valid_value() -> None:
    """Test `Param.validate_value` does not raise for a valid value."""
    param = create_integer_param_between(0, 10, name=mock_identifier("x", 1))

    param.validate_value(5)


def test_validate_value_rejects_inadmissible_value() -> None:
    """Test `Param.validate_value` reports an inadmissible value distinctly."""
    param = create_integer_param_between(0, 10, name=mock_identifier("x", 1))

    with pytest.raises(ParamError, match="not admissible"):
        param.validate_value("not an int")


def test_validate_value_rejects_constraint_violating_value() -> None:
    """Test `Param.validate_value` reports a constraint violation distinctly."""
    param = create_integer_param_between(0, 10, name=mock_identifier("x", 1))

    with pytest.raises(ParamError, match="violates constraint"):
        param.validate_value(20)
