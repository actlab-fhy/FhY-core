"""Tests for the `Param` ABC and shared base behavior."""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.constraint import EquationConstraint, InSetConstraint
from fhy_core.error import _COMPILER_ERRORS
from fhy_core.param import (
    CategoricalParam,
    IntParam,
    NatParam,
    NumericParam,
    OrdinalParam,
    Param,
    ParamError,
    PermParam,
    RealParam,
)
from fhy_core.serialization import SerializedDict
from fhy_core.symbol_type import SymbolType
from fhy_core.traits import FrozenMutationError, StructuralEquivalence
from fhy_core.utils.override import override

from .conftest import assert_all_satisfied, assert_none_satisfied, mock_identifier

# =============================================================================
# Error registration
# =============================================================================


def test_param_error_is_value_error_subclass() -> None:
    """Test `ParamError` subclasses `ValueError` for legacy call-site compatibility."""
    assert issubclass(ParamError, ValueError)


def test_param_error_is_registered_in_compiler_error_registry() -> None:
    """Test `ParamError` is registered via the `@register_error` decorator."""
    assert ParamError in _COMPILER_ERRORS


# =============================================================================
# Structural equivalence
# =============================================================================


def test_param_implementation_satisfies_structural_equivalence_protocol() -> None:
    """Test `Param` implementations satisfy the `StructuralEquivalence` protocol."""
    assert isinstance(IntParam(), StructuralEquivalence)


def test_structural_equivalence_is_true_for_constraint_reordering() -> None:
    """Test params compare structurally when equivalent constraints are reordered."""
    shared_name = mock_identifier("x", 1)
    # A separately-constructed identifier with the same id is equal but distinct,
    # standing in for a serialize/deserialize round-tripped copy.
    shared_name_copy = mock_identifier("x", 1)

    left_base = IntParam(name=shared_name)
    right_base = IntParam(name=shared_name_copy)

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

    left_base = IntParam(name=shared_name)
    right_base = IntParam(name=shared_name_copy)

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


def test_structural_equivalence_is_false_for_non_param_other() -> None:
    """Test `is_structurally_equivalent` returns ``False`` for a non-`Param` other."""
    assert not IntParam().is_structurally_equivalent(object())
    assert not IntParam().is_structurally_equivalent("not a param")


def test_structural_equivalence_is_false_when_variable_identifiers_differ() -> None:
    """Test params with different variable identifiers are not structurally equivalent.

    Both params have no constraints, so the only discriminator is the variable.
    Pins down the early-return when ``self.variable != other.variable``.
    """
    left = IntParam(name=mock_identifier("x", 1))
    right = IntParam(name=mock_identifier("y", 2))
    assert not left.is_structurally_equivalent(right)


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
    bound_builders: list[Callable[[IntParam], EquationConstraint]] = [
        lambda p: EquationConstraint(p.variable, p.variable_expression >= 0),
        lambda p: EquationConstraint(p.variable, p.variable_expression <= 10),
    ]
    left_base = IntParam(name=shared_name)
    right_base = IntParam(name=shared_name_copy)
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
            RealParam.with_lower_bound(0.0), 1.5, [], -1.0, id="real-lower-bounded"
        ),
        pytest.param(IntParam.with_lower_bound(0), 3, 1.2, -1, id="int-lower-bounded"),
        pytest.param(OrdinalParam([1, 2, 3]), 2, "2", 4, id="ordinal-1-2-3"),
        pytest.param(CategoricalParam({"a", "b"}), "a", 3, "z", id="categorical-a-b"),
        pytest.param(
            PermParam(["n", "c", "h", "w"]),
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
    param = OrdinalParam([1, 2, 3])
    updated = param.add_constraint(InSetConstraint(param.variable, {1, 2}))
    assert_all_satisfied(updated, [1, 2])
    assert_none_satisfied(updated, [3])
    assert_all_satisfied(param, [1, 2, 3])


def test_add_constraints_applies_multiple_constraints_in_one_call() -> None:
    """Test `add_constraints` adds multiple constraints in a single call."""
    param = RealParam()
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
    original = IntParam.with_lower_bound(0)
    result = original.add_constraints(empty)  # type: ignore[arg-type]  # test: empty variants
    assert result is original
    assert result.is_structurally_equivalent(original)


def test_add_constraints_validates_each_subclass_constraint_rule() -> None:
    """Test `add_constraints` enforces subclass-specific constraint validation."""
    param = OrdinalParam([1, 2, 3])
    with pytest.raises(ParamError):
        param.add_constraints(
            [
                InSetConstraint(param.variable, {1, 2}),
                EquationConstraint(param.variable, param.variable_expression > 1),
            ]
        )


def test_add_constraint_rejects_constraint_with_mismatched_variable() -> None:
    """Test `add_constraint` rejects a constraint built on a different variable."""
    param = IntParam(name=mock_identifier("x", 1))
    other = mock_identifier("y", 2)
    with pytest.raises(ParamError):
        param.add_constraint(EquationConstraint(other, param.variable_expression >= 0))


def test_add_constraint_deduplicates_structurally_equivalent_constraints() -> None:
    """Test `add_constraint` skips a structurally-equivalent existing constraint.

    Adding the same logical constraint twice should not pollute the internal
    constraint tuple. Returning ``self`` is the explicit "no-op" signal.
    """
    param = IntParam(name=mock_identifier("x", 1))
    constraint = EquationConstraint(param.variable, param.variable_expression >= 0)

    once = param.add_constraint(constraint)
    twice = once.add_constraint(constraint)

    assert len(once._constraints) == 1
    assert len(twice._constraints) == 1
    assert twice is once


def test_add_constraints_deduplicates_within_input_and_against_existing() -> None:
    """Test `add_constraints` dedups within the new list and against existing.

    A list containing three structurally-identical constraints results in
    a single added constraint. If the same logical constraint is already
    present, no new entry is added.
    """
    param = IntParam(name=mock_identifier("x", 1))
    constraint = EquationConstraint(param.variable, param.variable_expression >= 0)

    bulk = param.add_constraints([constraint, constraint, constraint])
    assert len(bulk._constraints) == 1

    again = bulk.add_constraints([constraint])
    assert len(again._constraints) == 1
    assert again is bulk


def test_constructor_constraints_kwarg_stores_validated_constraints() -> None:
    """Test `Param.__init__` accepts and stores a ``constraints`` sequence."""
    variable = mock_identifier("x", 1)
    lower = EquationConstraint(
        variable, IntParam(name=variable).variable_expression >= 0
    )
    upper = EquationConstraint(
        variable, IntParam(name=variable).variable_expression <= 9
    )

    param = IntParam(name=variable, constraints=(lower, upper))

    assert len(param.constraints) == 2


def test_constructor_constraints_kwarg_rejects_mismatched_variable() -> None:
    """Test `Param.__init__` validates each constraint against its variable."""
    variable = mock_identifier("x", 1)
    other = mock_identifier("y", 2)
    bad = EquationConstraint(other, IntParam(name=variable).variable_expression >= 0)

    with pytest.raises(ParamError):
        IntParam(name=variable, constraints=(bad,))


def test_constructor_constraints_kwarg_deduplicates_structurally_equivalent() -> None:
    """Test `Param.__init__` dedups structurally-equivalent incoming constraints."""
    variable = mock_identifier("x", 1)
    constraint = EquationConstraint(
        variable, IntParam(name=variable).variable_expression >= 0
    )

    param = IntParam(name=variable, constraints=(constraint, constraint))

    assert len(param.constraints) == 1


def test_add_constraint_returns_frozen_immutable_param() -> None:
    """Test `add_constraint` returns a frozen parameter that rejects mutation."""
    param = IntParam(name=mock_identifier("x", 1))
    constraint = EquationConstraint(param.variable, param.variable_expression >= 0)

    updated = param.add_constraint(constraint)

    assert updated.is_frozen is True
    with pytest.raises(FrozenMutationError):
        updated._variable = mock_identifier("hacked", 99)


def test_constraint_builder_methods_return_frozen_params() -> None:
    """Test the public constraint-builder methods return frozen parameters."""
    assert IntParam.with_lower_bound(0).is_frozen is True
    assert IntParam.between(0, 10).is_frozen is True


def test_constraints_property_returns_constraint_tuple() -> None:
    """Test the public ``constraints`` property exposes the constraint tuple."""
    variable = mock_identifier("x", 1)
    constraint = EquationConstraint(
        variable, IntParam(name=variable).variable_expression >= 0
    )

    param = IntParam(name=variable, constraints=(constraint,))

    assert param.constraints == (constraint,)
    assert isinstance(param.constraints, tuple)


def test_with_new_constraints_replaces_constraints_keeping_definition() -> None:
    """Test ``with_new_constraints`` replaces constraints without changing shape."""
    param = OrdinalParam([3, 1, 2])
    first = InSetConstraint(param.variable, {1, 2})
    second = InSetConstraint(param.variable, {2, 3})
    with_first = param.add_constraint(first)

    replaced = with_first.with_new_constraints((second,))

    assert replaced.constraints == (second,)
    assert replaced.possible_values == (1, 2, 3)
    assert replaced.is_frozen is True
    assert with_first.constraints == (first,)


def test_with_new_constraints_deduplicates_structurally_equivalent_constraints() -> (
    None
):
    """Test ``with_new_constraints`` drops structurally-equivalent duplicates."""
    param = IntParam(name=mock_identifier("x", 1))
    constraint = EquationConstraint(param.variable, param.variable_expression >= 0)

    deduped = param.with_new_constraints((constraint, constraint))

    assert len(deduped.constraints) == 1


def test_with_new_constraints_rejects_constraint_with_mismatched_variable() -> None:
    """Test ``with_new_constraints`` validates each constraint against the variable."""
    param = IntParam(name=mock_identifier("x", 1))
    mismatched = EquationConstraint(
        mock_identifier("y", 2), param.variable_expression >= 0
    )

    with pytest.raises(ParamError):
        param.with_new_constraints((mismatched,))


def test_add_constraint_preserves_ordinal_values() -> None:
    """Test ``add_constraint`` preserves an ``OrdinalParam``'s possible values."""
    param = OrdinalParam([3, 1, 2])

    updated = param.add_constraint(InSetConstraint(param.variable, {1, 2}))

    assert updated.possible_values == (1, 2, 3)
    assert updated.is_frozen is True
    assert len(updated.constraints) == 1
    assert param.constraints == ()


def test_add_constraint_preserves_categorical_categories() -> None:
    """Test ``add_constraint`` preserves a ``CategoricalParam``'s categories."""
    param = CategoricalParam({"a", "b"})

    updated = param.add_constraint(InSetConstraint(param.variable, {"a"}))

    assert updated.categories == frozenset({"a", "b"})
    assert updated.is_frozen is True


def test_add_constraint_preserves_perm_members() -> None:
    """Test ``add_constraint`` preserves a ``PermParam``'s ordered members."""
    param = PermParam(["x", "y", "z"])

    updated = param.add_constraint(InSetConstraint(param.variable, {("x", "y", "z")}))

    assert updated.members == ("x", "y", "z")
    assert updated.is_frozen is True


def test_nat_param_reconstructed_via_constraints_kwarg_does_not_double_basic() -> None:
    """Test constructing a `NatParam` with its own constraints keeps one basic.

    The auto-added ``var >= 0`` basic constraint dedups against an incoming
    copy of itself, so passing a ``NatParam``'s own constraints back through
    the ``constraints`` argument does not accumulate duplicates.
    """
    base = NatParam()
    assert len(base.constraints) == 1

    rebuilt = NatParam(name=base.variable, constraints=base.constraints)

    assert len(rebuilt.constraints) == 1


def test_nat_param_explicit_lower_bound_zero_does_not_duplicate_basic_constraint() -> (
    None
):
    """Test ``NatParam(is_zero_included=False)`` dedups an explicit ``var > 0`` add.

    The basic constraint ``var > 0`` is inserted during ``__init__``. An
    explicit ``add_lower_bound_constraint(0, is_inclusive=False)`` builds
    the same constraint, so structural-equivalence dedup leaves the
    result with a single constraint.
    """
    base = NatParam(is_zero_included=False)
    assert len(base._constraints) == 1

    extended = base.add_lower_bound_constraint(0, is_inclusive=False)
    assert len(extended._constraints) == 1
    assert extended is base


def test_param_default_is_structurally_equivalent_returns_false_for_non_param() -> None:
    """Test the unsubclassed `Param.is_structurally_equivalent` is ``False`` off-class.

    Concrete `Param` subclasses override `is_structurally_equivalent` to inject
    an `isinstance` short-circuit, so the base-class branch ``if not
    isinstance(other, Param): return False`` is unreachable through them.
    Define a `Param` subclass that does *not* override the method and assert
    the base-class branch returns ``False`` for a non-`Param` other.
    """

    class _NoEquivalenceOverride(Param[int]):
        @override
        def is_value_admissible(self, value: Any) -> bool:
            return isinstance(value, int) and not isinstance(value, bool)

        @override
        def is_value_set_subset(self, other: Param[int]) -> bool:
            return True

        @override
        def is_subset(self, other: Param[int]) -> bool:
            return True

        @override
        def _get_param_set_str(self) -> str:
            return "Z"

        @classmethod
        @override
        def deserialize_data_from_dict(
            cls, data: SerializedDict
        ) -> "_NoEquivalenceOverride":
            return cls()

    assert not _NoEquivalenceOverride().is_structurally_equivalent("not a param")


# =============================================================================
# Symbol types
# =============================================================================


def test_real_param_symbol_type_is_real() -> None:
    """Test `RealParam.get_symbol_type` returns `SymbolType.REAL`."""
    assert RealParam().get_symbol_type() == SymbolType.REAL


def test_int_param_symbol_type_is_int() -> None:
    """Test `IntParam.get_symbol_type` returns `SymbolType.INT`."""
    assert IntParam().get_symbol_type() == SymbolType.INT


# =============================================================================
# NumericParam intermediate class membership
# =============================================================================


def test_real_param_is_numeric_param() -> None:
    """Test ``RealParam`` is a ``NumericParam`` subclass."""

    assert isinstance(RealParam(), NumericParam)


def test_int_param_is_numeric_param() -> None:
    """Test ``IntParam`` is a ``NumericParam`` subclass."""

    assert isinstance(IntParam(), NumericParam)


def test_ordinal_param_is_not_numeric_param() -> None:
    """Test ``OrdinalParam`` is not a ``NumericParam`` subclass."""

    assert not isinstance(OrdinalParam([1, 2, 3]), NumericParam)


def test_categorical_param_is_not_numeric_param() -> None:
    """Test ``CategoricalParam`` is not a ``NumericParam`` subclass."""

    assert not isinstance(CategoricalParam([1, 2]), NumericParam)


def test_perm_param_is_not_numeric_param() -> None:
    """Test ``PermParam`` is not a ``NumericParam`` subclass."""

    assert not isinstance(PermParam([1, 2]), NumericParam)


# =============================================================================
# Abstract method enforcement
# =============================================================================


def test_numeric_param_subclass_must_implement_get_symbol_type() -> None:
    """Test a ``NumericParam`` subclass without ``get_symbol_type`` is uninstantiable.

    ``get_symbol_type`` is declared abstract on ``NumericParam``; a
    subclass that omits it must remain abstract and refuse instantiation.
    """

    class _MissingSymbolTypeNumeric(NumericParam[int]):
        @override
        def is_value_admissible(self, value: Any) -> bool:
            return isinstance(value, int)

        @override
        def is_value_set_subset(self, other: Param[int]) -> bool:
            return True

        @override
        def _get_param_set_str(self) -> str:
            return "Z"

        @classmethod
        @override
        def deserialize_data_from_dict(
            cls, data: SerializedDict
        ) -> "_MissingSymbolTypeNumeric":
            return cls()

    with pytest.raises(TypeError):
        _MissingSymbolTypeNumeric()  # type: ignore[abstract]  # test: ABC instantiation


def test_param_subclass_must_implement_is_value_admissible() -> None:
    """Test a `Param` subclass without `is_value_admissible` cannot be instantiated.

    The subclass implements every other abstract method on `Param` so that
    ``is_value_admissible`` is the *only* unimplemented abstract slot.
    """

    class _MissingIsValueAdmissible(Param[int]):
        @override
        def is_value_set_subset(self, other: Param[int]) -> bool:
            return True

        @override
        def is_subset(self, other: Param[int]) -> bool:
            return True

        @override
        def _get_param_set_str(self) -> str:
            return "Z"

        @classmethod
        @override
        def deserialize_data_from_dict(
            cls, data: SerializedDict
        ) -> "_MissingIsValueAdmissible":
            return cls()

    with pytest.raises(TypeError):
        _MissingIsValueAdmissible()  # type: ignore[abstract]  # test: ABC instantiation


def test_param_subclass_must_implement_get_param_set_str() -> None:
    """Test a `Param` subclass without `_get_param_set_str` cannot be instantiated.

    The subclass implements every other abstract method on `Param` so that
    ``_get_param_set_str`` is the *only* unimplemented abstract slot.
    """

    class _MissingGetParamSetStr(Param[int]):
        @override
        def is_value_admissible(self, value: Any) -> bool:
            return isinstance(value, int)

        @override
        def is_value_set_subset(self, other: Param[int]) -> bool:
            return True

        @override
        def is_subset(self, other: Param[int]) -> bool:
            return True

        @classmethod
        @override
        def deserialize_data_from_dict(
            cls, data: SerializedDict
        ) -> "_MissingGetParamSetStr":
            return cls()

    with pytest.raises(TypeError):
        _MissingGetParamSetStr()  # type: ignore[abstract]  # test: ABC instantiation


def test_param_subclass_must_implement_is_value_set_subset() -> None:
    """Test a `Param` subclass without `is_value_set_subset` cannot be instantiated.

    The subclass implements every other abstract method on `Param` so that
    ``is_value_set_subset`` is the *only* unimplemented abstract slot.
    """

    class _MissingIsValueSetSubset(Param[int]):
        @override
        def is_value_admissible(self, value: Any) -> bool:
            return isinstance(value, int)

        @override
        def is_subset(self, other: Param[int]) -> bool:
            return True

        @override
        def _get_param_set_str(self) -> str:
            return "Z"

        @classmethod
        @override
        def deserialize_data_from_dict(
            cls, data: SerializedDict
        ) -> "_MissingIsValueSetSubset":
            return cls()

    with pytest.raises(TypeError):
        _MissingIsValueSetSubset()  # type: ignore[abstract]  # test: ABC instantiation


def test_param_subclass_must_implement_is_subset() -> None:
    """Test a ``Param`` subclass without ``is_subset`` cannot be instantiated.

    ``is_subset`` is abstract on ``Param``: concrete subclasses must
    implement it (either via ``NumericParam``'s Z3 path or via their own
    set-semantics path).
    """

    class _MissingIsSubset(Param[int]):
        @override
        def is_value_admissible(self, value: Any) -> bool:
            return isinstance(value, int)

        @override
        def is_value_set_subset(self, other: Param[int]) -> bool:
            return True

        @override
        def _get_param_set_str(self) -> str:
            return "Z"

        @classmethod
        @override
        def deserialize_data_from_dict(cls, data: SerializedDict) -> "_MissingIsSubset":
            return cls()

    with pytest.raises(TypeError):
        _MissingIsSubset()  # type: ignore[abstract]  # test: ABC instantiation


# =============================================================================
# repr
# =============================================================================


def test_repr_omits_param_set_separator_when_param_set_repr_is_empty() -> None:
    """Test `__repr__` does not insert a stray ``", "`` for an empty param-set repr."""
    text = repr(IntParam(name=mock_identifier("x", 1)))
    assert ", , " not in text
    assert "constraints=" in text


def test_repr_inserts_param_set_separator_when_param_set_repr_is_non_empty() -> None:
    """Test `__repr__` inserts ``", "`` after a non-empty param-set repr."""
    text = repr(OrdinalParam([1, 2, 3], name=mock_identifier("x", 1)))
    assert "}, constraints=" in text


# =============================================================================
# Misc - serialize_data_to_dict shape sanity
# =============================================================================


def test_serialize_data_to_dict_includes_variable_and_constraints_keys() -> None:
    """Test `Param.serialize_data_to_dict` exposes ``variable`` and ``constraints``."""
    param = IntParam.with_lower_bound(0)
    data: SerializedDict = param.serialize_data_to_dict()
    assert "variable" in data
    assert "constraints" in data
