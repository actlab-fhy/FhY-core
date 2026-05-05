"""Tests for the `Param` ABC and shared base behavior."""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.constraint import EquationConstraint, InSetConstraint
from fhy_core.error import _COMPILER_ERRORS
from fhy_core.expression import SymbolType
from fhy_core.identifier import Identifier
from fhy_core.param import (
    CategoricalParam,
    IntParam,
    OrdinalParam,
    Param,
    ParamError,
    PermParam,
    RealParam,
)
from fhy_core.serialization import SerializedDict
from fhy_core.trait import StructuralEquivalence

from .conftest import assert_all_satisfied, assert_none_satisfied

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
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())

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
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())

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
    left = IntParam(name=Identifier("x"))
    right = IntParam(name=Identifier("y"))
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
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())
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


def test_add_constraints_validates_each_subclass_constraint_rule() -> None:
    """Test `add_constraints` enforces subclass-specific constraint validation."""
    param = OrdinalParam([1, 2, 3])
    with pytest.raises(ValueError):
        param.add_constraints(
            [
                InSetConstraint(param.variable, {1, 2}),
                EquationConstraint(param.variable, param.variable_expression > 1),
            ]
        )


def test_add_constraint_rejects_constraint_with_mismatched_variable() -> None:
    """Test `add_constraint` rejects a constraint built on a different variable."""
    param = IntParam(name=Identifier("x"))
    other = Identifier("y")
    with pytest.raises(ParamError):
        param.add_constraint(EquationConstraint(other, param.variable_expression >= 0))


def test_param_default_is_structurally_equivalent_returns_false_for_non_param() -> None:
    """Test the unsubclassed `Param.is_structurally_equivalent` is ``False`` off-class.

    Concrete `Param` subclasses override `is_structurally_equivalent` to inject
    an `isinstance` short-circuit, so the base-class branch ``if not
    isinstance(other, Param): return False`` is unreachable through them.
    Define a `Param` subclass that does *not* override the method and assert
    the base-class branch returns ``False`` for a non-`Param` other.
    """

    class _NoEquivalenceOverride(Param[int]):
        def get_symbol_type(self) -> SymbolType:
            return SymbolType.INT

        def is_value_admissible(self, value: Any) -> bool:
            return isinstance(value, int) and not isinstance(value, bool)

        def _get_param_set_str(self) -> str:
            return "Z"

        @classmethod
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
# Abstract method enforcement
# =============================================================================


def test_param_subclass_must_implement_get_symbol_type() -> None:
    """Test a `Param` subclass without `get_symbol_type` cannot be instantiated.

    The subclass implements every other abstract method on `Param` (including
    those inherited from `WrappedFamilySerializable`) so that ``get_symbol_type``
    is the *only* unimplemented abstract slot. Removing the ``@abstractmethod``
    decorator on ``get_symbol_type`` would make the subclass instantiable.
    """

    class _MissingSymbolType(Param[int]):
        def is_value_admissible(self, value: Any) -> bool:
            return isinstance(value, int)

        def _get_param_set_str(self) -> str:
            return "Z"

        @classmethod
        def deserialize_data_from_dict(
            cls, data: SerializedDict
        ) -> "_MissingSymbolType":
            return cls()

    with pytest.raises(TypeError):
        _MissingSymbolType()  # type: ignore[abstract]  # test: ABC instantiation


def test_param_subclass_must_implement_is_value_admissible() -> None:
    """Test a `Param` subclass without `is_value_admissible` cannot be instantiated.

    The subclass implements every other abstract method on `Param` so that
    ``is_value_admissible`` is the *only* unimplemented abstract slot.
    """

    class _MissingIsValueAdmissible(Param[int]):
        def get_symbol_type(self) -> SymbolType:
            return SymbolType.INT

        def _get_param_set_str(self) -> str:
            return "Z"

        @classmethod
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
        def get_symbol_type(self) -> SymbolType:
            return SymbolType.INT

        def is_value_admissible(self, value: Any) -> bool:
            return isinstance(value, int)

        @classmethod
        def deserialize_data_from_dict(
            cls, data: SerializedDict
        ) -> "_MissingGetParamSetStr":
            return cls()

    with pytest.raises(TypeError):
        _MissingGetParamSetStr()  # type: ignore[abstract]  # test: ABC instantiation


# =============================================================================
# repr
# =============================================================================


def test_repr_omits_param_set_separator_when_param_set_repr_is_empty() -> None:
    """Test `__repr__` does not insert a stray ``", "`` for an empty param-set repr."""
    text = repr(IntParam(name=Identifier("x")))
    assert ", , " not in text
    assert "constraints=" in text


def test_repr_inserts_param_set_separator_when_param_set_repr_is_non_empty() -> None:
    """Test `__repr__` inserts ``", "`` after a non-empty param-set repr."""
    text = repr(OrdinalParam([1, 2, 3], name=Identifier("x")))
    assert "}, constraints=" in text


# =============================================================================
# Misc — serialize_data_to_dict shape sanity
# =============================================================================


def test_serialize_data_to_dict_includes_variable_and_constraints_keys() -> None:
    """Test `Param.serialize_data_to_dict` exposes ``variable`` and ``constraints``."""
    param = IntParam.with_lower_bound(0)
    data: SerializedDict = param.serialize_data_to_dict()
    assert "variable" in data
    assert "constraints" in data
