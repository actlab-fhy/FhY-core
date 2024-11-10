"""Tests memory instances."""

import pytest
from fhy_core.constraint import (
    Constraint,
    EquationConstraint,
)
from fhy_core.expression import Expression, IdentifierExpression
from fhy_core.identifier import Identifier
from fhy_core.memory_instance import (
    RowMajorMemoryInstance,
    ScalarMemoryInstance,
)


def test_scalar_memory_instance():
    """Test scalar memory instance initializes correctly."""
    address = 0
    memory_instance = ScalarMemoryInstance(address)

    assert memory_instance.address == address


def test_scalar_memory_instance_negative_address_fails():
    """Test scalar memory instance with negative address raises a ValueError."""
    address = -1

    with pytest.raises(ValueError):
        ScalarMemoryInstance(address)


@pytest.fixture
def one_to_ten_no_stride() -> Constraint:
    identifier = Identifier("one_to_ten_no_stride_index")
    index = IdentifierExpression(identifier)
    return EquationConstraint(identifier, (index <= 10) and (index >= 1))


@pytest.fixture
def one_to_ten_stride_2() -> Constraint:
    identifier = Identifier("one_to_ten_stride_2_index")
    index = IdentifierExpression(identifier)
    return EquationConstraint(
        identifier,
        Expression.logical_and(index <= 10, index >= 1, ((index - 1) % 2).equals(0)),
    )


def test_row_major_memory_instance(one_to_ten_no_stride: Constraint):
    """Test row-major memory instance initializes correctly."""
    base_address = 0
    shape = [10]
    element_size_in_bits = 8
    memory_instance = RowMajorMemoryInstance(
        [one_to_ten_no_stride], base_address, shape, element_size_in_bits
    )

    assert memory_instance.base_address == base_address
    assert memory_instance.shape == shape
    assert memory_instance.element_size_in_bits == element_size_in_bits


def test_row_major_memory_instance_negative_base_address_fails(
    one_to_ten_no_stride: Constraint,
):
    """Test row-major memory instance with negative base address raises a ValueError."""
    base_address = -1
    shape = [10]
    element_size_in_bits = 8

    with pytest.raises(ValueError):
        RowMajorMemoryInstance(
            [one_to_ten_no_stride], base_address, shape, element_size_in_bits
        )


def test_row_major_memory_instance_negative_shape_fails(
    one_to_ten_no_stride: Constraint,
):
    """Test row-major memory instance with negative shape raises a ValueError."""
    base_address = 0
    shape = [-1]
    element_size_in_bits = 8

    with pytest.raises(ValueError):
        RowMajorMemoryInstance(
            [one_to_ten_no_stride], base_address, shape, element_size_in_bits
        )


def test_row_major_memory_instance_zero_shape_fails(one_to_ten_no_stride: Constraint):
    """Test row-major memory instance with zero shape raises a ValueError."""
    base_address = 0
    shape = [0]
    element_size_in_bits = 8

    with pytest.raises(ValueError):
        RowMajorMemoryInstance(
            [one_to_ten_no_stride], base_address, shape, element_size_in_bits
        )


def test_row_major_memory_instance_float_shape_fails(one_to_ten_no_stride: Constraint):
    """Test row-major memory instance with float shape raises a TypeError."""
    base_address = 0
    shape = [1.0]
    element_size_in_bits = 8

    with pytest.raises(TypeError):
        RowMajorMemoryInstance(
            [one_to_ten_no_stride], base_address, shape, element_size_in_bits
        )


def test_row_major_memory_instance_negative_element_size_fails(
    one_to_ten_no_stride: Constraint,
):
    """Test row-major memory instance with negative element size raises a ValueError."""
    base_address = 0
    shape = [10]
    element_size_in_bits = -8

    with pytest.raises(ValueError):
        RowMajorMemoryInstance(
            [one_to_ten_no_stride], base_address, shape, element_size_in_bits
        )


def test_indices_in_instance(
    one_to_ten_no_stride: Constraint, one_to_ten_stride_2: Constraint
):
    """Test indices are within the instance."""
    memory_instance = RowMajorMemoryInstance([one_to_ten_no_stride], 0, [10], 8)
    assert memory_instance.is_indices_in_instance([one_to_ten_stride_2])


def test_indices_not_in_instance(
    one_to_ten_no_stride: Constraint, one_to_ten_stride_2: Constraint
):
    """Test indices are not within the instance."""
    memory_instance = RowMajorMemoryInstance([one_to_ten_stride_2], 0, [10], 8)
    assert not memory_instance.is_indices_in_instance([one_to_ten_no_stride])
