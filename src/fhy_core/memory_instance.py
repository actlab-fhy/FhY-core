"""Memory layouts."""

__all__ = [
    "ColumnMajorMemoryInstance",
    "MemoryInstance",
    "RowMajorMemoryInstance",
    "ScalarMemoryInstance",
]

from abc import ABC
from collections.abc import Iterable, Sequence

from .constraint import Constraint
from .expression import Expression, SymbolType, is_satisfiable, replace_identifiers
from .identifier import Identifier


class MemoryInstance(ABC):
    """Instance of data in memory."""


class ScalarMemoryInstance(MemoryInstance):
    """Scalar memory instance."""

    _address: int

    def __init__(self, address: int) -> None:
        if address < 0:
            raise ValueError("Address must be non-negative.")
        self._address = address

    @property
    def address(self) -> int:
        return self._address


class ArrayMemoryInstance(MemoryInstance, ABC):
    """Array memory instance."""

    _parent_array_indices: list[list[Constraint]]
    _base_address: int
    _shape: list[int]
    _element_size_in_bits: int

    def __init__(
        self,
        indices: Sequence[Iterable[Constraint] | Constraint],
        base_address: int,
        shape: list[int],
        element_size_in_bits: int,
    ) -> None:
        if isinstance(indices, Sequence):
            self._parent_array_indices = [
                list(index) if isinstance(index, Iterable) else [index]
                for index in indices
            ]
        else:
            raise TypeError("Indices must be a sequence.")

        if base_address < 0:
            raise ValueError("Base address must be non-negative.")

        if not all(isinstance(dim_size, int) for dim_size in shape):
            raise TypeError("Shape must be a list of integers.")
        if not all(dim_size > 0 for dim_size in shape):
            raise ValueError("Shape must be a list of positive integers.")

        if element_size_in_bits <= 0:
            raise ValueError("Element size must be positive.")

        self._base_address = base_address
        self._shape = shape
        self._element_size_in_bits = element_size_in_bits

    @property
    def base_address(self) -> int:
        return self._base_address

    @property
    def shape(self) -> list[int]:
        return self._shape

    @property
    def element_size_in_bits(self) -> int:
        return self._element_size_in_bits

    def is_indices_in_instance(
        self, sub_array_indices: Sequence[Iterable[Constraint] | Constraint]
    ) -> bool:
        """Check if the indices of the data are within the instance.

        Args:
            sub_array_indices: Indices to check. Each element of the sequence
                corresponds to a constraint on the index for the corresponding
                dimension.

        Returns:
            True if the indices are within the instance; False otherwise.

        """
        standardized_sub_array_indices = [
            list(index) if isinstance(index, Iterable) else [index]
            for index in sub_array_indices
        ]
        if not len(standardized_sub_array_indices) == len(self._parent_array_indices):
            raise ValueError("Number of indices must match the number of dimensions.")

        for instance_index_constraints, index_constraints in zip(
            self._parent_array_indices, standardized_sub_array_indices
        ):
            constrained_variable = Identifier("index")
            constraint_expressions = []
            for instance_index_constraint, index_constraint in zip(
                instance_index_constraints, index_constraints
            ):
                instance_index_constraint_expression = (
                    instance_index_constraint.convert_to_expression()
                )
                instance_index_constraint_expression = replace_identifiers(
                    instance_index_constraint_expression,
                    {instance_index_constraint.variable: constrained_variable},
                )

                index_constraint_expression = index_constraint.convert_to_expression()
                index_constraint_expression = replace_identifiers(
                    index_constraint_expression,
                    {index_constraint.variable: constrained_variable},
                )

                constraint_expressions.extend(
                    [
                        instance_index_constraint_expression.logical_not(),
                        index_constraint_expression,
                    ]
                )

            all_constraints_expression = Expression.logical_and(*constraint_expressions)
            if is_satisfiable(
                {constrained_variable},
                all_constraints_expression,
                {constrained_variable: SymbolType.INT},
            ):
                return False

        return True


class RowMajorMemoryInstance(ArrayMemoryInstance):
    """Row-major instance."""


class ColumnMajorMemoryInstance(ArrayMemoryInstance):
    """Column-major instance."""
