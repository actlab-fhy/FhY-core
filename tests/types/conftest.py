"""Shared fixtures and helpers for the `tests/types` sub-package."""

import pytest

from fhy_core.types import CoreDataType, PrimitiveDataType, TypeUnificationEnvironment

from ..conftest import mock_identifier  # noqa: F401  # re-exported below

__all__ = [
    "empty_environment",
    "float32_data_type",
    "int32_data_type",
    "mock_identifier",
]


@pytest.fixture
def int32_data_type() -> PrimitiveDataType:
    return PrimitiveDataType(CoreDataType.INT32)


@pytest.fixture
def float32_data_type() -> PrimitiveDataType:
    return PrimitiveDataType(CoreDataType.FLOAT32)


@pytest.fixture
def empty_environment() -> TypeUnificationEnvironment:
    return TypeUnificationEnvironment.empty()
