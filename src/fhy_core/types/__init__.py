"""Core type system, its extensible dispatchers, and type checking.

The vocabulary of the type system itself -- data types, qualifiers,
unification -- is re-exported here. Type checking is a family of its own
and keeps its own namespace: reach it at ``fhy_core.types.checking``.
"""

__all__ = [
    "CoreDataType",
    "DataType",
    "FhYCoreTypeError",
    "IndexType",
    "NumericalType",
    "PrimitiveDataType",
    "TemplateDataType",
    "Type",
    "TypeQualifier",
    "TypeUnificationEnvironment",
    "bind_data_template",
    "bind_template",
    "checking",
    "get_core_data_type_bit_width",
    "is_structurally_equivalent",
    "is_weak_core_data_type",
    "promote_core_data_types",
    "promote_primitive_data_types",
    "promote_type_qualifiers",
    "resolve_literal_core_data_type",
    "substitute_data_template",
    "substitute_template",
    "unify",
    "unify_expression",
]

from . import checking
from .core import (
    CoreDataType,
    DataType,
    FhYCoreTypeError,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    Type,
    TypeQualifier,
    get_core_data_type_bit_width,
    is_weak_core_data_type,
    promote_core_data_types,
    promote_primitive_data_types,
    promote_type_qualifiers,
    resolve_literal_core_data_type,
)
from .dispatch import (
    TypeUnificationEnvironment,
    bind_data_template,
    bind_template,
    is_structurally_equivalent,
    substitute_data_template,
    substitute_template,
    unify,
    unify_expression,
)
