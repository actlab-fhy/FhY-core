"""FhY compiler core utilities."""

__version__ = "0.0.2"


from .constraint import (
    Constraint,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from .error import FhYCoreTypeError, SymbolTableError, register_error
from .expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    ExpressionBasePass,
    ExpressionVisitor,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    collect_identifiers,
    copy_expression,
    is_satisfiable,
    parse_expression,
    pformat_expression,
    replace_identifiers,
    simplify_expression,
    substitute_identifiers,
)
from .identifier import Identifier
from .memory_instance import (
    ColumnMajorMemoryInstance,
    MemoryInstance,
    RowMajorMemoryInstance,
    ScalarMemoryInstance,
)
from .param import (
    CategoricalParam,
    IntParam,
    NatParam,
    OrdinalParam,
    Param,
    PermParam,
    RealParam,
)
from .symbol_table import (
    FunctionKeyword,
    FunctionSymbolTableFrame,
    ImportSymbolTableFrame,
    SymbolTable,
    SymbolTableFrame,
    VariableSymbolTableFrame,
)
from .types import (
    CoreDataType,
    DataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TupleType,
    Type,
    TypeQualifier,
    promote_core_data_types,
    promote_primitive_data_types,
    promote_type_qualifiers,
)
from .utils import (
    IntEnum,
    Lattice,
    PartiallyOrderedSet,
    Stack,
    StrEnum,
    invert_dict,
    invert_frozen_dict,
)
