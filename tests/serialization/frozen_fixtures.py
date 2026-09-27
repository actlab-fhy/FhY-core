"""One object of each serializable class, over fixed identifier ids.

The frozen corpora of slice S17 (``docs/design/python-switch.md``) were
written from these objects by the code before S17: the pickles of
``data/pickles_v1.json`` and the V1 payloads of ``data/v1_payloads.json``.
Both pin V1 only until V1 is removed (D-S17-16, D-S17-20), and are
deleted with it. Every identifier takes a fixed id in the reserved range
no shipped tag uses, so the objects are the same in every process.
"""

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

from fhy_core.diagnostic import OTHER_NOTE_KIND, Note
from fhy_core.identifier import Identifier
from fhy_core.op_attribute import COMMUTATIVE
from fhy_core.provenance import (
    CallSiteProvenance,
    FileProvenance,
    FusedProvenance,
    NamedProvenance,
    Position,
    Span,
    UnknownProvenance,
)
from fhy_core.serialization import Serializable
from fhy_core.symbol_table import (
    FunctionKeyword,
    FunctionSymbolTableFrame,
    ImportSymbolTableFrame,
    SymbolTable,
    VariableSymbolTableFrame,
)
from fhy_core.symbolic.constraint import (
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
    create_constraint_system,
)
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    IdentifierExpression,
    LiteralExpression,
    LogicalExpression,
    LogicalOperation,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.symbolic.param import (
    CategoricalDomain,
    IntegerDomain,
    IntervalIntegerDomain,
    OrdinalDomain,
    ParamAssignment,
    PermutationDomain,
    RealDomain,
    create_integer_param,
    create_ordinal_param,
)
from fhy_core.types import (
    CoreDataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TypeQualifier,
)
from fhy_core.value_domain import DATA_DOMAIN, ValueDomain

__all__ = ["build_fixtures", "is_equivalent"]


def _identifier(identifier_id: int, name_hint: str) -> Identifier:
    return Identifier.deserialize_from_dict(
        {"id": identifier_id, "name_hint": name_hint}
    )


def build_fixtures() -> dict[str, Serializable]:
    """Return one object per case, keyed by the case's name."""
    x = _identifier(61_000, "x")
    y = _identifier(61_001, "y")
    n = _identifier(61_002, "N")
    t = _identifier(61_003, "T")
    f = _identifier(61_004, "f")
    ns = _identifier(61_005, "ns")
    child_ns = _identifier(61_006, "child")
    param_variable = _identifier(61_007, "p")
    ordinal_variable = _identifier(61_008, "o")
    x_ref = IdentifierExpression(x)
    source = FileProvenance(
        Path("src/a.fhy"), Span(0, 3, Position(1, 1), Position(1, 4))
    )
    int32 = PrimitiveDataType(CoreDataType.INT32)
    scalar = NumericalType(int32)
    shared = x_ref + LiteralExpression(1)
    fixtures: dict[str, Serializable] = {
        "identifier": x,
        "position": Position(2, 8),
        "span": Span(0, 3, Position(1, 1), Position(1, 4)),
        "span_unknown": Span(),
        "note": Note("hello", OTHER_NOTE_KIND),
        "note_kind": OTHER_NOTE_KIND,
        "op_attribute": COMMUTATIVE,
        "value_domain": ValueDomain(
            _identifier(61_010, "frozen_domain"), "a child", DATA_DOMAIN
        ),
        "provenance_unknown": UnknownProvenance(),
        "provenance_file": source,
        "provenance_file_no_span": FileProvenance(Path("b.fhy")),
        "provenance_named": NamedProvenance("lib", source),
        "provenance_call_site": CallSiteProvenance(
            callee=source, caller=UnknownProvenance()
        ),
        "provenance_fused": FusedProvenance(
            sources=(source, FileProvenance(Path("c.fhy"))), metadata="fuse"
        ),
        "provenance_fused_unlabelled": FusedProvenance(
            sources=(source, FileProvenance(Path("c.fhy")))
        ),
        "unary_expression": UnaryExpression(UnaryOperation.NEGATE, x_ref),
        "binary_expression": BinaryExpression(
            BinaryOperation.FLOOR_DIVIDE, x_ref, LiteralExpression(3)
        ),
        "logical_expression": LogicalExpression(
            LogicalOperation.OR,
            (x_ref > 0, LiteralExpression(True), LiteralExpression(False)),
        ),
        "identifier_expression": x_ref,
        "literal_int": LiteralExpression(7),
        "literal_big_int": LiteralExpression(2**100),
        "literal_negative_int": LiteralExpression(-12),
        "literal_float": LiteralExpression(1.5),
        "literal_float_integral": LiteralExpression(2.0),
        "literal_float_tiny": LiteralExpression(5e-324),
        "literal_decimal": LiteralExpression(Decimal("1.25")),
        "literal_decimal_integral": LiteralExpression(Decimal("100")),
        "literal_bool": LiteralExpression(True),
        "piecewise_expression": PiecewiseExpression(
            (x_ref > 0, x_ref < -5),
            (LiteralExpression(1), LiteralExpression(2)),
            LiteralExpression(0),
        ),
        "call_expression": CallExpression("max", (x_ref, IdentifierExpression(y))),
        "shared_subtree": shared * shared,
        "primitive_data_type": int32,
        "template_data_type": TemplateDataType(t),
        "template_data_type_widths": TemplateDataType(t, widths=[16, 32]),
        "numerical_type": NumericalType(
            int32, [LiteralExpression(4), IdentifierExpression(n)]
        ),
        "numerical_type_scalar": scalar,
        "numerical_type_ellipsis": NumericalType(
            TemplateDataType(t), [LiteralExpression(2), ...]
        ),
        "numerical_type_wildcard": NumericalType(int32, [...]),
        "index_type": IndexType(
            LiteralExpression(0), IdentifierExpression(n), LiteralExpression(2)
        ),
        "import_frame": ImportSymbolTableFrame(f),
        "variable_frame": VariableSymbolTableFrame(x, scalar, TypeQualifier.INPUT),
        "function_frame": FunctionSymbolTableFrame(
            f,
            FunctionKeyword.PROCEDURE,
            [(TypeQualifier.INPUT, scalar), (TypeQualifier.OUTPUT, scalar)],
        ),
        "equation_constraint": EquationConstraint(x_ref > 0),
        "in_set_constraint": InSetConstraint(
            x, [1, 2.5, "a", True, (1, 2), frozenset({3, 4})]
        ),
        "not_in_set_constraint": NotInSetConstraint(x, {3}),
        "constraint_system": create_constraint_system(
            InSetConstraint(x, {1, 2}),
            NotInSetConstraint(y, {3}),
            EquationConstraint(x_ref > 0),
        ),
        "integer_domain": IntegerDomain(),
        "natural_domain": IntegerDomain(non_negative=True, zero_included=False),
        "real_domain": RealDomain(),
        "interval_integer_domain": IntervalIntegerDomain(),
        "ordinal_domain": OrdinalDomain((1, 2.5, 3)),
        "categorical_domain": CategoricalDomain(("b", "a")),
        "permutation_domain": PermutationDomain(("n", "c", "h", "w")),
        "param": create_integer_param(name=param_variable),
        "param_ordinal": create_ordinal_param([1, 2, 3], name=ordinal_variable),
        "param_assignment": ParamAssignment(
            create_integer_param(name=param_variable), 5
        ),
    }
    table = SymbolTable()
    table.add_namespace(ns)
    table.add_namespace(child_ns, ns)
    table.add_symbol(ns, x, VariableSymbolTableFrame(x, scalar, TypeQualifier.STATE))
    table.add_symbol(ns, f, fixtures["function_frame"])  # type: ignore[arg-type]
    table.add_symbol(child_ns, y, ImportSymbolTableFrame(y))
    fixtures["symbol_table"] = table
    return fixtures


def is_equivalent(left: Any, right: Any) -> bool:
    """Return whether two fixtures are the same value, by the class's relation."""
    if type(left) is not type(right):
        return False
    compare: Callable[[Any], bool] | None = getattr(
        left, "is_structurally_equivalent", None
    )
    if compare is not None:
        return bool(compare(right))
    return bool(left == right)
