"""Golden `type_id` and wire-shape pins for the expression/constraint/param tree.

Every `Serializable` in the `expression` / `constraint` / `param` tree
pins an explicit `type_id`. This module hard-codes all 18 pinned
strings and checks, for one representative instance of each class, that
`get_serialization_class_type_id()` equals the pinned literal (the
registry key regardless of wrapped-family vs. plain dict form) and that
a golden blob for that class deserializes into an equivalent instance.
A rename or wire-format change flips a pinned id or breaks a golden
blob's deserialization here, even when every other test still passes.
"""

import json

import pytest

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
)

from .conftest import mock_identifier

_x = mock_identifier("x", 0)

_EXPRESSION_FIXTURES: dict[str, object] = {
    "unary_expression": UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(1)),
    "binary_expression": BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    ),
    "identifier_expression": IdentifierExpression(_x),
    "literal_expression": LiteralExpression(1),
    "piecewise_expression": PiecewiseExpression(
        (
            BinaryExpression(
                BinaryOperation.GREATER, IdentifierExpression(_x), LiteralExpression(0)
            ),
        ),
        (LiteralExpression(1),),
        LiteralExpression(0),
    ),
    "call_expression": CallExpression("sqrt", (LiteralExpression(4),)),
}

_CONSTRAINT_FIXTURES: dict[str, object] = {
    "equation_constraint": EquationConstraint(_x, LiteralExpression(True)),
    "in_set_constraint": InSetConstraint(_x, {1, 2}),
    "not_in_set_constraint": NotInSetConstraint(_x, {1, 2}),
    "constraint_system": create_constraint_system(
        InSetConstraint(_x, {1, 2}), NotInSetConstraint(_x, {3})
    ),
}

_default_int_param = create_integer_param()

_PARAM_FIXTURES: dict[str, object] = {
    "integer_domain": IntegerDomain(),
    "real_domain": RealDomain(),
    "interval_integer_domain": IntervalIntegerDomain(),
    "ordinal_domain": OrdinalDomain((1, 2, 3)),
    "categorical_domain": CategoricalDomain(("a", "b")),
    "permutation_domain": PermutationDomain(("n", "c", "h", "w")),
    "param": _default_int_param,
    "param_assignment": ParamAssignment(_default_int_param, 5),
}

_ALL_FIXTURES: dict[str, object] = {
    **_EXPRESSION_FIXTURES,
    **_CONSTRAINT_FIXTURES,
    **_PARAM_FIXTURES,
}

# Golden serialized forms for the 18 classes covered by `_ALL_FIXTURES`. Each
# blob is the literal `serialize_to_dict()` output of one representative
# instance, held as a static literal independent of the current
# implementation. Regenerating this JSON from the current code would defeat
# the test's purpose: a wire-format regression that changes both the writer
# and the golden data in lockstep would still pass. Every identifier embeds a
# fixed id (0 for the shared variable `x`, 1 for the `Param` variable) rather
# than one drawn from the process-global identifier counter, so the blob is
# exactly reproducible; `param` and `param_assignment` are compared by alpha
# equivalence below because their variable's id is not expected to match the
# id `create_integer_param()` assigns in this process.
_GOLDEN_BLOBS_JSON = """
{
  "binary_expression": {
    "__data__": {
      "left": {
        "__data__": {
          "value": 1
        },
        "__type__": "literal_expression"
      },
      "operation": "add",
      "right": {
        "__data__": {
          "value": 2
        },
        "__type__": "literal_expression"
      }
    },
    "__type__": "binary_expression"
  },
  "call_expression": {
    "__data__": {
      "arguments": [
        {
          "__data__": {
            "value": 4
          },
          "__type__": "literal_expression"
        }
      ],
      "function_name": "sqrt"
    },
    "__type__": "call_expression"
  },
  "categorical_domain": {
    "__data__": {
      "categories": [
        {
          "__data__": "a",
          "__type__": "builtins.str"
        },
        {
          "__data__": "b",
          "__type__": "builtins.str"
        }
      ]
    },
    "__type__": "categorical_domain"
  },
  "constraint_system": {
    "__data__": {
      "constraints": [
        {
          "__data__": {
            "valid_values": [
              {
                "__data__": 1,
                "__type__": "builtins.int"
              },
              {
                "__data__": 2,
                "__type__": "builtins.int"
              }
            ],
            "variable": {
              "id": 0,
              "name_hint": "x"
            }
          },
          "__type__": "in_set_constraint"
        },
        {
          "__data__": {
            "invalid_values": [
              {
                "__data__": 3,
                "__type__": "builtins.int"
              }
            ],
            "variable": {
              "id": 0,
              "name_hint": "x"
            }
          },
          "__type__": "not_in_set_constraint"
        }
      ]
    },
    "__type__": "constraint_system"
  },
  "equation_constraint": {
    "__data__": {
      "expression": {
        "__data__": {
          "value": true
        },
        "__type__": "literal_expression"
      },
      "variable": {
        "id": 0,
        "name_hint": "x"
      }
    },
    "__type__": "equation_constraint"
  },
  "identifier_expression": {
    "__data__": {
      "identifier": {
        "id": 0,
        "name_hint": "x"
      }
    },
    "__type__": "identifier_expression"
  },
  "in_set_constraint": {
    "__data__": {
      "valid_values": [
        {
          "__data__": 1,
          "__type__": "builtins.int"
        },
        {
          "__data__": 2,
          "__type__": "builtins.int"
        }
      ],
      "variable": {
        "id": 0,
        "name_hint": "x"
      }
    },
    "__type__": "in_set_constraint"
  },
  "integer_domain": {
    "__data__": {
      "non_negative": false,
      "zero_included": true
    },
    "__type__": "integer_domain"
  },
  "interval_integer_domain": {
    "__data__": {
      "non_negative": false,
      "prefer_inclusive": true,
      "zero_included": true
    },
    "__type__": "interval_integer_domain"
  },
  "literal_expression": {
    "__data__": {
      "value": 1
    },
    "__type__": "literal_expression"
  },
  "not_in_set_constraint": {
    "__data__": {
      "invalid_values": [
        {
          "__data__": 1,
          "__type__": "builtins.int"
        },
        {
          "__data__": 2,
          "__type__": "builtins.int"
        }
      ],
      "variable": {
        "id": 0,
        "name_hint": "x"
      }
    },
    "__type__": "not_in_set_constraint"
  },
  "ordinal_domain": {
    "__data__": {
      "sorted_values": [
        {
          "__data__": 1,
          "__type__": "builtins.int"
        },
        {
          "__data__": 2,
          "__type__": "builtins.int"
        },
        {
          "__data__": 3,
          "__type__": "builtins.int"
        }
      ]
    },
    "__type__": "ordinal_domain"
  },
  "param": {
    "constraints": [],
    "domain": {
      "__data__": {
        "non_negative": false,
        "zero_included": true
      },
      "__type__": "integer_domain"
    },
    "variable": {
      "id": 1,
      "name_hint": "param"
    }
  },
  "param_assignment": {
    "param": {
      "constraints": [],
      "domain": {
        "__data__": {
          "non_negative": false,
          "zero_included": true
        },
        "__type__": "integer_domain"
      },
      "variable": {
        "id": 1,
        "name_hint": "param"
      }
    },
    "value": {
      "__data__": 5,
      "__type__": "builtins.int"
    }
  },
  "permutation_domain": {
    "__data__": {
      "ordered_members": [
        {
          "__data__": "n",
          "__type__": "builtins.str"
        },
        {
          "__data__": "c",
          "__type__": "builtins.str"
        },
        {
          "__data__": "h",
          "__type__": "builtins.str"
        },
        {
          "__data__": "w",
          "__type__": "builtins.str"
        }
      ]
    },
    "__type__": "permutation_domain"
  },
  "piecewise_expression": {
    "__data__": {
      "conditions": [
        {
          "__data__": {
            "left": {
              "__data__": {
                "identifier": {
                  "id": 0,
                  "name_hint": "x"
                }
              },
              "__type__": "identifier_expression"
            },
            "operation": "greater",
            "right": {
              "__data__": {
                "value": 0
              },
              "__type__": "literal_expression"
            }
          },
          "__type__": "binary_expression"
        }
      ],
      "otherwise": {
        "__data__": {
          "value": 0
        },
        "__type__": "literal_expression"
      },
      "values": [
        {
          "__data__": {
            "value": 1
          },
          "__type__": "literal_expression"
        }
      ]
    },
    "__type__": "piecewise_expression"
  },
  "real_domain": {
    "__data__": {},
    "__type__": "real_domain"
  },
  "unary_expression": {
    "__data__": {
      "operand": {
        "__data__": {
          "value": 1
        },
        "__type__": "literal_expression"
      },
      "operation": "negate"
    },
    "__type__": "unary_expression"
  }
}
"""

_GOLDEN_BLOBS: dict[str, object] = json.loads(_GOLDEN_BLOBS_JSON)

# `Param.variable` is a binder field (`compared_as_binder`): structural
# equivalence compares it nominally, which would require the golden blob's
# captured id to match whatever id `create_integer_param()` assigns in this
# process. Alpha equivalence renames it consistently instead, which is the
# right relation for a variable whose concrete id is incidental.
_ALPHA_EQUIVALENT_ONLY: frozenset[str] = frozenset({"param", "param_assignment"})

# The 18 `type_id` literals this module pins, hard-coded independently of
# `_ALL_FIXTURES` and `_GOLDEN_BLOBS`. Because this set does not derive from
# either table, a missing or mismatched fixture or golden-blob entry is
# caught by the two guard tests below rather than passing vacuously.
_EXPECTED_PINNED_TYPE_IDS: frozenset[str] = frozenset(
    {
        "unary_expression",
        "binary_expression",
        "identifier_expression",
        "literal_expression",
        "piecewise_expression",
        "call_expression",
        "equation_constraint",
        "in_set_constraint",
        "not_in_set_constraint",
        "constraint_system",
        "integer_domain",
        "real_domain",
        "interval_integer_domain",
        "ordinal_domain",
        "categorical_domain",
        "permutation_domain",
        "param",
        "param_assignment",
    }
)


def test_exactly_eighteen_pinned_type_ids_are_covered() -> None:
    """Test the fixture table covers exactly the 18 pinned `type_id` literals."""
    assert len(_EXPECTED_PINNED_TYPE_IDS) == 18
    assert set(_ALL_FIXTURES) == _EXPECTED_PINNED_TYPE_IDS


def test_golden_blobs_cover_exactly_the_pinned_fixtures() -> None:
    """Test the embedded golden blob table has one entry per pinned `type_id`."""
    assert set(_GOLDEN_BLOBS) == _EXPECTED_PINNED_TYPE_IDS


@pytest.mark.parametrize(
    "type_id, instance", list(_ALL_FIXTURES.items()), ids=list(_ALL_FIXTURES.keys())
)
def test_fixture_is_registered_under_its_pinned_type_id(
    type_id: str, instance: object
) -> None:
    """Test the class's registry key equals its pinned literal `type_id`."""
    actual = type(instance).get_serialization_class_type_id()  # type: ignore[attr-defined]
    assert actual == type_id


@pytest.mark.parametrize(
    "type_id, expected", list(_ALL_FIXTURES.items()), ids=list(_ALL_FIXTURES.keys())
)
def test_golden_blob_deserializes_to_an_equivalent_instance(
    type_id: str, expected: object
) -> None:
    """Test the golden blob deserializes into an equivalent fresh instance."""
    cls = type(expected)
    rebuilt = cls.deserialize_from_dict(_GOLDEN_BLOBS[type_id])  # type: ignore[attr-defined]
    if type_id in _ALPHA_EQUIVALENT_ONLY:
        assert rebuilt.is_alpha_equivalent(expected)
    else:
        assert rebuilt.is_structurally_equivalent(expected)
