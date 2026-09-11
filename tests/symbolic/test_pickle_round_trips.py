"""Pickle and deepcopy round-trips for the frozen types that hold identifiers.

An `EquationConstraint`, a `Param`, and a `ParamAssignment` are all frozen
value objects that reach an `Identifier` -- the constraint through the free
identifiers of its expression, the parameter through the variable it binds.
Duplicating one must return an independent object that is still frozen and
still equivalent to its source, so neither the freeze flag nor any derived
state is lost or shared on the way through.

A test identifier is a `Mock(spec=Identifier)`, which pickle refuses to
serialize. Handing every identifier to the pickler as a persistent
reference keeps the object under test itself on the real
``dumps``/``loads`` path.
"""

import copy
import io
import pickle
from collections.abc import Callable
from typing import Any, TypeVar

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import EquationConstraint
from fhy_core.symbolic.expression import IdentifierExpression, LiteralExpression
from fhy_core.symbolic.param import Param, ParamAssignment, create_integer_param
from fhy_core.utils.override import override

from .conftest import mock_identifier

_T = TypeVar("_T")


class _IdentifierByReferencePickler(pickle.Pickler):
    """Pickler that emits identifiers as external references."""

    referenced: dict[str, Identifier]

    def __init__(self, file: Any, referenced: dict[str, Identifier]) -> None:
        super().__init__(file)
        self.referenced = referenced

    @override
    def persistent_id(self, obj: Any) -> str | None:
        if isinstance(obj, Identifier):
            key = str(id(obj))
            self.referenced[key] = obj
            return key
        return None


class _IdentifierByReferenceUnpickler(pickle.Unpickler):
    """Unpickler resolving the external identifier references by key."""

    referenced: dict[str, Identifier]

    def __init__(self, file: Any, referenced: dict[str, Identifier]) -> None:
        super().__init__(file)
        self.referenced = referenced

    @override
    def persistent_load(self, pid: Any) -> Identifier:
        return self.referenced[pid]


def _round_trip_through_pickle(value: _T) -> _T:
    """Return the value after a ``pickle.dumps``/``loads`` round trip."""
    referenced: dict[str, Identifier] = {}
    buffer = io.BytesIO()
    _IdentifierByReferencePickler(buffer, referenced).dump(value)
    buffer.seek(0)
    restored = _IdentifierByReferenceUnpickler(buffer, referenced).load()
    assert isinstance(restored, type(value))
    return restored


Duplicator = Callable[[Any], Any]

_DUPLICATORS = [
    pytest.param(_round_trip_through_pickle, id="pickle"),
    pytest.param(copy.deepcopy, id="deepcopy"),
]


def _create_equation_constraint() -> EquationConstraint:
    """Create an equation constraint over one free identifier."""
    variable = mock_identifier("x", 0)
    return EquationConstraint(IdentifierExpression(variable) < LiteralExpression(5))


def _create_param() -> Param[int]:
    """Create an integer parameter binding one identifier."""
    return create_integer_param(name=mock_identifier("p", 1))


@pytest.mark.parametrize("duplicate", _DUPLICATORS)
def test_equation_constraint_survives_duplication(duplicate: Duplicator) -> None:
    """Test a constraint reaching an identifier duplicates frozen and equivalent."""
    constraint = _create_equation_constraint()

    duplicated = duplicate(constraint)

    assert duplicated is not constraint
    assert duplicated.is_frozen
    assert duplicated.get_free_identifiers() == constraint.get_free_identifiers()
    assert duplicated.is_structurally_equivalent(constraint)
    assert duplicated.is_alpha_equivalent(constraint)


@pytest.mark.parametrize("duplicate", _DUPLICATORS)
def test_param_survives_duplication(duplicate: Duplicator) -> None:
    """Test a parameter duplicates frozen, with its binder and domain intact."""
    param = _create_param()

    duplicated = duplicate(param)

    assert duplicated is not param
    assert duplicated.is_frozen
    assert duplicated.variable == param.variable
    assert duplicated.domain.is_structurally_equivalent(param.domain)
    assert duplicated.is_structurally_equivalent(param)
    assert duplicated.is_alpha_equivalent(param)


@pytest.mark.parametrize("duplicate", _DUPLICATORS)
def test_param_assignment_survives_duplication(duplicate: Duplicator) -> None:
    """Test an assignment duplicates frozen, keeping its parameter and value."""
    assignment = ParamAssignment(_create_param(), 5)

    duplicated = duplicate(assignment)

    assert duplicated is not assignment
    assert duplicated.is_frozen
    assert duplicated.value == assignment.value
    assert duplicated.param.is_structurally_equivalent(assignment.param)
    assert duplicated.is_structurally_equivalent(assignment)
    assert duplicated.is_alpha_equivalent(assignment)
