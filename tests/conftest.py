"""Testing utilitiy functions."""

from importlib.util import find_spec
from unittest.mock import Mock

import pytest

from fhy_core.identifier import Identifier


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if find_spec("z3") is None:
        skip_z3 = pytest.mark.skip(reason="z3-solver not installed")
        for item in items:
            if "z3" in item.keywords:
                item.add_marker(skip_z3)


def mock_identifier(name_hint: str, identifier_id: int) -> Identifier:
    """Create a mock identifier.

    Args:
        name_hint: Variable name.
        identifier_id: Identifier ID.

    Returns:
        Mock identifier.

    """
    identifier = Mock(spec=Identifier)
    identifier._name_hint = name_hint
    identifier._id = identifier_id
    identifier.name_hint = name_hint
    identifier.id = identifier_id
    identifier.__eq__ = lambda self, other: self.id == other.id  # type: ignore[method-assign,assignment,misc]
    identifier.__hash__ = lambda self: hash(self.id)  # type: ignore[method-assign,assignment,misc]
    identifier.serialize_to_dict = lambda: {
        "id": identifier.id,
        "name_hint": identifier.name_hint,
    }
    identifier.deserialize_from_dict = lambda data: mock_identifier(
        data["id"], data["name_hint"]
    )
    return identifier
