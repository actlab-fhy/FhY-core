"""Testing utilitiy functions."""

from unittest.mock import Mock

from fhy_core.identifier import Identifier


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
