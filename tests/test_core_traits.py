"""Tests core compiler traits."""

from dataclasses import dataclass

import pytest

from fhy_core.diagnostic import (
    Diagnostic,
    DiagnosticLevel,
    Note,
    ValidationFailedError,
    ValidationReport,
)
from fhy_core.trait import (
    HasType,
    HasTypeMixin,
    Verifiable,
    VerifiableMixin,
)


@dataclass
class _TypedValue(HasTypeMixin[str]):
    _type: str

    def get_type(self) -> str:
        return self._type


@dataclass
class _VerifiableNode(VerifiableMixin):
    is_valid: bool

    def verify(self) -> ValidationReport:
        if self.is_valid:
            return ValidationReport()
        return ValidationReport(
            diagnostics=(
                Diagnostic(
                    level=DiagnosticLevel.ERROR,
                    message=Note("Node invariant violation."),
                    source="_VerifiableNode",
                ),
            )
        )


def test_has_type_runtime_protocol() -> None:
    """Test `HasType` runtime protocol."""
    value = _TypedValue("i32")
    assert isinstance(value, HasType)


def test_has_type_mixin_contract() -> None:
    """Test `HasTypeMixin` contract."""
    value = _TypedValue("index")
    assert value.get_type() == "index"


def test_verifiable_runtime_protocol() -> None:
    """Test `Verifiable` runtime protocol."""
    node = _VerifiableNode(True)
    assert isinstance(node, Verifiable)


def test_verifiable_mixin_contract() -> None:
    """Test `VerifiableMixin` contract."""
    node = _VerifiableNode(True)
    report = node.verify()

    assert report.has_errors() is False


def test_verifiable_invariant_violation_reports_error() -> None:
    """Test `VerifiableMixin` returns a failing report on invariant violation."""
    node = _VerifiableNode(False)

    report = node.verify()

    assert report.has_errors() is True
    with pytest.raises(ValidationFailedError):
        report.raise_if_failed()
