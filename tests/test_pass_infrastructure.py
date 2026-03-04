"""Tests for generic compiler pass infrastructure."""

import pytest
from fhy_core.pass_infrastructure import (
    CompilerPass,
    DiagnosticLevel,
    PassExecutionError,
    PassInfo,
    PassRegistrationError,
    PassValidationError,
    register_pass,
)
from fhy_core.provenance import Note


def test_compiler_pass_executes_and_tracks_stats() -> None:
    """Test that a registered pass executes and tracks run statistics."""

    @register_pass("tests.append_pass", "Append sentinel value to a list.")
    class AppendPass(CompilerPass[list[int], list[int]]):
        def run_pass(self, ir: list[int]) -> list[int]:
            return [*ir, 9]

    total_before = CompilerPass.get_total_run_count()
    per_before = AppendPass.get_run_count()

    result = AppendPass().execute([1, 2])

    assert result.output == [1, 2, 9]
    assert result.changed is True
    assert result.diagnostics == ()
    assert AppendPass.get_run_count() == per_before + 1
    assert CompilerPass.get_total_run_count() == total_before + 1


def test_compiler_pass_wraps_internal_exceptions() -> None:
    """Test that internal exceptions are wrapped as PassExecutionError."""

    @register_pass("tests.exploding_pass", "Always raises during execution.")
    class ExplodingPass(CompilerPass[list[int], list[int]]):
        def run_pass(self, ir: list[int]) -> list[int]:
            raise ValueError("boom")

    compiler_pass = ExplodingPass()
    with pytest.raises(PassExecutionError):
        compiler_pass.execute([1])

    assert compiler_pass.diagnostics
    assert compiler_pass.diagnostics[0].level == DiagnosticLevel.ERROR
    assert isinstance(compiler_pass.diagnostics[0].message, Note)
    assert "boom" in compiler_pass.diagnostics[0].message_text


def test_compiler_pass_rejects_none_input() -> None:
    """Test that None input is rejected by default validation."""

    @register_pass("tests.identity_pass", "Identity pass for object IR.")
    class IdentityPass(CompilerPass[object, object]):
        def run_pass(self, ir: object) -> object:
            return ir

    compiler_pass = IdentityPass()
    with pytest.raises(PassValidationError):
        compiler_pass.execute(None)

    assert compiler_pass.diagnostics
    assert compiler_pass.diagnostics[0].level == DiagnosticLevel.ERROR
    assert isinstance(compiler_pass.diagnostics[0].message, Note)
    assert "does not accept None" in compiler_pass.diagnostics[0].message_text


def test_compiler_pass_registry_create_and_collision() -> None:
    """Test pass creation from registry and duplicate-name rejection."""

    @register_pass("tests.createable_pass", "Increment an integer by one.")
    class CreateablePass(CompilerPass[int, int]):
        def run_pass(self, ir: int) -> int:
            return ir + 1

    created = CompilerPass.create("tests.createable_pass")
    assert isinstance(created, CreateablePass)
    assert created(2) == 3
    registered = CompilerPass.get_registered_passes()["tests.createable_pass"]
    assert isinstance(registered, PassInfo)
    assert registered.description == "Increment an integer by one."

    with pytest.raises(PassRegistrationError):

        @register_pass("tests.createable_pass", "Duplicate named pass.")
        class DuplicatePass(CompilerPass[int, int]):
            def run_pass(self, ir: int) -> int:
                return ir
