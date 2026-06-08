"""Tests the lexical `Scope` utility."""

import pytest

from fhy_core.utils import Scope


def test_fresh_scope_has_single_root_frame() -> None:
    """Test a newly constructed scope has exactly one (root) frame."""
    scope: Scope[str, int] = Scope()
    assert scope.get_depth() == 1


def test_define_and_lookup_in_root_frame() -> None:
    """Test a binding defined in the root frame is found by lookup."""
    scope: Scope[str, int] = Scope()
    scope.define("a", 1)
    assert scope.lookup("a") == 1


def test_lookup_raises_key_error_for_unbound_key() -> None:
    """Test lookup of an unbound key raises `KeyError`."""
    scope: Scope[str, int] = Scope()
    with pytest.raises(KeyError):
        scope.lookup("missing")


def test_push_increases_depth_and_pop_restores_it() -> None:
    """Test push/pop adjust frame depth in LIFO order."""
    scope: Scope[str, int] = Scope()
    scope.push()
    assert scope.get_depth() == 2
    scope.pop()
    assert scope.get_depth() == 1


def test_pop_root_frame_raises_index_error() -> None:
    """Test popping the sole root frame raises `IndexError`."""
    scope: Scope[str, int] = Scope()
    with pytest.raises(IndexError):
        scope.pop()


def test_inner_binding_shadows_outer_binding() -> None:
    """Test an inner-frame binding shadows an outer-frame binding of the same key."""
    scope: Scope[str, int] = Scope()
    scope.define("x", 1)
    scope.push()
    scope.define("x", 2)
    assert scope.lookup("x") == 2


def test_pop_reveals_shadowed_outer_binding() -> None:
    """Test popping the inner frame reveals the shadowed outer binding."""
    scope: Scope[str, int] = Scope()
    scope.define("x", 1)
    scope.push()
    scope.define("x", 2)

    scope.pop()

    assert scope.lookup("x") == 1


def test_lookup_finds_outer_binding_from_inner_frame() -> None:
    """Test lookup walks outward to find a binding defined in an outer frame."""
    scope: Scope[str, int] = Scope()
    scope.define("outer", 1)
    scope.push()
    assert scope.lookup("outer") == 1


def test_lookup_local_only_searches_innermost_frame() -> None:
    """Test lookup_local ignores outer frames."""
    scope: Scope[str, int] = Scope()
    scope.define("outer", 1)
    scope.push()
    with pytest.raises(KeyError):
        scope.lookup_local("outer")


def test_lookup_local_returns_innermost_binding() -> None:
    """Test lookup_local returns a binding defined in the innermost frame."""
    scope: Scope[str, int] = Scope()
    scope.push()
    scope.define("inner", 5)
    assert scope.lookup_local("inner") == 5


def test_define_overwrites_same_frame_binding() -> None:
    """Test redefining a key in the same frame overwrites its value."""
    scope: Scope[str, int] = Scope()
    scope.define("a", 1)
    scope.define("a", 2)
    assert scope.lookup("a") == 2


def test_is_defined_true_for_outer_binding() -> None:
    """Test is_defined sees a binding in any frame."""
    scope: Scope[str, int] = Scope()
    scope.define("a", 1)
    scope.push()
    assert scope.is_defined("a") is True


def test_is_defined_false_for_unbound_key() -> None:
    """Test is_defined is False for a key bound nowhere."""
    scope: Scope[str, int] = Scope()
    assert scope.is_defined("missing") is False


def test_is_defined_local_false_for_outer_binding() -> None:
    """Test is_defined_local does not see outer-frame bindings."""
    scope: Scope[str, int] = Scope()
    scope.define("a", 1)
    scope.push()
    assert scope.is_defined_local("a") is False


def test_is_defined_local_true_for_innermost_binding() -> None:
    """Test is_defined_local sees a binding in the innermost frame."""
    scope: Scope[str, int] = Scope()
    scope.push()
    scope.define("a", 1)
    assert scope.is_defined_local("a") is True


def test_scope_context_manager_pushes_and_pops() -> None:
    """Test the `scope` context manager pushes on enter and pops on exit."""
    scope: Scope[str, int] = Scope()

    with scope.scope():
        assert scope.get_depth() == 2

    assert scope.get_depth() == 1


def test_scope_context_manager_pops_on_exception() -> None:
    """Test the `scope` context manager pops its frame even when the body raises."""
    scope: Scope[str, int] = Scope()

    with pytest.raises(RuntimeError):
        with scope.scope():
            assert scope.get_depth() == 2
            raise RuntimeError("body failure")

    assert scope.get_depth() == 1


def test_scope_context_manager_yields_same_scope() -> None:
    """Test the `scope` context manager yields the same scope it was called on."""
    scope: Scope[str, int] = Scope()
    with scope.scope() as inner:
        assert inner is scope


def test_bindings_in_context_manager_frame_are_discarded_on_exit() -> None:
    """Test bindings made inside a `scope` block do not survive the block."""
    scope: Scope[str, int] = Scope()

    with scope.scope():
        scope.define("temp", 9)
        assert scope.lookup("temp") == 9

    assert scope.is_defined("temp") is False


def test_nested_scopes_track_depth() -> None:
    """Test nested context-manager scopes stack and unwind in LIFO order."""
    scope: Scope[str, int] = Scope()
    with scope.scope():
        with scope.scope():
            assert scope.get_depth() == 3
        assert scope.get_depth() == 2
    assert scope.get_depth() == 1
