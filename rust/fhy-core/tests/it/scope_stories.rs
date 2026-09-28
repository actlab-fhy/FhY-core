//! Tests for `Scope`: frames, the root frame, shadowing, the local and
//! global queries, and the scoped frame of `with_frame`.
//!
//! Ported from `tests/test_scope.py`.

use std::panic::{AssertUnwindSafe, catch_unwind};

use fhy_core::scope::{RootFramePopError, Scope};

#[test]
fn new_scope_has_one_root_frame() {
    let scope: Scope<&str, i32> = Scope::new();

    assert_eq!(scope.depth(), 1);
    assert_eq!(scope, Scope::default());
}

#[test]
fn define_and_lookup_in_the_root_frame() {
    let mut scope = Scope::new();

    scope.define("a", 1);

    assert_eq!(scope.lookup("a"), Some(&1));
}

#[test]
fn lookup_of_an_unbound_key_is_none() {
    let scope: Scope<&str, i32> = Scope::new();

    assert_eq!(scope.lookup("missing"), None);
}

#[test]
fn keyed_queries_take_a_borrowed_form_of_the_key() {
    let mut scope: Scope<String, i32> = Scope::new();

    scope.define("a".to_owned(), 1);

    assert_eq!(scope.lookup("a"), Some(&1));
    assert_eq!(scope.lookup_local("a"), Some(&1));
    assert!(scope.is_defined("a"));
    assert!(scope.is_defined_local("a"));
}

#[test]
fn push_adds_a_frame_and_pop_removes_it() {
    let mut scope: Scope<&str, i32> = Scope::new();

    scope.push();
    assert_eq!(scope.depth(), 2);

    assert_eq!(scope.pop(), Ok(()));
    assert_eq!(scope.depth(), 1);
}

#[test]
fn popping_the_root_frame_is_refused_and_changes_nothing() {
    let mut scope = Scope::new();
    scope.define("a", 1);

    assert!(matches!(scope.pop(), Err(RootFramePopError { .. })));
    assert_eq!(scope.depth(), 1);
    assert_eq!(scope.lookup("a"), Some(&1));

    scope.push();
    assert_eq!(scope.pop(), Ok(()));
    assert!(matches!(scope.pop(), Err(RootFramePopError { .. })));
    assert_eq!(scope.depth(), 1);
}

#[test]
fn root_frame_pop_error_displays_one_lowercase_line() {
    let mut scope: Scope<&str, i32> = Scope::new();

    let error = scope.pop().expect_err("only the root frame is left");

    assert_eq!(error.to_string(), "cannot pop the root frame");
}

#[test]
fn inner_binding_shadows_outer_binding() {
    let mut scope = Scope::new();
    scope.define("x", 1);
    scope.push();

    scope.define("x", 2);

    assert_eq!(scope.lookup("x"), Some(&2));
}

#[test]
fn pop_reveals_the_shadowed_outer_binding() {
    let mut scope = Scope::new();
    scope.define("x", 1);
    scope.push();
    scope.define("x", 2);

    scope.pop().expect("an inner frame is left");

    assert_eq!(scope.lookup("x"), Some(&1));
}

#[test]
fn lookup_finds_an_outer_binding_from_an_inner_frame() {
    let mut scope = Scope::new();
    scope.define("outer", 1);
    scope.push();
    scope.push();

    assert_eq!(scope.lookup("outer"), Some(&1));
}

#[test]
fn lookup_local_ignores_outer_frames() {
    let mut scope = Scope::new();
    scope.define("outer", 1);
    scope.push();

    assert_eq!(scope.lookup_local("outer"), None);
}

#[test]
fn lookup_local_finds_an_innermost_binding() {
    let mut scope = Scope::new();
    scope.push();

    scope.define("inner", 5);

    assert_eq!(scope.lookup_local("inner"), Some(&5));
}

#[test]
fn define_overwrites_a_binding_of_the_same_frame() {
    let mut scope = Scope::new();

    scope.define("a", 1);
    scope.define("a", 2);

    assert_eq!(scope.lookup("a"), Some(&2));
    assert_eq!(scope.depth(), 1);
}

#[test]
fn is_defined_sees_an_outer_binding() {
    let mut scope = Scope::new();
    scope.define("a", 1);
    scope.push();

    assert!(scope.is_defined("a"));
}

#[test]
fn is_defined_is_false_for_an_unbound_key() {
    let scope: Scope<&str, i32> = Scope::new();

    assert!(!scope.is_defined("missing"));
}

#[test]
fn is_defined_local_ignores_outer_frames() {
    let mut scope = Scope::new();
    scope.define("a", 1);
    scope.push();

    assert!(!scope.is_defined_local("a"));
}

#[test]
fn is_defined_local_sees_an_innermost_binding() {
    let mut scope = Scope::new();
    scope.push();

    scope.define("a", 1);

    assert!(scope.is_defined_local("a"));
}

#[test]
fn with_frame_pushes_for_the_body_and_pops_after() {
    let mut scope: Scope<&str, i32> = Scope::new();

    let depth_inside = scope.with_frame(|scope| scope.depth());

    assert_eq!(depth_inside, 2);
    assert_eq!(scope.depth(), 1);
}

#[test]
fn with_frame_pops_when_the_body_panics() {
    let mut scope: Scope<&str, i32> = Scope::new();

    let outcome = catch_unwind(AssertUnwindSafe(|| {
        scope.with_frame(|scope| {
            assert_eq!(scope.depth(), 2);
            scope.define("temp", 9);
            panic!("body failure");
        });
    }));

    assert!(outcome.is_err());
    assert_eq!(scope.depth(), 1);
    assert!(!scope.is_defined("temp"));
}

#[test]
fn with_frame_gives_the_body_the_scope_and_returns_its_result() {
    let mut scope = Scope::new();
    scope.define("outer", 1);

    let seen = scope.with_frame(|scope| {
        scope.define("inner", 2);
        (
            scope.lookup("outer").copied(),
            scope.lookup("inner").copied(),
        )
    });

    assert_eq!(seen, (Some(1), Some(2)));
}

#[test]
fn bindings_of_the_scoped_frame_are_discarded() {
    let mut scope = Scope::new();

    scope.with_frame(|scope| {
        scope.define("temp", 9);
        assert_eq!(scope.lookup("temp"), Some(&9));
    });

    assert!(!scope.is_defined("temp"));
}

#[test]
fn nested_frames_unwind_last_in_first_out() {
    let mut scope: Scope<&str, i32> = Scope::new();

    scope.with_frame(|scope| {
        scope.with_frame(|scope| assert_eq!(scope.depth(), 3));
        assert_eq!(scope.depth(), 2);
    });

    assert_eq!(scope.depth(), 1);
}

#[test]
fn with_frame_restores_the_entry_depth_after_an_unbalanced_body() {
    let mut scope = Scope::new();
    scope.push();
    scope.define("kept", 1);

    scope.with_frame(|scope| {
        scope.push();
        scope.push();
        scope.define("left", 2);
    });
    assert_eq!(scope.depth(), 2);
    assert!(!scope.is_defined("left"));
    assert_eq!(scope.lookup_local("kept"), Some(&1));

    let outcome = catch_unwind(AssertUnwindSafe(|| {
        scope.with_frame(|scope| {
            scope.push();
            panic!("body failure");
        });
    }));
    assert!(outcome.is_err());
    assert_eq!(scope.depth(), 2);

    scope.with_frame(|scope| {
        scope.pop().expect("the scoped frame is left");
        scope.pop().expect("the entry frame is left");
    });
    assert_eq!(scope.depth(), 1);
    assert!(!scope.is_defined("kept"));
}
