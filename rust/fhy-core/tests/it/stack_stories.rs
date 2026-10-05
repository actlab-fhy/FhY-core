//! Tests for `Stack`: push, pop and peek, the empty stack, clearing, and
//! iteration.
//!
//! Mirrors `tests/test_stack.py`.

use fhy_core::stack::Stack;

/// Return the stack holding `"fhy"` at the bottom and `"test"` on top.
fn text_stack() -> Stack<&'static str> {
    let mut stack = Stack::new();
    stack.push("fhy");
    stack.push("test");
    stack
}

#[test]
fn new_stack_is_empty() {
    let stack: Stack<&str> = Stack::new();

    assert_eq!(stack.len(), 0);
    assert!(stack.is_empty());
    assert_eq!(stack, Stack::default());
}

#[test]
fn push_grows_the_stack_by_one() {
    let mut stack = Stack::new();

    stack.push("fhy");
    assert_eq!(stack.len(), 1);
    assert!(!stack.is_empty());

    stack.push("test");
    assert_eq!(stack.len(), 2);
}

#[test]
fn peek_returns_the_top_without_removing_it() {
    let stack = text_stack();

    assert_eq!(stack.peek(), Some(&"test"));
    assert_eq!(stack.len(), 2);
    assert_eq!(stack.peek(), Some(&"test"));
}

#[test]
fn peek_of_an_empty_stack_is_none() {
    let mut stack: Stack<&str> = Stack::new();

    assert_eq!(stack.peek(), None);
    assert_eq!(stack.peek_mut(), None);
}

#[test]
fn pop_returns_the_elements_last_in_first_out() {
    let mut stack = text_stack();

    assert_eq!(stack.pop(), Some("test"));
    assert_eq!(stack.len(), 1);
    assert_eq!(stack.pop(), Some("fhy"));
    assert_eq!(stack.len(), 0);
}

#[test]
fn pop_of_an_empty_stack_is_none() {
    let mut stack: Stack<&str> = Stack::new();

    assert_eq!(stack.pop(), None);
    assert!(stack.is_empty());
}

#[test]
fn clear_empties_the_stack() {
    let mut stack = text_stack();

    stack.clear();

    assert!(stack.is_empty());
    assert_eq!(stack.peek(), None);
}

#[test]
fn iteration_runs_bottom_to_top_and_repeats() {
    let stack = text_stack();

    assert_eq!(stack.iter().copied().collect::<Vec<_>>(), ["fhy", "test"]);
    assert_eq!((&stack).into_iter().count(), 2);
    assert_eq!(stack.iter().len(), 2);
    assert_eq!(stack.iter().next_back(), Some(&"test"));

    let mut iterator = stack.iter();
    assert_eq!(iterator.next(), Some(&"fhy"));
    assert_eq!(iterator.next(), Some(&"test"));
    assert_eq!(iterator.next(), None);
    assert_eq!(iterator.next(), None);
}

#[test]
fn nested_iterations_are_independent() {
    let stack = text_stack();

    let pairs: Vec<(&str, &str)> = stack
        .iter()
        .flat_map(|outer| stack.iter().map(move |inner| (*outer, *inner)))
        .collect();

    assert_eq!(
        pairs,
        [
            ("fhy", "fhy"),
            ("fhy", "test"),
            ("test", "fhy"),
            ("test", "test"),
        ]
    );
}

#[test]
fn peek_mut_changes_the_top_in_place() {
    let mut stack = Stack::new();
    stack.push(vec![1]);
    stack.push(vec![2]);

    stack.peek_mut().expect("the stack is not empty").push(3);

    assert_eq!(stack.len(), 2);
    assert_eq!(stack.pop(), Some(vec![2, 3]));
    assert_eq!(stack.pop(), Some(vec![1]));
}
