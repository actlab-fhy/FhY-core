//! Properties of `Stack` and `Scope`: any sequence of operations agrees
//! with a plain model, a `Vec<i32>` for the stack and a `Vec` of maps for
//! the scope (index 0 is the root frame), as `tests/test_stack_properties.py`
//! and `tests/test_scope_properties.py` check them in Python.
//!
//! Cases K-11 and S-21 of the shared case list; the traceability table is in
//! `docs/design/python-switch.md`, "S18: scope and stack".

use std::collections::HashMap;

use fhy_core::scope::{RootFramePopError, Scope};
use fhy_core::stack::Stack;
use proptest::prelude::*;

/// One operation on a stack.
#[derive(Debug, Clone)]
enum StackOp {
    Push(i32),
    Pop,
    Peek,
    Clear,
}

fn stack_op() -> impl Strategy<Value = StackOp> {
    prop_oneof![
        3 => (-1000..=1000_i32).prop_map(StackOp::Push),
        2 => Just(StackOp::Pop),
        2 => Just(StackOp::Peek),
        1 => Just(StackOp::Clear),
    ]
}

/// One operation on a scope, over the keys `'a'` to `'d'`.
#[derive(Debug, Clone)]
enum ScopeOp {
    Push,
    Pop,
    Define(char, i32),
    Lookup(char),
    LookupLocal(char),
    IsDefined(char),
    IsDefinedLocal(char),
}

fn key() -> impl Strategy<Value = char> {
    prop::sample::select(vec!['a', 'b', 'c', 'd'])
}

fn scope_op() -> impl Strategy<Value = ScopeOp> {
    prop_oneof![
        1 => Just(ScopeOp::Push),
        1 => Just(ScopeOp::Pop),
        2 => (key(), -1000..=1000_i32).prop_map(|(key, value)| ScopeOp::Define(key, value)),
        1 => key().prop_map(ScopeOp::Lookup),
        1 => key().prop_map(ScopeOp::LookupLocal),
        1 => key().prop_map(ScopeOp::IsDefined),
        1 => key().prop_map(ScopeOp::IsDefinedLocal),
    ]
}

/// Return the value bound to `key` in the innermost model frame binding it.
fn find_innermost_binding(model: &[HashMap<char, i32>], key: char) -> Option<&i32> {
    model.iter().rev().find_map(|frame| frame.get(&key))
}

proptest! {
    #[test]
    fn stack_agrees_with_a_vec_model(ops in prop::collection::vec(stack_op(), 0..64)) {
        let mut stack = Stack::new();
        let mut model: Vec<i32> = Vec::new();
        for op in ops {
            match op {
                StackOp::Push(value) => {
                    stack.push(value);
                    model.push(value);
                }
                StackOp::Pop => prop_assert_eq!(stack.pop(), model.pop()),
                StackOp::Peek => prop_assert_eq!(stack.peek(), model.last()),
                StackOp::Clear => {
                    stack.clear();
                    model.clear();
                }
            }
            prop_assert_eq!(stack.len(), model.len());
            prop_assert_eq!(stack.is_empty(), model.is_empty());
            prop_assert!(stack.iter().eq(model.iter()));
        }
    }

    #[test]
    fn scope_agrees_with_a_list_of_maps_model(
        ops in prop::collection::vec(scope_op(), 0..64),
    ) {
        let mut scope = Scope::new();
        let mut model: Vec<HashMap<char, i32>> = vec![HashMap::new()];
        for op in ops {
            let innermost = model.len() - 1;
            match op {
                ScopeOp::Push => {
                    scope.push();
                    model.push(HashMap::new());
                }
                ScopeOp::Pop => {
                    if model.len() == 1 {
                        let refused = matches!(scope.pop(), Err(RootFramePopError { .. }));
                        prop_assert!(refused);
                    } else {
                        prop_assert_eq!(scope.pop(), Ok(()));
                        model.pop();
                    }
                }
                ScopeOp::Define(key, value) => {
                    scope.define(key, value);
                    model[innermost].insert(key, value);
                }
                ScopeOp::Lookup(key) => {
                    prop_assert_eq!(scope.lookup(&key), find_innermost_binding(&model, key));
                }
                ScopeOp::LookupLocal(key) => {
                    prop_assert_eq!(scope.lookup_local(&key), model[innermost].get(&key));
                }
                ScopeOp::IsDefined(key) => {
                    let expected = model.iter().any(|frame| frame.contains_key(&key));
                    prop_assert_eq!(scope.is_defined(&key), expected);
                }
                ScopeOp::IsDefinedLocal(key) => {
                    prop_assert_eq!(
                        scope.is_defined_local(&key),
                        model[innermost].contains_key(&key)
                    );
                }
            }
            prop_assert_eq!(scope.depth(), model.len());
        }
    }
}
