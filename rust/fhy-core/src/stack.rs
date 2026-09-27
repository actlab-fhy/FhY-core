//! A last-in, first-out stack.
//!
//! [`Stack`] holds elements in the order they were pushed. [`pop`] and
//! [`peek`] reach the most recently pushed element, the top, and answer
//! `None` on an empty stack; iteration runs from the bottom, the oldest
//! element, to the top.
//!
//! [`pop`]: Stack::pop
//! [`peek`]: Stack::peek
//!
//! # Examples
//!
//! ```
//! use fhy_core::stack::Stack;
//!
//! let mut stack = Stack::new();
//! stack.push("fhy");
//! stack.push("test");
//!
//! assert_eq!(stack.peek(), Some(&"test"));
//! assert_eq!(stack.pop(), Some("test"));
//! assert_eq!(stack.pop(), Some("fhy"));
//! assert_eq!(stack.pop(), None);
//! ```

use std::iter::FusedIterator;
use std::slice;

/// A last-in, first-out stack of `T`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Stack<T> {
    elements: Vec<T>,
}

impl<T> Stack<T> {
    /// Return an empty stack.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            elements: Vec::new(),
        }
    }

    /// Push `item` onto the top of the stack.
    pub fn push(&mut self, item: T) {
        self.elements.push(item);
    }

    /// Remove and return the top element, or `None` if the stack is empty.
    pub fn pop(&mut self) -> Option<T> {
        self.elements.pop()
    }

    /// Return the top element without removing it, or `None` if the stack
    /// is empty.
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        self.elements.last()
    }

    /// Return the top element for changing it in place, or `None` if the
    /// stack is empty.
    #[must_use]
    pub fn peek_mut(&mut self) -> Option<&mut T> {
        self.elements.last_mut()
    }

    /// Remove every element.
    pub fn clear(&mut self) {
        self.elements.clear();
    }

    /// Return the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Return whether the stack holds no element.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Return an iterator over the elements from the bottom, the oldest, to
    /// the top.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            inner: self.elements.iter(),
        }
    }
}

impl<T> Default for Stack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> IntoIterator for &'a Stack<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An iterator over a [`Stack`]'s elements, from the bottom to the top.
///
/// Returned by [`Stack::iter`].
#[derive(Debug, Clone)]
pub struct Iter<'a, T> {
    inner: slice::Iter<'a, T>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> DoubleEndedIterator for Iter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}

impl<T> FusedIterator for Iter<'_, T> {}
