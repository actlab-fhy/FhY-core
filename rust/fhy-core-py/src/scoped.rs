//! One guard for the binding's thread-local stacks (R2-031, S-5 of
//! `docs/design/rust-port-fixes.md`).
//!
//! Each stack holds a frame per call in progress on its thread: the
//! type-system contexts, the simplifications, the pending exception of an
//! infallible comparison, the pass scopes and hook frames, the pattern
//! object tables, and the slot collections of the cycle collector's owners.
//! [`ScopedStack::push`] returns a [`ScopedGuard`] that pops its frame when
//! dropped, on unwind included, so a panic, which `PyO3` turns into a
//! `PanicException`, never leaves a stale frame behind for the thread's next
//! call to find.

use std::cell::RefCell;
use std::thread::LocalKey;

/// A thread-local stack of frames, innermost last.
pub(crate) struct ScopedStack<T: 'static>(RefCell<Vec<T>>);

impl<T: 'static> ScopedStack<T> {
    /// Return an empty stack, for a `thread_local!` initializer.
    pub(crate) const fn new() -> Self {
        Self(RefCell::new(Vec::new()))
    }

    /// Push `frame` onto `key`'s stack, and return the guard that pops it.
    pub(crate) fn push(key: &'static LocalKey<Self>, frame: T) -> ScopedGuard<T> {
        let depth = key.with(|stack| {
            let mut frames = stack.0.borrow_mut();
            frames.push(frame);
            frames.len()
        });
        ScopedGuard {
            key,
            depth,
            popped: false,
        }
    }

    /// Return `read` of the innermost frame, or of `None` when none is
    /// pushed.
    pub(crate) fn with_top<R>(
        key: &'static LocalKey<Self>,
        read: impl FnOnce(Option<&T>) -> R,
    ) -> R {
        key.with(|stack| read(stack.0.borrow().last()))
    }

    /// Return a clone of the innermost frame, or `None` when none is
    /// pushed.
    pub(crate) fn cloned_top(key: &'static LocalKey<Self>) -> Option<T>
    where
        T: Clone,
    {
        key.with(|stack| stack.0.borrow().last().cloned())
    }

    /// Return `read` of every frame, outermost first.
    pub(crate) fn with_frames<R>(key: &'static LocalKey<Self>, read: impl FnOnce(&[T]) -> R) -> R {
        key.with(|stack| read(&stack.0.borrow()))
    }

    /// Return `update` of the innermost frame, or of `None` when none is
    /// pushed.
    ///
    /// `update` must not push onto or pop from this stack, whose frames it
    /// borrows.
    pub(crate) fn with_top_mut<R>(
        key: &'static LocalKey<Self>,
        update: impl FnOnce(Option<&mut T>) -> R,
    ) -> R {
        key.with(|stack| update(stack.0.borrow_mut().last_mut()))
    }

    /// Return `update` of the innermost frame, pushing `base()` first as a
    /// frame no guard pops when the stack is empty: the state that holds
    /// outside every call.
    ///
    /// `update` must not push onto or pop from this stack.
    pub(crate) fn with_top_or_base_mut<R>(
        key: &'static LocalKey<Self>,
        base: impl FnOnce() -> T,
        update: impl FnOnce(&mut T) -> R,
    ) -> R {
        key.with(|stack| {
            let mut frames = stack.0.borrow_mut();
            if frames.is_empty() {
                frames.push(base());
            }
            let top = frames
                .last_mut()
                .unwrap_or_else(|| unreachable!("a frame was just pushed"));
            update(top)
        })
    }

    /// Return the number of frames, for the tests.
    #[cfg(test)]
    pub(crate) fn depth(key: &'static LocalKey<Self>) -> usize {
        key.with(|stack| stack.0.borrow().len())
    }
}

/// The frame a [`ScopedStack::push`] pushed, popped by [`pop`](Self::pop) or
/// when dropped.
#[must_use = "dropping the guard pops its frame at once"]
pub(crate) struct ScopedGuard<T: 'static> {
    key: &'static LocalKey<ScopedStack<T>>,
    depth: usize,
    popped: bool,
}

impl<T: 'static> ScopedGuard<T> {
    /// Pop the frame and return it.
    pub(crate) fn pop(mut self) -> T {
        self.popped = true;
        let (frame, above) = self.take();
        drop(above);
        frame
    }

    /// Remove the frame, and any frame left above it, from the stack.
    ///
    /// Guards are dropped in scope order, so the guard's frame is the
    /// innermost one; a frame above it can only be one whose guard was
    /// leaked, which goes with it. What is removed is dropped by the caller,
    /// outside the stack's borrow: a frame may hold Python objects, whose
    /// finalizers may run Python.
    fn take(&self) -> (T, Vec<T>) {
        let mut removed = self.key.with(|stack| {
            let mut frames = stack.0.borrow_mut();
            let start = self.depth.saturating_sub(1).min(frames.len());
            frames.split_off(start)
        });
        let above = removed.split_off(1.min(removed.len()));
        let frame = removed
            .pop()
            .unwrap_or_else(|| unreachable!("the guard's frame is on the stack"));
        (frame, above)
    }
}

impl<T: 'static> Drop for ScopedGuard<T> {
    fn drop(&mut self) {
        if !self.popped {
            drop(self.take());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    thread_local! {
        static STACK: ScopedStack<u32> = const { ScopedStack::new() };
    }

    #[test]
    fn frames_are_read_innermost_first_and_popped_by_their_guards() {
        let outer = ScopedStack::push(&STACK, 1);
        {
            let _inner = ScopedStack::push(&STACK, 2);
            assert_eq!(ScopedStack::cloned_top(&STACK), Some(2));
        }
        assert_eq!(ScopedStack::cloned_top(&STACK), Some(1));
        assert_eq!(outer.pop(), 1);
        assert_eq!(ScopedStack::depth(&STACK), 0);
    }

    #[test]
    fn a_panic_inside_a_scope_leaves_the_stack_empty() {
        let unwound = std::panic::catch_unwind(|| {
            let _guard = ScopedStack::push(&STACK, 7);
            panic!("inside a scope");
        });

        let _panic = unwound.unwrap_err();
        assert_eq!(ScopedStack::depth(&STACK), 0);
    }

    #[test]
    fn a_base_frame_holds_outside_every_scope() {
        thread_local! {
            static BASED: ScopedStack<u32> = const { ScopedStack::new() };
        }
        ScopedStack::with_top_or_base_mut(&BASED, || 0, |top| *top += 5);
        {
            let guard = ScopedStack::push(&BASED, 0);
            ScopedStack::with_top_or_base_mut(&BASED, || 0, |top| *top += 1);
            assert_eq!(guard.pop(), 1);
        }

        assert_eq!(ScopedStack::cloned_top(&BASED), Some(5));
    }
}
