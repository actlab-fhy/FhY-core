//! One guard for the binding's thread-local stacks.
//!
//! A class that keeps per-call state declares a stack in a `thread_local!`,
//! `static STACK: ScopedStack<Frame> = const { ScopedStack::new() };`, and
//! pushes a frame for the duration of each call.
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
use std::fmt;
use std::thread::LocalKey;

/// A thread-local stack of frames, innermost last.
#[derive(Debug)]
pub struct ScopedStack<T: 'static>(RefCell<Vec<T>>);

impl<T: 'static> Default for ScopedStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static> ScopedStack<T> {
    /// Return an empty stack, for a `thread_local!` initializer.
    #[must_use]
    pub const fn new() -> Self {
        Self(RefCell::new(Vec::new()))
    }

    /// Push `frame` onto `key`'s stack, and return the guard that pops it.
    ///
    /// # Panics
    ///
    /// Panics if called from a closure that a reader of this stack is
    /// running, since the stack is borrowed then, and if the thread's local
    /// storage is being destroyed.
    pub fn push(key: &'static LocalKey<Self>, frame: T) -> ScopedGuard<T> {
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
    ///
    /// # Panics
    ///
    /// Panics if `read` pushes onto or pops from this stack, whose frames it
    /// borrows, and if the thread's local storage is being destroyed.
    pub fn with_top<R>(key: &'static LocalKey<Self>, read: impl FnOnce(Option<&T>) -> R) -> R {
        key.with(|stack| read(stack.0.borrow().last()))
    }

    /// Return a clone of the innermost frame, or `None` when none is
    /// pushed.
    ///
    /// # Panics
    ///
    /// Panics if called from an `update` closure of this stack, whose frames
    /// it borrows mutably, and if the thread's local storage is being
    /// destroyed.
    #[must_use]
    pub fn cloned_top(key: &'static LocalKey<Self>) -> Option<T>
    where
        T: Clone,
    {
        key.with(|stack| stack.0.borrow().last().cloned())
    }

    /// Return `read` of every frame, outermost first.
    ///
    /// # Panics
    ///
    /// Panics if `read` pushes onto or pops from this stack, whose frames it
    /// borrows, and if the thread's local storage is being destroyed.
    pub fn with_frames<R>(key: &'static LocalKey<Self>, read: impl FnOnce(&[T]) -> R) -> R {
        key.with(|stack| read(&stack.0.borrow()))
    }

    /// Return `update` of the innermost frame, or of `None` when none is
    /// pushed.
    ///
    /// # Panics
    ///
    /// Panics if `update` reads, pushes onto or pops from this stack, whose
    /// frames it borrows mutably, and if the thread's local storage is being
    /// destroyed.
    pub fn with_top_mut<R>(
        key: &'static LocalKey<Self>,
        update: impl FnOnce(Option<&mut T>) -> R,
    ) -> R {
        key.with(|stack| update(stack.0.borrow_mut().last_mut()))
    }

    /// Return `update` of the innermost frame, pushing `base()` first as a
    /// frame no guard pops when the stack is empty: the state that holds
    /// outside every call.
    ///
    /// # Panics
    ///
    /// Panics if `base` or `update` reads, pushes onto or pops from this
    /// stack, whose frames it borrows mutably, and if the thread's local
    /// storage is being destroyed.
    pub fn with_top_or_base_mut<R>(
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

    /// Return the number of frames.
    #[must_use]
    pub fn depth(key: &'static LocalKey<Self>) -> usize {
        key.with(|stack| stack.0.borrow().len())
    }
}

/// The frame a [`ScopedStack::push`] pushed, popped by [`pop`](Self::pop) or
/// when dropped.
#[must_use = "dropping the guard pops its frame at once"]
pub struct ScopedGuard<T: 'static> {
    key: &'static LocalKey<ScopedStack<T>>,
    depth: usize,
    popped: bool,
}

impl<T: 'static> ScopedGuard<T> {
    /// Pop the frame and return it.
    ///
    /// # Panics
    ///
    /// Panics if the stack is borrowed, which only a closure of a reader of
    /// this stack that pops from it can cause, and if an outer guard dropped
    /// first has already removed the frame.
    #[must_use]
    pub fn pop(mut self) -> T {
        self.popped = true;
        let Some((frame, above)) = self.take() else {
            panic!("the frame was removed by a guard of an outer frame");
        };
        drop(above);
        frame
    }

    /// Remove the frame, and any frame left above it, from the stack.
    ///
    /// Guards are dropped in scope order, so the guard's frame is the
    /// innermost one; a frame above it can only be one whose guard was
    /// leaked, which goes with it. A guard dropped after the guard of a frame
    /// below it finds its frame gone, which the removal of that frame took
    /// along, and removes nothing: it answers `None`. What is removed is
    /// dropped by the caller, outside the stack's borrow: a frame may hold
    /// Python objects, whose finalizers may run Python.
    fn take(&self) -> Option<(T, Vec<T>)> {
        let mut removed = self.key.with(|stack| {
            let mut frames = stack.0.borrow_mut();
            let start = self.depth.saturating_sub(1).min(frames.len());
            frames.split_off(start)
        });
        let above = removed.split_off(1.min(removed.len()));
        removed.pop().map(|frame| (frame, above))
    }
}

impl<T: 'static> fmt::Debug for ScopedGuard<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScopedGuard")
            .field("depth", &self.depth)
            .field("popped", &self.popped)
            .finish_non_exhaustive()
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

    fn top_of(key: &'static LocalKey<ScopedStack<u32>>) -> Option<u32> {
        ScopedStack::with_top(key, |top: Option<&u32>| top.copied())
    }

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

    #[test]
    fn with_frames_reads_every_frame_outermost_first_and_with_top_the_innermost() {
        thread_local! {
            static FRAMES: ScopedStack<u32> = const { ScopedStack::new() };
        }
        assert_eq!(top_of(&FRAMES), None);
        let outer = ScopedStack::push(&FRAMES, 1);
        let inner = ScopedStack::push(&FRAMES, 2);

        assert_eq!(
            ScopedStack::with_frames(&FRAMES, <[u32]>::to_vec),
            vec![1, 2]
        );
        assert_eq!(top_of(&FRAMES), Some(2));
        assert_eq!(ScopedStack::depth(&FRAMES), 2);
        assert_eq!(inner.pop(), 2);
        assert_eq!(ScopedStack::cloned_top(&FRAMES), Some(1));
        assert_eq!(outer.pop(), 1);
        assert_eq!(ScopedStack::cloned_top(&FRAMES), None);
    }

    #[test]
    fn with_top_mut_updates_only_the_innermost_frame() {
        thread_local! {
            static FRAMES: ScopedStack<u32> = const { ScopedStack::new() };
        }
        assert!(ScopedStack::with_top_mut(&FRAMES, |top| top.is_none()));
        let outer = ScopedStack::push(&FRAMES, 10);
        let inner = ScopedStack::push(&FRAMES, 20);

        ScopedStack::with_top_mut(&FRAMES, |top| {
            if let Some(top) = top {
                *top += 1;
            }
        });

        assert_eq!(inner.pop(), 21);
        assert_eq!(outer.pop(), 10);
    }

    #[test]
    fn a_guard_that_is_leaked_above_another_goes_with_the_frame_below_it() {
        thread_local! {
            static FRAMES: ScopedStack<u32> = const { ScopedStack::new() };
        }
        let outer = ScopedStack::push(&FRAMES, 1);
        std::mem::forget(ScopedStack::push(&FRAMES, 2));

        assert_eq!(outer.pop(), 1);
        assert_eq!(ScopedStack::depth(&FRAMES), 0);
    }

    #[test]
    fn a_guard_dropped_after_the_guard_of_a_frame_below_it_removes_nothing() {
        thread_local! {
            static FRAMES: ScopedStack<u32> = const { ScopedStack::new() };
        }
        let outer = ScopedStack::push(&FRAMES, 1);
        let inner = ScopedStack::push(&FRAMES, 2);

        drop(outer);
        assert_eq!(ScopedStack::depth(&FRAMES), 0);
        drop(inner);

        assert_eq!(ScopedStack::depth(&FRAMES), 0);
    }

    #[test]
    fn a_stack_and_a_guard_are_debug_values() {
        thread_local! {
            static FRAMES: ScopedStack<u32> = const { ScopedStack::new() };
        }
        let guard = ScopedStack::push(&FRAMES, 3);

        assert!(format!("{guard:?}").starts_with("ScopedGuard"));
        assert_eq!(
            format!("{:?}", ScopedStack::<u32>::default()),
            "ScopedStack(RefCell { value: [] })"
        );
        assert_eq!(guard.pop(), 3);
    }
}
