//! Testing utilities for code that creates identifiers internally.
//!
//! Compiled for this crate's own tests and, for other crates, only when the
//! `testing` feature is enabled, normally from `[dev-dependencies]`.
//!
//! [`DeterministicIdentifierScope`] opens a scope in which constructing an
//! [`Identifier`](crate::identifier::Identifier) with a name hint seen before in the scope yields an
//! identifier equal to the first one, while each new name hint draws a fresh
//! id from the real counter. A test can then compare an object graph whose
//! identifiers were created inside the code under test with one it built
//! itself, including when those identifiers are map keys or set members.
//!
//! A scope belongs to the thread that entered it, so tests that `cargo test`
//! runs in parallel never see each other's scopes. Code under test that
//! creates identifiers on other threads needs those threads to join the scope
//! through a [`DeterministicIdentifierScopeHandle`]. This is where the scope
//! departs from the Python one, which patches `Identifier` itself and so
//! applies to every thread.
//!
//! A scope is only safe when every semantically distinct identifier created
//! inside it has a unique name hint. An identifier held by a lazily
//! initialized static counts as created inside the scope if the scope is what
//! first initializes it, so initialize such statics before entering a scope,
//! or keep their name hints out of it. This crate's own shipped constants are
//! never affected.
//!
//! [`Identifier::restore`](crate::identifier::Identifier::restore)
//! and cloning are unaffected: a deserialized identifier keeps its payload's
//! id.

use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, PoisonError};

use crate::identifier::allocate_id;

/// Ids a scope has handed out, by name hint.
#[derive(Debug, Default)]
struct ScopeTable {
    ids_by_name_hint: Mutex<HashMap<String, u64>>,
}

impl ScopeTable {
    /// Return the id recorded for `name_hint`, first recording one drawn from
    /// `allocate` if there is none.
    fn find_or_record_id(&self, name_hint: &str, allocate: impl FnOnce() -> u64) -> u64 {
        // `allocate` runs before the insert, so a panic inside it leaves the
        // map unchanged and a poisoned lock is safe to recover.
        let mut ids_by_name_hint = self
            .ids_by_name_hint
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(&id) = ids_by_name_hint.get(name_hint) {
            return id;
        }
        let id = allocate();
        ids_by_name_hint.insert(name_hint.to_owned(), id);
        id
    }
}

/// The scope the current thread is in, and how many of its guards are alive.
struct ThreadEntry {
    table: Arc<ScopeTable>,
    depth: usize,
}

thread_local! {
    static CURRENT_SCOPE: RefCell<Option<ThreadEntry>> = const { RefCell::new(None) };
}

/// Return the table of the scope recorded in `scope`, or `None` when the
/// thread is in no scope.
fn find_current_table(scope: &RefCell<Option<ThreadEntry>>) -> Option<Arc<ScopeTable>> {
    scope
        .borrow()
        .as_ref()
        .map(|entry| Arc::clone(&entry.table))
}

/// Return the id the current thread's scope records for `name_hint`, or
/// `None` when the thread is in no scope.
pub(crate) fn find_scoped_id(name_hint: &str) -> Option<u64> {
    let table = CURRENT_SCOPE
        .try_with(find_current_table)
        // An identifier created while this thread's locals are being
        // destroyed is outside any scope.
        .ok()
        .flatten()?;
    Some(table.find_or_record_id(name_hint, allocate_id))
}

/// Count one fewer live guard for the current thread's scope, leaving the
/// scope when none remain.
fn leave_scope(scope: &RefCell<Option<ThreadEntry>>) {
    let mut scope = scope.borrow_mut();
    if let Some(entry) = scope.as_mut() {
        entry.depth -= 1;
        if entry.depth == 0 {
            *scope = None;
        }
    }
}

/// Guard that keeps the current thread inside a deterministic-identifier
/// scope.
///
/// The thread leaves the scope when its last guard drops, including while a
/// panic unwinds.
///
/// # Examples
///
/// ```
/// use fhy_core::identifier::Identifier;
/// use fhy_core::testing::DeterministicIdentifierScope;
///
/// let scope = DeterministicIdentifierScope::enter();
/// let built_inside = Identifier::new("accumulator");
/// let built_by_test = Identifier::new("accumulator");
/// assert_eq!(built_inside, built_by_test);
/// drop(scope);
///
/// assert_ne!(Identifier::new("accumulator"), Identifier::new("accumulator"));
/// ```
///
/// A guard belongs to the thread that created it, so it is not `Send`:
///
/// ```compile_fail
/// fn require_send<T: Send>() {}
/// require_send::<fhy_core::testing::DeterministicIdentifierScope>();
/// ```
#[must_use = "the scope ends when the guard is dropped"]
#[derive(Debug)]
pub struct DeterministicIdentifierScope {
    table: Arc<ScopeTable>,
    // The guard counts toward one thread's scope, so it must not move to
    // another thread.
    not_send: PhantomData<*const ()>,
}

impl DeterministicIdentifierScope {
    /// Enter a scope on the current thread: start one with an empty table, or
    /// join the one the thread is already in.
    pub fn enter() -> Self {
        let current = CURRENT_SCOPE.with(find_current_table);
        Self::join(current.unwrap_or_else(|| Arc::new(ScopeTable::default())))
    }

    /// Return a handle another thread uses to join this scope.
    #[must_use]
    pub fn share(&self) -> DeterministicIdentifierScopeHandle {
        DeterministicIdentifierScopeHandle {
            table: Arc::clone(&self.table),
        }
    }

    /// Keep the current thread in `table`'s scope for the returned guard's
    /// lifetime.
    ///
    /// # Panics
    ///
    /// Panics if the current thread is already in a different scope.
    fn join(table: Arc<ScopeTable>) -> Self {
        CURRENT_SCOPE.with(|scope| {
            let mut scope = scope.borrow_mut();
            match scope.as_mut() {
                Some(entry) if Arc::ptr_eq(&entry.table, &table) => entry.depth += 1,
                Some(_) => panic!(
                    "the current thread is already in a different deterministic-identifier scope"
                ),
                None => {
                    *scope = Some(ThreadEntry {
                        table: Arc::clone(&table),
                        depth: 1,
                    });
                }
            }
        });
        Self {
            table,
            not_send: PhantomData,
        }
    }
}

impl Drop for DeterministicIdentifierScope {
    fn drop(&mut self) {
        // A thread whose locals are already destroyed has no scope to leave.
        CURRENT_SCOPE.try_with(leave_scope).ok();
    }
}

/// Handle that lets another thread join a deterministic-identifier scope.
///
/// A handle keeps its scope's table alive, so a thread may join after the
/// thread that shared the handle has left.
#[derive(Debug, Clone)]
pub struct DeterministicIdentifierScopeHandle {
    table: Arc<ScopeTable>,
}

impl DeterministicIdentifierScopeHandle {
    /// Join the handle's scope on the current thread.
    ///
    /// # Panics
    ///
    /// Panics if the current thread is already in a different scope.
    pub fn enter(&self) -> DeterministicIdentifierScope {
        DeterministicIdentifierScope::join(Arc::clone(&self.table))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::collections::HashSet;
    use std::panic::{self, AssertUnwindSafe};
    use std::sync::Barrier;
    use std::thread;

    use crate::identifier::Identifier;
    use crate::test_support::{assert_send_sync, compute_hash, reserve_pinned_id};

    #[test]
    fn identifiers_sharing_a_name_hint_behave_as_one_key() {
        let _scope = DeterministicIdentifierScope::enter();
        let a = Identifier::new("a");
        let a_again = Identifier::new("a");
        let b = Identifier::new("b");

        assert_eq!(a, a_again);
        assert_eq!(compute_hash(&a), compute_hash(&a_again));
        assert_ne!(a, b);
        let set: HashSet<Identifier> = [a.clone(), b.clone()].into_iter().collect();
        let map: HashMap<Identifier, &str> = [(a, "left"), (b, "right")].into_iter().collect();
        assert!(set.contains(&a_again));
        assert_eq!(map[&a_again], "left");
    }

    #[test]
    fn identifiers_sharing_a_name_hint_in_a_scope_are_equal() {
        let _scope = DeterministicIdentifierScope::enter();

        assert_eq!(Identifier::new("a"), Identifier::new("a"));
    }

    #[test]
    fn a_repeated_name_hint_gets_the_first_identifier() {
        let _scope = DeterministicIdentifierScope::enter();
        let first = Identifier::new("shared");
        let second = Identifier::new("shared");

        assert_eq!(second.id(), first.id());
        assert_eq!(second.name_hint(), first.name_hint());
    }

    #[test]
    fn a_new_name_hint_allocates_once_and_a_repeat_allocates_nothing() {
        let table = ScopeTable::default();
        let draws = Cell::new(0_u64);
        let allocate = || {
            draws.set(draws.get() + 1);
            100 + draws.get()
        };

        let ids = [
            table.find_or_record_id("a", allocate),
            table.find_or_record_id("b", allocate),
            table.find_or_record_id("a", allocate),
        ];

        assert_eq!(ids, [101, 102, 101]);
        assert_eq!(draws.get(), 2);
    }

    #[test]
    fn a_panic_inside_a_scope_ends_it() {
        let inside = RefCell::new(Vec::new());

        let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
            let _scope = DeterministicIdentifierScope::enter();
            inside.borrow_mut().push(Identifier::new("shared"));
            inside.borrow_mut().push(Identifier::new("shared"));
            panic!("user-raised");
        }));

        outcome.expect_err("the scope's body panics");
        let inside = inside.into_inner();
        assert_eq!(inside[0], inside[1]);
        assert_ne!(Identifier::new("shared"), Identifier::new("shared"));
    }

    #[test]
    fn nested_scopes_share_one_table_until_the_outermost_ends() {
        let outer = DeterministicIdentifierScope::enter();
        let outer_before = Identifier::new("shared");
        let inner = DeterministicIdentifierScope::enter();
        let from_inner = Identifier::new("shared");
        drop(inner);
        let outer_after = Identifier::new("shared");
        drop(outer);
        let after_exit = Identifier::new("shared");

        assert_eq!(from_inner, outer_before);
        assert_eq!(outer_after, outer_before);
        assert_ne!(after_exit, outer_before);
    }

    #[test]
    fn guards_dropped_out_of_order_keep_the_scope_until_the_last() {
        let outer = DeterministicIdentifierScope::enter();
        let inner = DeterministicIdentifierScope::enter();
        let before = Identifier::new("shared");
        drop(outer);
        let while_inner_lives = Identifier::new("shared");
        drop(inner);
        let after = Identifier::new("shared");

        assert_eq!(while_inner_lives, before);
        assert_ne!(after, before);
    }

    #[test]
    fn a_later_scope_forgets_an_earlier_scopes_hints() {
        let first = {
            let _scope = DeterministicIdentifierScope::enter();
            Identifier::new("shared")
        };
        let second = {
            let _scope = DeterministicIdentifierScope::enter();
            Identifier::new("shared")
        };

        assert_ne!(first, second);
    }

    #[test]
    fn deserialization_inside_a_scope_keeps_the_payload_id() {
        let far_id = reserve_pinned_id("deserialize-in-scope-anchor");
        let _scope = DeterministicIdentifierScope::enter();
        let constructed = Identifier::new("shared");

        let deserialized = Identifier::restore(far_id, "shared".to_string());

        assert_eq!(
            (deserialized.id(), deserialized.name_hint()),
            (far_id, "shared")
        );
        assert_ne!(deserialized, constructed);
    }

    #[test]
    fn clones_inside_a_scope_stay_equal() {
        let _scope = DeterministicIdentifierScope::enter();
        let original = Identifier::new("shared");

        let cloned = original.clone();

        assert_eq!(cloned, original);
        assert_eq!(cloned.name_hint(), "shared");
    }

    #[test]
    fn a_serde_round_trip_inside_a_scope_keeps_the_identifier() {
        let _scope = DeterministicIdentifierScope::enter();
        let original = Identifier::new("shared");

        let json = serde_json::to_string(&original).unwrap();
        let restored: Identifier = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, original);
        assert_eq!(restored.name_hint(), "shared");
    }

    #[test]
    fn threads_joining_a_scope_share_one_identifier_per_hint() {
        const THREADS: usize = 16;
        let scope = DeterministicIdentifierScope::enter();
        let handle = scope.share();
        let barrier = Barrier::new(THREADS);

        let ids: Vec<u64> = thread::scope(|threads| {
            let workers: Vec<_> = (0..THREADS)
                .map(|_| {
                    threads.spawn(|| {
                        let _joined = handle.enter();
                        barrier.wait();
                        Identifier::new("shared").id()
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect()
        });

        let on_this_thread = Identifier::new("shared").id();
        assert!(
            ids.iter().all(|&id| id == on_this_thread),
            "{ids:?} vs {on_this_thread}"
        );
        drop(scope);
    }

    #[test]
    fn a_thread_that_does_not_join_draws_fresh_identifiers() {
        let _scope = DeterministicIdentifierScope::enter();
        let on_this_thread = Identifier::new("shared");

        let (first, second) =
            thread::spawn(|| (Identifier::new("shared"), Identifier::new("shared")))
                .join()
                .unwrap();

        assert_ne!(first, second);
        assert_ne!(first, on_this_thread);
    }

    #[test]
    fn a_handle_joins_after_the_sharing_thread_leaves() {
        let scope = DeterministicIdentifierScope::enter();
        let shared = Identifier::new("shared");
        let handle = scope.share();
        drop(scope);

        let joined = thread::spawn(move || {
            let _joined = handle.enter();
            Identifier::new("shared")
        })
        .join()
        .unwrap();
        let fresh_scope = DeterministicIdentifierScope::enter();
        let in_fresh_scope = Identifier::new("shared");
        drop(fresh_scope);

        assert_eq!(joined, shared);
        assert_ne!(in_fresh_scope, shared);
    }

    #[test]
    fn joining_the_same_scope_again_nests() {
        let scope = DeterministicIdentifierScope::enter();
        let handle = scope.share();
        let before = Identifier::new("shared");
        let rejoined = handle.enter();
        drop(rejoined);
        let still_inside = Identifier::new("shared");
        drop(scope);
        let after = Identifier::new("shared");

        assert_eq!(still_inside, before);
        assert_ne!(after, before);
    }

    #[test]
    #[should_panic(expected = "already in a different deterministic-identifier scope")]
    fn joining_a_different_scope_panics() {
        let first = DeterministicIdentifierScope::enter();
        let first_handle = first.share();
        drop(first);
        let _second = DeterministicIdentifierScope::enter();

        let _joined = first_handle.enter();
    }

    #[test]
    fn new_unscoped_ignores_the_scope() {
        let _scope = DeterministicIdentifierScope::enter();
        let scoped = Identifier::new("shared");

        let unscoped = Identifier::new_unscoped("shared");
        let unscoped_again = Identifier::new_unscoped("shared");

        assert_ne!(unscoped, scoped);
        assert_ne!(unscoped_again, unscoped);
        assert_eq!(Identifier::new("shared"), scoped);
    }

    /// A handle stays usable from other threads, which is its purpose. The
    /// scope itself is deliberately `!Send`; see its `compile_fail` example.
    const _: () = assert_send_sync::<DeterministicIdentifierScopeHandle>();
}
