//! The stories of the cycle-collector helpers of the util module, run in an
//! interpreter this test binary embeds. The ones that traverse use a class of
//! this module that holds its objects as the helpers prescribe, and run the
//! interpreter's collector over a cycle that runs through it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::types::PyList;

use crate::util::testing::with_stand_ins;

use super::*;

/// A class that holds Python objects in each place the module prescribes a
/// rule for, and records that it was freed.
#[pyclass(module = "fhy_core_util_tests")]
struct Holder {
    slots: Slots,
    fields: Vec<Py<PyAny>>,
    locked: Mutex<Vec<Py<PyAny>>>,
    freed: Arc<AtomicBool>,
}

impl Drop for Holder {
    fn drop(&mut self) {
        self.freed.store(true, Ordering::SeqCst);
    }
}

#[pymethods]
impl Holder {
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.slots.traverse(&visit)?;
        traverse_all(&visit, &self.fields)?;
        traverse_locked(&self.locked, |held| traverse_all(&visit, held))
    }

    fn __clear__(&mut self) {
        self.fields.clear();
        clear_locked(&self.locked);
    }
}

impl Holder {
    /// Return a holder that holds nothing.
    fn plain(freed: &Arc<AtomicBool>) -> Self {
        Self::holding(freed, Slots::default(), Vec::new(), Vec::new())
    }

    /// Return a holder of `slots`, `fields` and `locked`.
    fn holding(
        freed: &Arc<AtomicBool>,
        slots: Slots,
        fields: Vec<Py<PyAny>>,
        locked: Vec<Py<PyAny>>,
    ) -> Self {
        Self {
            slots,
            fields,
            locked: Mutex::new(locked),
            freed: Arc::clone(freed),
        }
    }
}

/// Run the interpreter's collector.
fn collect_garbage(py: Python<'_>) {
    py.import("gc")
        .and_then(|gc| gc.call_method0("collect"))
        .expect("the collector runs");
}

/// Return the flag that a holder sets when freed.
fn flag() -> Arc<AtomicBool> {
    Arc::new(AtomicBool::new(false))
}

#[test]
fn a_slot_made_inside_a_collection_is_owned_by_it_alone() {
    with_stand_ins(|py| {
        let (outer, outer_slots) = collect_slots(|| {
            let first = Slot::new(py.None());
            let ((), inner_slots) = collect_slots(|| {
                drop(Slot::new(py.None()));
            });
            (first, inner_slots)
        });
        let (_first, inner_slots) = outer;

        assert_eq!(outer_slots.len(), 1);
        assert_eq!(inner_slots.len(), 1);
        assert!(!outer_slots.is_empty());
    });
}

#[test]
fn a_collection_that_made_no_slot_is_empty() {
    with_stand_ins(|_py| {
        let (value, slots) = collect_slots(|| 5);

        assert_eq!(value, 5);
        assert!(slots.is_empty());
        assert_eq!(slots.len(), 0);
    });
}

#[test]
fn a_slot_made_outside_any_collection_has_no_owner() {
    with_stand_ins(|py| {
        let slot = Slot::new(py.None());

        assert!(matches!(slot.0, SlotKind::Unowned(_)));
        assert!(slot.get(py).is_none());
    });
}

#[test]
fn an_unowned_slot_stays_out_of_the_innermost_collection() {
    with_stand_ins(|py| {
        let ((), slots) = collect_slots(|| {
            let slot = Slot::unowned(py.None());
            assert!(matches!(slot.0, SlotKind::Unowned(_)));
            assert!(slot.get(py).is_none());
        });

        assert!(slots.is_empty());
    });
}

#[test]
fn a_collection_that_panics_leaves_no_collector_behind() {
    with_stand_ins(|py| {
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            collect_slots(|| -> () { panic!("inside a collection") })
        }));

        let _panic = unwound.unwrap_err();
        assert_eq!(ScopedStack::depth(&COLLECTORS), 0);
        let slot = Slot::new(py.None());
        assert!(matches!(slot.0, SlotKind::Unowned(_)));
    });
}

#[test]
fn a_slots_object_is_the_one_it_was_made_from_in_both_forms() {
    with_stand_ins(|py| {
        let list = PyList::empty(py).into_any().unbind();
        let ((slot, owned), _slots) = collect_slots(|| {
            (
                Slot::new(list.clone_ref(py)),
                Slot::unowned(list.clone_ref(py)),
            )
        });

        for slot in [&slot, &owned] {
            assert!(slot.object(py).bind(py).is(list.bind(py)));
            assert!(slot.get(py).is(list.bind(py)));
        }
    });
}

#[test]
fn a_cycle_through_an_owned_slot_is_collected() {
    with_stand_ins(|py| {
        let freed = flag();
        let list = PyList::empty(py);
        let ((), slots) = collect_slots(|| drop(Slot::new(list.clone().into_any().unbind())));
        let holder = Bound::new(py, Holder::holding(&freed, slots, Vec::new(), Vec::new()))
            .expect("a holder");
        list.append(&holder).expect("close the cycle");

        drop(holder);
        drop(list);
        collect_garbage(py);

        assert!(freed.load(Ordering::SeqCst), "the cycle leaked");
    });
}

#[test]
fn a_cycle_through_a_slot_nothing_owns_is_not_visited() {
    with_stand_ins(|py| {
        let freed = flag();
        let list = PyList::empty(py);
        let slot = Slot::unowned(list.clone().into_any().unbind());
        let holder = Bound::new(py, Holder::plain(&freed)).expect("a holder");
        list.append(&holder).expect("close the cycle");
        // The slot only keeps its object alive, as a Rust-held reference does.
        let held_elsewhere = slot.object(py);

        drop(holder);
        collect_garbage(py);

        assert!(
            !freed.load(Ordering::SeqCst),
            "the holder was freed while referenced"
        );
        drop(held_elsewhere);
        list.call_method0("clear").expect("break the cycle");
        drop(slot);
        collect_garbage(py);
        assert!(freed.load(Ordering::SeqCst));
    });
}

#[test]
fn a_cycle_through_a_field_is_collected() {
    with_stand_ins(|py| {
        let freed = flag();
        let list = PyList::empty(py);
        let fields = vec![list.clone().into_any().unbind()];
        let holder = Bound::new(
            py,
            Holder::holding(&freed, Slots::default(), fields, Vec::new()),
        )
        .expect("a holder");
        list.append(&holder).expect("close the cycle");

        drop(holder);
        drop(list);
        collect_garbage(py);

        assert!(freed.load(Ordering::SeqCst), "the cycle leaked");
    });
}

#[test]
fn a_cycle_through_a_locked_field_is_collected() {
    with_stand_ins(|py| {
        let freed = flag();
        let list = PyList::empty(py);
        let locked = vec![list.clone().into_any().unbind()];
        let holder = Bound::new(
            py,
            Holder::holding(&freed, Slots::default(), Vec::new(), locked),
        )
        .expect("a holder");
        list.append(&holder).expect("close the cycle");

        drop(holder);
        drop(list);
        collect_garbage(py);

        assert!(freed.load(Ordering::SeqCst), "the cycle leaked");
    });
}

#[test]
fn traversal_skips_a_mutex_that_is_locked_and_visits_a_poisoned_one() {
    let locked = Mutex::new(7);
    let guard = locked.lock().expect("lock");
    let skipped = traverse_locked(&locked, |_value| -> Result<(), PyTraverseError> {
        panic!("a locked mutex is not traversed")
    });
    assert!(matches!(skipped, Ok(())));
    drop(guard);

    let seen = std::cell::Cell::new(0);
    let visited = traverse_locked(&locked, |value| {
        seen.set(*value);
        Ok(())
    });
    assert!(matches!(visited, Ok(())));
    assert_eq!(seen.get(), 7);

    let poisoned = Arc::new(Mutex::new(9));
    let clone = Arc::clone(&poisoned);
    let _unwound = std::thread::spawn(move || {
        let _guard = clone.lock().expect("lock");
        panic!("poison the lock");
    })
    .join();
    let visited = traverse_locked(&poisoned, |value| {
        seen.set(*value);
        Ok(())
    });
    assert!(matches!(visited, Ok(())));
    assert_eq!(seen.get(), 9);
}

#[test]
fn clearing_drops_the_old_value_after_the_lock_is_released() {
    /// A value that takes the lock of its mutex when dropped.
    struct TakesTheLock(Arc<Mutex<Vec<TakesTheLock>>>, Arc<AtomicBool>);

    impl Drop for TakesTheLock {
        fn drop(&mut self) {
            self.1.store(self.0.try_lock().is_ok(), Ordering::SeqCst);
        }
    }

    let mutex = Arc::new(Mutex::new(Vec::new()));
    let lock_was_free = flag();
    mutex
        .lock()
        .expect("lock")
        .push(TakesTheLock(Arc::clone(&mutex), Arc::clone(&lock_was_free)));

    clear_locked(&mutex);

    assert!(
        lock_was_free.load(Ordering::SeqCst),
        "dropped under the lock"
    );
    assert!(mutex.lock().expect("lock").is_empty());
}

#[test]
fn clearing_recovers_a_poisoned_mutex() {
    let poisoned = Arc::new(Mutex::new(vec![1, 2]));
    let clone = Arc::clone(&poisoned);
    let _unwound = std::thread::spawn(move || {
        let _guard = clone.lock().expect("lock");
        panic!("poison the lock");
    })
    .join();

    clear_locked(&poisoned);

    assert!(
        poisoned
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    );
}
