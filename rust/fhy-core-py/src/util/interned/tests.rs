//! The stories of the interned-class helpers of the util module, run in an
//! interpreter this test binary embeds.

use std::thread;

use pyo3::exceptions::{PyKeyError, PyNotImplementedError};
use pyo3::types::{PyList, PyString, PyType};

use crate::util::testing::{define, entry, evaluate, with_stand_ins};

use super::*;

/// Return a new object, which is no other object.
fn fresh(py: Python<'_>) -> Bound<'_, PyAny> {
    PyList::empty(py).into_any()
}

/// Return a new class named `name`.
fn class_named<'py>(py: Python<'py>, name: &str) -> Bound<'py, PyType> {
    evaluate(py, &format!("type({name:?}, (), {{}})"))
        .cast_into::<PyType>()
        .expect("a class")
}

#[test]
fn a_cache_keyed_by_id_keeps_the_first_object_of_a_key() {
    with_stand_ins(|py| {
        let cache: IdentityCache = IdentityCache::new();
        assert!(cache.get(py, &7).is_none());

        let first = fresh(py);
        let kept = cache.insert(7, first.clone());
        let rival = cache.insert(7, fresh(py));

        assert!(kept.is(&first));
        assert!(rival.is(&first));
        assert!(cache.get(py, &7).expect("cached").is(&first));
        assert!(cache.get(py, &8).is_none());
    });
}

#[test]
fn a_cache_keyed_by_string_is_read_by_str() {
    with_stand_ins(|py| {
        let cache: IdentityCache<String> = IdentityCache::default();
        let alpha = fresh(py);
        let beta = fresh(py);

        cache.insert("alpha".to_owned(), alpha.clone());
        cache.insert("beta".to_owned(), beta.clone());
        let second = cache.insert("alpha".to_owned(), fresh(py));

        assert!(second.is(&alpha));
        assert!(cache.get(py, "alpha").expect("alpha").is(&alpha));
        assert!(cache.get(py, "beta").expect("beta").is(&beta));
        assert!(cache.get(py, "gamma").is_none());
        assert!(cache.get(py, &"alpha".to_owned()).is_some());
    });
}

#[test]
fn a_cache_keyed_by_a_tuple_of_strings_is_read_by_the_borrowed_tuple() {
    with_stand_ins(|py| {
        let cache: IdentityCache<(String, u32)> = IdentityCache::new();
        let object = fresh(py);

        cache.insert(("a".to_owned(), 1), object.clone());

        assert!(
            cache
                .get(py, &("a".to_owned(), 1))
                .expect("cached")
                .is(&object)
        );
        assert!(cache.get(py, &("a".to_owned(), 2)).is_none());
    });
}

#[test]
fn threads_racing_to_cache_one_key_agree_on_one_object() {
    static CACHE: IdentityCache<String> = IdentityCache::new();
    Python::initialize();
    let winners: Vec<Py<PyAny>> = thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    Python::attach(|py| {
                        let object = PyList::empty(py).into_any();
                        CACHE.insert("raced".to_owned(), object).unbind()
                    })
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("thread"))
            .collect()
    });

    Python::attach(|py| {
        let first = winners[0].bind(py);
        assert!(winners.iter().all(|winner| winner.bind(py).is(first)));
        assert!(CACHE.get(py, "raced").expect("cached").is(first));
    });
}

#[test]
fn a_cache_whose_lock_was_poisoned_still_works() {
    with_stand_ins(|py| {
        let cache: IdentityCache = IdentityCache::new();
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = cache.objects.lock().expect("lock");
            panic!("poison the lock");
        }));
        poisoned.unwrap_err();

        let object = fresh(py);
        cache.insert(1, object.clone());
        assert!(cache.get(py, &1).expect("cached").is(&object));
    });
}

#[test]
fn the_registry_operations_raise_not_implemented_naming_the_class() {
    with_stand_ins(|py| {
        let class = class_named(py, "Tag");

        let error = raise_registry_append_only(&class, "unregister").expect_err("raises");

        assert!(error.is_instance_of::<PyNotImplementedError>(py));
        assert!(
            error
                .value(py)
                .to_string()
                .starts_with("Tag.unregister is not supported")
        );
    });
}

#[test]
fn a_missing_instance_is_the_key_error_of_require_interned() {
    with_stand_ins(|py| {
        let class = class_named(py, "Tag");
        let key = PyString::new(py, "k").into_any();

        let error = build_not_interned_error(&class, &key).expect("builds");

        assert!(error.is_instance_of::<PyKeyError>(py));
        assert_eq!(
            error.value(py).str().unwrap().to_string(),
            "'No registered \"Tag\" instance for key \\'k\\'.'"
        );
    });
}

#[test]
fn a_conflicting_payload_is_a_deserialization_value_error() {
    with_stand_ins(|py| {
        let class = class_named(py, "Tag");
        let key = PyString::new(py, "k").into_any();
        let canonical = 1_i32.into_pyobject(py).unwrap().into_any();
        let payload = 2_i32.into_pyobject(py).unwrap().into_any();

        let error =
            build_conflict_error(&class, &key, "size", &canonical, &payload).expect("builds");

        assert!(DESERIALIZATION_VALUE_ERROR.is_instance_of(py, &error));
        assert_eq!(
            error.value(py).to_string(),
            "Payload for \"Tag\" key 'k' conflicts with the canonical instance on size \
             (canonical 1, payload 2)."
        );
    });
}

/// A logging handler that keeps the formatted messages of `fhy_core.traits.interned`.
const RECORDER: &str = "
import logging
messages = []
class Recorder(logging.Handler):
    def emit(self, record):
        messages.append(record.getMessage())
handler = Recorder()
logger = logging.getLogger('fhy_core.traits.interned')
logger.addHandler(handler)
logger.setLevel(logging.WARNING)
logger.propagate = False
";

#[test]
fn an_ignored_description_is_logged_only_when_it_differs() {
    with_stand_ins(|py| {
        let namespace = define(py, RECORDER);
        let class = class_named(py, "Tag");
        let key = PyString::new(py, "k").into_any();
        let canonical = PyString::new(py, "kept");

        warn_if_description_ignored(&class, &key, &canonical, &PyString::new(py, "kept"))
            .expect("equal");
        let messages = entry(&namespace, "messages");
        assert_eq!(messages.len().unwrap(), 0);

        warn_if_description_ignored(&class, &key, &canonical, &PyString::new(py, "other"))
            .expect("differs");

        assert_eq!(messages.len().unwrap(), 1);
        assert_eq!(
            messages.get_item(0).unwrap().to_string(),
            "Tag 'k' already canonical; keeping description='kept' and ignoring payload 'other'."
        );
        py.run(
            c"import logging\nlogging.getLogger('fhy_core.traits.interned').handlers.clear()",
            None,
            None,
        )
        .expect("remove the handler");
    });
}
