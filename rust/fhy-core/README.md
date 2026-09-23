# fhy-core

Core utilities for the [FhY](https://github.com/actlab-fhy) compiler infrastructure, in Rust.

This crate is the Rust implementation of the `fhy_core` Python package. It is being ported one module at a time. Each ported module keeps the Python module's behavior, and golden corpora recorded from the Python implementation check the two against each other.

## Modules

- `identifier`: `Identifier`, a name hint paired with a process-unique id drawn from a global counter.
- `interned`: `Interned`, `InternRegistry` and `Canonical`, which keep one canonical instance per key.
- `op_attribute`: `OpAttribute`, an open tag attached to compiler operations. The shipped defaults are `get_commutative()`, `get_associative()`, `get_pure()` and `get_elementwise()`.
- `value_domain`: `ValueDomain`, an open, hierarchical classification of the values an operation handles. The shipped defaults are `get_data_domain()` and `get_address_domain()`.
- `testing` (behind the `testing` feature): `DeterministicIdentifierScope`, in which identifiers created with the same name hint compare equal. Enable it only from `[dev-dependencies]`.

## One copy per process

The id counter and each interned type's registry are process-global statics. A process must therefore hold exactly one compiled copy of this crate. Link it into one Python extension module, and compile other FhY packages' Rust code into that same module rather than into a second one.

## Using it

The crate is not published to crates.io. Depend on it through git:

```toml
[dependencies]
fhy-core = { git = "https://github.com/actlab-fhy/FhY-core.git" }
```

The minimum supported Rust version is 1.85.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
