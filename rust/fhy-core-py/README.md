# fhy-core-py

The [PyO3](https://pyo3.rs/) binding of [`fhy-core`](https://crates.io/crates/fhy-core), as a library. It is the Rust side of the `fhy_core` Python package: `register` adds every `fhy_core._rs` class, function and piece of module state to an extension module it is given.

`fhy_core`'s own wheel builds its `fhy_core._rs` from this crate. A downstream *FhY* product with Rust code depends on it to build one combined extension module (an aggregate) for the whole process, instead of an extension of its own that links `fhy-core`. Its `#[pymodule]` calls `fhy_core_py::register`, then its own registration functions. The crate's public surface for that is:

- `register`, and `VERSION_ATTRIBUTE`, the module attribute that carries the version;
- `convert`, which converts `fhy_core`'s Python objects to and from the `fhy-core` values (expressions, params, search-space parts) and registers the search-space kinds a downstream crate defines;
- `util`, the helpers `fhy_core`'s own Rust-backed classes are written with;
- `util::testing`, behind the test-only `testing` feature, which is not stable API. Enable it in `[dev-dependencies]` only.

## Versions

Pin the exact release that matches the `fhy_core` Python package your product is installed with:

```toml
[dependencies]
fhy-core-py = "=0.2.0"
```

`fhy_core` refuses to load an aggregate built from another version. Depend on `fhy-core` itself, if at all, at the same version, so the process holds one copy of its statics.

The rules an aggregate follows, and how `fhy_core` loads one in place of its own extension, are in the repository's [CONTRIBUTING.md](https://github.com/actlab-fhy/FhY-core/blob/main/CONTRIBUTING.md) ("Porting to Rust").

## License

BSD-3-Clause.
