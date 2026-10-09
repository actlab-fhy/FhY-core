# fhy-core-py

[![crates.io](https://img.shields.io/crates/v/fhy-core-py.svg)](https://crates.io/crates/fhy-core-py)
[![docs.rs](https://img.shields.io/docsrs/fhy-core-py)](https://docs.rs/fhy-core-py)

[PyO3](https://pyo3.rs/) bindings of [`fhy-core`](https://crates.io/crates/fhy-core),
packaged as a library rather than an extension module. `register` adds
every class, function and piece of module state of `fhy_core._rs` to a
module you pass in.

The [`fhy_core`](https://pypi.org/project/fhy_core/) wheel builds its own
`fhy_core._rs` from this crate. You need this crate directly only if you
write a *FhY* package with its own Rust code. Pure-Rust users want
`fhy-core`.

## Why a library

`fhy-core` keeps the identifier counter and the interned registries in
process-global statics, and PyO3 creates distinct Python types per
extension module. A second extension that linked `fhy-core` would issue
colliding ids, hold registries whose canonical values never match, and fail
`isinstance` checks against `fhy_core`'s classes. A downstream package
therefore does not ship its own extension. It builds one combined module,
an aggregate, that registers `fhy_core`'s bindings and its own:

```rust
use pyo3::prelude::*;

#[pymodule(name = "_my_product_native", gil_used = true)]
fn aggregate(module: &Bound<'_, PyModule>) -> PyResult<()> {
    fhy_core_py::register(module.py(), module)?;
    my_product_py::register(module)
}
```

The product's wheel advertises the module through the `fhy_core.native`
entry-point group, and `fhy_core` loads it in place of its own `_rs`:

```toml
[project.entry-points."fhy_core.native"]
my_product = "_my_product_native"
```

An aggregate must also:

- set `module = "..."` on its own `#[pyclass]`es (the bindings here use
  `module = "fhy_core._rs"`, so `pickle` and `repr` are unchanged inside an
  aggregate);
- import no `fhy_core` Python code when the module itself is imported;
- have its Python package import `fhy_core` before calling into the module;
- depend on the same `fhy-core` source and version as this crate.

The loader, the version check and the full set of rules are documented in
[CONTRIBUTING.md](https://github.com/actlab-fhy/FhY-core/blob/main/CONTRIBUTING.md#one-extension-module-per-process).

## Versions

Pin the release that matches the `fhy_core` package you install with:

```toml
[dependencies]
fhy-core-py = "=0.2.0"
fhy-core = "=0.2.0"   # only if you use it directly; same version
```

`register` sets `__fhy_core_version__` (`VERSION_ATTRIBUTE`) on the module,
and `fhy_core` refuses an aggregate whose version differs from its own.

## Public API

| Item | Purpose |
| :--- | :--- |
| `register`, `VERSION_ATTRIBUTE` | Register the bindings into a module; the version attribute name |
| `convert` | Convert between `fhy_core` Python objects and `fhy-core` values: identifiers, expressions, types, params, param assignments, value domains, op attributes, diagnostics, validation reports |
| `convert::numpy` | NumPy arrays to and from the core evaluator's values, NumPy-backed kernels for the transcendental natives. No `rust-numpy` types in signatures. |
| `convert::param` | `with_param_context`: answer a param question with the same solver, function registry and observer `fhy_core` uses |
| `convert::search_space` | Search-space conversions, and registration of downstream `Variable` and `Alternative` kinds |
| `util` | Helpers `fhy_core`'s own Rust-backed classes are built with: cached imports, exception classes, identity caches for interned values, dataclass-style equality and `repr`, payload readers, pending exceptions for infallible hooks, GC slots |
| `util::testing` | Stand-in `fhy_core` modules for embedded-interpreter tests. Behind the `testing` feature; not stable API, enable it in `[dev-dependencies]` only. |

The bindings have not been verified safe without the GIL. `fhy_core`'s own
module declares `gil_used = true`, so a free-threaded interpreter
re-enables the GIL on import; declare the same on an aggregate.

## License

BSD-3-Clause. See
[LICENSE](https://github.com/actlab-fhy/FhY-core/blob/main/LICENSE).
