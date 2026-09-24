# fhy-core

Core IR building blocks for the [FhY](https://github.com/actlab-fhy) compiler, in Rust: identifiers, interned vocabularies, diagnostics and provenance, symbolic expressions with patterns and rewrite rules, tree traversals, and a compiler-pass framework.

This crate is the Rust implementation of the `fhy_core` Python package. Where a concept is defined in both languages (`identifier`, `interned`), the Rust behavior matches Python's and a golden corpus checks it. Elsewhere Rust defines the behavior.

## Modules

Each module depends only on the modules listed before it, except that `expr` and `pass` are independent of each other and `expr::passes` joins them. Each public item has exactly one public path.

- `identifier`: `Identifier`, a name hint paired with a process-unique id.
- `interned`: `Interned`, `InternRegistry` and `Canonical`, which keep one canonical value per key.
- `described_tag`: `DescribedTag<K>`, an open vocabulary entry named by an `Identifier`, and the sealed `TagKind` of its vocabularies.
- `diagnostic`: `Diagnostic`, `Note` and its `NoteKind`, and `ValidationReport`.
- `op_attribute`: `OpAttribute`, an open tag attached to compiler operations. The shipped defaults are `OpAttribute::commutative()`, `OpAttribute::associative()`, `OpAttribute::pure()` and `OpAttribute::elementwise()`.
- `value_domain`: `ValueDomain`, an open, hierarchical classification of the values an operation handles. The shipped defaults are `ValueDomain::data()` and `ValueDomain::address()`.
- `provenance`: `Position`, `Span` and `Provenance`, where a value came from.
- `tree`: the `Tree` trait, and iterative walks (`walk_tree`) and memoized rewrites (`rewrite_tree`) over any tree-shaped IR.
- `expr`: `Expression`, its node kinds, builders, literals and analyses, and `BooleanScreen`.
  - `expr::builtins`: the catalogue of built-in functions and constants.
  - `expr::pattern`: `Pattern`, `Capture`, the `Rule` trait and `RewriteRule`, and `apply_rewrite_rules`.
  - `expr::passes`: `RewriteRuleApplier`, `ExpressionPrettyFormatter` and `register_expression_passes`.
- `pass`: `CompilerPass`, `PassManager`, `FixpointPassGroup`, analyses, `Validator`s and the owned `PassRegistry`.

## One copy per process

The identifier id counter and each interned type's registry are process-global statics, and both are append-only: an id is never reissued and a canonical value is never replaced. A process must therefore hold exactly one compiled copy of this crate. Link it into one Python extension module, and compile other FhY packages' Rust code into that same module rather than into a second one. Everything else, including the pass registry, is an owned value.

## Using it

```toml
[dependencies]
fhy-core = "0.2"
```

The minimum supported Rust version is 1.85.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
