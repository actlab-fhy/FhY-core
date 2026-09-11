# Mutation testing configs

One `cosmic-ray.toml` per targeted module, named after the module. Each
config differs from the others only in `module-path`. Splitting per module keeps a run
scoped, since cosmic-ray mutates and re-tests one file at a time and a
combined config would force every mutation through the whole target list.

Run one config with the wrapper script from the repo root:

```bash
scripts/run-mutation.sh lattice
```

or directly with `nox`:

```bash
uv run nox -s mutation -- lattice
```

Both default to `lattice` when no module name is given. See the script and
the `mutation` nox session for the exact `cosmic-ray init` / `exec` / report
pipeline.

Cosmic-ray applies each mutant to the source file in place and restores it
after the mutant's test run, so do not run any other test command, commit,
or edit files under `src/` while a mutation run is active; a run that is
killed mid-mutant can also leave the mutated file behind, so check
`git status` afterwards.
