# Mutation testing configs

One `cosmic-ray.toml` per targeted module, named after the module. Each
config differs from the others only in `module-path`. Splitting per module keeps a run
scoped, since cosmic-ray mutates and re-tests one file at a time and a
combined config would force every mutation through the whole target list.

Run one config from the repo root:

```bash
uv run nox -s mutation -- lattice
```

The module defaults to `lattice` when no name is given, and a name without a
config aborts the session with the list of available ones. See the `mutation`
nox session for the exact `cosmic-ray baseline` / `init` / `exec` / report
pipeline.

The session first runs `cosmic-ray baseline`, which stops the run when the
unmutated suite fails or overruns the config's `timeout`, and it selects the
`mutation` Hypothesis profile (derandomized, no example database) so every
mutant sees the same draws. Cosmic-ray scores a mutant whose tests overrun
the timeout as killed, so each config's `timeout` leaves roughly four times
the suite's serial run time; keep that margin if the suite grows.

Cosmic-ray applies each mutant to the source file in place and restores it
after the mutant's test run, so do not run any other test command, commit,
or edit files under `src/` while a mutation run is active; a run that is
killed mid-mutant can also leave the mutated file behind, so check
`git status` afterwards.
