#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

module="${1:-lattice}"
config="cosmic-ray/${module}.toml"

if [ ! -f "$config" ]; then
  echo "usage: $0 [module]" >&2
  echo "no such config: $config" >&2
  echo "available modules:" >&2
  for f in cosmic-ray/*.toml; do
    name="$(basename "$f" .toml)"
    echo "  $name" >&2
  done
  exit 1
fi

# Every mutant has to see the same Hypothesis draws, and none may replay a
# counterexample saved while testing another, so the run uses the
# derandomized, database-free `mutation` profile from tests/conftest.py.
export HYPOTHESIS_PROFILE=mutation

# Cosmic-ray scores a mutant whose test run overruns the config's timeout as
# killed, so stop before mutating anything if the unmutated suite fails or
# does not finish within that timeout.
cosmic-ray baseline "$config"

rm -f session.sqlite
cosmic-ray init "$config" session.sqlite
cr-filter-pragma session.sqlite
cosmic-ray exec "$config" session.sqlite
cr-html session.sqlite > mutation-report.html
cr-report session.sqlite

echo "Report: mutation-report.html"
