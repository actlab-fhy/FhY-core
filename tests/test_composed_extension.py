"""Tests that ``fhy_core`` runs on a combined extension module and shares it.

``rust/example-aggregate`` is a test-only aggregate: it registers all of
``fhy-core-py`` and one extra class, ``Tagger``, into one native module. The
fixture builds it with cargo, as a downstream product's aggregate would be
built, and installs it in a temporary directory with a ``fhy_core.native``
entry point, so a fresh interpreter finds it the way it finds an installed
product. The tests then check what composition promises: one copy of the Rust
code, so one identifier counter, one registry per interned type and one set
of classes, under the qualified names users see today.
"""

import importlib.machinery
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap

import pytest

from fhy_core._extension import NATIVE_MODULE_ENVIRONMENT_VARIABLE
from tests.native_modules import (
    read_import_error,
    run_python,
    write_distribution,
    write_native_module,
)

ROOT = pathlib.Path(__file__).parent.parent
PACKAGE = "fhy-core-example-aggregate"
MODULE = "_fhy_example_aggregate"
# The layers the aggregate holds, one entry point each.
LAYERS = ("fhy_example", "fhy_example_search")

pytestmark = [pytest.mark.slow, pytest.mark.subprocess]


def _library_name() -> str:
    """Return the file name cargo gives the aggregate's library."""
    stem = "fhy_example_aggregate"
    if sys.platform == "win32":
        return f"{stem}.dll"
    if sys.platform == "darwin":
        return f"lib{stem}.dylib"
    return f"lib{stem}.so"


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Build the example aggregate and install it in a temporary directory.

    Returns:
        A directory that holds the aggregate's extension module and the
        metadata of a distribution that advertises it.

    """
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not installed, so the example aggregate cannot be built")
    version = ".".join(map(str, sys.version_info[:2]))
    # A target directory of its own per interpreter: pyo3 is built against
    # one interpreter, and this keeps the editable install's build intact.
    target = ROOT / "target" / f"composition-{version}"
    environment = {
        **os.environ,
        "CARGO_TARGET_DIR": str(target),
        "PYO3_PYTHON": sys.executable,
        # What maturin sets: leave libpython unlinked in an extension.
        "PYO3_BUILD_EXTENSION_MODULE": "1",
    }
    command = ["cargo", "rustc", "-p", PACKAGE, "--lib", "--locked"]
    if sys.platform == "darwin":
        # What maturin passes on macOS: an extension resolves the
        # interpreter's symbols when it is loaded, so the linker must not
        # require them.
        command += ["--", "-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]
    built = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    site = tmp_path_factory.mktemp("site")
    extension_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    shutil.copy(
        target / "debug" / _library_name(), site / f"{MODULE}{extension_suffix}"
    )
    write_distribution(site, PACKAGE, dict.fromkeys(LAYERS, MODULE))
    return site


def _run(site: pathlib.Path, program: str) -> str:
    """Run `program` in a fresh interpreter that finds the aggregate.

    Returns:
        Its standard output.

    """
    completed = run_python(textwrap.dedent(program), python_path=[site])
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_the_aggregate_is_the_extension_of_the_process(site: pathlib.Path) -> None:
    """Test the aggregate is loaded and installed as `fhy_core._rs`."""
    output = _run(
        site,
        f"""
        import sys
        import fhy_core
        import {MODULE} as aggregate
        from fhy_core import _rs

        assert _rs is aggregate
        assert sys.modules["fhy_core._rs"] is aggregate
        assert aggregate.__name__ == "{MODULE}"
        assert fhy_core.symbolic.param.Param.__mro__[1] is aggregate.Param
        print("ok")
        """,
    )

    assert output.split() == ["ok"]


def test_the_classes_keep_their_fhy_core_qualified_names(site: pathlib.Path) -> None:
    """Test classes of the aggregate are `fhy_core._rs` classes to users."""
    output = _run(
        site,
        f"""
        import pickle
        import {MODULE} as aggregate
        import fhy_core
        from fhy_core.op_attribute import COMMUTATIVE, PURE, OpAttribute
        from fhy_core.symbolic.constraint import ConstraintSystem  # noqa
        from fhy_core.symbolic.expression import LiteralExpression

        assert aggregate.Param.__module__ == "fhy_core._rs"
        assert repr(aggregate.Param) == "<class 'fhy_core._rs.Param'>"
        assert aggregate.Tagger.__module__ == "fhy_example_aggregate"
        attribute = COMMUTATIVE
        assert isinstance(attribute, aggregate.OpAttribute)
        assert pickle.loads(pickle.dumps(attribute)) is attribute
        expression = LiteralExpression(3)
        assert isinstance(expression, aggregate.Expression)
        assert pickle.loads(pickle.dumps(expression)).is_structurally_equivalent(
            expression
        )
        print("ok")
        """,
    )

    assert output.split() == ["ok"]


def test_the_identifier_counter_is_shared(site: pathlib.Path) -> None:
    """Test ids drawn in Python and in the aggregate's Rust never collide."""
    output = _run(
        site,
        f"""
        import {MODULE} as aggregate
        from fhy_core.identifier import Identifier

        first = Identifier("first")
        derived = aggregate.Tagger.derive(first, "_derived")
        second = Identifier("second")

        assert type(derived) is Identifier
        assert derived.name_hint == "first_derived"
        assert [derived.id - first.id, second.id - derived.id] == [1, 1]
        echoed = aggregate.Tagger.echo(first)
        assert (echoed.id, echoed.name_hint) == (first.id, first.name_hint)
        print("ok")
        """,
    )

    assert output.split() == ["ok"]


def test_the_interned_registry_is_shared(site: pathlib.Path) -> None:
    """Test a canonical attribute is the same value on both sides."""
    output = _run(
        site,
        f"""
        import {MODULE} as aggregate
        from fhy_core.identifier import Identifier
        from fhy_core.op_attribute import COMMUTATIVE, PURE, OpAttribute

        tagger = aggregate.Tagger
        assert tagger.is_commutative(COMMUTATIVE)
        assert not tagger.is_commutative(PURE)

        name = Identifier("shared_tag")
        from_rust = tagger.tag(name, "registered by the aggregate")
        assert type(from_rust) is OpAttribute
        assert from_rust is OpAttribute.get_interned(name)
        assert tagger.tag(name, "ignored") is from_rust
        assert not tagger.is_commutative(from_rust)
        print("ok")
        """,
    )

    assert output.split() == ["ok"]


def test_a_value_of_another_type_is_refused(site: pathlib.Path) -> None:
    """Test the conversions raise `TypeError` for a foreign object."""
    output = _run(
        site,
        f"""
        import fhy_core
        import {MODULE} as aggregate

        for call in (
            lambda: aggregate.Tagger.echo("not an identifier"),
            lambda: aggregate.Tagger.is_commutative(object()),
        ):
            try:
                call()
            except TypeError as error:
                assert "must be an Identifier" in str(error) or "expected" in str(error)
            else:
                raise AssertionError("no TypeError")
        print("ok")
        """,
    )

    assert output.split() == ["ok"]


def test_the_environment_variable_selects_the_aggregate(
    site: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Test naming the aggregate overrides what the entry points advertise."""
    write_distribution(tmp_path, "other-product", {"other": "native_other"})

    completed = run_python(
        "import fhy_core\nprint(fhy_core._rs.__name__)",
        python_path=[site, tmp_path],
        environment={NATIVE_MODULE_ENVIRONMENT_VARIABLE: MODULE},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == [MODULE]


def test_a_second_native_module_is_refused_naming_both(site: pathlib.Path) -> None:
    """Test an aggregate is refused where `fhy_core._rs` is already another module."""
    completed = run_python(
        "import sys, types\n"
        "sys.modules['fhy_core._rs'] = types.ModuleType('fhy_core._rs')\n"
        "import fhy_core\n",
        python_path=[site],
    )

    message = read_import_error(completed)

    assert "one native extension module per process" in message
    assert f"'{MODULE}'" in message
    assert "'fhy_core._rs'" in message


def test_the_aggregate_holds_fhy_core_state_once(site: pathlib.Path) -> None:
    """Test the aggregate reports the `fhy_core` version and holds its state."""
    output = _run(
        site,
        f"""
        import importlib.metadata
        import {MODULE} as aggregate
        import fhy_core

        assert aggregate.__fhy_core_version__ == importlib.metadata.version("fhy_core")
        assert not hasattr(aggregate, "__version__")
        assert hasattr(aggregate, "_verification_registry")
        print("ok")
        """,
    )

    assert output.split() == ["ok"]


def test_each_layer_of_the_aggregate_reports_its_version(site: pathlib.Path) -> None:
    """Test the aggregate sets one `__<layer>_version__` per layer."""
    output = _run(
        site,
        f"""
        import importlib.metadata
        import {MODULE} as aggregate
        import fhy_core

        from fhy_core._extension import normalize_pep440_version

        # The crates report a Cargo version; the package's is PEP 440 normal.
        version = importlib.metadata.version("fhy_core")
        assert normalize_pep440_version(aggregate.__fhy_example_version__) == version
        assert (
            normalize_pep440_version(aggregate.__fhy_example_search_version__)
            == version
        )
        selection = fhy_core.native.get_selection()
        assert selection.module == "{MODULE}"
        assert selection.reason == "entry_points"
        assert selection.layers == {{"fhy_core", "fhy_example", "fhy_example_search"}}
        print("ok")
        """,
    )

    assert output.split() == ["ok"]


def test_a_layer_is_required_and_installed_for_its_package(
    site: pathlib.Path,
) -> None:
    """Test `require_layer` finds each layer and installs the module."""
    output = _run(
        site,
        f"""
        import importlib.metadata
        import sys
        import types
        import {MODULE} as aggregate
        import fhy_core

        # The packages that own the layers, as they are while importing.
        for package in ("fhy_example", "search_package", "not_installed"):
            sys.modules[package] = types.ModuleType(package)
        version = importlib.metadata.version("fhy_core")
        module = fhy_core.native.require_layer("fhy_example", version)
        assert module is aggregate
        assert sys.modules["fhy_example._rs"] is aggregate
        assert sys.modules["fhy_example"]._rs is aggregate
        module = fhy_core.native.require_layer(
            "fhy_example_search", version, package="search_package"
        )
        assert sys.modules["search_package._rs"] is aggregate
        assert sys.modules["search_package"]._rs is aggregate
        try:
            fhy_core.native.require_layer(
                "fhy_example_search", "999.0.0", package="search_package"
            )
        except ImportError as error:
            assert "is stale" in str(error)
            assert "{PACKAGE}" in str(error)
        else:
            raise AssertionError("no ImportError")
        try:
            fhy_core.native.require_layer("not_installed", version)
        except ImportError as error:
            assert "is not in the native module" in str(error)
        else:
            raise AssertionError("no ImportError")
        print("ok")
        """,
    )

    assert output.split() == ["ok"]


def test_a_lower_module_next_to_the_aggregate_is_never_imported(
    site: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Test the aggregate, which holds every layer, is selected over a lower one.

    The lower module claims one of the aggregate's layers and raises when it
    is imported, so the aggregate is selected only if the lower one is not
    imported, and composition still works on it.
    """
    write_distribution(tmp_path, "lower-product", {"fhy_example": "native_lower"})
    write_native_module(
        tmp_path,
        "native_lower",
        {},
        fhy_core_version="0",
        on_import="raise RuntimeError('the lower module was imported')",
    )

    completed = run_python(
        textwrap.dedent(
            f"""
            import sys
            import fhy_core
            import {MODULE} as aggregate
            from fhy_core.identifier import Identifier

            assert fhy_core._rs is aggregate
            assert "native_lower" not in sys.modules
            selection = fhy_core.native.get_selection()
            assert selection.module == "{MODULE}"
            derived = aggregate.Tagger.derive(Identifier("first"), "_derived")
            assert derived.name_hint == "first_derived"
            print("ok")
            """
        ),
        python_path=[site, tmp_path],
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["ok"]
