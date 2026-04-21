"""Tests the logging utility."""

import logging
from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from fhy_core.logger import (
    add_file_handler,
    configure_logging,
    get_logger,
    install_null_handler,
    reset,
)


@pytest.fixture(autouse=True)
def isolate_logging_state() -> Generator[Callable[[str], logging.Logger], None, None]:
    root_logger = logging.getLogger()
    root_logger_handlers = list(root_logger.handlers)
    root_logger_level = root_logger.level

    touched_names: set[str] = set()

    def touch(name: str) -> logging.Logger:
        touched_names.add(name)
        return logging.getLogger(name)

    yield touch

    root_logger.handlers[:] = root_logger_handlers
    root_logger.setLevel(root_logger_level)

    for name in touched_names:
        logger = logging.getLogger(name)
        logger.handlers[:] = []
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

    reset()


def _get_non_file_stream_handlers(
    logger: logging.Logger,
) -> list[logging.StreamHandler]:  # type: ignore[type-arg]
    return [
        h
        for h in logger.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]


def _get_file_handlers(logger: logging.Logger) -> list[logging.FileHandler]:
    return [h for h in logger.handlers if isinstance(h, logging.FileHandler)]


def test_get_logger_returns_named_logger() -> None:
    """Test `get_logger` returns a logger with the correct name."""
    name = "some.pkg.mod"

    logger = get_logger(name)

    assert isinstance(logger, logging.Logger)
    assert logger.name == name


def test_get_logger_does_not_configure_handlers_or_level() -> None:
    """Test `get_logger` does not modify root logger configuration."""
    root_logger = logging.getLogger()
    before_handlers = list(root_logger.handlers)
    before_level = root_logger.level

    _ = get_logger("x.y.z")

    assert list(root_logger.handlers) == before_handlers
    assert root_logger.level == before_level


def test_install_null_handler_adds_only_if_no_handlers(
    isolate_logging_state: Callable[[str], logging.Logger],
) -> None:
    """Test `install_null_handler` adds a NullHandler only if no handlers exist."""
    namespace = "my_namespace"
    logger = isolate_logging_state(namespace)
    assert logger.handlers == []

    install_null_handler(namespace)

    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.NullHandler)


def test_install_null_handler_does_not_add_if_handlers_exist(
    isolate_logging_state: Callable[[str], logging.Logger],
) -> None:
    """Test `install_null_handler` does not add a NullHandler if handlers exist."""
    namespace = "my_namespace"
    logger = isolate_logging_state(namespace)

    sentinel = logging.StreamHandler()
    logger.addHandler(sentinel)

    install_null_handler(namespace)

    assert logger.handlers == [sentinel]


def test_install_null_handler_is_idempotent(
    isolate_logging_state: Callable[[str], logging.Logger],
) -> None:
    """Test `install_null_handler` is idempotent."""
    namespace = "my_namespace"
    logger = isolate_logging_state(namespace)

    install_null_handler(namespace)
    install_null_handler(namespace)

    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.NullHandler)


def test_configure_logging_adds_single_console_handler_and_sets_levels(
    isolate_logging_state: Callable[[str], logging.Logger],
) -> None:
    """Test `configure_logging` adds a console handler with correct level."""
    namespace = "mycompiler"
    logger = isolate_logging_state(namespace)

    out = configure_logging(namespace, console_level=logging.WARNING)

    assert out is logger
    stream_handlers = _get_non_file_stream_handlers(logger)
    assert len(stream_handlers) == 1
    assert stream_handlers[0].level == logging.WARNING
    assert _get_file_handlers(logger) == []


def test_configure_logging_is_idempotent_for_console_handler(
    isolate_logging_state: Callable[[str], logging.Logger],
) -> None:
    """Test `configure_logging` does not duplicate handlers on multiple calls."""
    namespace = "mycompiler"
    logger = isolate_logging_state(namespace)

    _ = configure_logging(namespace, console_level=logging.INFO)
    _ = configure_logging(namespace, console_level=logging.ERROR)

    stream_handlers = _get_non_file_stream_handlers(logger)
    assert len(stream_handlers) == 1
    assert stream_handlers[0].level == logging.ERROR


def test_configure_logging_adds_file_handler_and_is_idempotent_by_resolved_path(
    isolate_logging_state: Callable[[str], logging.Logger], tmp_path: Path
) -> None:
    """Test `configure_logging` adds file handler and does not duplicate."""
    namespace = "mycompiler"
    logger = isolate_logging_state(namespace)

    log_path = tmp_path / "build.log"
    format1 = logging.Formatter("A:%(message)s")
    format2 = logging.Formatter("B:%(message)s")

    configure_logging(
        namespace,
        file_path=log_path,
        file_level=logging.DEBUG,
        formatter=format1,
    )
    configure_logging(
        namespace,
        file_path=str(log_path),
        file_level=logging.ERROR,
        formatter=format2,
    )

    file_handlers = _get_file_handlers(logger)
    assert len(file_handlers) == 1
    assert file_handlers[0].level == logging.ERROR
    assert file_handlers[0].formatter is format2
    assert Path(file_handlers[0].baseFilename).resolve() == log_path.resolve()


def test_configure_logging_does_not_duplicate_if_existing_streamhandler_present(
    isolate_logging_state: Callable[[str], logging.Logger],
) -> None:
    """Test `configure_logging` updates existing StreamHandler."""
    namespace = "mycompiler"
    logger = isolate_logging_state(namespace)

    existing = logging.StreamHandler()
    existing.setLevel(logging.CRITICAL)
    logger.addHandler(existing)

    configure_logging(namespace=namespace, console_level=logging.INFO)

    stream_handlers = _get_non_file_stream_handlers(logger)
    assert len(stream_handlers) == 1
    assert stream_handlers[0] is existing
    assert stream_handlers[0].level == logging.INFO


def test_add_file_handler_adds_and_is_idempotent(
    isolate_logging_state: Callable[[str], logging.Logger], tmp_path: Path
) -> None:
    """Test `add_file_handler` adds FileHandler and is idempotent by absolute path."""
    namespace = "mycompiler"
    logger = isolate_logging_state(namespace)

    log_path = tmp_path / "x.log"
    add_file_handler(logger, log_path, level=logging.WARNING)

    file_handlers = _get_file_handlers(logger)
    assert len(file_handlers) == 1
    assert file_handlers[0].level == logging.WARNING
    assert Path(file_handlers[0].baseFilename).resolve() == log_path.resolve()

    add_file_handler(logger, str(log_path), level=logging.ERROR)

    file_handlers2 = _get_file_handlers(logger)
    assert len(file_handlers2) == 1
    assert file_handlers2[0].level == logging.ERROR


def test_add_file_handler_uses_explicit_formatter_when_provided(
    isolate_logging_state: Callable[[str], logging.Logger], tmp_path: Path
) -> None:
    """Test that `add_file_handler` uses the provided formatter."""
    logger = isolate_logging_state("mycompiler")

    formatter = logging.Formatter("X:%(message)s")
    log_path = tmp_path / "x.log"

    add_file_handler(logger, log_path, formatter=formatter)
    file_handler = _get_file_handlers(logger)[0]
    assert file_handler.formatter is formatter


def test_add_file_handler_infers_formatter_from_existing_handler(
    isolate_logging_state: Callable[[str], logging.Logger], tmp_path: Path
) -> None:
    """Test `add_file_handler` uses existing handler's formatter if none provided."""
    logger = isolate_logging_state("mycompiler")

    formatter = logging.Formatter("INFER:%(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = tmp_path / "x.log"
    add_file_handler(logger, log_path, formatter=None)

    file_handler = _get_file_handlers(logger)[0]
    assert file_handler.formatter is formatter


def test_configure_logging_propagate_flag(
    isolate_logging_state: Callable[[str], logging.Logger],
) -> None:
    """Test `configure_logging` sets the propagate flag correctly."""
    namespace = "mycompiler"
    logger = isolate_logging_state(namespace)

    configure_logging(namespace, propagate=True)
    assert logger.propagate is True

    configure_logging(namespace, propagate=False)
    assert logger.propagate is False


def test_stream_handler_detection_does_not_count_filehandler_as_console(
    isolate_logging_state: Callable[[str], logging.Logger], tmp_path: Path
) -> None:
    """Test `configure_logging` detects existing handler when a handler exists."""
    namespace = "mycompiler"
    logger = isolate_logging_state(namespace)

    log_path = tmp_path / "only_file.log"
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    logger.addHandler(file_handler)

    configure_logging(namespace=namespace, console_level=logging.INFO)

    assert len(_get_file_handlers(logger)) == 1
    assert len(_get_non_file_stream_handlers(logger)) == 1
