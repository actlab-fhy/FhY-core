"""Core logging utilities.

Expected behavior
-----------------
1) When used in a library/component package
   - Importing your package must not configure global logging.
   - This module does not attach output handlers automatically.
   - A NullHandler is installed on the project namespace logger to avoid
     "No handler could be found" warnings in embedding applications.

   Component code should simply acquire a logger and emit messages:

       >>> from fhy_core import get_logger
       >>> log = get_logger(__name__)
       >>> log.debug("Lowering AST node")

   (Using `logging.getLogger(__name__)` directly is also fine; `get_logger()`
   is provided for convenience and for consistent naming.)

2) When used for CLI / full pipeline tools with explicit logging configuration
   - The *application* (your stitched "full compiler" package or a third-party
     compiler that embeds these components) is responsible for configuring
     logging once, typically in the CLI entrypoint.
   - `configure_logging()` installs at most one console handler and optional
     file handler on the chosen project/package namespace logger, and disables
     propagation to avoid duplicate output when root logging is configured.

   Example (CLI entrypoint):

       >>> from fhy_core import configure_logging
       >>> configure_logging(
       ...     namespace="mycompiler",
       ...     console_level=logging.INFO,
       ...     file_path="build.log",
       ...     file_level=logging.DEBUG,
       ... )

3) Handler-driven filtering
   - Namespace loggers are set to DEBUG so handlers control what is emitted.
   - Typical compiler defaults are:
       - Console: INFO (or WARNING in quiet mode)
       - File:    DEBUG (full traces)

"""

__all__ = [
    "add_file_handler",
    "configure_logging",
    "get_logger",
    "install_null_handler",
    "reset",
]

import logging
from pathlib import Path

_DEFAULT_FORMAT = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s():%(lineno)d | %(message)s"
)
_NULLHANDLER_ATTACHED: set[str] = set()


def reset() -> None:
    """Reset module state."""
    _NULLHANDLER_ATTACHED.clear()


def get_logger(name: str) -> logging.Logger:
    """Return a logger for internal use.

    This function is intentionally non-configuring:
    - It does not set levels.
    - It does not add StreamHandlers/FileHandlers.

    Components should call `get_logger(__name__)` (or `logging.getLogger(__name__)`)
    and rely on the embedding application / CLI to call `configure_logging()`.
    """
    return logging.getLogger(name)


def install_null_handler(namespace: str) -> None:
    """Install a NullHandler on the namespace logger to keep library imports quiet."""
    if namespace in _NULLHANDLER_ATTACHED:
        return

    log = logging.getLogger(namespace)
    if not log.handlers:
        log.addHandler(logging.NullHandler())

    _NULLHANDLER_ATTACHED.add(namespace)


def configure_logging(  # noqa: C901
    namespace: str,
    *,
    console_level: int = logging.INFO,
    file_path: str | Path | None = None,
    file_level: int = logging.DEBUG,
    formatter: logging.Formatter | None = None,
    propagate: bool = False,
) -> logging.Logger:
    """Configure logging for a compiler application/CLI.

    This should be called once from the CLI entrypoint of the full end-to-end
    compiler, or by any third-party compiler that embeds FhY components.
    a
    Args:
        namespace: The logger namespace to configure (e.g., "moga").
        console_level: Log level for the console (StreamHandler).
        file_path: If provided, path to a log file to create (FileHandler).
        file_level: Log level for the file handler.
        formatter: Log message formatter to use for all handlers.
        propagate: If True, allow log messages to propagate to ancestor loggers.

    Returns:
        The configured namespace logger.

    """
    log = logging.getLogger(namespace)

    log.setLevel(logging.DEBUG)
    log.propagate = propagate

    if formatter is None:
        formatter = _DEFAULT_FORMAT

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in log.handlers
    ):
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    else:
        for h in log.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler
            ):
                h.setLevel(console_level)
                h.setFormatter(formatter)

    if file_path is not None:
        file_path = Path(file_path).expanduser().resolve()
        existing = False
        for h in log.handlers:
            if isinstance(h, logging.FileHandler):
                try:
                    existing_path = Path(h.baseFilename).resolve()
                except (OSError, RuntimeError, ValueError) as exc:
                    log.debug(
                        "Failed to resolve existing log file handler path %r: %s",
                        getattr(h, "baseFilename", None),
                        exc,
                    )
                if existing_path == file_path:
                    existing = True
                    h.setLevel(file_level)
                    h.setFormatter(formatter)
                    break

        if not existing:
            fh = logging.FileHandler(str(file_path), mode="a", encoding="utf-8")
            fh.setLevel(file_level)
            fh.setFormatter(formatter)
            log.addHandler(fh)

    return log


def add_file_handler(
    log: logging.Logger,
    path: str | Path,
    *,
    level: int = logging.DEBUG,
    formatter: logging.Formatter | None = None,
) -> None:
    """Append a FileHandler to an existing logger (idempotent by absolute path).

    Prefer calling `configure_logging(..., file_path=...)` from the CLI, but this
    helper is useful when a library embedding wants to add a file log after the
    fact.

    Args:
        log: Logger instance to modify.
        path: Log file path.
        level: File handler log level.
        formatter: If None, uses the first non-None formatter found on existing
                   handlers, otherwise falls back to the module default.

    """
    file_path = Path(path).expanduser().resolve()

    for h in log.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if Path(h.baseFilename).resolve() == file_path:
                    h.setLevel(level)
                    if formatter is not None:
                        h.setFormatter(formatter)
                    return
            except (OSError, ValueError) as exc:
                logging.getLogger(__name__).debug(
                    "Failed to resolve file handler path %r: %s",
                    getattr(h, "baseFilename", None),
                    exc,
                )

    if formatter is None:
        for h in log.handlers:
            if getattr(h, "formatter", None) is not None:
                formatter = h.formatter
                break
        if formatter is None:
            formatter = _DEFAULT_FORMAT

    fh = logging.FileHandler(str(file_path), mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    log.addHandler(fh)
