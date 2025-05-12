"""
Logging utilities.

Utilities for configuring and ensuring a logger with console and file handlers,
including support for rotating log files.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import sys
import threading
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

_DEFAULT_LOGGER = None


def set_default_logger(logger: logging.Logger) -> None:
    global _DEFAULT_LOGGER
    _DEFAULT_LOGGER = logger


def get_default_logger() -> logging.Logger:
    global _DEFAULT_LOGGER
    if _DEFAULT_LOGGER is not None:
        return _DEFAULT_LOGGER
    msg = 'No default logger has been set. Please call `ensure_logger()` and `set_default_logger()` first.'
    raise RuntimeError(msg)


def flush_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()
        if hasattr(handler, 'stream') and hasattr(handler.stream, 'fileno'):
            with contextlib.suppress(OSError):
                os.fsync(handler.stream.fileno())


class LoggerAutoFlusher:
    """Automatically flush the logger's handlers at regular intervals."""

    def __init__(self, logger, interval=10):
        self._logger = logger
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)

    def _flush_loop(self):
        while not self._stop_event.is_set():
            self.flush()
            time.sleep(self._interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        self.flush()  # <--- FORCE FINAL FLUSH when stopping

    def flush(self):
        """Flush all handlers manually."""
        for handler in self._logger.handlers:
            try:
                handler.flush()
                if hasattr(handler, 'stream') and hasattr(handler.stream, 'fileno'):
                    import os

                    os.fsync(handler.stream.fileno())
            except Exception:  # noqa: BLE001, PERF203, S110
                pass  # ignore any errors silently


def log_exception(
    logger: logging.Logger, ex: Exception, message: str = 'Unhandled Exception'
) -> None:
    """
    Log an exception with full traceback to the provided logger.

    Parameters
    ----------
    logger : logging.Logger
        Logger to write the exception details to.
    ex : Exception
        Exception instance.
    message : str, optional
        Custom message to add before traceback.
    """
    tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
    logger.error(f'{message}:\n{tb_str}')


def ensure_logger(
    *,
    logger: logging.Logger | None = None,
    log_filename: str | Path = 'logFile.log',
    console_level: int = logging.INFO,
    file_level: int = logging.INFO,
    prefix: str | None = None,
    format_str: str | None = None,
    style: str = 'compact',
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    use_prefix: bool = True,
    justify: int = 0,
) -> logging.Logger:
    """
    Ensure a usable logger with console and file handlers, supporting different styles.

    Parameters
    ----------
    style : str, optional
        'full' for detailed logging format, 'compact' for minimal format. Default is 'compact'.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not logger.handlers:
        # Format construction
        if format_str:
            base_format = format_str
        elif style == 'compact':
            base_format = '%(message)s'
        else:  # 'full'
            base_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

        if use_prefix and prefix:
            padded_prefix = prefix.ljust(justify)
            base_format = f'{padded_prefix} {base_format}'

        formatter = logging.Formatter(base_format)

        # Console handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(console_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File handler
        log_filename = Path(log_filename)

        # Do NOT accept a directory — must be a file path
        if log_filename.is_dir():
            msg = f'Expected a file path, but got a directory: {log_filename}'
            raise ValueError(msg)

        log_filename = log_filename.resolve()
        log_filename.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Logger base level
        logger.setLevel(min(console_level, file_level))

        logger.propagate = False

        logger.debug(
            f'Logger configured with console output (level {logging.getLevelName(console_level)}) '
            f'and rotating file {log_filename} (max {max_bytes} bytes, backups {backup_count}) '
            f'with level {logging.getLevelName(file_level)}.'
        )

    return logger


def get_module_logger(
    *,
    prefix: str | None = None,
    log_filename: str | Path | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    style: str = 'full',
    use_prefix: bool = True,
    justify: int = 0,
) -> logging.Logger:
    """
    Quickly get a logger for a module with standardized setup.

    If no log_filename is provided, it is set based on module name.

    Parameters
    ----------
    style : str, optional
        'full' for detailed format, 'compact' for minimal format.

    Returns
    -------
    logging.Logger
    """
    try:
        import inspect

        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame else None
        module = inspect.getmodule(caller_frame)
        if module:
            module_name = module.__name__
            if module_name == '__main__':
                module_base = 'MAIN_SCRIPT'
            else:
                module_base = module_name.split('.')[-1].upper()
        else:
            module_base = 'UNKNOWN_MODULE'
    except Exception:  # noqa: BLE001
        module_base = 'UNKNOWN_MODULE'

    if prefix is None:
        prefix = module_base

    if log_filename is None:
        log_filename = f'logFile{module_base}.txt'

    return ensure_logger(
        logger=None,
        log_filename=log_filename,
        console_level=console_level,
        file_level=file_level,
        prefix=prefix,
        style=style,
        use_prefix=use_prefix,
        justify=justify,
    )


# =========================================================
# Step Logging Helpers with Indentation
# =========================================================
# Thread-local context to track nesting depth
_log_context = threading.local()

# Use only ASCII-safe symbols to ensure compatibility across terminals and Qt GUIs
USE_ASCII_SYMBOLS = True

# ASCII-safe fallback symbols by indent level
_ASCII_SYMBOLS_DEFAULT = ['->', '-->', '--->', '==>', '===>']
_ASCII_SYMBOLS_BY_LEVEL = {
    logging.DEBUG: ['.', '..', '...', '....', '.....'],
    logging.INFO: _ASCII_SYMBOLS_DEFAULT,
    logging.WARNING: ['[!]', '[!]', '[!]', '[!]', '[!]'],
    logging.ERROR: ['[X]', '[X]', '[X]', '[X]', '[X]'],
}

# Unicode version (optional, retained for reference or future use)
_UNICODE_SYMBOLS_DEFAULT = ['→', '↳', '⤷', '⇢', '⇨']
_UNICODE_SYMBOLS_BY_LEVEL = {
    logging.DEBUG: ['•', '◦', '▪', '▫', '‣'],
    logging.INFO: _UNICODE_SYMBOLS_DEFAULT,
    logging.WARNING: ['⚠', '⚠', '⚠', '⚠', '⚠'],
    logging.ERROR: ['✖', '✖', '✖', '✖', '✖'],
}


def _get_indent_level() -> int:
    return getattr(_log_context, 'indent_level', 0)


def _increase_indent():
    _log_context.indent_level = _get_indent_level() + 1


def _decrease_indent():
    _log_context.indent_level = max(0, _get_indent_level() - 1)


def _get_indent_string() -> str:
    return '  ' * _get_indent_level()


def _get_log_symbol_for_indent(level: int) -> str:
    indent = _get_indent_level()
    if USE_ASCII_SYMBOLS:
        symbols = _ASCII_SYMBOLS_BY_LEVEL.get(level, _ASCII_SYMBOLS_DEFAULT)
    else:
        symbols = _UNICODE_SYMBOLS_BY_LEVEL.get(level, _UNICODE_SYMBOLS_DEFAULT)
    return symbols[indent] if indent < len(symbols) else symbols[-1]


@contextlib.contextmanager
def log_step_auto(
    message: str,
    logger: logging.Logger | None = None,
    warn_if_longer_than: float | None = 300.0,
):
    """
    Log a step with duration, using DEBUG if active, else INFO.

    Parameters
    ----------
    message : str
        Message describing the step.
    logger : logging.Logger, optional
        Logger to use; defaults to the default logger.
    warn_if_longer_than : float or None
        Threshold in seconds. Logs a warning if exceeded.
    """
    if logger is None:
        logger = get_default_logger()

    level = logger.getEffectiveLevel()
    use_debug = level <= logging.DEBUG
    log_fn = logger.debug if use_debug else logger.info

    indent = _get_indent_string()
    symbol = _get_log_symbol_for_indent(level)
    log_fn(f'{indent}{symbol} {message}')

    start = time.time()
    _increase_indent()
    try:
        yield
    finally:
        _decrease_indent()
        duration = time.time() - start
        duration_str = _format_duration(duration)

        if warn_if_longer_than is not None and duration > warn_if_longer_than:
            logger.warning(
                f'{indent}{symbol} Took {duration_str} (exceeded threshold: {_format_duration(warn_if_longer_than)}).'
            )
        else:
            log_fn(f'{indent}{symbol} Done in {duration_str}.')


def log_step_auto_decorator(
    message: str,
    logger: logging.Logger | None = None,
    warn_if_longer_than: float | None = 300.0,
):
    """
    Log function steps with timing and optional warnings.

    Parameters
    ----------
    message : str
        Description of the operation.
    logger : logging.Logger, optional
        Logger to use; defaults to get_default_logger().
    warn_if_longer_than : float or None, optional
        Threshold in seconds to trigger a warning.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            used_logger = logger or get_default_logger()
            with log_step_auto(
                message, used_logger, warn_if_longer_than=warn_if_longer_than
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def _format_duration(seconds: float) -> str:
    """
    Format elapsed time into a human-readable string.

    Parameters
    ----------
    seconds : float
        Time duration in seconds.

    Returns
    -------
    str
        Formatted string like '12 sec', '2.05 min', or '1.5 hr'.
    """
    if seconds < 1:
        return f'{seconds:.3f} seconds'
    if seconds < 60:
        if seconds < 10:
            return f'{seconds:.2f} seconds'
        return f'{seconds:.1f} seconds'
    if seconds < 3600:
        minutes = seconds / 60
        if minutes < 10:
            return f'{minutes:.2f} minutes'
        return f'{minutes:.1f} minutes'
    hours = seconds / 3600
    return f'{hours:.2f} hours'


def make_logger_context(logger: logging.Logger):
    """
    Return a pre-bound version of log_step_auto that doesn't require passing the logger repeatedly.

    Usage
    -----
    log_step = make_logger_context(logger)
    with log_step("Step message"):
        ...
    """
    return lambda *args, **kwargs: log_step_auto(*args, logger=logger, **kwargs)


def make_log_step_decorator(logger: logging.Logger):
    """
    Return a pre-bound version of log_step_auto_decorator using the provided logger.

    Usage
    -----
    log_step = make_log_step_decorator(logger)

    @log_step("Running expensive operation")
    def do_work(...):
        ...
    """

    def decorator_factory(message: str, warn_if_longer_than: float | None = 300.0):
        return log_step_auto_decorator(
            message=message,
            logger=logger,
            warn_if_longer_than=warn_if_longer_than,
        )

    return decorator_factory


def decorate_methods_with_log_step(
    instance,
    method_names: list[str],
    logger=None,
    warn_if_longer_than: float | None = 300.0,
):
    """
    Decorate selected methods of an object instance using `log_step_auto_decorator`.

    Parameters
    ----------
    instance : object
        The class instance whose methods should be decorated.
    method_names : list[str]
        List of method names (as strings) to wrap.
    logger : logging.Logger, optional
        Logger to use. Defaults to get_default_logger().
    warn_if_longer_than : float or None, optional
        Threshold for logging warnings if time exceeds this.
    """
    if logger is None:
        logger = get_default_logger()

    for name in method_names:
        if hasattr(instance, name):
            method = getattr(instance, name)
            decorated = log_step_auto_decorator(
                message=f'{instance.__class__.__name__}.{name}()',
                logger=logger,
                warn_if_longer_than=warn_if_longer_than,
            )(method)
            setattr(instance, name, decorated)
        else:
            msg = f"{instance} has no method named '{name}'"
            raise AttributeError(msg)
