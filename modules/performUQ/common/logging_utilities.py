from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import time
from functools import partial
from logging.handlers import RotatingFileHandler
from pathlib import Path

_INDENT_LEVEL = threading.local()
_INDENT_LEVEL.value = 0

USE_ASCII_SYMBOLS = True

_UNICODE_SYMBOLS_BY_LEVEL = {
    logging.INFO: ['→', '↳', '↪', '⇨', '⇾'],
    logging.DEBUG: ['→', '↳', '↪', '⇨', '⇾'],
}
_UNICODE_SYMBOLS_DEFAULT = ['-']

# _ASCII_SYMBOLS_BY_LEVEL = {
#     logging.INFO: ['->', '-->', '--->', '---->', '     >'],
#     logging.DEBUG: ['->', '-->', '--->', '---->', '     >'],
# }
# _ASCII_SYMBOLS_BY_LEVEL = {
#     logging.INFO: ['>'],
#     logging.DEBUG: ['>'],
# }
_ASCII_SYMBOLS_BY_LEVEL = {
    logging.INFO: [''],
    logging.DEBUG: [''],
}
_ASCII_SYMBOLS_DEFAULT = ['-']

_BASE_INDENT = '  '


def _get_indent_level():
    return getattr(_INDENT_LEVEL, 'value', 0)


def _set_indent_level(value):
    _INDENT_LEVEL.value = value


def _increment_indent_level():
    _set_indent_level(_get_indent_level() + 1)


def _decrement_indent_level():
    _set_indent_level(max(0, _get_indent_level() - 1))


def _get_indent_string():
    return _BASE_INDENT * _get_indent_level()


def _get_log_symbol_for_indent(level: int) -> str:
    indent = _get_indent_level()
    if USE_ASCII_SYMBOLS:
        symbols = _ASCII_SYMBOLS_BY_LEVEL.get(level, _ASCII_SYMBOLS_DEFAULT)
    else:
        symbols = _UNICODE_SYMBOLS_BY_LEVEL.get(level, _UNICODE_SYMBOLS_DEFAULT)
    return symbols[indent % len(symbols)]


class LoggerAutoFlusher:
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
        self.flush()

    def flush(self):
        for handler in self._logger.handlers:
            try:
                handler.flush()
                if hasattr(handler, 'stream') and hasattr(handler.stream, 'fileno'):
                    os.fsync(handler.stream.fileno())
            except Exception:
                pass


def setup_logger(
    log_filename='logFile.txt',
    prefix='',
    style='compact',
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
):
    logger = logging.getLogger(prefix or __name__)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
            if style == 'full'
            else '%(message)s'
        )

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(console_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = RotatingFileHandler(
            log_filename, maxBytes=max_bytes, backupCount=backup_count
        )
        fh.setLevel(file_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def flush_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()
        if hasattr(handler, 'stream') and hasattr(handler.stream, 'fileno'):
            with contextlib.suppress(OSError):
                os.fsync(handler.stream.fileno())


class BoundLogger:
    def __init__(self, logger):
        self.logger = logger

    def __call__(self, msg: str) -> None:
        indent = _get_indent_string()
        symbol = _get_log_symbol_for_indent(self.logger.getEffectiveLevel())
        self.logger.info(f'{indent}{symbol} {msg}')


def make_log_info(logger: logging.Logger) -> BoundLogger:
    return BoundLogger(logger)


_HIGHLIGHT_SYMBOLS = {
    True: '=',  # fallback for highlight=True
    'default': '=',
    'minor': '-',
    'major': '#',
    'submajor': '*',
    'success': '+',
    'error': '!',
}


def _log_highlight_block(
    logger: logging.Logger,
    msg: str,
    style: str | bool = 'default',
    width: int = 60,
) -> None:
    """
    Log a multi-line banner around a message with configurable style.

    Example:
      > ============================================================
      >                         Iteration 2
      > ============================================================
    """
    indent = _get_indent_string()
    symbol = _HIGHLIGHT_SYMBOLS.get(style, '=')  # fallback to '='
    border = symbol * width
    centered_msg = msg.center(width)
    log_symbol = _get_log_symbol_for_indent(logger.getEffectiveLevel())

    logger.info(f'{indent}{log_symbol} {border}')
    logger.info(f'{indent}{log_symbol} {centered_msg}')
    logger.info(f'{indent}{log_symbol} {border}')


class LogStepContext:
    def __init__(
        self,
        msg: str,
        logger: logging.Logger,
        *,
        min_duration: float = 5.0,
        highlight: bool | str = False,
    ):
        self.msg = msg
        self.logger = logger
        self.min_duration = min_duration
        self.highlight = highlight

    def __enter__(self):
        if self.highlight:
            _log_highlight_block(self.logger, self.msg, style=self.highlight)
        else:
            indent = _get_indent_string()
            symbol = _get_log_symbol_for_indent(self.logger.getEffectiveLevel())
            self.logger.info(f'{indent}{symbol} {self.msg}')

        _increment_indent_level()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if duration >= self.min_duration:
            indent = _get_indent_string()
            symbol = _get_log_symbol_for_indent(self.logger.getEffectiveLevel())
            duration_string = _format_duration(duration)
            self.logger.info(f'{indent}{symbol} Done in {duration_string}.')
        _decrement_indent_level()


def make_logger_context(logger: logging.Logger):
    return partial(LogStepContext, logger=logger)


def decorate_methods_with_log_step(
    instance, method_names, logger, warn_if_longer_than=None
):
    for method_name in method_names:
        original_method = getattr(instance, method_name)

        def wrapper(method=original_method):
            def timed(*args, **kwargs):
                msg = _prettify_method_name(method.__name__)
                with LogStepContext(msg, logger):
                    return method(*args, **kwargs)

            return timed

        setattr(instance, method_name, wrapper())


def log_exception(logger, ex: Exception, message=''):
    logger.error(f'{message}\nException: {ex}', exc_info=True)


def _prettify_method_name(name: str) -> str:
    name = name.lstrip('_')
    if name.startswith('step_'):
        parts = name.split('_')
        if parts[1].isdigit():
            step = parts[1]
            label = ' '.join(word.capitalize() for word in parts[2:])
            return f'Step {step}: {label}'
    return name.replace('_', ' ').capitalize()


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
