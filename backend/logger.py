"""
logger.py
---------
Structured logging for the HPSM pipeline.
Logs to both console and rotating log files.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler

# Import config paths (but avoid circular imports by using defaults)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

# Defaults (overridden if config is available)
_LOG_LEVEL = "INFO"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_loggers = {}


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Get or create a named logger with console + file handlers.

    Parameters
    ----------
    name : str
        Logger name (typically __name__).
    log_file : str or None
        Log file name. Defaults to 'pipeline.log'.

    Returns
    -------
    logging.Logger
    """
    if name in _loggers:
        return _loggers[name]

    if log_file is None:
        log_file = "pipeline.log"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    logger.propagate = False

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
               for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
        logger.addHandler(ch)

    # File handler (rotating, max 5 MB, keep 3 backups)
    log_path = os.path.join(_LOG_DIR, log_file)
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
        logger.addHandler(fh)

    _loggers[name] = logger
    return logger


def get_training_logger() -> logging.Logger:
    """Logger specifically for CNN training, writes to training.log."""
    return get_logger("hpsm.training", log_file="training.log")


def get_evaluation_logger() -> logging.Logger:
    """Logger specifically for evaluation, writes to evaluation.log."""
    return get_logger("hpsm.evaluation", log_file="evaluation.log")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log = get_logger("test")
    log.info("Logger test message — INFO")
    log.debug("Logger test message — DEBUG")
    log.warning("Logger test message — WARNING")
    print(f"Log directory: {_LOG_DIR}")
