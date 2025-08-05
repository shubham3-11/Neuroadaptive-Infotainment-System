"""
Logging utilities for KSS prediction pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import datetime


def setup_logger(
    level: str = "INFO", output_dir: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger for the KSS prediction pipeline.

    Parameters:
    -----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    output_dir : str, optional
        Directory to save log files

    Returns:
    --------
    logger : logging.Logger
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("kss_pipeline")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if output directory specified)
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_path / f"kss_pipeline_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Parameters:
    -----------
    name : str
        Module name

    Returns:
    --------
    logger : logging.Logger
        Logger for the module
    """
    return logging.getLogger(f"kss_pipeline.{name}")
