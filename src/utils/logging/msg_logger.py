import sys
from pathlib import Path

from loguru import logger


def setup_logging(exp_dir: str, log_filename: str) -> None:
    """
    configure loguru for the project with color output to console and file output.

    Args:
        exp_dir: experiment directory where log file will be saved
        log_filename: log file name
    """
    # Get the log directory from Hydra
    log_dir = Path(exp_dir)
    log_file = log_dir / f"{log_filename}.log"

    # Remove any existing handlers
    logger.remove()

    # Add console handler with colors (enabled by default)
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # Add file handler
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
    )

    logger.info(f"Logging initialized. Log file: {log_file}")