import logging
from typing import Optional


def get_logger(
    rank: int, name: str = "lizardist", level: int = logging.INFO
) -> logging.Logger:
    """Get a rank-aware logger.

    Args:
        rank: The process rank
        name: Logger name
        level: Logging level

    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        f"[{rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
