import os
import logging

LOGGER_NAME = "mlkit"
LOG_DIR = "logs"
LOG_LEVEL = logging.INFO


def create_logger() -> logging.Logger:
    """
    Configure and return the model registry logger.

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(LOG_LEVEL)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = logging.FileHandler(f"./{LOG_DIR}/{LOGGER_NAME}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = create_logger()
logger.propagate = False
