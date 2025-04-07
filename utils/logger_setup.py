# utils/logger_setup.py

import logging
from colorlog import ColoredFormatter

def setup_logger(level=logging.INFO, log_file=None):
    """
    Настраивает корневой логгер для вывода цветных логов в консоль и
    (опционально) записи в файл.

    :param level: Уровень логирования (например, logging.DEBUG)
    :param log_file: Путь к файлу лога (если не None, логи будут писаться в этот файл)
    """
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    # Консольный хендлер с colorlog
    console_handler = logging.StreamHandler()
    log_format = (
        "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(blue)s%(message)s"
    )
    console_formatter = ColoredFormatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red"
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Если указан log_file, добавляем файловый хендлер
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_format = "%(asctime)s [%(levelname)s] %(message)s"
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level)
    return logger
