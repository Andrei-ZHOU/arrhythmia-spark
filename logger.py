import logging
import sys

APP_LOGGER_NAME = "TPG Test"


class ContextFilter(logging.Filter):
    """Filter to remove TPG Test from the logger name"""

    def filter(self, record):
        split_name = record.name.split(".", 1)
        if split_name[0] == "TPG Test":
            if len(split_name) > 1:
                record.name = split_name[1]
        return record


def setup_logger(logger_name=APP_LOGGER_NAME, is_debug=True, file_name=None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG if is_debug else logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.addFilter(ContextFilter())
    logger.handlers.clear()
    logger.addHandler(sh)

    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def get_logger(module_name):
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)
