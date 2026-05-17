from __future__ import annotations

import logging


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("neuro_path_ai")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
