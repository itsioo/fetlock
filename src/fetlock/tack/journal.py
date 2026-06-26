import logging
import os
from typing import Dict

_CONFIGURED: Dict[str, bool] = {}


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if _CONFIGURED.get(name):
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("FETLOCK_LOGLEVEL", "INFO").upper())
    logger.propagate = False
    _CONFIGURED[name] = True
    return logger
