import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"app_{datetime.now():%Y-%m-%d}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        # logging.StreamHandler(),
    ],
)


def get_logger(name):
    return logging.getLogger(name)