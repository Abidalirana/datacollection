# workflow/utils.py
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("workflow_utils")

def log_info(message: str):
    logger.info(message)

def pretty_print(data):
    print(json.dumps(data, indent=2, default=str))
