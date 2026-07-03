from tenacity import retry, wait_random_exponential, stop_after_attempt
import logging
import re
import json
from local_llm import infer, infer_json

RETRY_TIMES = 3
WAIT_TIME_LOWER = 1
WAIT_TIME_UPPER = 5

def llm_request(prompt):
    return infer(prompt)

def llm_request_for_json(prompt):
    return infer_json(prompt)
