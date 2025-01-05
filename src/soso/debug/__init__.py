'''
Debugging utilities to be used throughout the SOSO project.
'''

import logging
from functools import wraps


def debug(func):
    '''
    Executes the decorated function only if the function's module logger is in
    debug mode. If the function's module logger is in debug mode, the function
    will not be executed. Use this for generation of complex logs.
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        if logger.isEnabledFor(logging.DEBUG):
            return func(*args, **kwargs)
    return wrapper
