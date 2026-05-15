from functools import cache
import logging
from logging import Logger
from typing import Any, Dict

import colorlog


__version__ = '0.2.0'


#########
# TYPES #
#########

JSONDict = Dict[str, Any]


###########
# LOGGING #
###########

@cache
def get_logger(name: str) -> Logger:
    """Gets a default logger with the given name."""
    handler = colorlog.StreamHandler()
    fmt = '%(log_color)s%(levelname)s - %(name)s - %(message)s'
    log_colors = {
        'DEBUG' : 'cyan',
        'INFO' : 'black',
        'WARNING' : 'yellow',
        'ERROR' : 'red',
        'CRITICAL' : 'red,bg_white'
    }
    formatter = colorlog.ColoredFormatter(fmt, log_colors = log_colors)
    handler.setFormatter(formatter)
    logger = colorlog.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

# global logger
LOGGER = get_logger('labrat')
