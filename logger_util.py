# encoding=utf8
"""
Created on  : 11/10/17
@author: chenxf@chinaskycloud.com
"""
import logging
from file_operator import FileOperator
import os


def get_logger(refresh=False):
    log_path = "./log"
    log_file = os.path.join(log_path, "logs")
    FileOperator.makesure_file_exits(log_path)
    if refresh:
        get_logger.logger = None
    if get_logger.logger:
        return get_logger.logger
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    file_hadler = logging.FileHandler(log_file)
    file_hadler.setLevel(logging.WARNING)
    stream_hadler = logging.StreamHandler()
    stream_hadler.setLevel(logging.WARNING)
    formater = logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s[line:%(lineno)d]: %(message)s"
        )
    file_hadler.setFormatter(formater)
    stream_hadler.setFormatter(formater)
    logger.addHandler(file_hadler)
    logger.addHandler(stream_hadler)
    get_logger.logger = logger
    return get_logger.logger

get_logger.logger = None