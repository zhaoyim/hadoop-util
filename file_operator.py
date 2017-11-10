# encoding=utf8
"""
Created on  : 11/10/17
@author: chenxf@chinaskycloud.com
"""
import json
import os
import csv
from logger_util import get_logger

logger = get_logger()


class FileOperator(object):

    @staticmethod
    def write_to_csv(headers, data, file, header=True, model="w"):
        """
        write data to csv
        :param headers: the csv headers
        :param data: type list
        :return:
        """
        with open(file, model) as f:
            try:
                f_csv = csv.DictWriter(f, headers)
                if header:
                    f_csv.writeheader()
                f_csv.writerows(data)
            except Exception as error:
                raise error
            else:
                logger.info("write data to {0} success".format(file))

    @staticmethod
    def write_to_json(data, file, model="w"):
        with open(file, model) as f:
            json.dump(data, f)

    @staticmethod
    def file_exits(file):
        if os.path.isfile(file):
            return True
        return False