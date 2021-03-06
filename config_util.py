# encoding=utf8
"""
Created on  : 11/3/17
@author: chenxf@chinaskycloud.com
"""
import os
try:
    import ConfigParser as configparser
except Exception:
    import configparser as configparser


class ConfigUtil(object):
    """read conf from file"""

    def __init__(self, config_file_name):
        self.config_file_name = config_file_name
        self.cf = ConfigUtil.load_config(self.config_file_name)

    @staticmethod
    def load_config(config_file_name):
        if not os.path.exists(config_file_name):
            raise IOError("no such file {0}".format(config_file_name))
        cf = configparser.ConfigParser()
        cf.read(config_file_name)
        return cf

    def get_options(self, option_name, key=None):
        """
        load hadoop_url from properties.conf
        if key is None return get
        """
        if key:
            try:
                value = self.cf.get(option_name, key).replace(" ", "").replace("\n", "")
            except configparser.NoOptionError as error:
                raise KeyError("load url error", error)
            except configparser.NoSectionError as error:
                raise KeyError("load url error", error)
            return value
        else:
            try:
                value = self.cf.items(option_name)
            except configparser.NoOptionError as error:
                raise KeyError("load url error", error)
            except configparser.NoSectionError as error:
                raise KeyError("load url error", error)
            return value
