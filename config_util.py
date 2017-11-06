# encoding=utf8
"""
Created on  : 11/3/17
@author: chenxf@chinaskycloud.com
"""
import ConfigParser


class ConfigUtil(object):
    """read conf from file"""

    def __init__(self, config_file_name):
        self.config_file_name = config_file_name
        self.cf = ConfigUtil.load_config(self.config_file_name)

    @staticmethod
    def load_config(config_file_name):
        cf = ConfigParser.ConfigParser()
        try:
            cf.read(config_file_name)
        except ConfigParser.NoSectionError as error:
            raise IOError("read properties.conf error", error)
        else:
            return cf

    def get_options(self, option_name, key=None):
        """
        load hadoop_url from properties.conf
        if key is None return get
        """
        if key:
            try:
                value = self.cf.get(option_name, key)
            except ConfigParser.NoOptionError as error:
                raise KeyError("load url error", error)
            return value
        else:
            try:
                value = self.cf.items(option_name)
            except ConfigParser.NoOptionError as error:
                raise KeyError("load url error", error)
            return value
