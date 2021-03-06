# encoding=utf8
"""
Created on  : 11/6/17
@author: chenxf@chinaskycloud.com
"""
import unittest
from config_util import ConfigUtil


class TestConfigUtil(unittest.TestCase):
    """
    test ConfigUtil's methods
    """

    def test_load_config(self):
        """
        test ConfigUtil.load_config's exception
        """
        self.assertRaises(IOError, ConfigUtil.load_config, "./conf/conf.conf")

    def test_get_options(self):
        """
        test if  ConfigUtil.get_options can resolve exception correct
        """
        config_file = "../conf/properties.conf"
        config_util = ConfigUtil(config_file)
        self.assertEqual(config_util.get_options("url", "hadoop_url"),
                         "http://10.10.236.203:8088/ws/v1/cluster/")
        self.assertRaises(KeyError,
                          config_util.get_options,"url", "hadoop_ur2l")
        self.assertRaises(KeyError,
                          config_util.get_options, "url2", "hadoop_url")
        self.assertRaises(KeyError,
                          config_util.get_options, "url2")

if __name__ == "__main__":
    unittest.main()




