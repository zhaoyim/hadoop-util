# encoding=utf8
"""
Created on  : 11/6/17
@author: chenxf@chinaskycloud.com
"""
import unittest
from time_util import TimeUtil


class TestTimeUtil(unittest.TestCase):
    def test_changetime(self):
        time_input = [1509522118051, 1509522060561, 1509518440216]
        time_result = ["2017-11-01 15:41:58", "2017-11-01 15:41:00", "2017-11-01 14:40:40"]
        for input, pre_result in zip(time_input, time_result):
            result = TimeUtil.change_to_time(input)
            self.assertAlmostEqual(pre_result, str(result))
