# encoding=utf8
"""
Created on:17-11-1 上午11:18
@author:chenxf@chinaskycloud.com
"""
import datetime
import time


class TimeUtil(object):
    """
    获取时间工具，获取当前时间和当前时间之前的周，天，小时，秒的时间
    返回13位时间戳
    """
    @staticmethod
    def __get_time_now():
        end_time = datetime.datetime.now()
        return end_time

    @staticmethod
    def __chang_time(times):
        microsecond = times.microsecond
        times = int(round(time.mktime(times.timetuple()) * 1000)) + microsecond
        return times

    @staticmethod
    def get_time_weeks(weeks=1):
        end_time = TimeUtil.__get_time_now()
        start_time = end_time - datetime.timedelta(weeks=1)
        end_time = TimeUtil.__chang_time(end_time)
        start_time = TimeUtil.__chang_time(start_time)
        return start_time, end_time

    @staticmethod
    def get_time_days(days=1):
        end_time = TimeUtil.__get_time_now()
        start_time = end_time - datetime.timedelta(days=days)

        end_time = TimeUtil.__chang_time(end_time)
        start_time = TimeUtil.__chang_time(start_time)
        return start_time, end_time

    @staticmethod
    def get_time_hours(hours=1):
        end_time = TimeUtil.__get_time_now()
        start_time = end_time - datetime.timedelta(hours=1)
        end_time = TimeUtil.__chang_time(end_time)
        start_time = TimeUtil.__chang_time(start_time)
        return start_time, end_time

    @staticmethod
    def get_time_minutes(minutes=5):
        end_time = TimeUtil.__get_time_now()
        start_time = end_time - datetime.timedelta(minutes=minutes)

        end_time = TimeUtil.__chang_time(end_time)
        start_time = TimeUtil.__chang_time(start_time)
        return start_time, end_time

    @staticmethod
    def get_time_seconds(seconds=10):
        end_time = TimeUtil.__get_time_now()
        start_time = end_time - datetime.timedelta(seconds=seconds)
        end_time = TimeUtil.__chang_time(end_time)
        start_time = TimeUtil.__chang_time(start_time)
        return start_time, end_time

    @staticmethod
    def change_to_time(time_num):
        time_stamp = float(time_num / 1000)
        time_array = time.localtime(time_stamp)
        style_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
        return style_time
