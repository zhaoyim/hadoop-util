# !/usr/bin/env python
# encoding=utf8
"""
Created on  : 17-10-30 上午10:23
@author: chenxf@chinaskycloud.com
"""

import json
import os
import argparse
import threading

from config_util import ConfigUtil
from time_util import TimeUtil
import multiprocessing
try:
    from urllib2 import urlopen as urlopen
    from urllib2 import URLError as urlerror
except Exception:
    from urllib.request import urlopen as urlopen
    from urllib.error import URLError as urlerror
from logger_util import get_logger
from file_operator import FileOperator


'''
 通过hadoop API获取成功执行完成的applications, queue信息。
'''

PWD = os.getcwd()
CONFIG_FILE = PWD + "/conf/properties.conf"
FLAGS = None
logger = get_logger(refresh=True)


class HadoopUtil(object):
    """
    获取hadoop 的相关信息，主要包括队列信息，job信息
    """

    def __init__(self, sparkjob_file, commonjob_file, app_file, scheduler_file, cluster_file):
        """
        :param hadoop_url: 初始化集群url
        """

        self.confing_util = ConfigUtil(CONFIG_FILE)
        self.hadoop_url = self.confing_util.get_options("url", "hadoop_url")
        self.app_file = app_file
        self.sparkjob_file = sparkjob_file
        self.commonjob_file = commonjob_file
        self.scheduler_file = scheduler_file
        self.cluster_file = cluster_file
        self.application_url = self.confing_util.get_options("url", "application_url")
        self.job_metrics = self.confing_util.get_options("job", "job_metrices")
        self.job_url = self.confing_util.get_options("url", "job_url")

    def get_cluster_information(self):
        url = self.hadoop_url + "metrics"
        header = True
        if FileOperator.file_exits(self.cluster_file):
            header = False
        try:
            results = urlopen(url, timeout=2000).read()
            results = [json.loads(results)["clusterMetrics"]]
        except Exception as error:
            logger.error(error)
        headers = results[0].keys()
        FileOperator.write_to_csv(headers, results, self.cluster_file, header=header, model="a+")
        FileOperator.write_to_csv(headers, results, "./output/cluster2.csv", model="w")
        #FileOperator.write_to_json(results, "./output/cluster.json", model="a+")

    def get_cluster_scheduler(self):
        """
        获取hadoop 集群信息
        :param file: 输出文件保存路径
        """
        url = self.hadoop_url + "scheduler"

        try:
            results = urlopen(url, timeout=2000).read()

            results = json.loads(results)

            results = results['scheduler']['schedulerInfo']['queues']['queue']

            results_copy = results[0].copy()

            for key, value in results_copy['resourcesUsed'].items():
                results[0][key] = value
        except KeyError as error:
            logger.error("key error {0}".format(error))
        except Exception as error:
            logger.error(error)
        if FileOperator.file_exits(self.scheduler_file):
            headers = None
        else:
            headers = results[0].keys()
        FileOperator.write_to_csv(results, self.scheduler_file, headers=headers, model="a+")
        FileOperator.write_to_csv(headers, results, "./output/scheduler2.csv", model="w+")
        #FileOperator.write_to_json(results, "./output/scheduler.json", model="a+")

    @staticmethod
    def request_url(url):
        try:
            result = urlopen(url, timeout=2000).read()
        except urlerror as error:
            raise urlerror("urlopen {0} error:{1}".format(url, error.reason))
        else:
            return result

    def get_applications_information(self, query_parametes=None):
        """
        :param query_parametes: dict 过滤条件，默认为成功执行完成 默认搜索所有
          * state [deprecated] - state of the application
          * states - applications matching the given application states,
                specified as a comma-separated list.
          * finalStatus - the final status of the application -
                reported by the application itself
          * user - user name
          * queue - queue name
          * limit - total number of app objects to be returned
          * startedTimeBegin -
                applications with start time beginning with this time,
                specified in ms since epoch
          * startedTimeEnd -
                applications with start time ending with this time,
                specified in ms since epoch
          * finishedTimeBegin -
                applications with finish time beginning with this time,
                specified in ms since epoch
          * finishedTimeEnd -
                applications with finish time ending with this time,
                specified in ms since epoch
          * applicationTypes -
                applications matching the given application types,
                specified as a comma-separated list.
          * applicationTags -
                applications matching any of the given application tags,
                specified as a comma-separated list.
        :param file: 输出文件保存位置
        example:
           query_parametes = {"finalStaus": "SUCCEEDED"}
           get_job(query_parametes=query_parametes)
        """
        hadoop_rest_url = self.hadoop_url + "apps?"

        try:
            for key, value in query_parametes.items():
                hadoop_rest_url += key + "=" + str(value) + "&"
        except AttributeError:
            logger.warn("didn't get any query_parametes, so ,collect all apps")

        json_result = HadoopUtil.request_url(hadoop_rest_url)
        try:
            list_result = json.loads(json_result)['apps']['app']
            headers = list_result[0].keys()
        except KeyError as error:
            logger.error("key error {0}".format(error))
        except TypeError:
            logger.warn("dit not get any data from parameters "
                        "{0}".format(query_parametes))
        except Exception as error:
            logger.error(error)
        else:
            FileOperator.write_to_csv(list_result, self.app_file, headers=headers)
            #FileOperator.write_to_json(list_result, "./output/apps.json")
            self.get_sparkjobs_information(list_result)

    def get_sparkjobs_information(self, applications):
        """
        get each application's jobs information
        :param applications: list contains applications information
        """
        app_jobs = []
        self.job_metrics = self.job_metrics.replace("\n", "").split(',')
        for application_items in applications:
            application_id = application_items["id"]
            application_rest_url = self.application_url + application_id + "/1/jobs"
            try:
                application_jobs_list = HadoopUtil.request_url(application_rest_url)
                application_jobs_list = json.loads(application_jobs_list)
            except urlerror:
                logger.warning("this application {0} is not "
                            "a spark type application".format(application_items["id"]))
            else:
                for applications in application_jobs_list:
                    apps = {key: value for key, value in applications.items()
                            for applications in application_jobs_list
                            if key in self.job_metrics}
                    app_jobs.append(dict(apps, **application_items))
        headers = app_jobs[0].keys()
        FileOperator.write_to_json(app_jobs, "./output/sparkjob.json")
        #FileOperator.write_to_csv(app_jobs, self.sparkjob_file, headers=headers)

    def get_commonjobs_information(self):
        jobs_url = self.job_url
        result = HadoopUtil.request_url(jobs_url)
        result = json.loads(result)["jobs"]["job"]
        headers = result[0].keys()
        FileOperator.write_to_csv(result, self.commonjob_file, headers=headers)
        FileOperator.write_to_json(result, "./output/commonjob.json")
        print(type(result))


def thread_main(query_parameters):
    pools = list()
    pools.append(multiprocessing.Process(
        target=hadoop_util.get_cluster_information))
    pools.append(multiprocessing.Process(
        target=hadoop_util.get_cluster_scheduler))
    pools.append(multiprocessing.Process(
        target=hadoop_util.get_applications_information, args=(query_parameters,)))
    pools.append(multiprocessing.Process(
        target=hadoop_util.get_commonjobs_information)
    )
    for pool in pools:
        pool.start()
    t = threading.Timer(
        FLAGS.time_period, thread_main, args=(query_parameters,))
    t.start()


def main():
    query_parameters = {"state": FLAGS.state}
    sw = {
        'w': lambda: TimeUtil.get_time_weeks(FLAGS.time_interval),
        'd': lambda: TimeUtil.get_time_days(FLAGS.time_interval),
        'h': lambda: TimeUtil.get_time_hours(FLAGS.time_interval),
        'm': lambda: TimeUtil.get_time_minutes(FLAGS.time_interval),
        's': lambda: TimeUtil.get_time_seconds(FLAGS.time_interval),
    }
    if FLAGS.time_interval > 0:
        finished_time_begin, finished_time_end = sw[FLAGS.time_format]()
        query_parameters["finishedTimeBegin"] = finished_time_begin
        query_parameters["finishedTimeEnd"] = finished_time_end

    thread_main(query_parameters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--sparkjob_file",
        type=str,
        default="./output/sparkjob.csv",
        help="the output file of the spark job's information."
    )
    parser.add_argument(
        "--commonjob_file",
        type=str,
        default="./output/commonjob.csv",
        help="the output file of the common job's information."
    )
    parser.add_argument(
        "--app_file",
        type=str,
        default="./output/app.csv",
        help="the output file of the job's information."
    )
    parser.add_argument(
        "--scheduler_file",
        type=str,
        default="./output/scheduler.csv",
        help="the output file of the scheduler's information."
    )
    parser.add_argument(
        "--cluster_file",
        type=str,
        default='./output/cluster.csv',
        help="the output file of the cluster's information."
    )
    parser.add_argument(
        "--time_format",
        type=str,
        choices=['w', 'd', 'h', 'm', 's'],
        default='m',
        help="w: week, d:day, h:hour, m:minutes, s:second"
    )
    parser.add_argument(
        "--time_interval",
        type=int,
        default=10,
        help="to collector job's information which job's finished time begin "
             "before now.time_format:m , time_interval:20 means collectors "
             "job's information which finished in lasted 20 minutes, "
             "if time_interval < 0 then collecotrs all"
    )
    parser.add_argument(
        "--state",
        type=str,
        choices=["finished", "accepted", "running"],
        default="finished",
        help="the job's state"
    )
    parser.add_argument(
        "--time_period",
        type=int,
        default=600,
        help="the scripts run's time period"
    )

    FLAGS = parser.parse_args()

    hadoop_util = HadoopUtil(
        FLAGS.sparkjob_file,
        FLAGS.commonjob_file,
        FLAGS.app_file,
        FLAGS.scheduler_file,
        FLAGS.cluster_file)
    main()
