#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/17/17 2:21 PM
# @Author  : chenxf
from os import path
import numpy as np
file = path.join("../output/hive_pre.csv")
# spark is ../output/spark_pre.csv
for data in open(file):
    data = np.array(data.replace("\n", "").split(','))
    time_serices = data[0]
    print("----time_serices----")
    print(time_serices)
    mem_cpu = np.array(data[1][1:-1].split(), dtype=np.float64)
    print("----mem----")
    print(mem_cpu[0])
    print("------cpu----")
    print(mem_cpu[1])