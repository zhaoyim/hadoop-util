#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/21/17 6:53 PM
# @Author  : chenxf
from os import path
import numpy as np
file = path.join("../model_out/prediction.csv")
# spark is ../output/spark_pre.csv
for data in open(file):
    data = np.array(data.replace("\n", "").split(','))
    time_serices = data[0]
    print("----time_serices----")
    print(time_serices)
    print(data[1])
    mem_cpu = np.array(data[2][1:-1].split(), dtype=np.float64)
    print("----mem----")
    print(mem_cpu[0])
    print("------cpu----")
    print(mem_cpu[1])