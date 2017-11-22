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
    print("----mem----")
    print(data[1])
    print("------cpu----")
    print(data[2])
    print(data[3])