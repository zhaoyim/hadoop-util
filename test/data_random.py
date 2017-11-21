#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/20/17 3:24 PM
# @Author  : chenxf
import numpy as np
from file_operator import FileOperator
memory = np.random.rand(1000, 2)
#i = 0
data = list()

for i, m in enumerate(memory):
    data.append([i, m[0], m[1]])

print(data)
FileOperator.write_list_tocsv(data, "../output/spark.csv")