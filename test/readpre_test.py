#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/17/17 2:21 PM
# @Author  : chenxf
from os import path
import numpy as np
file = path.join("../output/pre.txt")
for data in open(file):
    data = np.array(data.split(','))
    print(data)
    print("---------")