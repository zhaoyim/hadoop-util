#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/20/17 3:24 PM
# @Author  : chenxf
import numpy as np
import os
from file_operator import FileOperator
from train_lstm import train
from project_dir import project_dir


memory = np.random.rand(500, 2)
data = list()


FileOperator.makesure_file_exits("../model_out")


train_file = os.path.join(project_dir, "model_input/spark.csv")
pre_file = os.path.join(project_dir, "model_out/pre.csv")
model_dir = os.path.join(project_dir, "model/")


train(queue_name='spark', csv_file=train_file,
      pre_file=pre_file,model_dir=model_dir, train_step=1000, predict_step=50)

