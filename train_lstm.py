# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
from os import path

import multiprocessing
import threading
import tensorflow as tf

from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import model as ts_model
import matplotlib
import argparse
import sys

matplotlib.use("agg")
import matplotlib.pyplot as plt

from config_util import ConfigUtil
import numpy as np
from file_operator import FileOperator


class _LSTMModel(ts_model.SequentialTimeSeriesModel):
    """A time series model-building example using an RNNCell."""

    def __init__(self, num_units, num_features, dtype=tf.float32):
        """Initialize/configure the model object.
        Note that we do not start graph building here. Rather, this object is a
        configurable factory for TensorFlow graphs which are run by an Estimator.
        Args:
          num_units: The number of units in the model's LSTMCell.
          num_features: The dimensionality of the time series (features per
            timestep).
          dtype: The floating point data type to use.
        """
        super(_LSTMModel, self).__init__(
            # Pre-register the metrics we'll be outputting (just a mean here).
            train_output_names=["mean"],
            predict_output_names=["mean"],
            num_features=num_features,
            dtype=dtype)
        self._num_units = num_units
        # Filled in by initialize_graph()
        self._lstm_cell = None
        self._lstm_cell_run = None
        self._predict_from_lstm_output = None

    def initialize_graph(self, input_statistics):
        """Save templates for components, which can then be used repeatedly.
        This method is called every time a new graph is created. It's safe to start
        adding ops to the current default graph here, but the graph should be
        constructed from scratch.
        Args:
          input_statistics: A math_utils.InputStatistics object.
        """
        super(_LSTMModel, self).initialize_graph(input_statistics=input_statistics)
        self._lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self._num_units)
        # Create templates so we don't have to worry about variable reuse.
        self._lstm_cell_run = tf.make_template(
            name_="lstm_cell",
            func_=self._lstm_cell,
            create_scope_now_=True)
        # Transforms LSTM output into mean predictions.
        self._predict_from_lstm_output = tf.make_template(
            name_="predict_from_lstm_output",
            func_=lambda inputs: tf.layers.dense(inputs=inputs, units=self.num_features),
            create_scope_now_=True)

    def get_start_state(self):
        """Return initial state for the time series model."""
        return (
            # Keeps track of the time associated with this state for error checking.
            tf.zeros([], dtype=tf.int64),
            # The previous observation or prediction.
            tf.zeros([self.num_features], dtype=self.dtype),
            # The state of the RNNCell (batch dimension removed since this parent
            # class will broadcast).
            [tf.squeeze(state_element, axis=0)
             for state_element
             in self._lstm_cell.zero_state(batch_size=1, dtype=self.dtype)])

    def _transform(self, data):
        """Normalize data based on input statistics to encourage stable training."""
        mean, variance = self._input_statistics.overall_feature_moments
        return (data - mean) / variance

    def _de_transform(self, data):
        """Transform data back to the input scale."""
        mean, variance = self._input_statistics.overall_feature_moments
        return data * variance + mean

    def _filtering_step(self, current_times, current_values, state, predictions):
        """Update model state based on observations.
        Note that we don't do much here aside from computing a loss. In this case
        it's easier to update the RNN state in _prediction_step, since that covers
        running the RNN both on observations (from this method) and our own
        predictions. This distinction can be important for probabilistic models,
        where repeatedly predicting without filtering should lead to low-confidence
        predictions.
        Args:
          current_times: A [batch size] integer Tensor.
          current_values: A [batch size, self.num_features] floating point Tensor
            with new observations.
          state: The model's state tuple.
          predictions: The output of the previous `_prediction_step`.
        Returns:
          A tuple of new state and a predictions dictionary updated to include a
          loss (note that we could also return other measures of goodness of fit,
          although only "loss" will be optimized).
        """
        state_from_time, prediction, lstm_state = state
        with tf.control_dependencies(
                [tf.assert_equal(current_times, state_from_time)]):
            transformed_values = self._scale_data(current_values)
            #transformed_values = self._transform(current_values)
            # Use mean squared error across features for the loss.
            predictions["loss"] = tf.reduce_mean(
                (prediction - transformed_values) ** 2, axis=-1)
            # Keep track of the new observation in model state. It won't be run
            # through the LSTM until the next _imputation_step.
            new_state_tuple = (current_times, transformed_values, lstm_state)
        return new_state_tuple, predictions

    def _prediction_step(self, current_times, state):
        """Advance the RNN state using a previous observation or prediction."""
        _, previous_observation_or_prediction, lstm_state = state
        lstm_output, new_lstm_state = self._lstm_cell_run(
            inputs=previous_observation_or_prediction, state=lstm_state)
        next_prediction = self._predict_from_lstm_output(lstm_output)
        new_state_tuple = (current_times, next_prediction, new_lstm_state)
        #return new_state_tuple, {"mean": self._de_transform(next_prediction)}
        return new_state_tuple, {"mean": self._scale_back_data(next_prediction)}

    def _imputation_step(self, current_times, state):
        """Advance model state across a gap."""
        # Does not do anything special if we're jumping across a gap. More advanced
        # models, especially probabilistic ones, would want a special case that
        # depends on the gap size.
        return state

    def _exogenous_input_step(
            self, current_times, current_exogenous_regressors, state):
        """Update model state based on exogenous regressors."""
        raise NotImplementedError(
            "Exogenous inputs are not implemented for this example.")


def read_scv():

    config_util = ConfigUtil("conf/properties.conf")

    sch_metrices = config_util.get_options("scheduler", "sch_metrices").split(',')
    spark = []
    # hive = []
    # queue1_name = "spark"
    # queue2_name = "hive"
    total_mem = 0
    total_cpu = 0
    with open(CLUSTER_INFILE) as cluster:
        reader = csv.DictReader(cluster)
        for row in reader:
            total_mem = float(row.get("totalMB", 0))
            total_cpu = float(row.get("totalVirtualCores"))
    if not total_mem:
        raise ZeroDivisionError
    # print("totalMB", total_mem)
    # total_mem = 11776
    with open(SCHEDULER_INFILE) as csv_file:
        reader = csv.DictReader(csv_file)
        spark_line_num = 0
        # hive_line_num = 0
        for row in reader:
            spark_tmp = []
            # hive_tmp = []
            if row.get("queueName") == FLAGS.queue_name:
                for metrices in sch_metrices:
                    values = row.get(metrices, 0)
                    if not values:
                        break
                    if metrices == "memory":
                        values = float(values) / total_mem
                    elif metrices == "vCores":
                        values = float(values) / total_cpu
                    spark_tmp.append(values)
                spark_tmp.insert(0, spark_line_num)
                spark_line_num += 1
                spark.append(spark_tmp)
            # elif row.get("queueName") == queue2_name:
            #     for metrices in sch_metrices:
            #         values = row.get(metrices, 0)
            #         if not values:
            #             break
            #         if metrices == "memory":
            #             values = float(values) / total_mem
            #         elif metrices == "vCores":
            #             values = float(values) / total_cpu
            #         hive_tmp.append(values)
            #     hive_tmp.insert(0, hive_line_num)
            #     hive_line_num += 1
            #     hive.append(hive_tmp)

    FileOperator.write_list_tocsv(spark, FLAGS.out_file)
    # FileOperator.write_list_tocsv(hive, HIVE_INFILE)


def train(csv_file_name, pre_file_name, model_dir):
    tf.logging.set_verbosity(tf.logging.INFO)
    read_scv()
    csv_file_name = path.join(csv_file_name)
    pre_file_name = path.join(pre_file_name)
    reader = tf.contrib.timeseries.CSVReader(
        csv_file_name,
        column_names=((tf.contrib.timeseries.TrainEvalFeatures.TIMES,)
                      + (tf.contrib.timeseries.TrainEvalFeatures.VALUES,) * 2))

    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
        reader, batch_size=8, window_size=32)

    estimator = ts_estimators.TimeSeriesRegressor(
        model=_LSTMModel(num_features=2, num_units=256),
        optimizer=tf.train.AdamOptimizer(0.001), model_dir=model_dir)

    estimator.train(input_fn=train_input_fn, steps=1000)
    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1,
                                    )

    # Predict starting after the evaluation
    (predictions,) = tuple(estimator.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=100)))

    observed_times = evaluation["times"][0]
    observed = evaluation["observed"][0, :, :]
    evaluated_times = evaluation["times"][0]
    evaluated = evaluation["mean"][0]
    predicted_times = predictions['times']
    predicted = predictions["mean"]

    result = zip(predicted_times, predicted)
    FileOperator.write_list_tocsv(result, pre_file_name)

    plt.figure(figsize=(50, 10))
    plt.axvline(99, linestyle="dotted", linewidth=4, color='r')
    observed_lines = plt.plot(observed_times, observed, label="observation", color="k")
    evaluated_lines = plt.plot(evaluated_times, evaluated, label="evaluation", color="g")
    predicted_lines = plt.plot(predicted_times, predicted, label="prediction", color="r")
    plt.legend(handles=[observed_lines[0], evaluated_lines[0], predicted_lines[0]],
               loc="upper left")
    plt.savefig('predict_result.png')


def main(_):

    pools = list()
    pools.append(multiprocessing.Process(
        target=train, args=(FLAGS.input_file, FLAGS.out_file, FLAGS.model_dir)))
    for pool in pools:
        pool.start()
    t = threading.Timer(FLAGS.time_period, main)
    t.start()

SCHEDULER_INFILE = path.join("./output/scheduler.csv")

CLUSTER_INFILE = path.join("./output/cluster2.csv")

# from CLUSTER_INFILE get hive information's file
HIVE_INFILE = path.join("./output/hive.csv")

# from CLUSTER_INFILE get spark information's file
SPARK_INFILE = path.join("./output/spark.csv")

# the predictions file of queue hive
HIVE_PREFILE = path.join("./output/hive_pre.csv")

# the predictions file of spark hive
SPARK_PREFILE = path.join("./output/spark_pre.csv")

# the dir of hive model' saved
HIVE_MODEDIR = path.join("./model/hive/")

# the dir of spark model' saved
SPARK_MODEDIR = path.join("./model/spark/")

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda  v: v.lower() == "true")
    parser.add_argument(
        "--input_file",
        type=str,
        default=SPARK_INFILE,
        help="input file for model trainning."
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=SPARK_PREFILE,
        help="output file of predict."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=SPARK_MODEDIR,
        help="to save model's dir."
    )
    parser.add_argument(
        "--queue_name",
        type=str,
        default="spark",
        help="the queue name "
    )
    parser.add_argument(
        "--time_period",
        type=int,
        default=600,
        help="the time interval for the scripts to run "
    )

    FLAGS = parser.parse_args()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]])

