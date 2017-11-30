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
import pandas as pd


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


def train(queue_name, csv_file_name, pre_file_name, model_dir):
    """
    :param queue_name: the queue_name of the hadoop
    :param csv_file_name: the input file to trainning model
    :param pre_file_name: the output file of predict
    :param model_dir: the dir to save model
    """
    tf.logging.set_verbosity(tf.logging.INFO)

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

    estimator.train(input_fn=train_input_fn, steps=FLAGS.train_step)
    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1,
                                    )

    # Predict starting after the evaluation
    # (predictions,) = tuple(estimator.predict(
    #     input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
    #         evaluation, steps=FLAGS.predict_step)))

    (predictions,) = tuple(estimator.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=FLAGS.predict_step)))

    observed_times = evaluation["times"][0]
    observed = evaluation["observed"][0, :, :]
    evaluated_times = evaluation["times"][0]
    evaluated = evaluation["mean"][0]
    predicted_times = predictions['times']


    predicted = predictions["mean"]
    df = pd.DataFrame(predicted)
    df.insert(0, "times", predicted_times)
    df.insert(3, "queue", queue_name)
    df.to_csv(pre_file_name, header=None, mode="a", index=False)


    plt.figure(figsize=(15, 2))
    plt.axvline(99, linestyle="dotted", linewidth=4, color='r')
    observed_lines = plt.plot(observed_times, observed, label="observation", color="k")
    evaluated_lines = plt.plot(evaluated_times, evaluated, label="evaluation", color="g")
    predicted_lines = plt.plot(predicted_times, predicted, label="prediction", color="r")
    plt.legend(handles=[observed_lines[0], evaluated_lines[0], predicted_lines[0]],
               loc="upper left")
    plt.savefig('predict_result.png')


def thread_main():
    """
    for queue to trainning model and predict
    """

    cluster_df = pd.read_csv(CLUSTER_INFILE)
    # total_mem = cluster_df["totalMB"].values[0]
    # total_cpu = cluster_df["totalVirtualCores"].values[0]

    scheduler_df = pd.read_csv(SCHEDULER_INFILE)
    scheduler_df = scheduler_df.dropna(how="any", axis=0)

    # scheduler_df["memory"] = scheduler_df["memory"] / total_mem
    # scheduler_df["vCores"] = scheduler_df["vCores"] / total_cpu

    queue_names = set(scheduler_df["queueName"].values)
    scheduler_df = scheduler_df.set_index("queueName")

    FileOperator.makesure_file_exits("model_input")
    FileOperator.makesure_file_exits("model_out")
    FileOperator.makesure_file_exits("model")

    # empty the pre_file
    FileOperator.write_list_tocsv([], PRE_FILE)

    for queue_name in queue_names:
        queue_information = scheduler_df.ix[queue_name, ["memory", "vCores"]]
        queue_information.insert(0, "times", range(queue_information.shape[0]))
        model_input_file = "./model_input/{0}.csv".format(queue_name)
        queue_information.to_csv(
            model_input_file,
            index=False,
            header=False)
        model_dir = "./model/{0}".format(queue_name)

        train(queue_name, model_input_file, PRE_FILE, model_dir)


def main(_):

    pools = list()
    pools.append(multiprocessing.Process(
        target=thread_main))
    for pool in pools:
        pool.start()
    t = threading.Timer(FLAGS.time_period, main)
    t.start()
    t.join()
    # call lichongmin's predict and core function
    # .....

SCHEDULER_INFILE = path.join("./output/scheduler.csv")

CLUSTER_INFILE = path.join("./output/cluster2.csv")

PRE_FILE = path.join("./model_out/prediction.csv")

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda  v: v.lower() == "true")

    parser.add_argument(
        "--time_period",
        type=int,
        default=7200,
        help="the time interval for the scripts to run "
    )
    parser.add_argument(
        "--train_step",
        type=int,
        default=100,
        help="the step to training  "
    )
    parser.add_argument(
        "--predict_step",
        type=int,
        default=100,
        help="the step to predict "
    )

    FLAGS = parser.parse_args()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]])

