# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader
from tensorflow.python.training import training as train


tf.logging.set_verbosity(tf.logging.INFO)

#load data
data = pd.read_csv('netflow_minute_5467_mean_0317_0521.csv')

df = data[0:360]

df['y'] = np.log(df['y'])

len_df = len(df)
len_train = len(df)-24

x_source_train = np.array(range(len_df))
x_train = np.array(x_source_train[:len_train])

y_source_train = np.array(df['y'])
y_train = y_source_train[:len_train]

import time
start = time.clock()

data = {tf.contrib.timeseries.TrainEvalFeatures.TIMES: x_train,
        tf.contrib.timeseries.TrainEvalFeatures.VALUES: y_train}

reader = NumpyReader(data)

train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
    reader, batch_size=4, window_size=40)

ar = tf.contrib.timeseries.ARRegressor(
        periodicities=24, input_window_size=30, output_window_size=10,
        num_features=1,
#        hidden_layer_sizes=[20,20,20],
#        optimizer = train.AdamOptimizer(0.001),
#        model_dir='./model',
        loss=tf.contrib.timeseries.ARModel.SQUARED_LOSS)

ar.train(input_fn=train_input_fn, steps=100)
evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)

(predictions, ) = tuple(ar.predict(input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
    evaluation, steps=24)))


times = evaluation["times"][0]
observed = evaluation["observed"][0, :, 0]
mean = np.squeeze(np.concatenate([evaluation["mean"][0], predictions["mean"]], axis=0))
variance = np.squeeze(np.concatenate([evaluation["covariance"][0], predictions["covariance"]], axis=0))
all_times = np.concatenate([times, predictions["times"]], axis=0)
upper_limit = mean + np.sqrt(variance)
lower_limit = mean - np.sqrt(variance)

#observed_times = evaluation["times"][0]
#observed = evaluation["observed"][0, :, :]
#evaluated_times = evaluation["times"][0]
#evaluated = evaluation["mean"][0]
#predicted_times = predictions['times']
#predicted = predictions["mean"]

end = time.clock()

training_times = times

plt.figure(figsize=(10, 5))
plt.plot(training_times, observed, "b", label="training series")
plt.plot(all_times, mean, "r", label="forecast")
plt.plot(all_times, upper_limit, "g", label="forecast upper bound")
plt.plot(all_times, lower_limit, "g", label="forecast lower bound")
plt.fill_between(all_times, lower_limit, upper_limit, color="grey",
                      alpha="0.2")
plt.axvline(training_times[-1], color="k", linestyle="--")
plt.xlabel("time")
plt.ylabel("observations")
plt.legend(loc=0)
plt.show()

#plt.figure(figsize=(10, 5))
#len_axv = len_train -1
#plt.axvline(len_axv, linestyle="dotted", linewidth=4, color='r')
#observed_lines = plt.plot(observed_times, observed, label="observation", color='k')
#evaluated_lines = plt.plot(evaluated_times, evaluated, label="evaluation", color='g')
#predicted_lines = plt.plot(predicted_times, predicted, label='prediction', color='r')
#plt.legend(handles=[observed_lines[0], evaluated_lines[0], predicted_lines[0]], loc='upper left')

#print(predicted)
'''
#模型评估
test = pd.Series(df[len_train:len_df]['y'].values)
pre_y = pd.Series([x[0] for x in predicted])

#
test = np.exp(test)
pre_y  =np.exp(pre_y)

abs_ = (pre_y-test).abs()
mae_ = abs_.mean()
rmse_ = ((abs_**2).mean())**0.5
relative_err = abs_/test
mape_ = (abs_/test).mean()

print('平均绝对误差：%.4f,\n均方根误差：%.4f,\n平均相对误差：%.4f' % (mae_,rmse_,mape_))

print('绝对误差分布：',abs_.describe())
print('相对误差分布：',relative_err.describe())

print('耗时统计:',  (end-start))
'''


