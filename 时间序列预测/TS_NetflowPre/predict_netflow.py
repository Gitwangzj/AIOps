import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.contrib.timeseries.python.timeseries import NumpyReader
from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from train_lstm import _LSTMModel


x_train = np.array(range(168))
y_train = [17820652, 18050364, 17911606, 16831316, 8759626, 8546053, 17573559, 17957414, 8579513, 8947416, 8896293, 8513787, 9689336, 7644881, 8187614, 8157440, 14690066, 13122113, 8189606, 8128143, 9928045, 12186196, 11087630, 8487980, 11550187, 15780109, 14416526, 12554730, 9677747, 11517624, 18042087, 17725251, 16918745, 13981266, 9067449, 7621155, 8233936, 6407346, 5129027, 5072525, 5379080, 2749548, 3386364, 2606616, 5753418, 11276798, 6900994, 3827738, 6735140, 10346079, 10970506, 9039637, 6130244, 7863363, 17851643, 16621718, 15299360, 6380432, 5777692, 5496734, 5051249, 4673845, 3902700, 2850843, 5616231, 2150059, 1836288, 2328483, 4835081, 7806310, 6582895, 3822465, 7020531, 9781196, 10954961, 5855414, 6374146, 7007517, 17624561, 11252861, 15389236, 7100923, 6288682, 5557393, 5507696, 4839543, 3215167, 2107288, 5284018, 1677593, 1975131, 2127995, 4371311, 7579893, 6320845, 3379414, 6971598, 10663938, 11656811, 9904060, 6615442, 7785164, 17951288, 18241499, 17124873, 7860011, 5068894, 6072006, 7082042, 5178642, 3504864, 3124104, 5363476, 2672646, 3056433, 2628075, 5603036, 8108877, 7231927, 4471953, 7084985, 12227811, 13953487, 9176698, 7023547, 13697545, 18330191, 18341313, 18270395, 9819666, 6596462, 7404952, 7209089, 4442841, 2527758, 3934994, 7647949, 3578724, 2257577, 3076758, 5145234, 8768478, 7397283, 4147334, 5192010, 5990813, 6407392, 5855151, 6151765, 5421948, 17275831, 13503030, 9697451, 7223771, 6569367, 4713950, 6340608, 3813842, 3705733, 3428458, 6596959, 2956116, 1902686, 2577152, 5127546, 7551840, 7490062, 4156895]
y_train = np.array(y_train)

data = {tf.contrib.timeseries.TrainEvalFeatures.TIMES: x_train,
        tf.contrib.timeseries.TrainEvalFeatures.VALUES: y_train}

reader = NumpyReader(data)

train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
    reader, batch_size=4, window_size=100)

estimator = ts_estimators.TimeSeriesRegressor(model=_LSTMModel(num_features=1, num_units=128),
                                              optimizer=tf.train.AdamOptimizer(0.001))

estimator.train(input_fn=train_input_fn, steps=2000)
evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)

(predictions, ) = tuple(estimator.predict(input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
    evaluation, steps=24)))

observed_times = evaluation["times"][0]
observed = evaluation["observed"][0, :, :]
evaluated_times = evaluation["times"][0]
evaluated = evaluation["mean"][0]
predicted_times = predictions['times']
predicted = predictions["mean"]

# plt.figure(figsize=(15, 5))
# plt.axvline(167, linestyle="dotted", linewidth=4, color='r')
# observed_lines = plt.plot(observed_times, observed, label="observation", color='k')
# evaluated_lines = plt.plot(evaluated_times, evaluated, label="evaluation", color='g')
# predicted_lines = plt.plot(predicted_times, predicted, label='prediction', color='r')
#
# plt.legend(handles=[observed_lines[0], evaluated_lines[0], predicted_lines[0]], loc='upper left')
# plt.savefig('netflow_prediction.jpg')


plt.figure()
plt.plot(x_train, y_train)

predicted_times = list(predicted_times)
plt.plot(predicted_times, predicted)
plt.axvline(168, linestyle="dotted", linewidth=2, color='r')

plt.savefig('netflow_prediction.jpg')

