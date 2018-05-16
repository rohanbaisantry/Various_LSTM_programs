

# START

# IMPORTS

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import csv
# GLOBAL VARIABLES
sequence_length = 4 # predict the 5th number given the previous sequence_length numbers.
batch_sizes = []
n_epochs = 100
n_neurons = 5
lstm_size = 3
learning_rate = 0.001

# CALCULATE ALL POSSIBLE BATCH SIZES
def calculate_batch_sizes(n_train):
	global batch_size
	for i in range(2, int(n_train/2)):	
		if n_train % i == 0 and n_train / i > 1: 
			batch_sizes.append(i)

# NORMALIZING THE DATA
def normalize_data(train, test):
	m11 = float(max(train))
	m12 = float(min(train))
	normalized_train = [float((x-m12)/(m11-m12)) for x in train]
	m21 = float(max(test))
	m22 = float(min(train))
	normalized_test = [float((x-m22)/(m21-m22)) for x in test]
	return normalized_train, normalized_test

# LOADING THE DATA
def load_data(train, test):
	with open(train, 'r') as csvfile1:
		reader1 = csv.reader(csvfile1, delimiter = ',')
		train = [ float(row[1]) for row in reader1]
	with open(test, 'r') as csvfile2:
		reader2 = csv.reader(csvfile2, delimiter = ',')
		test = [ float(row[1]) for row in reader2 ]
	normalized_train, normalized_test = normalize_data(train, test)
	global sequence_length
	trainx = [ normalized_train[i:i+sequence_length] for i in range(len(normalized_train) - sequence_length-1)]
	testx = [ normalized_test[i:i+sequence_length] for i in range(len(normalized_test) - sequence_length-1)]
	trainy = [ normalized_train[i] for i in range(sequence_length+1, len(normalized_train))]
	testy = [ normalized_test[i] for i in range(sequence_length+1, len(normalized_test))]
	return trainx, testx, trainy, testy

# Create Dict function
def create_dict(x,y):
	feature_dict = dict()
	global sequence_length
	for i in range(sequence_length):
		temp = "number"+str(i+1)
		feature_dict[temp] = x[:,i]
	labels = y
	return feature_dict, labels

# creates two files for stroring the testing and training data in the correct format
def create_newfile(feature_dict_test, n_test, testy, trainy, n_train, feature_dict_train):
	f= open("train_data_new.csv", "w+")
	csvwriter = csv.writer(f, delimiter = ",")
	headers = [ i for i in feature_dict_train]
	headers.append("target")
	csvwriter.writerow(headers)
	del headers[-1]
	temp_list = []
	for i in range(n_train):
		for j in feature_dict_train:
			temp_list.append(feature_dict_train[j][i])
		temp_list.append(trainy[i])
		csvwriter.writerow(temp_list)
		del temp_list[:]
	f.close()
	f= open("test_data_new.csv", "w+")
	csvwriter = csv.writer(f, delimiter = ",")
	headers.append("target")
	csvwriter.writerow(headers)
	del headers[-1]
	del temp_list[:]
	for i in range(n_test):
		for j in feature_dict_train:
			temp_list.append(feature_dict_test[j][i])
		temp_list.append(testy[i])
		csvwriter.writerow(temp_list)
		del temp_list[:]
	f.close()

# define the model_fn
def my_lstm_model(features, labels, mode, params):
	x = features["x"]

	# make X the correct shape
	#x = tf.reshape(x, [-1, params["batch_size"], sequence_length])

	# define the layers
	rnn_lstm_layers = []
	rnn_lstm_layers.append(tf.nn.rnn_cell.BasicRNNCell(params["n_input_neurons"]))
	rnn_lstm_layers.append(tf.nn.rnn_cell.LSTMCell(params["n_lstm_neurons"]))
	multi_rnn_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_lstm_layers)
	rnn_outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_lstm_cell, inputs=x, dtype=tf.float32)

	print("\n shape f output of the network: ", rnn_outputs.shape, "\n")

	"""
	stacked_rnn_output = tf.reshape(rnn_output, [-1, 1])
	stacked_outputs = tf.layers.dense(stacked_rnn_output, 1)
	outputs = tf.reshape(stacked_outputs, [n_train, 1, 1])
	"""

	if mode == tf.estimator.ModeKeys.PREDICT:
		# in test mode
		spec = tf.estimator.EstimatorSpec(mode = mode, predictions = outputs )
	else:
		# in train or evaulation mode

		loss = tf.reduce_sum(tf.square(outputs - y))
		optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
		training_op = optimizer.minimize(loss)
		metrics = { 'accuracy': tf.metrics.accuracy(labels, y_pred_cls) }
		
		spec = tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op, eval_metric = metrics)

	return spec

def main():

	global sequence_length, batch_sizes, n_epochs, n_neurons, lstm_size, learning_rate
	sequence_length = int(input("\n\n enter a sequence length: "))

	# load the data and reshape it in the correct format:
	trainx, testx, trainy, testy = load_data("train_data.csv", "test_data.csv")
	trainx, testx, trainy, testy = np.array(trainx), np.array(testx), np.array(trainy), np.array(testy )
	n_train, n_test = len(trainx), len(testx)
	feature_dict_train, label_train = create_dict(trainx, trainy)
	feature_dict_test, label_test = create_dict(testx, testy)
	print("\n shape of trainx: ", trainx.shape)
	print("\n shape of trainy: ", trainy.shape)

	# create a new file with the format we give as input to the LSTM RNN
	create_newfile(feature_dict_test, n_test, testy, trainy, n_train, feature_dict_train)

	calculate_batch_sizes(n_train)
	batch_size = 1
	while batch_size not in batch_sizes:
		print("\n The following batch sizes can be used\n", batch_sizes)
		batch_size = int(input("\n enter a batch size: "))
	print("\n number of batches: ", int(n_train/batch_size), "\n")

	# Define the parameters
	params = dict()
	params["n_data_points"] = n_train
	params["learning_rate"] = learning_rate
	params["optimizer"] = "AdamOptimizer"
	params["n_epochs"] = n_epochs
	params["n_input_neurons"] = sequence_length
	params["n_lstm_neurons"] = sequence_length * 3
	params["batch_size"] = batch_size
	params["n_train"] = n_train
	params["n_test"] = n_test
	params["sequence_length"] = sequence_length
	params["n_batches"] = int(n_train / batch_size)

	# Define the feature_columns and the input functions:
	train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {'x' : trainx.reshape(params["n_batches"], params["batch_size"], params["sequence_length"]) }, y = trainy.reshape(params["n_batches"], params["batch_size"], 1), num_epochs = None, shuffle = True)
	test_input_fn = tf.estimator.inputs.numpy_input_fn(x = {'x' : testx}, num_epochs = 1, shuffle = False)
	feature_columns = tf.feature_column.numeric_column("x", shape = trainx.shape)

	# reset the graph to default
	tf.reset_default_graph()

	# Create an instance of the model
	sequence_predictor = tf.estimator.Estimator(model_fn = my_lstm_model, params = params, model_dir = "C:\\Users\\Rohan Baisantry\\Desktop\\Python, ML, Dl, RL and AI\\pythonfiles\\LSTM\\custom_estimator")

	# Train the model
	sequence_predictor.train(input_fn = train_input_fn, steps = 1000)

	# Evaluate the model
	test_result = sequence_predictor.evaluate(input_fn = test_input_fn)
	print("\n\n Result:\n\t", test_result)

	# Make predictions using the model
	predictions = sequence_predictor.predict(input_fn = test_input_fn)
	print("\n\n", predictions)

# RUN
main()

# END