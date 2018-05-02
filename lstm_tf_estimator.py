

# START

# IMPORTS

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import csv
# GLOBAL VARIABLES
sequence_length = 4 # predict the 5th number given the previous sequence_length numbers.
batch_size = 10
n_epochs = 100
n_neurons = 5
lstm_size = 3
learning_rate = 0.001

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
	trainx = [ normalized_train[i:i+sequence_length] for i in range(len(normalized_train) - sequence_length-2)]
	testx = [ normalized_test[i:i+sequence_length] for i in range(len(normalized_test) - sequence_length-2)]
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


# define the model_fn
def my_lstm_model(features, labels, mode, params):
	x = features["x"]
	# make X the correct shape
	x = x.reshape(, params["batch_size"], )

	# define the layers
	rnn_lstm_layers = []
	rnn_lstm_layers.append(tf.nn.rnn_cell.BasicRNNCell(params["n_input_neurons"]))
	rnn_lstm_layers.append(tf.nn.rnn_cell.LSTMCell(params["n_lstm_neurons"]))
	multi_rnn_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_lstm_layers)
	outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_lstm_cell, inputs=data, dtype=tf.float32)

	if mode = tf.estimator.ModeKeys.PREDICT:
		# in test mode
		spec = tf.estimator.EstimatorSpec(mode = mode, predictions = outputs )
	else:
		# in train or evaluate mode

		# define loss function

		optimizer = tf.train.AdamOptimizer(learning_rate = params["learning_rate"])
		metrics = { 'accuracy': tf.metrics.accuracy(labels, y_pred_cls) }
		spec = tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op, eval_metric = metrics)

	return spec



def main():
	global sequence_length, batch_size, n_epochs, n_neurons, lstm_size, learning_rate

	# load the data and reshape it in the correct format:
	trainx, testx, trainy, testy = load_data("train.csv", "test.csv")
	trainx, testx, trainy, testy = np.array(trainx), np.array(testx), np.array(trainy), np.array(testy )
	n_train, n_test = len(trainx), len(testx)
	feature_dict_train, label_train = create_dict(trainx, trainy)
	feature_dict_test, label_test = create_dict(testx, testy)

	# create a new file with the format we give as input to the LSTM RNN
	create_newfile(feature_dict_test, n_test, testy, trainy, n_train, feature_dict_train) 
	
	# Define the feature_columns and the input functions:
	train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {'x' : trainx }, y = trainy, num_epochs = None, shuffle = True)
	test_input_fn = tf.estimator.inputs.numpy_input_fn(x = {'x' : testx}, num_epochs = 1, shuffle = False)
	feature_columns = tf.feature_column.numeric_column("x", shape = trainx.shape)
    
    # Define the parameters
    params = dict()
    params["learning_rate"] = learning_rate
    params["optimizer"] = "AdamOptimizer"
    params["n_epochs"] = n_epochs
    params["n_input_neurons"] = sequence_length
    params["n_lstm_neurons"] = sequence_length * 3
    params["batch_size"] = batch_size
    params["n_train"] = n_train
    params["n_test"] = n_test

    # Create an instance of the model
	sequence_predictor = tf.estimator.Estimator(model_fn = my_lstm_model, params = params, model_dir = "C:\Users\Rohan Baisantry\Desktop\Python, ML, Dl, RL and AI\pythonfiles\LSTM\custom_estimator")
	
	# Train the model
	sequence_predictor.train(input_fn = train_input_fn, steps = 1000)
	# Evaluate the model
	test_result = sequence_predictor.evaluate(input_fn = test_input_fn)
	print("\n\n Result:\n\t", test_result)
	# Make predictions using the model
	predictions = sequence_predictor.predict(input_fn = test_input_fn)
	print("\n\n", predictions)

main()


