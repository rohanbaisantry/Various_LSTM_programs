import tensorflow as tf
import csv
import numpy as np
import tensorflow.contrib.rnn as rnn
from random import shuffle
from tensorflow.python import debug as tf_debug
# import 

sequence_length = 4 # default
batch_sizes = []

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
	return normalized_train, normalized_test, m21, m22, m11, m12

# LOADING THE DATA
def load_data(train, test):
	with open(train, 'r') as csvfile1:
		reader1 = csv.reader(csvfile1, delimiter = ',')
		train = [ float(row[1]) for row in reader1]
	with open(test, 'r') as csvfile2:
		reader2 = csv.reader(csvfile2, delimiter = ',')
		test = [ float(row[1]) for row in reader2 ]
	normalized_train, normalized_test, max_test, min_test, max_train, min_train = normalize_data(train, test)
	global sequence_length
	trainx = [ normalized_train[i:i+sequence_length] for i in range(len(normalized_train) - sequence_length-1)]
	testx = [ normalized_test[i:i+sequence_length] for i in range(len(normalized_test) - sequence_length-1)]
	trainy = [ normalized_train[i] for i in range(sequence_length+1, len(normalized_train))]
	testy = [ normalized_test[i] for i in range(sequence_length+1, len(normalized_test))]
	return trainx, trainy, testx, testy, max_test, min_test, max_train, min_train

def create_model(params):
	rnn_layers = []
	rnn_layers.append(tf.nn.rnn_cell.LSTMCell(params["neurons_layer1"]))
	rnn_layers.append(tf.nn.rnn_cell.LSTMCell(params["neurons_layer2"]))
	rnn_layers.append(tf.nn.rnn_cell.BasicRNNCell(params["neurons_layer3"]))
	multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
	return multi_rnn_cell

def main():
	
	global sequence_length, batch_sizes

	sequence_length = int(input("\n\n enter a sequence length: "))

	x_train, y_train, x_test, y_test, max_test, min_test, max_train, min_train = load_data("train_data.csv", "test_data.csv")
	n_train = len(x_train)
	n_test = len(x_test)

	# shuffling
	temp_train = list(zip(x_train,y_train))
	shuffle(temp_train)
	x_train, y_train = zip(*temp_train)
	temp_test = list(zip(x_test, y_test))
	shuffle(temp_test)
	x_test, y_test = zip(*temp_test)

	x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
	# x = ( n*sequence_length ) || y = ( n*1 )
	print("\n shape of the input: ", x_train.shape)
	print("\n shape of the output: ", y_train.shape)

	calculate_batch_sizes(n_train)
	batch_size = 1
	while batch_size not in batch_sizes:
		print("\n Choose one of the following batch sizes to be used \n", batch_sizes)
		batch_size = int(input("\n enter a batch size: "))

	tf.reset_default_graph()

	n_batches = int(len(y_train) / batch_size)
	print("\n number of batches: ", n_batches, "\n")
	x = tf.placeholder(tf.float32, [None, batch_size, sequence_length])
	x_batches = np.reshape(x_train ,[n_batches, batch_size, sequence_length])
	y = tf.placeholder(tf.float32, [None, batch_size, 1])
	y_batches = np.reshape(y_train, [n_batches, batch_size, 1])

	params = dict()
	params["n_layers"] = 3
	params["neurons_layer1"] = sequence_length
	params["neurons_layer2"] = sequence_length * 3
	params["neurons_layer3"] = 1
	params["learning_rate"] = 0.01
	params["n_epochs"] = 4500
	params["optimizer"] = "AdamOptimizer"

	print("\n Parameters of the network are:\n")
	for key in params:
		print("\t",key, ": ", params[key])
	print("\n\n")

	my_lstm_cell = create_model(params)
	print("\n x's shape: ", x.shape)
	print("\n y's shape: ", y.shape, "\n")
	rnn_output, state = tf.nn.dynamic_rnn(cell=my_lstm_cell, inputs=x, dtype=tf.float32)
	print("\n shape of Network's output: ", rnn_output.shape, "\n")

	"""
	stacked_rnn_output = tf.reshape(rnn_output, [-1, 1])
	stacked_outputs = tf.layers.dense(stacked_rnn_output, 1)
	outputs = tf.reshape(stacked_outputs, [n_train, 1, 1])
	"""

	#outputs = tf.reshape(rnn_output, [n_train, 1, 1])
	loss = tf.reduce_sum(tf.square(rnn_output - y))
	optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
	training_op = optimizer.minimize(loss)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		sess = tf_debug.TensorBoardDebugWrapperSession(sess, "rohanasus:6006")
		init.run()
		avg_loss = float(0)
		i = 0
		for ep in range(params["n_epochs"] + 1):
			sess.run(training_op, feed_dict= {x: x_batches, y: y_batches})
			if ep % 100 == 0:
				i += 1
				mse = loss.eval(feed_dict = {x: x_batches, y: y_batches})
				if ep != 0:
					print("Epoch: ",ep, "\tLoss(MSE): ", mse)
				else:
					print("Epoch: ",ep, "  \tLoss(MSE): ", mse)
					avg_loss = float(avg_loss - mse)
				avg_loss = float(avg_loss + mse)
		print("\n Average loss while training: ", avg_loss)
		avg_loss = float(avg_loss/i)
		remove = x_test.shape[0] % batch_size
		x_test = x_test[:-remove]
		y_test = y_test[:-remove]
		print("\n x_test's shape: ", x_test.shape, "\n")
		print("\nPredicted value \tActal value\n")
		y_pred = sess.run(rnn_output, feed_dict= {x: x_test.reshape(-1, batch_size, sequence_length)})
		temp_pred = np.reshape(np.array(y_pred), [-1,1])
		avg_error = float(0)
		for i in range(len(y_test)):
			print("Predicted: ", int(temp_pred[i][0]*(max_test-min_test) + min_test), "\tActual: ", int(y_test[i]*(max_test-min_test) + min_test))
			avg_error = avg_error + abs(temp_pred[i][0] - y_test[i] )*(max_test-min_test)
		avg_error = float(avg_error / len(y_test))
		accuracy = float( 100 - avg_error)
		print("\n Average error while Testing: ", avg_error)
		print("\n Accuracy while testing: ", accuracy)
		all_tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
		tf.train.write_graph(sess.graph_def, '.', 'hellotensor.pbtxt')
		
# run
main()
