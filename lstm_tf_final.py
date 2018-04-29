
"""
LSTM and RNN

using TensorFlow:
1) Dataset
2) Feature column
3) Debugger
4) TensorBoard
5) Estimators

using other concepts:
1) Map and Reduce
2) Lamba Functions
"""

# START

# IMPORTS
import numpy as np
import tensorflow as tf
import csv
from tensorflow.contrib.learn import DynamicRnnEstimator
from tensorflow.python import debug as tf_debug

# GLOBAL VARIABLES
sequence_length = 4 # predict the 5th number give the previous sequence_length numbers.
batch_size = 10
n_epochs = 100
n_neurons = 5


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
		test = [ float(row[1]) for row in reader2]
	normalized_train, normalized_test = normalize_data(train, test)
	global sequence_length
	trainx = [ normalized_train[i:i+sequence_length] for i in range(len(normalized_train) - sequence_length-2)]
	testx = [ normalized_test[i:i+sequence_length] for i in range(len(normalized_test) - sequence_length-2)]
	trainy = [ normalized_train[i] for i in range(sequence_length+1, len(normalized_train))]
	testy = [ normalized_test[i] for i in range(sequence_length+1, len(normalized_test))]
	return trainx, testx, trainy, testy

# INPUT_FUNCTION FOR TRAIN
def input_fn():
	feature_dict = dict()
	global sequence_length
	for i in range(sequence_length):
		temp = "number"+str(i+1)
		feature_dict[temp] = trainx[:,i]

# DICT FUNCTION USING DATASET
def create_dict(trainx, testx):
	feature_dict_train = dict()
	feature_dict_test = dict()
	global sequence_length
	for i in range(sequence_length):
		temp = "number"+str(i+1)
		feature_dict_train[temp] = trainx[:,i]
		feature_dict_test[temp] = testx[:,i]
	return feature_dict_train, feature_dict_test

# creates two files for stroring the testing and training data i the correct format
def create_dataset(feature_dict_test, n_test, testy, trainy, n_train, feature_dict_train):
	f= open("train_new.csv", "w+")
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
	f= open("test_new.csv", "w+")
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

# MAIN
def main():
	trainx, testx, trainy, testy = load_data("train.csv", "test.csv")
	trainx, testx, trainy, testy = np.array(trainx), np.array(testx), np.array(trainy), np.array(testy )
	n_train = len(trainx)
	n_test = len(testx)
	global sequence_length, batch_size, n_epochs, n_neurons
	feature_dict_train, feature_dict_test = create_dict(trainx, testx)
	create_dataset(feature_dict_test, n_test, testy, trainy, n_train, feature_dict_train)
	feature_column = []
	for i in range(sequence_length):
		feature_column.append(tf.feature_column.numeric_column(key="number"+str(i+1)))
	# input_fn definition

	# define the training inputs
	train_input_fn = tf.estimator.inputs.numpy_input_fn( x = trainx, y = trainy , num_epochs = None, shuffle = True)
	estimator = DynamicRnnEstimator( 
		problem_type = "ProblemType.LINEAR_REGRESSION",
		prediction_type = "PredictionType.SINGLE_VALUE",
		optimizer = 'Adam',
		learning_rate = 0.001,
		num_units = sequence_length,
		cell_type = 'lstm',
		sequence_feature_columns = feature_column 
		)

	tf.feature_column = feature_column
	
	estimator.train(input_fn = train_input_fn, steps = 2000)
	hooks = [tf_debug.LocalCLIDebugHook()]
	# Define the test inputs
	test_input_fn = tf.estimator.inputs.numpy_input_fn( x = testx, y = testy, num_epochs = 1, shuffle = False)
	# Evaluate accuracy.
	accuracy_score = estimator.evaluate(input_fn = test_input_fn, hooks = hooks)["accuracy"]
	print("\nTest Accuracy: {0:f}\n".format(accuracy_score))



main()