

# LSTM [ Many to One ]

# START

# imports
import csv
import numpy as np
import tensorflow as tf
import sys
import os
import json
from random import shuffle
from tensorflow.python import debug as tf_debug

# CALCULATE ALL POSSIBLE BATCH SIZES
def calculate_batch_sizes(n_train):
	batch_sizes = []
	for i in range(2, int(n_train/2)):	
		if n_train % i == 0 and n_train / i > 1: 
			batch_sizes.append(i)
	return batch_sizes

class lstm_network():
	name = "lstm_"

	# initialization function
	def __init__(self, config_params):
		self.sequence_length = config_params["sequence_length"]
		self.batch_size = config_params["batch_size"]
		self.hidden_layers_size = config_params["hidden_layers_size"]
		self.data_path = config_params["data_path"]
		self.n_epochs = config_params["no_of_epochs"]
		self.learning_rate = config_params["learning_rate"]
		self.w_igate, self.w_fgate, self.w_ogate, self.w_cgate = tf.get_variable('w_igate', shape = [self.sequence_length, self.hidden_layers_size], initializer = tf.contrib.layers.xavier_initializer()), tf.get_variable('w_fgate', shape = [self.sequence_length, self.hidden_layers_size], initializer = tf.contrib.layers.xavier_initializer()), tf.get_variable('w_ogate', shape = [self.sequence_length, self.hidden_layers_size], initializer = tf.contrib.layers.xavier_initializer()), tf.get_variable('w_cgate', shape = [self.sequence_length, self.hidden_layers_size], initializer = tf.contrib.layers.xavier_initializer())
		self.u_igate, self.u_fgate, self.u_ogate, self.u_cgate = tf.get_variable('u_igate', shape = [self.hidden_layers_size, self.hidden_layers_size], initializer = tf.contrib.layers.xavier_initializer()), tf.get_variable('u_fgate', shape = [self.hidden_layers_size, self.hidden_layers_size], initializer = tf.contrib.layers.xavier_initializer()), tf.get_variable('u_ogate', shape = [self.hidden_layers_size, self.hidden_layers_size], initializer = tf.contrib.layers.xavier_initializer()), tf.get_variable('u_cgate', shape = [self.hidden_layers_size, self.hidden_layers_size], initializer = tf.contrib.layers.xavier_initializer())
		self.outputs = [0.0] * self.batch_size
		self.testing_loss = float(0)
		self.training_loss = float(0)
		self.ft, self.ct, self._ct, self.it = [0.0]*(self.hidden_layers_size), [0.0]*(self.hidden_layers_size), [0.0]*(self.hidden_layers_size), [0.0]*(self.hidden_layers_size)
		self.ot, self.ht, self.ct_prev, self.ht_prev = [0.0]*(self.hidden_layers_size), [0.0]*(self.hidden_layers_size), np.array([0.0]*(self.hidden_layers_size)).reshape(1, self.hidden_layers_size), np.array([0.0]*(self.hidden_layers_size)).reshape(1, self.hidden_layers_size)
		self.w_output_layer = tf.get_variable('w_output_layer', shape = [self.hidden_layers_size, 1], initializer = tf.contrib.layers.xavier_initializer())
		print("\n Object of class lstm_network initialized with the given configuration")

	# print values function
	def print_model_info(self):
		print("\n\n\n\t\t MODEL INFORMATION\n\n")
		print("\n Weights of the LSTM layer: ")
		print("\n\n   input Gate Weights: \n    w: ", self.w_igate,"\n    u: ", self.u_igate)
		print("\n\n   Forget Gate Weights: \n    w: ", self.w_fgate,"\n    u: ", self.u_fgate)
		print("\n\n   Context Gate Weights: \n    w: ", self.w_cgate,"\n    u: ", self.u_cgate)
		print("\n\n   Output Gate Weights: \n    w: ", self.w_ogate,"\n    u: ", self.u_ogate)
		print("\n\n Average loss while training: ", self.training_loss)
		print("\n\n Average loss while testing: ", self.testing_loss)

	# loading function
	def load_data(self):
		with open(self.data_path, 'r') as data_file:
			data_reader = csv.reader(data_file, delimiter = ',')
			self.data = [float(row[1]) for row in data_reader]
		self.data_max, self.data_min, self.n_data = float(max(self.data)), float(min(self.data)), len(self.data)
		for i in range(len(self.data)):
			self.data[i] = float( (self.data[i]-self.data_min)/(self.data_max-self.data_min) )
		self.data_x = [ self.data[i:i+self.sequence_length] for i in range(self.n_data - self.sequence_length-1)]
		self.data_y = [ self.data[i] for i in range(self.sequence_length+1, self.n_data)]
		self.n_data = len(self.data_x)
		temp = list(zip(self.data_x,self.data_y))
		shuffle(temp)
		test_size = 0.25
		self.data_x, self.data_y = zip(*temp)
		self.trainx, self.trainy, self.testx, self.testy = self.data_x[:-int(test_size*self.n_data)], self.data_y[:-int(test_size*self.n_data)], self.data_x[-int(test_size*self.n_data):], self.data_y[-int(test_size*self.n_data):] 
		self.n_train, self.n_test = len(self.trainx), len(self.testx)
		batch_sizes = []
		batch_sizes.extend(calculate_batch_sizes(self.n_train))
		while self.batch_size not in batch_sizes:
			print("\n batch size provided in the initial configuration cannot be used, please select one from the following batch sizes:\n",batch_sizes)
			self.batch_size = int(input("\n enter a batch size: "))
		self.n_train_batches = int( self.n_train/self.batch_size ) 
		self.trainx, self.trainy, self.testx, self.testy = np.float32(self.trainx), np.float32(self.trainy), np.float32(self.testx), np.float32(self.testy)
		self.trainx_batches, self.trainy_batches = self.trainx.reshape(self.n_train_batches, self.batch_size, self.sequence_length), self.trainy.reshape(self.n_train_batches,self.batch_size, 1)
		print("\n data loaded succesfully")

	# graph building and training function
	def build_graph_train(self):
		outputs = [0.0]*self.batch_size#tf.placeholder(tf.float32, shape = [1, self.batch_size])
		x = tf.placeholder(tf.float32, shape = [self.n_train_batches, self.batch_size, self.sequence_length], name = 'x')
		ht_prev = tf.placeholder(tf.float32, shape = [1, self.hidden_layers_size], name = 'ht_prev')
		ct_prev = tf.placeholder(tf.float32, shape = [1, self.hidden_layers_size], name = 'ct_prev')
		self.ht_prev = np.array([0.0]*(self.hidden_layers_size)).reshape(1, self.hidden_layers_size)
		self.ct_prev = np.array([0.0]*(self.hidden_layers_size)).reshape(1, self.hidden_layers_size)
		for i1 in range(self.n_train_batches):
			for i2 in range(self.batch_size):
				#self.ht_prev = [self.ht_prev[i:i+9] for i in range(0, self.hidden_layers_size, 9)]
				self.ft = tf.sigmoid( tf.matmul([x[i1][i2]], self.w_fgate) + tf.matmul(ht_prev, self.u_fgate) )
				self.it = tf.sigmoid( tf.matmul([x[i1][i2]], self.w_igate) + tf.matmul(ht_prev, self.u_igate) )
				self.ot = tf.sigmoid( tf.matmul([x[i1][i2]], self.w_ogate) + tf.matmul(ht_prev, self.u_ogate) )
				self._ct = tf.sigmoid( tf.matmul([x[i1][i2]], self.w_cgate) + tf.matmul(ht_prev, self.u_cgate) )
				self.ct = tf.tanh(tf.multiply(self.ft, ct_prev) + tf.multiply(self.it, self._ct))
				self.ht = tf.multiply(self.ot, self.ct)
				ht_prev = self.ht
				ct_prev = self.ct
				outputs[i2] = tf.nn.relu( tf.matmul(self.ht, self.w_output_layer) )
			loss = tf.reduce_mean(tf.square(tf.subtract(outputs, self.trainy_batches[i1])))
			self.ht_prev = ht_prev
			self.ct_prev = ct_prev
			self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
		print("\n Graph built \n\n Now training begins...\n")

		#training 
		i = 0
		avg_loss = float(0)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
			for ep in range(self.n_epochs):
				import pdb
				pdb.set_trace()

				sess.run(self.train_op, feed_dict= { x: self.trainx_batches, ht_prev: np.float32([0]*(self.hidden_layers_size)).reshape(1, self.hidden_layers_size), ct_prev: np.float32([0.0]*(self.hidden_layers_size)).reshape(1, self.hidden_layers_size) })
				if ep % 10 == 0:
					i += 1
					mse = loss.eval()
					avg_loss = float(avg_loss + mse)
					print("\n Epoch: ", ep, "\t Loss: ", mse)
			avg_loss = float(avg_loss/i)
			self.training_loss = avg_loss
		print("\n Training Loss: ", avg_loss)

	# Predict function
	def predict(self):
		x_test_row = tf.placeholder(tf.float32, shape = [1, self.sequence_length])
		avg_loss = float(0)
		input_row = []
		output_row = [0.0]
		predictions = []
		ht_prev = tf.placeholder(tf.float32, shape = [1, self.hidden_layers_size])
		ct_prev = tf.placeholder(tf.float32, shape = [1, self.hidden_layers_size])
		# one forward pass
		self.ft = tf.sigmoid( tf.matmul([x_test_row], self.w_fgate) + tf.matmul(self.ht_prev, self.u_fgate) )
		self.it = tf.sigmoid( tf.matmul([x_test_row], self.w_igate) + tf.matmul(self.htprev, self.u_igate ) )
		self.ot = tf.sigmoid( tf.matmul([x_test_row], self.w_ogate) + tf.matmul(self.ht_prev, self.u_ogate) )
		self._ct = tf.sigmoid( tf.matmul([x_test_row], self.w_cgate) + tf.matmul(self.ht_prev, self.u_cgate) )
		self.ct = yf.tanh(tf.multiply(self.ft, self.ct_prev) + tf.multiply(self.it, self._ct))
		self.ht = tf.matmul(self.ot,self.ct)
		pred_output = tf.nn.relu( tf.matmul(self.ht, self.w_output_layer) )
		with tf.Session() as sess:
			for i1 in range(self.n_test):
				del input_row[:]
				output_row[0] = float(self.testy[i1])
				for i2 in range(self.sequence_length):
					input_row.append(self.testx[i1][i2])
				sess.run(pred_output, feed_dict = { x_test_row: np.array(input_row).reshape(1, self.sequence_length), ht_prev:self.ht_prev, ct_prev: self.ct_prev })
				predictions.append(pred_output)
				avg_error += abs(pred_output - output_row)
			avg_error = float( avg_error/i1 )
			self.testing_loss = avg_error
		print("\n testing Error: ", avg_error)
		return np.array(predictions)

	# save model function
	def save_model(self):
		f = open("model_info.txt", "w+")
		f.write("\n\n\n\n\t\t MODEL INFORMATION\n\n")
		f.write("\n\n Weights of the LSTM layer: ")
		f.write("\n\n\n   input Gate Weights: \n    w: ", self.w_igate,"\n    u: ", self.u_igate)
		f.write("\n\n\n   Forget Gate Weights: \n    w: ", self.w_fgate,"\n    u: ", self.u_fgate)
		f.write("\n\n\n   Context Gate Weights: \n    w: ", self.w_cgate,"\n    u: ", self.u_cgate)
		f.write("\n\n\n   Output Gate Weights: \n    w: ", self.w_ogate,"\n    u: ", self.u_ogate)
		f.write("\n\n\n Average loss while training: ", self.training_loss)
		f.write("\n\n\n Average loss while testing: ", self.testing_loss)
		f.close()
		print("\n\n model's information saved in model_info.txt and weights stored in model.json\n\n")
		f = open("model.json", "w+")
		model_dict = { 'w_output_layer': self.w_output_layer, 'w_igate': self.w_igate, 'u_igate': self.u_igate, 'w_fgate': self.w_fgate, 'u_fgate': self.u_fgate, 'w_cgate': self.w_cgate, 'u_cgate': self.u_cgate, 'w_ogate': self.w_ogate, 'u_ogate': self.u_gate }
		f.write(str(model_dict))
		f.close()

# main function()
def main():

	# parameters of the network
	config_params = dict()
	config_params["sequence_length"] = 3
	config_params["batch_size"] = 33
	config_params["hidden_layers_size"] = 9
	config_params["data_path"] = "C:\\Users\\Rohan Baisantry\\Desktop\\Python, ML, Dl, RL and AI\\pythonfiles\\LSTM\\data.csv"
	config_params["no_of_epochs"] = 1500
	config_params["learning_rate"] = 0.01

	# object of class lstm_network
	test_object = lstm_network(config_params)
	test_object.load_data()
	test_object.build_graph_train()
	predictions = test_object.predict()
	print("\n predictions are: \n", predictions)
	test_object.save_model()

# run
main()