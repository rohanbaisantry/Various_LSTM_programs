

# LSTM [ Many to One ]

"""
Parameters to take from the configuration file:
  ~ number of layers
  ~ sequence length
  ~ batch size
  ~ data path
  ~ number of epochs
  ~ hidden layer nuero configurations [ number of neurons in the layer and the activation function to be used ]

"""
# START

# imports
import csv
import numpy as np
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import sys
import os
import json

class lstm_network():
	name = "lstm_"

	# initialization function
	def __init__(self, config_params):
		config_params = f.read()
		# operations to be done here
		self.n_layers = config_params["no_of_layers"] # including the input layer and the output layer (min=3)
		self.sequence_length = config_params["sequence_length"]
		self.batch_size = config_params["batch_size"]
		self.hidden_layers_config = config_params["hidden_layers_config"]
		self.data_path = config_params["data_path"]
		self.n_epochs = config_params["no_of_epochs"]
		self.learning_rate = config_params["learning_rate"]
		f.close()
		self.w_igate, self.w_fgate, self.w_ogate, self.w_gate = dict(), dict(), dict(), dict()
		self.u_igate, self.u_igate, self.u_igate, self.u_igate = dict(), dict(), dict(), dict()
		for layer_no in range(1, self.n_layers-2):
			self.w_igate[layer_no], self.u_igate[layer_no] = tf.variable(), tf.variable()
			self.w_fgate[layer_no], self.u_fgate[layer_no] = tf.variable(), tf.variable()
			self.w_ogate[layer_no], self.u_ogate[layer_no] = tf.variable(), tf.variable()
			self.w_cgate[layer_no], self.u_cgate[layer_no] = tf.variable(), tf.variable()
		self.weights_output_layer = tf.variable() 
		self.outputs = [0.0] * self.batch_size
		self.testing_loss = float(0)
		self.training_loss = float(0)
		self.ft, self.ct, self._ct, self.it, self.ot, self.ht = dict(), dict(), dict(), dict(), dict(), dict() #list(self.n_layers), list(self.n_layers), list(self.n_layers), list(self.n_layers), list(self.n_layers), list(self.n_layers), list(self.n_layers), list(self.n_layers)
		for layer_no in range(self.n_layers):
			self.ft[layer_no] = [0.0] * ()
			self.it[layer_no] = [0.0] * ()
			self.ot[layer_no] = [0.0] * ()
			self.ct[layer_no] = [0.0] * ()
			self._ct[layer_no] = [0.0] * () 
			self.ht[layer_no] = [0.0] * ()
			self.ct_prev[layer_no] = [0.0] * () 
			self.ht_prev[layer_no] = [0.0] * ()

	# print values function
	def print_model_info(self):
		print("\n\n\n\t\t MODEL INFORMATION\n\n")
		for layer_no in range(1,self.n_layers-2):
			print("\n Weights of the LSTM layer ", layer_no)
			print("\n\n   input Gate Weights: \n    w: ", self.w_igate,"\n    u: ", self.u_igate)
			print("\n\n   Forget Gate Weights: \n    w: ", self.w_fgate,"\n    u: ", self.u_fgate)
			print("\n\n   Context Gate Weights: \n    w: ", self.w_cgate,"\n    u: ", self.u_cgate)
			print("\n\n   Output Gate Weights: \n    w: ", self.w_ogate,"\n    u: ", self.u_ogate)
		print("\n\n Weights of the output layer: \n", self.weights_output_layer)
		print("\n\n Average loss while training: ", self.training_loss)
		print("\n\n Average loss while testing: ", self.testing_loss)

	# loading function
	def load_data(self):
		with open(train, 'r') as data_file:
			data_reader = csv.reader(data_file, delimiter = ',')
			self.data = [fload(row[1]) for row in data_reader]
		self.data_max, self.data_min, self.n_data = float(max(self.data)), float(min(self.data)), len(self.data)
		for i in len(self.data):
			self.data[i] = float( (self.data[i]-self.data_min)/(self.data_max-self.data_min) )
		self.data_x = [ self.data[i:i+self.sequence_length] for i in range(self.n_data - self.sequence_length-1)]
		self.data_y = [ self.data[i] for i in range(self.sequence_length+1, self.n_data)]
		self.trainx, self.trainy, self.testx, self.testy = train_test_split( self.data_x, self.data_y, test_size=0.25, random_state=30 )
		self.n_train, self.n_test = len(self.trainx), len(self.testx)
		self.n_train_batches = int( self.n_train/self.batch_size ) 
		self.trainx, self.trainy, self.testx, self.testy = np.array(self.trainx), np.array(self.trainy), np.array(self.testx), np.array(self.testy)
		self.trainx_batches, self.trainy_batches = self.trainx.reshape(self.n_train_batches, self.batch_size, self.sequence_length), self.trainy.reshape(self.n_train_batches,self.batch_size, self.sequence_length)

	# graph building function
	def build_graph(self):
		outputs = np.array( [float(0.0)]*self.batch_size )
		x = tf.variable(tf.float32, [0.0]*self.sequence_length)
		for i1 in range(self.n_batches):
			for i2 in range(len(self.trainx_batches[i1])):
				self.x = self.trainx[i1][i2]
				layer_no = self.n_layers - 2
				ht_prev, ct_pev = dict(), dict()
				ht_prev[1] = [0.0] * ()
				ct_prev[1] = [0.0] * ()
				for layer_no in range(1, self.n_layer-2):
					self.ft[layer_no] = tf.sigmoid( tf.matmul(self.w_fgate[layer_no], self.x) + tf.matmul(self.u_fgate[layer_no], self.ht_prev[layer_no]) )
					self.it[layer_no] = tf.sigmoid( tf.matmul(self.w_igate[layer_no], self.x) + tf.matmul(self.u_igate[layer_no], self.ht_prev[layer_no]) )
					self.ot[layer_no] = tf.sigmoid( tf.matmul(self.w_ogate[layer_no], self.x) + tf.matmul(self.u_ogate[layer_no], self.ht_prev[layer_no]) )
					self._ct[layer_no] = tf.sigmoid( tf.matmul(self.w_cgate[layer_no], self.x) + tf.matmul(self.u_cgate[layer_no], self.ht_prev[layer_no]) )
					self.ct[layer_no] = yf.tanh(tf.matmul(self.ft[layer_no], self.ct_prev[layer_no]) + tf.matmul(self.it[layer_no], self._ct[layer_no]))
					self.ht[layer_no] = tf.matmul(self.ot[layer_no], self.ct[layer_no])
					self.ht_prev[layer_no+1] = self.ht[layer_no]
					self.ct_prev[layer_no]+1 = self.ct[layer_no]
					self.x = self.ht[layer_no]
				outputs[i2] = tf.sigmoid( tf.matmul(ht[self.n_layers-2], weights_output_layer) )
			self.loss = tf.reduce_mean(tf.square(outputs - self.trainy_batches[i1]))
			self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

	# training function
	def train(self):
		i = 0
		avg_loss = float(0)
		with tf.session() as sess:
			sess.run(tf.global_variables_initializer())
			for ep in self.n_epochs:
				sess.run(self.train_op)
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
		x_test_row = tf.placeholder(tf.float32, )
		avg_loss = float(0)
		input_row = []
		output_row = []
		predictions = []
		# one forward pass
		for layer_no in range(1,self.n_layer-2):
			self.ft[layer_no] = tf.sigmoid( tf.matmul(self.w_fgate[layer_no], self.x_test_row) + tf.matmul(self.u_fgate[layer_no], self.ht_prev[layer_no]) )
			self.it[layer_no] = tf.sigmoid( tf.matmul(self.w_igate[layer_no], self.x_test_row) + tf.matmul(self.u_igate[layer_no], self.htprev[layer_no]) )
			self.ot[layer_no] = tf.sigmoid( tf.matmul(self.w_ogate[layer_no], self.x_test_row) + tf.matmul(self.u_ogate[layer_no], self.ht_prev[layer_no]) )
			self._ct[layer_no] = tf.sigmoid( tf.matmul(self.w_cgate[layer_no], self.x_test_row) + tf.matmul(self.u_cgate[layer_no], self.ht_prev[layer_no]) )
			self.ct[layer_no] = yf.tanh(tf.matmul(self.ft[layer_no], self.ct_prev[layer_no]) + tf.matmul(self.it[layer_no], self._ct[layer_no]))
			self.ht[layer_no] = tf.matmul(self.ot,self.ct)
		pred_output = tf.sigmoid( tf.matmul(self.ht[self.n_layers], self.weights_output_layer) )
		with tf.session() as sess:
			for i1 in range(self.n_test):
				del input_row[:]
				output_row = self.trainy[i1]
				for i2 in range(self.sequence_length):
					input_row.append(self.testx[i1][i2])
				sess.run(pred_output, feed_dict = { x_test_row: np.array(input_row).reshape(1, self.sequence_length)})
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
		for layer_no in range(1,self.n_layers-2):
			f.write("\n\n Weights of the LSTM layer ", layer_no)
			f.write("\n\n\n   input Gate Weights: \n    w: ", self.w_igate,"\n    u: ", self.u_igate)
			f.write("\n\n\n   Forget Gate Weights: \n    w: ", self.w_fgate,"\n    u: ", self.u_fgate)
			f.write("\n\n\n   Context Gate Weights: \n    w: ", self.w_cgate,"\n    u: ", self.u_cgate)
			f.write("\n\n\n   Output Gate Weights: \n    w: ", self.w_ogate,"\n    u: ", self.u_ogate)
		f.write("\n\n\n Weights of the output layer: \n", self.weights_output_layer)
		f.write("\n\n\n Average loss while training: ", self.training_loss)
		f.write("\n\n\n Average loss while testing: ", self.testing_loss)
		f.close()
		print("\n\n model's information saved in model_info.txt and weights stored in model.json\n\n")
		f = open("model.json", "w+")
		f.write(str(self.w_igate))
		f.write(str(self.u_igate))
		f.write(str(self.w_gate))
		f.write(str(self.u_gate))
		f.write(str(self.w_gate))
		f.write(str(self.u_gate))
		f.write(str(self.w_gate))
		f.write(str(self.u_gate))
		f.write(str(self.weights_output_layer))
		f.close()

# main function()
def main():
	config_params = dict()
	config_params["no_of_layers"] = 3
	config_params["sequence_length"] = 3
	config_params["batch_size"] = 21
	config_params["hidden_layers_config"] = [] 
	config_params["data_path"] = "C:\\Users\\Rohan Baisantry\\Desktop\\Python, ML, Dl, RL and AI\\pythonfiles\\LSTM\\test_data.csv"
	config_params["no_of_epochs"] = 1500
	config_params["learning_rate"] = 0.01