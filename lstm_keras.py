

# START

#IMPORTS 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout
from math import tan
# GLOBALS

# PARAMETERS TO TUNE
batch_size = 0
n_epochs = 0
n_neurons = 0
sequence_length = 4
optimize = "Adam"
 


# LOADING THE DATA
def load_data(file_name):
	data = np.array(pd.read_csv(file_name, sep=",",header= None))
	n = len(data)
	# normalize the dataset
	m1 = float(max(data[:,1]))
	m2 = float(min(data[:,1]))
	d = []
	for i in range(len(data)):
		d.append( float( (data[i][1]-m2)/(m1-m2) ) )
	global sequence_length
	i = 0
	
	sequences = []
	s2 =[]
	while i <= (n - sequence_length):
		s = [0]*sequence_length
		for j in range(sequence_length):
			s[j] = d[ i + j ]
		sequences.append(s)
		i += 1
	for i in range(320,400):
		s2.append(sequences.pop())
	# X_train, X_test, y_train, y_test = train_test_split(data[0], data[1])
	return sequences, s2,m1,m2

# MAIN
def main():
	#load the data
	sequences, s2,m1,m2 = load_data("train_data.csv")
	trainx = [ [ 0.0 for x in range(sequence_length-1)] for y in range(len(sequences)) ]
	trainy = [ 0.0 for y in range(len(sequences)) ]
	testx = [ [0.0 for x in range(sequence_length-1)] for y in range(len(s2))]
	testy =  [0.0 for x in range(len(s2))]
	trainx = np.array(trainx)
	trainy = np.array(trainy)
	testx=np.array(testx)
	testy=np.array(testy)
	#print(trainx.shape,trainy.shape)
	for i in range(len(sequences)):
		for j in range(sequence_length-1):
			trainx[i][j] = sequences[i][j]
		trainy[i] = sequences[i][-1]

	for i in range(len(s2)):
		for j in range(sequence_length-1):
			testx[i][j] = s2[i][j]
		testy[i] = s2[i][-1]
	#print(trainx)
	#print(testy)
	trainx = trainx.reshape(-1,3,1) # needs to be 3d
	trainy = trainy.reshape(-1,1) # needs to be 2d
	testx = testx.reshape(-1,3,1)
	testy = testy.reshape(-1,1)

	# import pdb
	# pdb.set_trace()

	# Create the model
	model = Sequential()
	model.add(LSTM(3, input_shape=(3,1), activation='tanh'))
	model.add(Dropout(0.02))
	model.add(Dense(output_dim = 1))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	model.fit(trainx,trainy, epochs=50, batch_size=4)

	output = model.predict(testx)
	loss=[0.0]*len(testy)
	sum=0
	for i in range(len(testy)):
		loss[i]=abs(output[i][0] - testy[i][0])
		sum+=loss[i]
		print(loss[i],(tan(output[i][0])*(m1-m2))+m2,(tan(testy[i][0])*(m1-m2)+1))
	sum/=len(testy)
	print("\n average loss: ",sum)

main()