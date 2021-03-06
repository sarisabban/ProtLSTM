#!/usr/bin/python3

import numpy
#import keras
import pandas
import random
import sys

def SS(calc):
	'''
	This function trains or generates secondary structure 
	protein sequences. This script is based on the
	nietzsche LSTM example by Keras.
	'''
	#Import text
	data = pandas.read_csv('SS.csv' , sep = ';')
	column = data['Secondary_Structures']
	text = '\n'.join(column)
	chars = sorted(list(set(text)))
	chars_indices = dict((c , i) for i , c in enumerate(chars))
	indices_chars = dict((i , c) for i , c in enumerate(chars))

	#Generate sentences and next characters
	maxlen = 70
	step = 1
	sentences = []
	next_chars = []
	for i in range(0 , len(text) - maxlen , step):
		sent = text[i : i + maxlen]
		char = text[i + maxlen]
		sentences.append(sent)
		next_chars.append(char)

	#Vectorise - (sentances , sentance length , characters)
	X = numpy.zeros((len(sentences) , maxlen , len(chars)) , numpy.int)
	Y = numpy.zeros((len(sentences) , len(chars)) , numpy.int)

	#One-hot encoding
	for i , sentence in enumerate(sentences):
		for t , char in enumerate(sentence):
			X[i , t , chars_indices[char]] = 1
		Y[i , chars_indices[next_chars[i]]] = 1

	#Setup neural network
	model = keras.models.Sequential()
	model.add(keras.layers.LSTM(128 , input_shape = (maxlen , len(chars))))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.core.Dense(200 , activation = 'relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dense(len(chars) , activation = 'softmax'))

	#Compile model
	model.compile(keras.optimizers.Adam(lr = 0.01) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

	if calc == 'train':
		#TensorBoard log (tensorboard --logdir=./logs)
		tensorboard = keras.callbacks.TensorBoard(log_dir = './' , histogram_freq = 1 , write_grads = True)
		#Early stopping
		stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss' , patience = 50)
		#Train model
		model.summary()
		model.fit(X , Y , batch_size = 8192 , epochs = 1000 , validation_split = 0.2 , verbose = 2 , callbacks = [tensorboard , stopping])
		#Save Model
		model.save('SS.h5')

	elif calc == 'generate':
		#Load Model
		model.load_weights('SS.h5')
		#Generate
		print('--------------------')
		start_index = random.randint(0 , len(text) - maxlen - 1)
		sentence = text[start_index : start_index + maxlen]
		print('Starting sequence:' , sentence)
		for iter in range(1000):
			x_pred = numpy.zeros((1 , maxlen , len(chars)))
			for t , char in enumerate(sentence):
				x_pred[0 , t , chars_indices[char]] = 1.0
			preds = model.predict(x_pred , verbose = 0)[0]
			preds = preds#[-1]
			temperature = 1.0
			preds = numpy.asarray(preds).astype('float64')
			preds[preds == 0.0] = 0.0000001
			preds = numpy.log(preds) / temperature
			exp_preds = numpy.exp(preds)
			preds = exp_preds / numpy.sum(exp_preds)
			probas = numpy.random.multinomial(1 , preds , 1)
			next_index = numpy.argmax(probas)
			next_char = indices_chars[next_index]
			sentence = sentence[1 : ] + next_char
			sys.stdout.write(next_char)
			sys.stdout.flush()

def FASTA(calc):
	'''
	This function trains or generates FASTA protein sequences.
	This script is based on the nietzsche LSTM example by Keras.
	'''
	#Import text
	data = pandas.read_csv('FASTA.csv' , sep = ';')
	column = data['Sequence']
	text = '\n'.join(column)
	chars = sorted(list(set(text)))
	chars_indices = dict((c , i) for i , c in enumerate(chars))
	indices_chars = dict((i , c) for i , c in enumerate(chars))

	#Generate sentences and next characters
	maxlen = 70
	step = 1
	sentences = []
	next_chars = []
	for i in range(0 , len(text) - maxlen , step):
		sent = text[i : i + maxlen]
		char = text[i + maxlen]
		sentences.append(sent)
		next_chars.append(char)

	#Vectorise - (sentances , sentance length , characters)
	X = numpy.zeros((len(sentences) , maxlen , len(chars)) , numpy.int)
	Y = numpy.zeros((len(sentences) , len(chars)) , numpy.int)

	#One-hot encoding
	for i , sentence in enumerate(sentences):
		for t , char in enumerate(sentence):
			X[i , t , chars_indices[char]] = 1
		Y[i , chars_indices[next_chars[i]]] = 1

	#Setup neural network
	model = keras.models.Sequential()
	model.add(keras.layers.LSTM(128 , input_shape = (maxlen , len(chars))))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.core.Dense(200 , activation = 'relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dense(len(chars) , activation = 'softmax'))

	#Compile model
	model.compile(keras.optimizers.Adam(lr = 0.01) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

	if calc == 'train':
		#TensorBoard log (tensorboard --logdir=./logs)
		tensorboard = keras.callbacks.TensorBoard(log_dir = './' , histogram_freq = 1 , write_grads = True)
		#Early stopping
		stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss' , patience = 50)
		#Train model
		model.summary()
		model.fit(X , Y , batch_size = 8192 , epochs = 1000 , validation_split = 0.2 , verbose = 2 , callbacks = [tensorboard , stopping])
		#Save Model
		model.save('FASTA.h5')

	elif calc == 'generate':
		#Load Model
		model.load_weights('FASTA.h5')
		#Generate
		print('--------------------')
		start_index = random.randint(0 , len(text) - maxlen - 1)
		sentence = text[start_index : start_index + maxlen]
		print('Starting sequence:' , sentence)
		for iter in range(1000):
			x_pred = numpy.zeros((1 , maxlen , len(chars)))
			for t , char in enumerate(sentence):
				x_pred[0 , t , chars_indices[char]] = 1.0
			preds = model.predict(x_pred , verbose = 0)[0]
			preds = preds#[-1]
			temperature = 1.0
			preds = numpy.asarray(preds).astype('float64')
			preds[preds == 0.0] = 0.0000001
			preds = numpy.log(preds) / temperature
			exp_preds = numpy.exp(preds)
			preds = exp_preds / numpy.sum(exp_preds)
			probas = numpy.random.multinomial(1 , preds , 1)
			next_index = numpy.argmax(probas)
			next_char = indices_chars[next_index]
			sentence = sentence[1 : ] + next_char
			sys.stdout.write(next_char)
			sys.stdout.flush()




if __name__ == '__main__':
	calc = sys.argv[1]
	sets = sys.argv[2]

	if sets == 'SS' or sets == 'ss':
		SS(calc)

	elif sets == 'FASTA' or sets == 'fasta':
		FASTA(calc)
