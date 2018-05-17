#!/usr/bin/python3

import numpy
import keras
import random
import sys

#Import text
text = open('./Datasets/SS.txt' , 'r').read()										# Import dataset
chars = sorted(list(set(text)))												# Make a list of all different characters within the text
chars_indices = dict((c , i) for i , c in enumerate(chars))								# Give each character a number
indices_chars = dict((i , c) for i , c in enumerate(chars))								# The opposite, give each number a character
#Generate sentences and next charachters
maxlen = 5														#?Length of sentences
step = 1														#?Number of charachters to move
sentences = []														# List of sentences
next_chars = []														# List of the next characters that come after each sentence
for i in range(0 , len(text) - maxlen , step):										# range (from , to end of text - length of text , move in step increments)
	sent = text[i : i + maxlen]											#?Slice sentences 40 character long
	char = text[i + 1 : i + maxlen + 1]										#?Slice sentances + next character that comes after each sentence
	sentences.append(sent)												# Append to list
	next_chars.append(char)												# Append to list
	#print(sent , char)												# Print to view
#Vectorise - (sentances , sentance length , charachters)
X = numpy.zeros((len(sentences) , maxlen , len(chars)) , dtype = numpy.bool)						# Training set:	Convert all sentences				into True/False (number of sentances , Maximum sentance length , number of available characters)
Y = numpy.zeros((len(sentences) , maxlen , len(chars)) , dtype = numpy.bool)						# Lable set:	Convert all sentences + next character		into True/False (number of sentances , Maximum sentance length , number of available characters)
#One-hot encoding
for i , sentence in enumerate(sentences):										# Loop through all sentences
	for t , char in enumerate(sentence):										# Loop through each character in each sentece
		X[i , t , chars_indices[char]] = 1									#?
for i , sentence in enumerate(next_chars):										# Loop through all sentences + next character
	for t , char in enumerate(sentence):										# Loop through each character in each sentece + next character
		Y[i , t , chars_indices[char]] = 1									#?

#TensorBoard log (tensorboard --logdir=./logs)
tensorboard = keras.callbacks.TensorBoard(log_dir = './logs')

#Setup neural network
model = keras.models.Sequential()											# Sequential model
model.add(keras.layers.LSTM(128 , input_shape = (maxlen , len(chars)) , return_sequences = True))			#?return_sequences = True because
model.add(keras.layers.core.Dropout(0.25))										# 25% drop rate
model.add(keras.layers.TimeDistributed(keras.layers.Dense(len(chars))))							#?TimeDistributed because
model.add(keras.layers.Activation('softmax'))										# Call softmax activation

#Compile model
model.compile(keras.optimizers.Adam(lr = 0.001) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])		# Compile model
model.summary()														# Print model summary

#Train model
#model.fit(X , Y , batch_size = 128 , epochs = 10 , validation_split = 0.2 , verbose = 2 , callbacks = [tensorboard])	# Train model

#Save Model
#model.save('model.h5')													# Save weights

#Load Model
#model.load_weights('model.h5')												# Load Weights

#Generate
print('--------------------')
start_index = random.randint(0 , len(text) - maxlen - 1)								# Get a random number that is between 0 and 1 charachter less and 1 sentence less than the leangth of the text
sentence = text[start_index : start_index + maxlen]									# Get the sentence of that represents that random number, this is the starting sentence
print('Starting sequence:' , sentence)											# Print the sentence
for iter in range(100):													# Generate this number of characters
	x_pred = numpy.zeros((1 , maxlen , len(chars)))									# Generate a tensor that has the same shape of a sentece. The tensor is just filled with zeros
	for t , char in enumerate(sentence):										# Loop through sentence
		x_pred[0 , t , chars_indices[char]] = 1.0								# One-hot encode the randomly generated sentance (put a 1.0 for each charachter as available from the list of characters)
	preds = model.predict(x_pred , verbose = 0)[0]									# Run a prediction and generate a sentence + next charachter
	preds = preds[-1]												# Choose last character from this generated sentence
	temperature = 1.0												# Temperature is used to make lower probabilities lower and higher probabilities higher. 1.0 does nothing
	preds = numpy.asarray(preds).astype('float64')									# Make sure all tensor values are float64
	preds[preds == 0.0] = 0.0000001											# Otherwise numpy.log will warn when preds contains 0
	preds = numpy.log(preds) / temperature										# Log each tensor value and then divide each value by the temperature
	exp_preds = numpy.exp(preds)											# Turn each tensor value into an exponant
	preds = exp_preds / numpy.sum(exp_preds)									# Re-Normalise (all values add up to 1) by dividing the exponant values by the sum of all the values
	probas = numpy.random.multinomial(1 , preds , 1)								# Randomly choose one index based on probability (most times it will choose the index with the highest probability, but sometimes it will randomly choose a slightly lower one)
	next_index = numpy.argmax(probas)										# Choose the largest value's number location in the vector, which will correspond to the identify of the charachter from the charachter list "indices_chars"
	next_char = indices_chars[next_index]										# Get the value's corresponding charachter
	sentence = sentence[1 : ] + next_char										# Add new charachter to sentance and remove 1 charachter from start of the sentence to maintain its length
	sys.stdout.write(next_char)											# print next character
	sys.stdout.flush()												# Flush buffer to print characters
