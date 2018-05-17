#!/usr/bin/python3

import numpy
import keras
import random
import sys

#Import text
text = open('./Datasets/SS.txt' , 'r').read()
chars = sorted(list(set(text)))
chars_indices = dict((c , i) for i , c in enumerate(chars))
indices_chars = dict((i , c) for i , c in enumerate(chars))
maxlen = 40
step = 40
#Generate sentences and next charachters
sentences = []
next_chars = []
for i in range(0 , len(text) - maxlen , step):
	char = text[i + maxlen]
	sent = text[i : i + maxlen]
	sentences.append(sent)
	next_chars.append(char)
#Vectorise - (sentances , sentance length , charachters)
X = numpy.zeros((len(sentences) , maxlen , len(chars)) , dtype = numpy.bool)
Y = numpy.zeros((len(sentences) , maxlen , len(chars)) , dtype = numpy.bool)
#One-hot encoding
for i , sentence in enumerate(sentences):
	for t , char in enumerate(sentence):
		X[i , t , chars_indices[char]] = 1
for i , sentence in enumerate(next_chars):
	for t , char in enumerate(sentence):
		Y[i, t, chars_indices[char]] = 1

#TensorBoard log (tensorboard --logdir=./logs)
tensorboard = keras.callbacks.TensorBoard(log_dir = './logs')

#Setup neural network
model = keras.models.Sequential()
model.add(keras.layers.LSTM(128 , input_shape = (maxlen , len(chars)) , return_sequences = True))
model.add(keras.layers.core.Dropout(0.25))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(len(chars))))
model.add(keras.layers.Activation('softmax'))

#Compile model
model.compile(keras.optimizers.Adam(lr = 0.001) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

#Train model
model.fit(X , Y , batch_size = 64 , epochs = 10 , verbose = 2 , callbacks = [tensorboard])

#Save Model
model.save('model.h5')

#Load Model
#model.load_weights('model.h5')

#Generate
print('--------------------')
start_index = random.randint(0 , len(text) - maxlen - 1)
sentence = text[start_index : start_index + maxlen]
print('Starting sequence:' , sentence)
for iter in range(100):
	#One-hot encode that sentance
	x_pred = numpy.zeros((1 , maxlen , len(chars)))
	for t , char in enumerate(sentence):
		x_pred[0 , t , chars_indices[char]] = 1.0
	#Use that tensor to make a prediction of the next charachter
	preds = model.predict(x_pred , verbose = 0)[0]
	preds = preds[-1] #To get just the last charachter
	#Decode that charachter
	temperature = 1.0
	preds = numpy.asarray(preds).astype('float64')
	preds = numpy.log(preds) / temperature
	preds[preds == 0.0] = 0.0000001 #Otherwise np.log will warn when preds contains 0
	exp_preds = numpy.exp(preds)
	preds = exp_preds / numpy.sum(exp_preds)
	probas = numpy.random.multinomial(1 , preds , 1)
	next_index = numpy.argmax(probas)
	next_char = indices_chars[next_index]
	sentence = sentence[1 : ] + next_char
	sys.stdout.write(next_char)
	sys.stdout.flush()
print('\n--------------------')