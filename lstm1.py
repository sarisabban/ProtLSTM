#!/usr/bin/python3

import numpy
import keras
import random
import sys

#Import text
text = open('SS_raw.txt' , 'r').read()
chars = sorted(list(set(text)))
chars.remove('\n')
chars_indices = dict((c , i) for i , c in enumerate(chars))
indices_chars = dict((i , c) for i , c in enumerate(chars))
maxlen = 150
step = 1
#Generate sentences and next charachters
text2 = open('SS_raw.txt' , 'r')
sentences = []
next_chars = []
for line in text2:
	line = line.strip()
	#Generate sentences
	for i in range(0 , len(line)-1 , step):
		sentence = (line[0 : i+1])
		sentences.append(sentence)
		#Generate next charachter
		next_char = line[i+1]
		next_chars.append(next_char)
#Vectorise - (sentances , sentance length , charachters)
X = numpy.zeros((len(sentences) , maxlen , len(chars)) , dtype = numpy.bool)
Y = numpy.zeros((len(sentences) , len(chars)) , dtype = numpy.bool)
#One-hot encoding
for i , sentence in enumerate(sentences):
	for t , char in enumerate(sentence):
		X[i , t , chars_indices[char]] = 1
Y[i , chars_indices[next_chars[i]]] = 1

#TensorBoard log (tensorboard --logdir=./logs)
tensorboard = keras.callbacks.TensorBoard(log_dir = './logs')

#Setup neural network
model = keras.models.Sequential()
model.add(keras.layers.LSTM(3 , input_shape = (maxlen , len(chars)) , activation = 'relu'))
model.add(keras.layers.Dense(len(chars) , activation = 'softmax'))

#Compile model
model.compile(keras.optimizers.Adam(lr = 0.1) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

#Train model
model.fit(X , Y , batch_size = 1 , epochs = 10 , verbose = 1 , callbacks = [tensorboard])

#Save Model
model.save('model.h5')

#Load Model
#model.load_weights('model.h5')

#Generate
print('--------------------')
while True:
	x = random.randint(0 , len(sentences))
	sentence = sentences[x]
	if 10 > len(sentence) > 3:
		break
	else:
		continue
print(sentence)
for iter in range(100):
	#One-hot encode that sentance
	x_pred = numpy.zeros((1 , maxlen , len(chars)))
	for t , char in enumerate(sentence):
		x_pred[0 , t , chars_indices[char]] = 1.0
	#Use that tensor to make a prediction of the next charachter
	preds = model.predict(x_pred , verbose = 0)[0]
	#Decode that charachter
	temperature = 0.2
	preds = numpy.asarray(preds).astype('float64')
	preds = numpy.log(preds) / temperature
	exp_preds = numpy.exp(preds)
	preds = exp_preds / numpy.sum(exp_preds)
	probas = numpy.random.multinomial(1 , preds , 1)
	next_index = numpy.argmax(probas)
	next_char = indices_chars[next_index]
	sentence = sentence[1 : ] + next_char
	sys.stdout.write(next_char)
	sys.stdout.flush()
print('\n--------------------')
