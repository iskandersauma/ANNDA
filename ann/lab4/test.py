from __future__ import print_function
import keras
import numpy as np
import csv
import keras.utils as keras_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

def create_data(filename):
    with open(filename, 'r') as f:
	rows = csv.reader(f)
	ls = list(rows)
	ls = [[int(i) for i in row] for row in ls]
    return ls

def create_datasets():
    x_train = create_data("bindigit_trn.csv")
    x_test = create_data("bindigit_tst.csv")
    y_train = create_data("targetdigit_trn.csv")
    y_test = create_data("targetdigit_tst.csv")

    return np.array(x_train), np.array(x_test), keras_utils.to_categorical(y_train,10), keras_utils.to_categorical(y_test, 10)

def create_matrix(input_dim, nodes_dim):
    weight_matrix = np.ndarray(shape=(input_dim,nodes_dim))
    for i in range(weight_matrix.shape[0]):
	for j in range(weight_matrix.shape[1]):
	    weight_matrix[i,j] = np.random.normal(0,0.01)
    return weight_matrix

batch_size = 128
num_classes = 10
epochs = 100

x_train, x_test, y_train, y_test = create_datasets()

model = Sequential()
model.add(Dense(150, activation='relu', input_shape=(784,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
	optimizer=RMSprop(),
	metrics=['accuracy'])
history = model.fit(x_train, y_train,
	batch_size=batch_size,
	verbose=1,
	validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss: " + str(score[0]))
print("Test accuracy: " + str(score[1]))













