import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM as bRBM

def create_datasets():
    x_train = load_data("bindigit_trn.csv")
    x_test = load_data("bindigit_tst.csv")
    y_train = load_data("targetdigit_trn.csv")
    y_test = load_data("targetdigit_tst.csv")

    return x_train, x_test, y_train, y_test

def load_data(filename):
    with open(filename, 'r') as f:
	rows = csv.reader(f)
	ls = list(rows)
	ls = [[int(i) for i in row] for row in ls]

    return ls

def create_weights(n_nodes):
    matrix = np.ndarray(shape=(728,n_nodes))
    for i in range(matrix.shape[0]):
	for j in range(matrix.shape[1]):
	    matrix[i,j] = np.random.normal(0,0.01)

    return matrix

def sigmoid(x):
    return 1/(1+np.exp(-x))




def main(n_nodes):
    x_train, x_test, y_train, y_test = create_datasets()
    x_train = np.array(x_train)
    weight_matrix = create_weights(n_nodes)
    visible_bias = np.zeros(shape=(1,784))
    hidden_bias = np.zeros(shape=(1,n_nodes))

    bern = bRBM(n_components=n_nodes,
	learning_rate = 0.05,
	batch_size=1,
	n_iter = 20)

    bern.intercept_hidden_ = hidden_bias
    bern.intercept_visible = visible_bias
    bern.components_ = weight_matrix
    bern.fit(x_train)

    errors = []

    for i in range(20):
	acc = 0
	for j in range(x_train.shape[0]):
	    res = bern.gibbs(visible_bias)
	    print(np.sum(np.abs(x_train[j] - res)))
	    acc+= np.sum(np.abs(x_train[j] - res))/float(x_train.shape[1])
	errors.append(acc/float(x_train.shape[0]))
	
    epochs = range(20)
    plt.plot(epochs, errors)
    plt.show()
   

main(10)








