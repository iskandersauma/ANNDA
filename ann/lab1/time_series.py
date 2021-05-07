import tensorflow as tf
from math import sqrt
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def timeSeries():
    x = [1]
    for t in range(1,1801):
	if t-26 > 0:
	    x.append(0.9*x[t-1] + (0.2*x[t-26])/(1+np.power(x[t-26],10)) )
	else:
	    x.append(0.9*x[t-1])
    return x

def Noise(array, mu, cov):
    for i in range(len(array)):
	array[i] = array[i] + np.random.normal(mu, cov)
    return array

def generateData(mu,cov,noise):
    train_output = []
    train_input = []
    test_input = []
    test_output = []

    x = timeSeries()

    if noise == 'on':
	x = Noise(x,mu,cov)

    for t in range(301,1301):
	train_input.append(np.array([x[t-25],x[t-20],x[t-15],x[t-10],x[t-5] ]))
	train_output.append(x[t])

    train_input = np.array(train_input)
    train_output = np.array(train_output).reshape(len(train_output),1)

    for t in range(1302,len(x)):
	test_input.append(np.array([x[t-25],x[t-20],x[t-15],x[t-10],x[t-5] ]))
	test_output.append(x[t])

    test_input = np.array(test_input)
    test_output = np.array(test_output).reshape(len(test_output),1)

    return train_input, train_output, test_input, test_output

class MGTS():

    def __init__(self,x,y,x_test,y_test):
	self.x_train,self.x_val,self.y_train,self.y_val = train_test_split(x,y,shuffle=False, test_size = 0.5)
	self.x_test = x_test
	self.y_test = y_test

    def net(self,layers, eta, epochs, earlyStop, regularization):
	w = []
	b = []
	x = tf.placeholder(tf.float32, [None, self.x_train.shape[1]], name = 'x')
	y = tf.placeholder(tf.float32, [None, self.y_train.shape[1]], name = 'y')

	layers.append(self.y_train.shape[1])
	w.append(tf.Variable(tf.random_normal([self.x_train.shape[1],layers[0]],stddev = 0.01)))
	b.append(tf.Variable(tf.random_normal([layers[0]],stddev = 0.01)))

	out = tf.add(tf.matmul(x,w[0]),b[0])
	for i in range(1, len(layers)):
	    w.append(tf.Variable(tf.random_normal([layers[i-1], layers[i]], stddev = 0.01)))
	    b.append(tf.Variable(tf.random_normal([layers[i]], stddev = 0.01)))
	    out = tf.add(tf.matmul(out,w[i]),b[i])

	mse = tf.losses.mean_squared_error(labels = y, predictions = out)

	if regularization == 'L2 reg':
	    reg = 0
	    beta = 0.01
	    for i in range(0,len(layers)):
		reg += tf.nn.l2_loss(w[i])
	    loss = tf.reduce_mean(mse+beta*reg)
	else:
	    loss = mse

	optimizer = tf.train.AdamOptimizer(learning_rate = eta).minimize(loss)
	init_var = tf.global_variables_initializer()
	
	with tf.Session() as sess:
	    sess.run(init_var)
	    maximum = 20
	    value = 0
	    minDiff = 0.01
	    train_error = []
	    val_error = []

	    for k in range (epochs):

		avg_val_error = 0
		_, cost = sess.run([optimizer, loss], feed_dict = {x: self.x_train, y: self.y_train})
		train_error.append(cost)
		val_error.append(sqrt(sess.run(loss, feed_dict = {x: self.x_val, y: self.y_val})))

		print("Epoch: " + str(k+1) + "train_error = " + str(train_error[k]))
		print("validation_error = " + str(val_error[k]))
		print("-----------------------")
		
		if earlyStop == 'on':
		    if k > 0 and (val_error[k-1] - val_error[k] < minDiff):
			cool_count = 0
		    else:
			value += 1
		    if value > maximum and train_error[k] < 0.01:
			print("can stop early")
			break

            print("Test_error = " + str(sqrt(sess.run(loss, feed_dict = {x: self.x_test, y: self.y_test}))))
            pred = sess.run(out, feed_dict = {x:self.x_test})
            plt.plot(pred, 'b', label = 'Prediction')
            plt.plot(self.y_test, 'r', label = 'Original')
            plt.legend(('Prediction', 'Original'))
            plt.show()
            
            plt.plot(train_error, 'b', label = 'Training')
            plt.plot(val_error, 'r', label = 'Validation')
            plt.legend(('Training error', 'Validation error', 'Test error'))
	    plt.show()








def main():
    mu = 0
    cov = 0.18
    train_input, train_output, test_input,test_output = generateData(mu, cov, noise = 'off')
    mgts = MGTS(train_input, train_output, test_input, test_output)
    mgts.net(layers = [8,4], eta = 0.05, epochs = 1000, earlyStop = 'off', regularization = 'L2')


main()











