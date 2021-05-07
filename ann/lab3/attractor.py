import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    x1 = [-1, -1, 1, -1, 1, -1, -1, 1] 
    x2 = [-1, -1, -1, -1, -1, 1, -1, -1] 
    x3 = [-1, 1, 1, -1, -1, 1, -1, 1]
    x = np.vstack([x1,x2,x3])

    x1d = [1, -1, 1, -1, 1, -1, -1, 1]
    x2d = [1, 1, -1, -1, -1, 1, -1, -1]
    x3d = [1, 1, 1, -1, 1, 1, -1, 1]
    xd = np.vstack([x1d, x2d, x3d])

    x1h = [-1, 1, 1, 1, 1, -1, -1, 1]
    x2h = [1, -1, 1, 1, -1, 1, -1, -1]
    x3h = [1, 1, -1, -1, 1, 1, -1, 1]
    xh = np.vstack([x1h, x2h, x3h])

    return x, xd, xh

def create_weights(X):
    p = len(X[0])
    W = np.zeros((p,p))
    for x in X:
	w = np.outer(x,x.T)
	W +=w

    return w

def update(w, origin, activation, iterations):
    pattern = activation
    activation_list = np.array(activation)
    print('origin')
    print(origin)
    print('noisy data')
    print(activation)
    
    for i in range(iterations):
	temp = np.copy(activation_list)
	for j in range(len(pattern)):
	    activation = np.where(np.sum(w*temp[j], axis=1) > 0, 1, -1)
	    activation_list[j] = activation

    print('prediction')
    print(activation_list)

    if np.array_equal(origin, activation_list):
	print('stable fixed point reached')
    else:
	print(np.equal(origin,activation_list))

def main():
    inputs, input_dis, input_wrong = generate_data()
    w = create_weights(np.copy(inputs))
    iterations = 100
    update(w, np.copy(inputs), np.copy(input_wrong), iterations)

main()

















