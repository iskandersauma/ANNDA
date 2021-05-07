import numpy as np
import matplotlib.pyplot as plt

def create_weights():
    return np.random.rand(10,2)

def read_city():
    city = np.ndarray(shape=(10,2))
    data = open('cities.dat','r').readlines()
    step = 0
    for line in data:
	temp = line.split(',')
	city[step,0] = float(temp[0])
	city[step,1] = float(temp[1])
	step +=1

    return city

def similarity(x,w):
    return np.dot((x-w).T, x-w)

def SOM():
    eta = 0.2
    cities = read_city()
    weights = create_weights()
    neighbour = 1

    for i in range(1,51):
	for j in range(cities.shape[0]):
	    minimum_similarity = 999999999
	    for k in range(0,weights.shape[0]):
		temp = similarity(cities[j,:], weights[k,:])
		
		if temp < minimum_similarity:
		    minimum_similarity = temp
		    index = k

	if neighbour > 0:
	    if index - neighbour < 0:
		left = 10 + index - neighbour
	    else:
		left = index - neighbour

	    if index + neighbour:
		right = index + neighbour - 10
	    else:
		right = index + neighbour

	    if left > right:
		for k in range(left,10):
		    weights[k,:] = weights[k,:] + eta*(cities[j,:] - weights[k,:])
		for k in range(0, right+1):
		    weights[k,:] = weights[k,:] + eta*(cities[j,:] - weights[k,:])
	    else:
		for k in range(left,right+1):
		    weights[k,:] = weights[k,:] + eta*(cities[j,:] - weights[k,:])
	
	else:
	    weights[index,:] = weights[index,:] + eta*(cities[j,:] - weights[k,:])

	if i > 8:
	    neighbour = 1
	else:
	    neighbour = 0

    pos = []
    for i in range(cities.shape[0]):
	minimum_similarity = 99999999
	for j in range(0, weights.shape[0]):
	    temp = similarity(cities[i,:],weights[j,:])
	    if temp < minimum_similarity:
		minimum_similarity = temp
		index = j

	pos.append(index)
    pos_ordered = np.argsort(pos)

    x=np.array([cities[pos_ordered[0],0],cities[pos_ordered[1],0]])
    y=np.array([cities[pos_ordered[0],1],cities[pos_ordered[1],1]])
    plt.plot(x,y,'b',label='1')
    x=np.array([cities[pos_ordered[1],0],cities[pos_ordered[2],0]])
    y=np.array([cities[pos_ordered[1],1],cities[pos_ordered[2],1]])
    plt.plot(x,y,'b',label='2')
    x=np.array([cities[pos_ordered[2],0],cities[pos_ordered[3],0]])
    y=np.array([cities[pos_ordered[2],1],cities[pos_ordered[3],1]])
    plt.plot(x,y,'b',label='3')
    x=np.array([cities[pos_ordered[3],0],cities[pos_ordered[4],0]])
    y=np.array([cities[pos_ordered[3],1],cities[pos_ordered[4],1]])
    plt.plot(x,y,'b',label='4')
    x=np.array([cities[pos_ordered[4],0],cities[pos_ordered[5],0]])
    y=np.array([cities[pos_ordered[4],1],cities[pos_ordered[5],1]])
    plt.plot(x,y,'b',label='5')
    x=np.array([cities[pos_ordered[5],0],cities[pos_ordered[6],0]])
    y=np.array([cities[pos_ordered[5],1],cities[pos_ordered[6],1]])
    plt.plot(x,y,'b',label='6')
    x=np.array([cities[pos_ordered[6],0],cities[pos_ordered[7],0]])
    y=np.array([cities[pos_ordered[6],1],cities[pos_ordered[7],1]])
    plt.plot(x,y,'b',label='7')
    x=np.array([cities[pos_ordered[7],0],cities[pos_ordered[8],0]])
    y=np.array([cities[pos_ordered[7],1],cities[pos_ordered[8],1]])
    plt.plot(x,y,'b',label='8')
    x=np.array([cities[pos_ordered[8],0],cities[pos_ordered[9],0]])
    y=np.array([cities[pos_ordered[8],1],cities[pos_ordered[9],1]])
    plt.plot(x,y,'b',label='9')
    x=np.array([cities[pos_ordered[9],0],cities[pos_ordered[0],0]])
    y=np.array([cities[pos_ordered[9],1],cities[pos_ordered[0],1]])
    plt.plot(x,y,'b',label='10')
    plt.xlim(0,1.5)


    plt.show()

SOM()
