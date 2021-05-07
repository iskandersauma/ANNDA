import numpy as np

def create_props():
    props = np.ndarray(shape=(32,84))
    data = open('animals.dat','r').readline()
    current = 0
    for i in range(0,len(data),168):
	acc = []
	temp = data[current*168:(current+1)*168-1]
	for element in temp:
	    if element != ',':
		acc.append(element)
	props[current,:] = np.array(acc)
	current +=1

    return props

def create_weights():
    return np.random.rand(100,84)

def similarity(x,w):
    return np.dot((x-w).T,x-w)

def similarity2(x,w):
    return np.abs(np.sum(x-w))

def print_names(indices):
    names = open('animalnames.txt').readlines()
    for name in indices:
	print(names[name])

def SOM():
    eta = 0.2
    props = create_props()
    weights = create_weights()
    neighbourhood_size = 50

    pos = []

    for i in range(props.shape[0]):
	minimum_similarity = 99999999
	for k in range(0,weights.shape[0]):
	    temp = similarity(props[i,:], weights[k,:])
	    
	    if temp < minimum_similarity:
		minimum_similarity = temp
		index = k

	pos.append(index)
    animal_order = np.argsort(pos)
    print_names(animal_order)

SOM()	








