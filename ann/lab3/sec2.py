import numpy as np
import matplotlib.pyplot as plt
import random
from colorama import Fore
from colorama import Style

def generate_data():
    with open('pict.dat','r') as f:
	lines = f.read().split(',')
	lines = list(map(int, lines))
	
    p = []
    for i in range(0,len(lines), 1024):
	p.append(lines[i: i + 1024])

    return np.array(p)

def create_noise(image, numDis):
    im = np.copy(image)
    np.random.seed(100)
    idx = random.sample(range(0,len(im)), numDis)
    for i in idx:
	im[i] *= -1
    return im

def visualization(pattern, predicted):
    titles = ['pattern', 'prediction']
    pattern = np.reshape(pattern,(32,32))
    predicted = np.reshape(predicted,(32,32))
    image = [pattern, predicted]
    fig = plt.figure(figsize = (20,5))
    for i in range(len(image)):
	ax = fig.add_subplot(1,2,i+1)
	ax.set_title(titles[i])
	ax.imshow(image[i].T)
	ax.axis('off')
    plt.pause(1)

def create_weights(X):
    N = X[0].size
    W = np.zeros((N,N))
    for x in X:
	w = np.outer(x,x.T)
	W += w
    return W/N

def rand_weights(X):
    N = X[0].size
    return np.random.normal(0,0.001,(N,N)) #np.random.randn(N,N)

def energy(s,W):
    return -np.dot(np.dot(s.T,W),s)

def update(w,origin, activation, mode):
    patterns = np.copy(activation)
    updated = 1
    energy_list = []
    if mode == 'synchronous':
	visualization(origin, activation)
	while True:
	    patterns = np.where(np.sum(w*patterns,axis=1)> 0, 1, -1)
 	    E = energy(patterns, w)
	    if np.array_equal(origin, patterns):
		print('num of updates: ' + str(updated))
		print('Energy: ' + str(E))
		print('-----------------')
		print(Fore.GREEN)
		print('stable fixed point reached')
		print(Style.RESET_ALL)
		visualization(origin, patterns)
		break
	    elif updated%10 == 0:
		print('updates:' + str(updated) + 'energy: ' + str(E))
		visualization(origin, patterns)
	    updated +=1

    elif mode == 'asynchronous':
	neurons = len(activation)
	unique_idx = set()
	value = 0
	visualization(origin, patterns)
	try:
	    while True:
		idx = random.randrange(neurons)
		if idx not in unique_idx:
		    unique_idx.add(idx)
		    patterns[idx] = np.where(np.dot(w[idx,:], patterns.T) > 0, 1, -1)
		    E = energy(patterns, w)
		    energy_list.append(E)

		    if np.array_equal(origin, patterns) and len(unique_idx) == neurons:
			print('num of updates: ' + str(value))
			print('Energy: ' + str(E))	
			visualization(origin, patterns)
			break

		    elif len(unique_idx) == neurons:
			updated += 1
			print('updates: ' + str(updated - 1))
			print('Energy: ' + str(E))
			unique_idx.clear()

	except KeyboardInterrupt:
	    fig2 = plt.figure()
	    ax2 = fig2.add_subplot(111)
	    plt.plot(energy_list)
	    plt.xlim([0,len(energy_list)])
	    plt.xlabel('iterations')
	    plt.ylabel('energy')
	    plt.show()
	





def main():
    data = generate_data()
    learn = data[:3]
    image = create_noise(learn[2], numDis=256)
    W = create_weights(np.copy(learn))
    update(W, np.copy(data[2]), np.copy(image), mode = 'synchronous')


main()



