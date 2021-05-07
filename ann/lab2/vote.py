import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches

def choice():
    prob = random.uniform(0,1)
    if prob >= 0.5:
	return 1
    else:
	return -1

def read_votes():
    votes = np.ndarray(shape=(349, 31))
    stringed_data = open('votes.dat', 'r').readline()

    current_line = 0

    acc = []
    elem_index = 0
    while elem_index <len(stringed_data)-1:

        if (elem_index==len(stringed_data)-2):
            acc.append(stringed_data[elem_index])
            votes[current_line, :] = np.asarray(acc)
            break

        else:
            if (stringed_data[elem_index] !=','):

                if (stringed_data[elem_index+1] =='.'):
                    acc.append(0.5)
                    elem_index+=3
                else:
                    acc.append(float(stringed_data[elem_index]))
                    elem_index+=1
            else:
                elem_index+=1

        if len(acc)==31:
            votes[current_line, :] = np.asarray(acc)
            current_line += 1
            acc=[]

    return votes	


def read_party():

    file = open('mpparty.dat', 'r')
    acc =[]
    while(len(acc)<349):
        acc.append(int(file.readline()))
    return acc

def read_sex():

    file = open('mpsex.dat', 'r')
    acc =[]
    while(len(acc)<349):

        acc.append(int(file.readline()))

    return acc

def read_district():

    file = open('mpdistrict.dat', 'r')

    acc =[]
    while(len(acc)<349):

        acc.append(int(file.readline()))

    return acc

def create_weights():
    weights = []
    for i in range(10):
	temp = []
	for j in range(10):
	    vec = np.random.rand(1,31)
	    temp.append(vec)
	weights.append(temp)

    return weights		

def similarity(x,w):
    return np.dot((x-w[0]).T, x - w[0])

def create_visual(votes, weights, data, color, names, title):
    for votes_index in range(votes.shape[0]):
	votes_member = votes[votes_index,:]
	index_x = 0
	index_y = 0
	minimum_similarity = 99999999
	for x_index in range(0,len(weights)):
	    for y_index  in range(len(weights[x_index])):
		temp = similarity(votes_member, weights[x_index][y_index])
		if temp < minimum_similarity:
		    minimum_similarity = temp
		    index_x = x_index
		    index_y = y_index

	step_x = 1 + float(random.randrange(1,50))/100*choice()
	step_y = 1 + float(random.randrange(1,50))/100*choice()

	try:
            plt.scatter(index_x+step_x, index_y+step_y, c=color[int(data[votes_index])])
        except AttributeError:
	    print((votes_index))

    plt.xlim(0,20)
    recs = []
    for i in range(0,len(names)):
	recs.append(mpatches.Rectangle((0,0),1,1,fc=color[i]))
    plt.legend(recs,names,loc=4)
    plt.show()

def run():
    eta = 0.2
    neighbours = [5,4,3,2,1]
    votes = read_votes()
    weights = create_weights()
    parties = read_party()
    sex = read_sex()
    districts = read_district()

    for i in range(50):
	neighbour = neighbours[int(i/10)]
	for j in range(votes.shape[0]):
	    votes_member = votes[j,:]
	    minimum_similarity = 999999999
	    index_x = 0
	    index_y = 0

	    for x_k in range(1,len(weights)):
		for y_k in range(1,len(weights[x_k])):

		    temp = similarity(votes_member, weights[x_k][y_k])
		    if temp < minimum_similarity:
			minimum_similarity = temp
			index_x = x_k
			index_y = y_k

		minimum_x = max(0,index_x - neighbour)
		maximum_x = min(len(weights[0]) - 1, index_x + neighbour)
		
		minimum_y = max(0, index_y - neighbour)
		maximum_y = min(len(weights[0]) - 1, index_y + neighbour)

		for x_k in range(minimum_x, maximum_x+1):
		    weights[x_k][index_y] += eta*(votes_member - weights[x_k][index_y])

		for y_k in range(minimum_y, maximum_y+1):
		    weights[index_x][y_k] += eta*(votes_member - weights[index_x][y_k])

    color_scheme =['black', 'lightblue', 'blue', 'red', 'tomato', 'green',  'darkblue', 'darkgreen' ]
    names = ['No party', 'Moderate', 'Liberals', 'Swedish Socialist Party', 'Left Party', 'Green Party', 'Christian Democrats', 'Centre Party' ]

    create_visual(votes, weights,data=parties, color=color_scheme, names=names, title="parties.png")

run()

