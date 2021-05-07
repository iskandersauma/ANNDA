import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class Params:

    def __init__(self,Dsize,eta,epochs,mu,cov):
	self.Dsize = Dsize
	self.eta = eta
	self.epochs = epochs
	self.mu = mu
	self.cov = cov

def generate_data(muA, muB, cov, bias,Dsize,vis,mode):

    if mode == 'linear':
	x1, y1 = np.random.multivariate_normal(muA[0],cov,Dsize).T
	classA = np.array([x1,y1,bias])

	x2, y2 = np.random.multivariate_normal(muB,cov,Dsize).T
	classB = np.array([x2,y2,-bias])

    elif mode == 'nonlinear':
	x1_, y1_ = np.random.multivariate_normal(mu[0],cov,int(Dsize/2)).T
	x_1, y_1 = np.random.multivariate_normal(mu[1],cov,int(Dsize/2)).T
	x1 = np.concatenate([x1_,x_1])
	y1 = np.concatenate([y1_,y_1])
	classA = np.array([x1,y1,bias])

	x2, y2 = np.random.multivariate_normal(muB,cov,Dsize).T
	classB = np.array([x2,y2,-bias])


    data = np.concatenate([classA,classB],axis=1)

    np.random.shuffle(data.T)
    patterns = data[:2,:]
    targets = data[2]

    if vis == 'on':
	plt.plot(patterns[0],patterns[1],'k.',markersize=20)
	plt.plot(x1,y1,'*r',markersize=5)
	plt.plot(x2,y2,'+g',markersize=5)
	plt.title("separble dataset")
	plt.show()

    return patterns, targets, classA, classB

def weightInit(N,M):
    return np.random.rand(M,N)

def perceptron(x,w,T,eta,epochs,classA,classB, vis):
    errors = []
    miss = []
    for i in range(epochs):
	pred = np.dot(w,x)
	pred = np.where(pred[0]>=0.,1,-1)
	error = pred-T
	update = -eta*np.dot(error,x.T)
	w+=update
	
	if vis == 'on':
	     visAnim(w,x,classA,classB)

	missclassified = np.count_nonzero(error)
	miss.append(missclassified)
	errors.append(missclassified/x.shape[1])

    plt.show(block=False)
    return update, errors, miss

def deltaRule(X,W,T,eta,epochs,classA,classB,vis):
    errors = []
    miss = []
    for i in range(epochs):
	pred = np.dot(W,X)
	delta_error = pred-T
	pred = np.where(pred[0]>=0.,1,-1)
	update = -eta*np.dot(delta_error,X.T)
	W +=update

	if vis == 'on':
	     visAnim(W,X,classA,classB)

	missclassified = np.count_nonzero(error)
	miss.append(missclassified)
	errors.append(missclassified/X.shape[1])

    plt.show(block=False)
    return update, errors, miss

def visAnim(W, X, classA, classB):
    linelenghth = np.sqrt(np.dot(W[0,:],W[0,:].T))*0.2
    plt.plot(X[0],X[1],'k.',markersize=20)
    plt.plot(classA[0],classA[1],'*r',markersize=5)
    plt.plot(classB[0],classB[1],'+g',markersize=5)
    plt.plot(np.array([-W[0,1], W[0,1]])/linelenghth,np.array([W[0,0], -W[0,0]])/linelenghth)
    plt.title("Anime of 2 classes")
    plt.autoscale(enable=False)
    plt.pause(0.01)

def main():
    parameters = Params(Dsize=100,eta = 0.0005,epochs = 20, mu = [[[-4,-2],[4,2]],[2,1]], cov = np.array([[1,0],[0,1]]))

    bias = np.ones([parameters.Dsize])
    X, T, classA, classB = generate_data(parameters.mu[0],parameters.mu[1], parameters.cov, bias, parameters.Dsize, vis= 'off', mode = 'linear')

    outputs = 1
    W = weightInit(len(X),outputs)

    #update, errors, miss = perceptron(X,W,T,parameters.eta,parameters.epochs,classA,classB,vis='on')
    update, errors, miss = deltaRule(X,W,T,parameters.eta,parameters.epochs,classA,classB,vis= 'on')

    best_error = np.argmin(errors)
    missclassed = miss[best_error]

    print('---------------------')
    print("Error rate = " '%.2f' % errors[best_error] + '%')
    print("Missclassified Points: " + str(missclassed) + " out of " + str(X.shape[1]))
    print('---------------------')
    #--- Visualize error 
    ax = plt.figure().gca()
    ax.plot(errors)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()



main()

