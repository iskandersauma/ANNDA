import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d, Axes3D


class Params:

    def __init__(self,Dsize,eta,nodes,outputNodes, alpha, epochs, mu, cov):
	self.Dsize = Dsize
	self.eta = eta
	self.epochs = epochs
	self.mu = mu
	self.cov = cov
	self.nodes = nodes
	self.outputNodes = outputNodes
	self.alpha = alpha

def generate_data(muA, muB, cov, bias,Dsize,vis,mode):

    if mode == 'linear':
	x1, y1 = np.random.multivariate_normal(muA[0],cov,Dsize).T
	classA = np.array([x1,y1,bias])

	x2, y2 = np.random.multivariate_normal(muB,cov,Dsize).T
	classB = np.array([x2,y2,-bias])

    elif mode == 'nonlinear':
	x1_, y1_ = np.random.multivariate_normal(muA[0],cov,int(Dsize/2)).T
	x_1, y_1 = np.random.multivariate_normal(muA[1],cov,int(Dsize/2)).T
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
    return np.random.rand(N,M)

def phi(x):
    return (2/(1+np.exp(-x))) - 1

def phiPrime(x):
    return ((1+phi(x))*(1-phi(x)))/2

def forwardPass(X,W,V):
    bias = np.array([np.ones([X.shape[1]])])
    hin = np.dot(W.T,X)
    hout = phi(hin)
    hout = np.concatenate([hout,bias])
    oin = np.dot(V.T,hout)
    out = phi(oin)

    return out, oin, hin, hout

def backwardPass(X,T,W,V,nodes):
    out, oin, hin, hout = forwardPass(X,W,V)
    delta = (out - T)*phiPrime(oin)
    delta_h = np.dot(V,delta)[0:nodes,:]*phiPrime(hin)

    return delta,delta_h, hout, hin


def backPropagation(X,X_test,bias,T,T_test,W,V, a, eta, nodes, epochs, test):
    dW, dV = 0, 0
    listW = [] 
    listV = [] 
    training_error = [] 
    test_error = []

    for i in range(epochs):
	delta, delta_h, hout, out = backwardPass(X, T, W, V, nodes)
	
	dW = (dW*a) - np.dot(delta_h, X.T)*(1-a)
	dV = (dV*a) - np.dot(delta, hout.T)*(1-a)
	W = W + dW.T*eta
	V = V + dV.T*eta
	listW.append(W)
	listV.append(V)
	
	pred = np.where(out>=0,1,-1)
	training_error.append((np.sum(pred-T)**2)/2)

	if test == 'on':
	    out_test, _, _ ,_ = forwardPass(X_test, listW[i], listV[i])
	    pred = np.where(out_test>=0,1,-1)
	    test_error.append((np.sum((pred-T_test)**2))/2)

    min_error = np.argmin(training_error)
    W_best = listW[min_error]
    V_best = listV[min_error]

    return listW, listV, W_best, V_best, training_error, test_error	


def evaluate(X, T, W_best, V_best): 
    out_best, _, _, _ = forwardPass(X, W_best, V_best)
    pred = np.where(out_best >= 0., 1, -1)
    missclassed = np.count_nonzero(pred - T)
    errorRate = missclassed / X.shape[1]
    print("Error rate = " '%.2f' % errorRate + '%')
    print("Missclassified Points: " + str(missclassed) + " out of " + str(X.shape[1]))


def grid(X, classA, classB, W, V, title):
    xsize = np.arange(-8, 10, 0.01)
    ysize = np.arange(-8, 10, 0.01)
    grid = np.array(np.meshgrid(xsize, ysize))
    grid = np.array([grid[0].flatten(), grid[1].flatten(), np.ones(len(grid[0].flatten()))])

    bias = np.array([np.ones(len(grid[0].flatten()))])
    l1 = phi(np.dot(W.T, grid))
    l1 = np.concatenate([l1, bias])
    l2 = np.dot(V.T, l1)
    l2 = np.where(phi(l2) >= 0., 1, -1)
    
    plt.pcolormesh(xsize, ysize, l2.reshape((len(xsize), len(ysize))), cmap = 'gist_ncar', alpha = 0.3)
    plt.contour(xsize, ysize, l2.reshape((len(xsize), len(ysize))), levels = [0])
    plt.plot(X[0],X[1],'k.',markersize = 20)
    plt.plot(classA[0],classA[1],'*r',markersize=10)
    plt.plot(classB[0],classB[1],'+g',markersize=10)
    plt.title(title)
    plt.show()

def getFuncData(step, vis):
    x = np.arange(-5, 5.1, step)
    y = np.arange(-5, 5.1, step)
    x, y = np.array(np.meshgrid(x, y))
    z = np.exp(-(x**2.+y**2.)/10.)-0.5
    X = np.array([x.reshape(1, x.size).flatten(), y.reshape(1, y.size).flatten()])
    T = z.reshape(1, z.size).flatten()

    if vis == 'on':
        functionVis(x, y, z, title = 'Function Visualization')
    return X, T, x, y

def functionVis(x, y, out, title):
    gridsize = [len(x), len(y)]
    result = out.reshape(gridsize[0], gridsize[1])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, result, cmap = plt.cm.coolwarm, linewidth = 0)
    ax.set_zlim(np.min(result) + np.min(result) / 5, np.max(result) - np.max(result) / 5)
    ax.text2D(0.05, 0.95, title, transform = ax.transAxes)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))    
    plt.show()



def main():
    parameters = Params(Dsize = 100, eta = 0.005, nodes = 10, outputNodes = 1, alpha = 0.9, epochs = 200, mu = [[[-3, -3],[4,4]], [0, 1]], cov = np.array([[1,0], [0,1]]))
    intargets = np.ones([parameters.Dsize])
    patterns, targets, classA, classB = generate_data(parameters.mu[0], parameters.mu[1], parameters.cov, intargets, parameters.Dsize, vis = 'Off', mode = 'nonlinear' )  

    n = int(patterns.shape[1] / 3)
    X_training = np.concatenate([[patterns[0][:n], patterns[1][:n]]])
    T_training = targets[:n]
    X_test = patterns
    T_test = targets

    W = weightInit(len(X_training) + 1, parameters.nodes)
    V = weightInit(parameters.nodes + 1, parameters.outputNodes)

    bias = np.array([np.ones([X_training.shape[1]])])
    bias_test = np.array([np.ones([X_test.shape[1]])])

    X_training = np.concatenate([X_training, bias])
    X_test = np.concatenate([X_test, bias_test])

    listW, listV, W_best, V_best, training_error, test_error = backPropagation(X_training, X_test, bias_test, T_training, T_test, W, V, parameters.alpha, parameters.eta, parameters.nodes, parameters.epochs, test = 'on')

    print('Training:')
    evaluate(X_training, T_training, W_best, V_best)
    print(' Testing:')
    evaluate(X_test, T_test, W_best, V_best)

    plt.plot(training_error, 'b', label = 'Training')
    plt.plot(test_error, 'r', label = 'Testing')
    plt.legend(('Training error', 'Testing error'),
                loc='upper right')
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title('Training VS Testing error')
    plt.show()

    grid(X_test, classA, classB, W_best, V_best, title = 'Test set')


main()









