import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans

def square(x):
    if x < np.pi:
	return 1
    else:
	return -1

def gaussian(nominator, cov):
    return np.exp(-nominator**2/(2*cov**2))

def phi_matrix(training_sample, rbfs, cov):
    phi = np.ndarray(shape=(training_sample.shape[0],rbfs.shape[0]))
    for i in range(0, training_sample.shape[0]):
	column = np.array([gaussian(rbf - training_sample[i],cov) for rbf in rbfs])
	phi[i,:] = column
    return np.array(phi)

def RBF(training_sample,N):
    size = training_sample.shape[0]
    steps = int(size/N)
    rbfs = []
    variance = int((steps/2)+3)*0.1
    for i in range(int(steps/2), training_sample.shape[0], steps):
	rbfs.append(training_sample[i])
    
    return np.array(rbfs), variance

def random_RBF(training_sample, N):
    rbfs = np.random.choice(training_sample, size=N, replace=False)
    cov = (np.amax(rbfs) - np.amin(rbfs))/np.sqrt(2*N)

    return rbfs, cov

def kmeans_RBF(training_sample, N):
    rbfs = kmeans(training_sample, N)
    cov = (rbfs.max()-rbfs.min())/np.sqrt(2*N)

    return rbfs, cov

def train_sin_function():
    training_sample = np.transpose(np.linspace(0, 2*np.pi, num=20*np.pi))
    training_sin = np.array([np.sin(2*x) for x in training_sample])

    found = False

    validation_sample = np.transpose(np.linspace(0.05, 2*np.pi, num = 20*np.pi))
    validation_sin = np.array([np.sin(2*x) for x in validation_sample])

    for n in range(4,25):
	if found:
	    break

	rbfs, cov = kmeans_RBF(training_sample, N=n)
	phi = phi_matrix(training_sample, rbfs, cov)
	weights = np.dot(np.linalg.pinv(phi), training_sin)

	for k in range(101):
	    predictions = np.dot(phi, weights.T)
	    residual = np.sum(np.abs(training_sin - predictions))/training_sin.shape[0]

            if residual < 0.0001 and not(found):
                print("residual error fell under 0.0001 for "+str(n)+" hidden units:"+str(residual)+" after "+str(k)+" epochs")
                found = True
                plt.plot(validation_sample, predictions, label="Function approximation")
                plt.plot(validation_sample, validation_sin , label="True function")
                plt.legend(loc='upper right')
                plt.ylim(-1.5,1.5)
                plt.title("Approximating sin(2x) function for target error<0.1")
                # plt.savefig("k-means sin(2x)_error_0_1.png")
                plt.show()
                plt.clf()
                break

            
            rbfs, cov = kmeans_RBF(training_sample, N=n)
            phi = phi_matrix(training_sample, rbfs, cov)

	    weights = np.dot(np.linalg.pinv(phi), training_sin)

def train_square_function():
    training_sample = np.transpose(np.linspace(0, 2*np.pi, num=20*np.pi))
    training_square = np.array([square(2*x) for x in training_sample])
    
    found = False

    validation_sample = np.transpose(np.linspace(0.05, 2*np.pi, num = 20*np.pi))
    validation_square = np.array([square(2*x) for x in validation_sample])

    for n in range(2,101):
	if found:
	    break

	rbfs, cov = kmeans_RBF(training_sample, N=n)
	phi = phi_matrix(training_sample, rbfs, cov)
	weights = np.dot(np.linalg.pinv(phi), training_square)

	for k in range(101):
	    predictions = np.dot(phi, weights.T)
	    residual = np.sum(np.abs(training_square - predictions))/training_square.shape[0]

            if residual < 0.1 and not(found):
                print("residual error fell under 0.1 for "+str(n)+" hidden units:"+str(residual)+" after "+str(k)+" epochs")
                found = True
                plt.plot(validation_sample, predictions, label="Function approximation")
                plt.plot(validation_sample, validation_square , label="True function")
                plt.legend(loc='upper right')
                plt.ylim(-1.5,1.5)
                plt.title("Approximating square(2x) function for target error<0.1")
                # plt.savefig("k-means square(2x)_error_0_1.png")
                plt.show()
                plt.clf()
                break

            
            rbfs, cov = kmeans_RBF(training_sample, N=n)
            phi = phi_matrix(training_sample, rbfs, cov)

	    weights = np.dot(np.linalg.pinv(phi), training_square)


train_square_function()
train_sin_function()











