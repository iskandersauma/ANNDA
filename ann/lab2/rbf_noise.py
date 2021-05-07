import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans

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
    for i in range(int(steps/2), training_sample.shape[0], steps):
	rbfs.append(training_sample[i])
    rbfs = np.array(rbfs)
    variance = 0.001*(rbfs.max()-rbfs.min())/np.sqrt(2*N)    

    return np.array(rbfs), variance

def random_RBF(training_sample, N):
    rbfs = np.random.permutation(training_sample)[:N]
    cov = (np.amax(rbfs) - np.amin(rbfs))/np.sqrt(2*N)

    return rbfs, cov

def kmeans_RBF(training_sample, N):
    rbfs = kmeans(training_sample, N)
    cov = 0.1*(rbfs.max()-rbfs.min())/np.sqrt(2*N)

    return rbfs, cov

def training_data():
    np.random.seed(940325)
    training_sample = np.transpose(np.linspace(0,2*np.pi,num=20*np.pi))
    training_sin = np.array([np.sin(2*x) for x in training_sample]) + np.random.normal(0,0.1,training_sample.shape[0])
    return training_sample, training_sin

def validation_data():
    np.random.seed(19940325)
    validation_sample = np.transpose(np.linspace(0,2*np.pi,num=20*np.pi))
    validation_sin = np.array([np.sin(2*x) for x in validation_sample]) + np.random.normal(0,0.1,validation_sample.shape[0])
    return validation_sample, validation_sin


def batch():
    training_sample, training_sin = training_data()
    validation_sample, validation_sin = validation_data()
    found = False

    minimal_residual_error_val = 2
    minimal_residual_error_train = 2
    train_error = []
    validation_error = []

    for i in range(4,40):
	if found:
	    break

	rbfs, cov = kmeans_RBF(training_sample, N=i)
	phi = phi_matrix(training_sample, rbfs, cov)
	weights = np.dot(np.linalg.pinv(phi), training_sin.T)
	
	for k in range(100):
	    prediction = np.dot(phi, weights.T)
	    total_residual_val = np.sum(np.abs(validation_sin-prediction))/validation_sin.shape[0]
	    minimal_residual_error_val = min(minimal_residual_error_val, total_residual_val)
	    total_residual_train = np.sum(np.abs(training_sin - prediction))/training_sin.shape[0]
	    minimal_residual_error_train = min(minimal_residual_error_train, total_residual_train)

	    if total_residual_val < 0.1 and not(found):
                print("Total residual error fell under 0.1 for "+str(n)+"hidden layers:"+str(total_residual_error)+" after "+str(epoch)+" epochs")
                found = True
                plt.plot(training_samples, predictions, label="Function approximation")
                plt.plot(training_samples, training_sin, label="True function")
                plt.ylim(-1.5,1.5)
                plt.title("Approximating sin(2x) function with noise added for target error<0.1")
                plt.show()

	    rbfs, cov = kmeans_RBF(training_sample, N=i)
	    phi = phi_matrix(training_sample, rbfs, cov)
	    weights = np.dot(np.linalg.pinv(phi), training_sin.T)
	
	train_error.append(minimal_residual_error_train)
	validation_error.append(minimal_residual_error_val)

    print("minimum residual error on validation set is :"+str(minimal_residual_error_val))
    print("minimum residual error on training set is :"+str(minimal_residual_error_train))
    plt.plot(np.linspace(4, 40, num=36),train_error, 'r--')
    plt.plot(np.linspace(4, 40, num=36),validation_error, 'g--')
    plt.show()

def sequential():
    training_sample, training_sin = training_data()
    validation_sample, validation_sin = validation_data()
    found = False

    for i in range(4,30):
	if found:
	    break

	rbfs, cov = random_RBF(training_sample,N=i)
	full_phi = phi_matrix(training_sample, rbfs, cov)
	weights = np.dot(np.linalg.pinv(full_phi), training_sin.T)
	eta = 0.1
	for j in range(100):
	    prediction = []
	    residual_training = []
	    residual_val = []
	    for k in range(training_sample.shape[0]):
		phi = np.array([gaussian(training_sample[k] - rbf,cov) for rbf in rbfs]).T
		pred = np.dot(phi,weights)
		prediction.append(pred)
		
		dev_train = training_sin[k] - pred
		dev_val = validation_sin[k] - pred
		residual_training.append(dev_train)
		residual_val.append(dev_val)
		err = 0.5*(dev_train**2)
		add = eta*err
		weights = add*phi

	    residual_training = np.array(residual_training)
	    resigual_val = np.array(residual_val)
	    prediction = np.array(prediction)
	    absolute_err_train = np.sum(residual_training)/len(residual_training)
	    absolute_err_val = np.sum(residual_val)/len(residual_val)

	    if absolute_err_train < 0.1 and not (found):
		print("Total residual error fell under 0.1 for "+str(i)+ " hidden layers and "+str(eta) +" learning rate value:"+str(absolute_err_train)+" after "+str(j)+" epochs")
		found = True
                plt.plot(training_sample, prediction, label="Function approximation")
                plt.plot(training_sample, training_sin, label="True function")
                plt.ylim(-1.5,1.5)
                plt.title("Approximating sin(2x) function with noise added for target error<0.1 with sequential training")
                plt.show()
                print("VALIDATION ", str(absolute_err_val))





#batch()
sequential()

















