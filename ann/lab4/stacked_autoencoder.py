import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, Callback, EarlyStopping
from keras import optimizers, losses, initializers, metrics
from colorama import Fore, Style
from keras.utils.vis_utils import plot_model

class PlotLosses(Callback):
    def on_train_begin(self,logs={}):
	self.i = 0
	self.x = []
	self.losses = []
	self.val_losses = []
	self.fig = plt.figure(1)
	self.logs = []

    def on_epoch_end(self,epoch, logs = {}):
	self.logs.append(logs)
	self.x.append(self.i)
	self.losses.append(logs.get('loss'))
	self.val_losses.append(logs.get('val_loss'))
	self.i += 1

	plt.plot(self.x, self.losses, 'b', label="loss")
	plt.plot(self.x, self.val_losses, 'r', label="val_loss")
	plt.legend(['loss','val_loss'])
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Loss evolution')

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
	self.losses = []

    def on_epoch_end(self, epoch, logs = {}):
	self.losses.append(logs.get('val_loss'))

def load_data(filename):
    with open(filename, 'r') as f:
	rows = csv.reader(f)
	ls = list(rows)
	ls = [[int(i) for i in row] for row in ls]
    return ls

def visualization(x, image, n):
    plt.figure(2,figsize=(n,5))
    i = 0
    count = 0
    for i in range(n):
	ax = plt.subplot(2,n,i+1)
	plt.imshow(np.reshape(x[i],(28,28)))	
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(2,n,i+1 + n)
	plt.imshow(np.reshape(x[i],(28,28)))	
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	i += 1
    plt.suptitle('original vs reconstructed')


def weight_visualization(weights):
    plt.figure(3,figsize=(10,6))
    weight_list = weights.tolist()
    for i in range(len(weight_list)):
	ax = plt.subplot(10,15,i+1)
	plt.imshow(np.reshape(weight_list[i], (28,28)))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

    plt.suptitle('final weights for each hidden unit')

def create_autoencoder(image_dim, encoded_list, decoded_list):
    input_image = Input(shape=(image_dim,))
    weight_init = initializers.RandomNormal(mean=0,stddev=0.001, seed=None)
    encoded = input_image
    for i in range(len(encoded_list)):
	if i == len(encoded_list)-1:
	    encoded = Dense(encoded_list[i],activation='relu',use_bias=True, kernel_initializer=weight_init,
		bias_initializer='zeros',kernel_regularizer=None, bias_regularizer=None,
		activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(encoded)

    	    decoded = Dense(decoded_list[i], activation='sigmoid', use_bias=True, kernel_initializer=weight_init,
		bias_initializer='zeros',kernel_regularizer=None, bias_regularizer=None,
		activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(encoded)
	else:
	    
    	    encoded = Dense(encoded_list[i],activation='relu',use_bias=True, kernel_initializer=weight_init,
		bias_initializer='zeros',kernel_regularizer=None, bias_regularizer=None,
		activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(encoded)

    	    decoded = Dense(decoded_list[i], activation='sigmoid', use_bias=True, kernel_initializer=weight_init,
		bias_initializer='zeros',kernel_regularizer=None, bias_regularizer=None,
		activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(encoded)

    autoencoder = Model(input_image, decoded)
    encoder = Model(input_image, encoded)
    encoded_input = Input(shape=(encoded_list[-1],))
    decoded_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoded_layer(encoded_input))

    return autoencoder, encoder, decoder

def train_test(x_train, x_test, loss_f, encoded_list, decoded_list, opt, epochs, batch_size, earlystop):
    autoencoder, encoder, decoder = create_autoencoder(len(x_train[0]), encoded_list, decoded_list)
    autoencoder.compile(optimizer=opt, loss=loss_f, metrics=[metrics.binary_accuracy])

    plot_losses = PlotLosses()
    history = LossHistory()

    if earlystop == 'on':
	earlyStopping = EarlyStopping(monitor='val_loss', patience= 0, verbose = 0, mode= 'auto')
	autoencoder.fit(x_train, x_train, batch_size, epochs, shuffle=True,
		validation_data=(x_test,x_test), callbacks=[plot_losses,earlyStopping, history])

    else:
	autoencoder.fit(x_train, x_train, batch_size, epochs, shuffle=True,
		validation_data=(x_test, x_test), callbacks=[plot_losses, history])

    optimal_val_loss = history.losses[-1]
    final_weights = autoencoder.layers[-1].get_weights()[0]
    encoded_im = encoder.predict(x_test)
    decoded_im = decoder.predict(encoded_im)

    return decoded_im, final_weights, optimal_val_loss


def main():
    mode = input('For training press 1, For param searching press 2: ')

    x_train = load_data("bindigit_trn.csv")
    y_train = load_data("targetdigit_trn.csv")
    x_test = load_data("bindigit_tst.csv")
    y_test = load_data("targetdigit_tst.csv")

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2 = 0.999, epsilon=1e-8, decay = 0)
    loss_f = losses.mean_squared_error

    ##change only these variables
    img_dim = len(x_train[0])
    encoded_list = [250,100,50]
    decoded_list = [100, 250, img_dim]
    opt = adam
    epochs = 500
    batch_size = 250
    earlystop = 'off'

    if mode == 1:
	decoded_im, final_weights, optimal_val_loss= train_test(np.array(x_train), np.array(x_test),loss_f, encoded_list, decoded_list, opt, epochs, batch_size, earlystop)

    elif mode == 2:
	unit_list = [50, 75, 100, 150]
	val_loss_list = []
	for units in unit_list:
	    decoded_im, final_weights, optimal_val_loss = train_test(np.array(x_train), np.array(x_test), loss_f, encoded_list, decoded_list, opt, epochs, batch_size, earlystop)
	    val_loss_list.append(optimal_val_loss)

	print(val_loss_list)
	best_model_idx = np.argmin(val_loss_list)
	best_model_loss = val_loss_list[best_model_idx]
	best_model_units = unit_list[best_model_idx]

	print('Best model:')
	print("Nodes = " + str(best_model_units))
	print("Loss = " + str(best_model_loss))

    else:
	print("wrong input")
	raise SystemExit


    visualization(x_test, decoded_im, n=20)
    weight_visualization(final_weights) 
    plt.show()


main()

