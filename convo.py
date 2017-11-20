mport os
import pylab


import matplotlib.pyplot as plt
import numpy as np

from lasagne import layers
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from lasagne.layers import get_all_params
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

def load_mnist(path):
    X = []
    y = []
    with open(path, 'rb') as f:
        next(f)  # skip header
        for line in f:
            yi, xi = line.split(',', 1)
            y.append(yi)
            X.append(xi.split(','))

    # Theano works with fp32 precision
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.int32)

    # apply some very simple normalization to the data
    X -= X.mean()
    X /= X.std()

    # For convolutional layers, the default shape of data is bc01,
    # i.e. batch size x color channels x image dimension 1 x image dimension 2.
    # Therefore, we reshape the X data to -1, 1, 28, 28.
    X = X.reshape(
        -1,  # number of samples, -1 makes it so that this number is determined automatically
        1,   # 1 color channel, since images are only black and white
        10,  # first image dimension (vertical)
        10,  # second image dimension (horizontal)
    )

    return X, y

for num in range(1001,1010):
	print num
	print '\n'
	# here you should enter the path to your MNIST data
	path = os.path.join(os.path.expanduser('~'), 'Desktop/processed/'+str(num)+'alignrightnopad.csv')

	X, y = load_mnist(path)

	#figs, axes = plt.subplots(4, 4, figsize=(6, 6))
	#for i in range(4):
	#    for j in range(4):
	#        axes[i, j].imshow(-X[i + 4 * j].reshape(28, 28), cmap='gray', interpolation='none')
	#        axes[i, j].set_xticks([])
	#        axes[i, j].set_yticks([])
	#        axes[i, j].set_title("Label: {}".format(y[i + 4 * j]))
	#        axes[i, j].axis('off')
	layers0 = [
	    # layer dealing with the input data
	    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

	    # first stage of our convolutional layers
	    (Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
 	    (Conv2DLayer, {'num_filters': 96, 'filter_size': 2}),

	  
	   (MaxPool2DLayer, {'pool_size': 2,'stride':1}),

	    # second stage of our convolutional layers
	   (Conv2DLayer, {'num_filters': 128, 'filter_size': 2}),
	  
	    (MaxPool2DLayer, {'pool_size': 2,'stride':1}),

	    # two dense layers with dropout
	    (DenseLayer, {'num_units': 64}),
	    (DropoutLayer,{}),
	    #('dropout', layers.DropoutLayer),
	    (DenseLayer, {'num_units': 64}),

	    # the output layer
	    (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
	]


	def regularization_objective(layers, lambda1=0., lambda2=0., *args, **kwargs):
	    # default loss
	    losses = objective(layers, *args, **kwargs)
	    # get the layers' weights, but only those that should be regularized
	    # (i.e. not the biases)
	    weights = get_all_params(layers[-1], regularizable=True)
	    # sum of absolute weights for L1
	    sum_abs_weights = sum([abs(w).sum() for w in weights])
	    # sum of squared weights for L2
	    sum_squared_weights = sum([(w ** 2).sum() for w in weights])
	    # add weights to regular loss
	    losses += lambda1 * sum_abs_weights + lambda2 * sum_squared_weights
	    return losses

	# split data.
	train_descs, test_descs,  train_acts, test_acts = train_test_split(X, y,test_size=0.3,random_state=0)
	net0 = NeuralNet(
	    layers=layers0,
	    max_epochs=100,
	    #dropout_p=0.25,
	    update=adam,
	    update_learning_rate=0.0002,

	    objective=regularization_objective,
	    objective_lambda2=0.0025,

	   
	    verbose=1,
	)


	net0.fit(train_descs, train_acts)
	pred=net0.predict(test_descs)
	 
	print(classification_report( test_acts, pred ))
	#plot_loss(net0)
	#plot_conv_weights(net0.layers_[1], figsize=(4, 4))
	x = X[0:1]
	#plot_conv_activity(net0.layers_[3], x)
	#pylab.show()

