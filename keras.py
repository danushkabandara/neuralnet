# 3. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.model_selection import StratifiedKFold
import os
from keras import backend as K

K.set_image_dim_ordering('th')
 
# 4. Load pre-shuffled MNIST data into train and test sets
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
        2,   # 1 color channel, since images are only black and white
        5,  # first image dimension (vertical)
        10,  # second image dimension (horizontal)
    )

    return X, y
for num in range(1001,1021):
	print num
	print '\n'
	# here you should enter the path to your MNIST data
	path = os.path.join(os.path.expanduser('~'), 'Desktop/keras/valence.labels/oxydeoxymiddle40s/'+str(num)+'alignrightnopad.csv')
	X, y = load_mnist(path)
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
	cvscores = []
	for train, test in kfold.split(X, y):

		# 5. Preprocess input data
		X_train = X[train].reshape(X[train].shape[0], 2, 5, 10)
		X_test = X[test].reshape(X[test].shape[0], 2, 5, 10)
		X_train = X_train.astype('float32')
		X_test = X_test.astype('float32')
		X_train /= 255
		X_test /= 255
		 
		# 6. Preprocess class labels
		Y_train = np_utils.to_categorical(y[train], 10)
		Y_test = np_utils.to_categorical(y[test], 10)
		 
		# 7. Define model architecture
		model = Sequential()
		model.reset_states
		 
		model.add(Convolution2D(32, 2, 2, activation='relu', input_shape=(2,5,10)))
		model.add(Convolution2D(32, 2, 2, activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))
		 
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.25))
		model.add(Dense(10, activation='softmax'))
		 
		# 8. Compile model
		model.compile(loss='categorical_crossentropy',
			      optimizer='adam',
			      metrics=['accuracy'])
		 
		# 9. Fit model on training data
		model.fit(X_train, Y_train, 
			  batch_size=32, epochs=150, verbose=0)
		 
		# 10. Evaluate model on test data
		scores = model.evaluate(X_test, Y_test, verbose=0)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
	print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))