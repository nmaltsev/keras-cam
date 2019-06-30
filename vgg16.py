from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Input, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import keras.backend as K

def VGG16(weights_path=None, without_end=False):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu')) # , name='ad1'
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='block5_conv3'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	if not without_end:
		model.add(Flatten())
		model.add(Dense(4096, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(4096, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(1000, activation='softmax'))

	if weights_path:
		model.load_weights(weights_path)

	return model

def compile_model(model):
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')
	return model

def global_average_pooling(x):
	return K.mean(x, axis = (2, 3))

def global_average_pooling_shape(input_shape):
	return input_shape[0:2]

def for_feature_detection(model):
	model.add(Lambda(global_average_pooling,output_shape=global_average_pooling_shape))
	model.add(Dense(2, activation = 'softmax', init='uniform'))
	return model

def load_model_weights(model, weights_path):
	import h5py
	print 'Loading model.'
	f = h5py.File(weights_path)
	for k in range(f.attrs['nb_layers']):
		if k >= len(model.layers):
			# we don't look at the last (fully-connected) layers in the savefile
			break
		g = f['layer_{}'.format(k)]
		weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
		model.layers[k].set_weights(weights)
		model.layers[k].trainable = False
	f.close()
	print 'Model loaded.'
	return model

