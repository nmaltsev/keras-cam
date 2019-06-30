from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Input, Lambda, Layer
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import keras.backend as K

class Softmax4D(Layer):
	def __init__(self, axis=-1,**kwargs):
		self.axis=axis
		super(Softmax4D, self).__init__(**kwargs)

	def build(self,input_shape):
		pass

	def call(self, x,mask=None):
		e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
		s = K.sum(e, axis=self.axis, keepdims=True)
		return e / s

	def get_output_shape_for(self, input_shape):
		return input_shape

def VGG16(weights_path=None, heatmap=False):
	model = Sequential()
	if heatmap:
		model.add(ZeroPadding2D((1,1),input_shape=(3,None,None)))
	else:
		model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	if heatmap:
		model.add(Convolution2D(4096,7,7,activation="relu",name="dense_1"))
		model.add(Convolution2D(4096,1,1,activation="relu",name="dense_2"))
		model.add(Convolution2D(1000,1,1,name="dense_3"))
		model.add(Softmax4D(axis=1,name="softmax"))
	else:
		model.add(Flatten(name="flatten"))
		model.add(Dense(4096, activation='relu', name='dense_1'))
		model.add(Dropout(0.5))
		model.add(Dense(4096, activation='relu', name='dense_2'))
		model.add(Dropout(0.5))
		model.add(Dense(1000, name='dense_3'))
		model.add(Activation("softmax",name="softmax"))

	if weights_path:
		model.load_weights(weights_path)
	return model
