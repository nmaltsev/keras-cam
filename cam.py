import numpy as np
import sys
import cv2

from vgg16 import VGG16, compile_model, for_feature_detection, load_model_weights
import keras.backend as K
import tensorflow as tf

K.set_image_dim_ordering('tf')
# fix seed for reproducible results (only works on CPU, not GPU)
# seed = 9
seed = 4
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

def load_image(img_path):
	im = cv2.imread(img_path)
	im = cv2.resize(im, (224, 224)).astype(np.float32)
	im[:,:,0] -= 103.939
	im[:,:,1] -= 116.779
	im[:,:,2] -= 123.68
	im = im.transpose((2,0,1))
	im = np.expand_dims(im, axis=0)
	return im
	
def getFileName(path_s):
	items = path_s.split('/')
	file_name_s = items[-1] if len(items) > 1 else items[0] 
	items = file_name_s.split('.')
	file_name_s = items[-2] if len(items) > 1 else items[0] 
	return file_name_s

# The heatmap of the max class
# CAM - Class Activation Map
def get_cam(model, img):
	class_weights = model.layers[-1].get_weights()[0]
	final_conv_layer = model.get_layer('block5_conv3')
	get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
	[conv_outputs, predictions] = get_output([img])
	conv_outputs = conv_outputs[0, :, :, :]
	# Create the class activation map.
	cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:3])

	for i, w in enumerate(class_weights[:, 1]):
		cam += w * conv_outputs[i, :, :]
	
	print("predictions", predictions)
	cam /= np.max(cam)
	return cam

def save_cam(cam, original_img, output_path):    
	width, height, _ = original_img.shape
	cam = cv2.resize(cam, (height, width))
	heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
	heatmap[np.where(cam < 0.2)] = 0
	img = heatmap*0.5 + original_img
	cv2.imwrite(output_path, img)

   
def testCam(weights_path):
	model = VGG16(None, True)
	model = load_model_weights(model, weights_path)
	model = for_feature_detection(model)
	model = compile_model(model)

	test_images = [
		'images/dog.jpg',
		'images/cars.jpg',
		'images/dog_from_yolo.jpg'
	]

	for img_path in test_images:
		preprocessed_input = load_image(img_path)
		original_img = cv2.imread(img_path, 1)
		cam = get_cam(model, preprocessed_input)
		save_cam(cam, original_img, '{}.heatmap.jpg'.format(getFileName(img_path)))


if __name__ == '__main__':
	weights_path = '/root/tfplayground/datasets/vgg16_weights.h5'
	testCam(weights_path)
