#Loading the saved_model
import tensorflow as tf
import pytesseract
import time
import os
import glob
from matplotlib import pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

IMAGE_SIZE = (12, 8) # Output display size as you want
PATH_TO_SAVED_MODEL='./content/inference_graph/saved_model'
print('Loading model...', end='')
input_image = './test/test_data_1.jpg'
file_name = os.path.basename(input_image)[0:-4]

# Load saved model and build the detection function
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done.')

#Loading the label_map
category_index=label_map_util.create_category_index_from_labelmap('graph.pbtxt',use_display_name=False)

class Detection():
	def __init__(self):
		image_path = np.array(Image.open(input_image))
		image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
		image_np = np.array(image)

		# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
		input_tensor = tf.convert_to_tensor(image_np)
		# The model expects a batch of images, so add an axis with `tf.newaxis`.
		input_tensor = input_tensor[tf.newaxis, ...]
		detections = detect_fn(input_tensor)

		# All outputs are batch tensors.
		# Convert to NumPy arrays, and take index [0] to remove the batch dimension.
		# We are only interested in the first num_detections.

		num_detections = int(detections.pop('num_detections'))
		detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
		detections['num_detections'] = num_detections

		# detection_classes should be ints.
		detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
		image_np_with_detections = image_np.copy()

		self.image_np_with_detections = image_np_with_detections
		self.num_detections = detections['num_detections']
		self.classes = detections['detection_classes']
		self.boxes = detections['detection_boxes']
		self.scores = detections['detection_scores']
		self.image = image
		

def callDetection():
	return Detection()


def main():
	image_tensors = callDetection()
	image_np_with_detections = image_tensors.image_np_with_detections
	boxes = image_tensors.boxes
	classes = image_tensors.classes
	scores = image_tensors.scores
	image = image_tensors.image
	extractText(image, file_name)
	cropImage(image_np_with_detections, boxes, classes, scores)
	drawVisualization(image_np_with_detections, boxes, classes, scores, file_name)

def cropImage(image_np_with_detections, boxes, classes, scores):
	final_score = np.squeeze(scores)
	min_score_thresh = 0.50
	no_of_objects_detected = 0
	for i in range(len(np.squeeze(scores))):
		if scores is None or final_score[i] > min_score_thresh:
			no_of_objects_detected = no_of_objects_detected + 1

	class_names_detected = []
	for i in range(len(scores)):
		if(scores[i]>0.50):
			classes_detected = category_index.get(classes[i])
			class_names_detected.append(classes_detected)
	
	(frame_height, frame_width) = image_np_with_detections.shape[:2]
	count = 0
	cropped_images = []

	for i in range(no_of_objects_detected):
		class_name = class_names_detected[i]['name']
		ymin = int((np.squeeze(boxes)[i][0]*frame_height))
		xmin = int((np.squeeze(boxes)[i][1]*frame_width))
		ymax = int((np.squeeze(boxes)[i][2]*frame_height))
		xmax = int((np.squeeze(boxes)[i][3]*frame_width))
		
		cropped_img = np.array(image_np_with_detections[ymin:ymax,xmin:xmax])
		cropped_images.append(cropped_img)
		count += 1
		print("Extracting charts...")
		cv2.imwrite('output/' + file_name + '_' + class_name + str(count) + '.jpg', cropped_img)

	return cropped_images


def drawVisualization(image_np_with_detections, boxes, classes, scores, file_name):
	viz_utils.visualize_boxes_and_labels_on_image_array(
	      image_np_with_detections,
	      boxes,
	      classes,
	      scores,
	      category_index,
	      use_normalized_coordinates=True,
	      line_thickness=8,
	      max_boxes_to_draw=200,
	      min_score_thresh=.50, # Adjust this value to set the minimum probability boxes to be classified as True
	      agnostic_mode=False)

	cv2.imwrite('./output/' + file_name + '_detected.jpg', image_np_with_detections)
	return image_np_with_detections

def extractText(image, file_name):
	text = pytesseract.image_to_string(image)
	with open('output/' + file_name + '_text.txt', mode = 'w') as f:
	    f.write(text)

if __name__ == '__main__':
	main()