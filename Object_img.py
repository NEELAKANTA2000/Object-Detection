import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from utils import label_map_util
from utils import visualization_utils as vis_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#It is used to on the environment of tensorflow version 2..
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#predefined model name
MODEL_FILE = MODEL_NAME + '.tar.gz'
#tarbal .tar.gz \\\ it is similar to zip
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
#download the path from tensorflow version 
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
#saving the path to a variable 
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
#It is used to label the path.  mscoco_label_map.pbtxt ----->module in coco used for labels

NUM_CLASSES = 90
#check whether path exists ?
if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
	print ('Downloading the model')
	# for web Request and download the module we use urllib.request.URLopener()
 	opener = urllib.request.URLopener()
 	#for retrieving the module we use retrieve(Base loc + file ,file)
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	#tarfile is opened after downloading module
	tar_file = tarfile.open(MODEL_FILE)
	#for loop is used because....many number of files gets downloaded...we get what we want
	for file in tar_file.getmembers():
	  file_name = os.path.basename(file.name)
	  if 'frozen_inference_graph.pb' in file_name:
	    tar_file.extract(file, os.getcwd())
	print ('Coco Model Download complete!!!')
else:
	print ('COCO Model already exists')

#now starts tensorflow
# inference graph is loaded
detection_graph = tf.Graph()
with detection_graph.as_default():
	#Defining a graph
  od_graph_def = tf.GraphDef()
  #gfile--->graphFile
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
  	#reading it serially
    serialized_graph = fid.read()
    #convert to string
    od_graph_def.ParseFromString(serialized_graph)

    tf.import_graph_def(od_graph_def, name='')
    #Till here the graph gets loaded

#from here label map will get loaded...
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#categarize the labels
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#indexing the categories
category_index = label_map_util.create_category_index(categories)

#Images converts to numpy array
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  #3---->rgb image
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Image gets loaded from now..  
#Testing images gets loaded....
PATH_TO_TEST_IMAGES_DIR = 'C:/Users/MY LENOVO/PycharmProjects/test/object_recognition_detection/test_images/'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
#for loop is for just naming the images...

#output image initial size
IMAGE_SIZE = (12, 8)
#Detection gets started.....
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)

      image_np = load_image_into_numpy_array(image)
      #to expand dimensions...
      image_np_expanded = np.expand_dims(image_np, axis=0)
      #axis=0--->colimn wise

      #tensor-->1-d array
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      #feed_dict is uded to feed values


      #visualizing the output
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      plt.show()