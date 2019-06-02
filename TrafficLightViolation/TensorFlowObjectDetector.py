import calendar
import time
import random

import numpy as np
import tensorflow as tf

from utils import label_map_util
from utils import visualization_utils as vis_util
from matplotlib import pyplot as plt

class TensorFlowObjectDetector:
    def __init__(self, model_name, path_to_labels, num_classes):
        self.model_name = model_name
        self.path_to_labels = path_to_labels
        self.num_classes = num_classes

        PATH_TO_CKPT = self.model_name + '/frozen_inference_graph.pb'

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.label_map = label_map_util.load_labelmap(self.path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                         max_num_classes=self.num_classes,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        with self.detection_graph.as_default():
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            self.sess = tf.Session(graph=self.detection_graph)

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = 640, 480
        return np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)

    def detectTF(self, image):
        image_np = self.load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        return (boxes, scores, classes, num)

    def closeSessionTF(self):
        self.sess.close()

    def plot_origin_image(self, image_np, boxes, classes, scores, category_index):
        IMAGE_SIZE = (12, 8)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            min_score_thresh=.5,
            use_normalized_coordinates=True,
            line_thickness=1)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        # save augmented images into hard drive
        ts = calendar.timegm(time.gmtime())
        plt.savefig('./output_images/' + str(ts) + str(random.randint(1000, 9999)) + ' .png')
        plt.close()