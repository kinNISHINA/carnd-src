import rospy
import numpy as np
import tensorflow as tf
from PIL import Image
from styx_msgs.msg import TrafficLight
import yaml

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        MODEL_NAME = 'ssd_site' if self.config['is_site'] else 'ssd_styx'
        # rospy.logwarn("is_site: {0}".format(self.config['is_site']))
        PATH_TO_MODEL = 'model/' + MODEL_NAME + '/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
    
        """
        #TODO implement light color prediction
        with self.detection_graph.as_default():
            img_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        if (int(classes[0][0])==1) and (scores[0][0]>0.4):
            return TrafficLight.RED
        return TrafficLight.UNKNOWN
