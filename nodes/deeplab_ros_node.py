#!/usr/bin/env python
import os
import tempfile
import threading
from six.moves import urllib

import PIL
import numpy as np
from skimage import transform

import tensorflow as tf

from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image

from deeplab_ros import deeplab

class DeepLabNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()

        self._last_msg = None
        self._msg_lock = threading.Lock()

        rgb_input = rospy.get_param('~rgb_input', '/camera/rgb/image_color')

        rospy.Subscriber(rgb_input, Image, self._image_callback, queue_size=1)

        self.label_pub = rospy.Publisher('~segmentation', Image, queue_size=1)
        self.vis_pub = rospy.Publisher('~segmentation_viz', Image, queue_size=1)

        self._publish_rate = rospy.get_param('~publish_rate', 100)
        self._visualize = rospy.get_param('~visualize', True)

        # Valid model choices:
        #'mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval'.
        MODEL_NAME = rospy.get_param('~model', 'mobilenetv2_coco_voctrainaug')

        _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
        _MODEL_URLS = {
            'mobilenetv2_coco_voctrainaug':
                'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
            'mobilenetv2_coco_voctrainval':
                'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
            'xception_coco_voctrainaug':
                'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
            'xception_coco_voctrainval':
                'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
        }
        _TARBALL_NAME = 'deeplab_model.tar.gz'

        model_dir = tempfile.mkdtemp()
        tf.gfile.MakeDirs(model_dir)
        download_path = os.path.join(model_dir, _TARBALL_NAME)
        rospy.logdebug('downloading model, this might take a while...')
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                           download_path)
        rospy.logdebug('download completed! loading DeepLab model...')

        self.model = deeplab.DeepLabModel(download_path)
        rospy.logdebug('model loaded successfully!')

    def run(self):
        rate = rospy.Rate(self._publish_rate)

        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                rgb_image = self._cv_bridge.imgmsg_to_cv2(msg, "passthrough")

                # Run detection
                seg_map = self.detect(rgb_image)

                rospy.logdebug("Publishing semantic labels.")
                label_msg = self._cv_bridge.cv2_to_imgmsg(seg_map, 'mono16')
                label_msg.header = msg.header
                self.label_pub.publish(label_msg)

                if self._visualize:
                    # Visualize results
                    seg_image = deeplab.label_to_color_image(seg_map.astype(
                        np.int64)).astype(np.uint8)
                    label_color_msg = self._cv_bridge.cv2_to_imgmsg(seg_image, 'rgb8')
                    label_color_msg.header = msg.header
                    self.vis_pub.publish(label_color_msg)


            rate.sleep()

    def detect(self, rgb_image):
        rgb_image = PIL.Image.fromarray(rgb_image)

        resized_im, seg_map = self.model.run(rgb_image)
        target_size = rgb_image.size[::-1]
        seg_map = transform.resize(
            seg_map, target_size, order=0, preserve_range=True).astype(
                np.uint16)

        return seg_map

    def _image_callback(self, msg):
        rospy.logdebug("Got an image.")

        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()


def main():
    rospy.init_node('deeplab_ros_node')

    node = DeepLabNode()
    node.run()


if __name__ == '__main__':
    main()
