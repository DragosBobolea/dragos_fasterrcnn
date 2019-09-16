import os
import numpy as np
from anchors import generate_anchors, classify_anchors, generate_minibatch
from rpn_builder import build_rpn
import unittest
import cv2
import tensorflow as tf
keras = tf.keras

class RpnTest(unittest.TestCase):
    def test_rpn(self):
        DEBUG = True
        if DEBUG:
            scales = [1]
            ratios = [1]
            image = cv2.imread(os.path.join('images','1.jpg'))

            image_shape = (512, 512, 3)
            input_tensor = tf.placeholder(dtype=float, shape=(None, 512, 512, 3))
            base_model = keras.applications.ResNet50(input_tensor=input_tensor, include_top=False)
            base_model.trainable = False
            rpn = build_rpn(base_model)
        
if __name__ == '__main__':
    unittest.main()
