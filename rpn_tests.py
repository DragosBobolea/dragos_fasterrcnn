import os
import numpy as np
from rpn_builder import RegionProposalNetwork
import unittest
import cv2
import tensorflow as tf
keras = tf.keras

class RpnTest(unittest.TestCase):
    def test_anchors(self):
        DEBUG = True
        if DEBUG:
            scales = [0.5, 1, 2]
            ratios = [0.5, 1, 2]
            image = cv2.imread(os.path.join('images','1.jpg'))
            image_tensor = image.astype(np.float32) \
                .reshape((1,image.shape[0],image.shape[1],3))

            backbone = None
            rpn = RegionProposalNetwork(backbone, scales, ratios)
            image_feature_map = np.zeros((1, image.shape[0] // rpn.stride, image.shape[1] // rpn.stride, 2048))
            anchors = rpn.generate_anchors(image_feature_map)
            anchors = np.array(anchors).reshape(-1,4)
            for anchor in anchors:
                cv2.rectangle(image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (0,0,255),1)
            cv2.imshow('anchors', image)
            cv2.waitKey(0)
            
    def test_rpn(self):
        DEBUG = False
        if DEBUG:
            scales = [1]
            ratios = [1]
            image = cv2.imread(os.path.join('images','1.jpg'))
            image = image.astype(np.float32) \
                .reshape((1,image.shape[0],image.shape[1],3))

            backbone = keras.applications.ResNet50(include_top=False)
            rpn = RegionProposalNetwork(backbone, scales, ratios)
            image_feature_map = backbone(image)
            anchors = rpn.generate_anchors(image_feature_map)
            print(anchors)
if __name__ == '__main__':
    unittest.main()
