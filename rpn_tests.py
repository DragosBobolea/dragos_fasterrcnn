import os
import numpy as np
from rpn_builder import RegionProposalNetwork
from helpers import intersection_over_union
import unittest
import cv2
import tensorflow as tf
keras = tf.keras

class RpnTest(unittest.TestCase):
    def test_anchors(self):
        DEBUG = False
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

    def test_iou(self):
        boxes1 = np.array([[0, 0, 10, 10], [50, 50, 60, 60]])      
        boxes2 = np.array([ [0, 0, 5, 5], [50, 50, 60, 60], [55, 55, 65, 65] ])
        ret = intersection_over_union(boxes1, boxes2)
        assert ret[0,1] == 0
        assert ret[1,1] == 1

    def test_assign_anchors(self):
        DEBUG = True
        if DEBUG:
            image = np.ones((500,500,3))
            box_size = 60
            bounding_boxes = np.array([[100,100,100+box_size,100+box_size],[300,300,300+box_size,300+box_size]])
            for box in bounding_boxes:
                image[box[1]:box[3],box[0]:box[2]] = 0
            
            backbone = None
            scales = [0.5, 1, 2]
            ratios = [0.5, 1, 2]
            rpn = RegionProposalNetwork(backbone, scales, ratios)
            image_feature_map = np.zeros((1, image.shape[0] // rpn.stride, image.shape[1] // rpn.stride, 2048))
            anchors = rpn.generate_anchors(image_feature_map)
            positive_anchor_indices, positive_ground_truth_indices, negative_anchor_indices = rpn.assign_anchors_to_ground_truths(anchors, np.expand_dims(bounding_boxes,axis=0))
            positive_anchor_indices = np.array(positive_anchor_indices).reshape((3,-1))
            positive_ground_truth_indices = np.array(positive_ground_truth_indices).reshape((-1))
            positive_anchors = np.squeeze(anchors)[positive_anchor_indices[0],positive_anchor_indices[1],positive_anchor_indices[2]]
            positive_ground_truths = bounding_boxes[positive_ground_truth_indices]

            for anchor in positive_anchors:
                cv2.rectangle(image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (0,0,255),1)
            for anchor in positive_ground_truths:
                cv2.rectangle(image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (0,255,0),1)
            
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
