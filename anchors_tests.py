import os
import numpy as np
from anchors import generate_anchors, classify_anchors, generate_minibatch_mask
import unittest
import cv2

class AnchorsTest(unittest.TestCase):
    def test_generate_anchors(self):
        DEBUG = False
        if DEBUG:
            image = cv2.imread(os.path.join('images','1.jpg'))
            anchors = generate_anchors(image.shape, scales=[1/2,2], base_size=32, stride=32)
            print(anchors.shape)
            anchors = anchors.reshape(-1,4)
            for anchor in anchors:
                cv2.rectangle(image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (0,0,255),1)
            cv2.imshow('anchors', image)
            cv2.waitKey(0)

    def test_classify_anchors(self):
        DEBUG = False
        if DEBUG:
            image = np.ones((500,500,3))
            box_size = 60
            bounding_boxes = np.array([[100,100,100+box_size,100+box_size],[300,300,300+box_size,300+box_size]])
            for box in bounding_boxes:
                image[box[1]:box[3],box[0]:box[2]] = 0
            anchors = generate_anchors(image.shape).reshape(-1,4)
            anchors_classes = classify_anchors(bounding_boxes, anchors)
            for anchor in anchors[anchors_classes == -1]:
                cv2.rectangle(image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (0,0,255),1)
            for anchor in anchors[anchors_classes == 0]:
                cv2.rectangle(image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (255,0,0),1)
            for anchor in anchors[anchors_classes == 1]:
                cv2.rectangle(image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (0,255,0),1)
            
            cv2.imshow('anchors', image)
            cv2.waitKey(0)

    def test_generate_minibatch(self):
        DEBUG = True
        if DEBUG:
            image = np.ones((500,500,3))
            box_size = 60
            bounding_boxes = np.array([[100,100,100+box_size,100+box_size],[300,300,300+box_size,300+box_size]])
            for box in bounding_boxes:
                image[box[1]:box[3],box[0]:box[2]] = 0
            anchors = generate_anchors(image.shape)
            anchors_batch_indices, _, _ = generate_minibatch_mask(anchors, bounding_boxes,batch_size=64)
            anchors = anchors.reshape(-1,4)
            anchors_indices = anchors_batch_indices.reshape(-1,)
            for anchor in anchors[anchors_indices == -1,:]:
                cv2.rectangle(image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (0,0,255),1)
            for anchor in anchors[anchors_indices == 1,:]:
                cv2.rectangle(image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (0,255,0),1)
            
            cv2.imshow('anchors', image)
            cv2.waitKey(0)

        
if __name__ == '__main__':
    unittest.main()
