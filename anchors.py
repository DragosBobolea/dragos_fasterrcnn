import numpy as np
import tensorflow as tf
from helpers import IoUTensorial

'''
Returns array of anchors in XYXY format
return shape: (image_height/base_size, image_width/base_size, scales*ratios, 4)
'''
def generate_anchors(image_shape, scales = [1/2, 1, 2], ratios = [1/2, 1, 2], stride = 32, base_size = 64):
    
    height, width = image_shape[0], image_shape[1]
    anchors = np.zeros((height//stride, width//stride, len(scales)*len(ratios), 4))
    for scale_idx in range(len(scales)):
        scale = scales[scale_idx]
        for ratio_idx in range(len(ratios)):
            ratio = ratios[ratio_idx]
            area = base_size * base_size * scale
            anchor_width = np.sqrt(ratio * area)
            anchor_height = area / anchor_width
            for i in range(0, height//stride*stride, stride):
                for j in range(0, width//stride*stride, stride):
                    anchor = [j-anchor_width/2, i-anchor_height/2, j+anchor_width/2, i+anchor_height/2]
                    anchors[i//stride, j//stride,len(scales)*scale_idx + ratio_idx] = anchor
    return np.array(anchors)

'''
    ground_truths: np array, XYXY format, shape (n, 4)
    anchors: np array, XYXY format, shape (feature_map_height, feature_map_width, num_anchors, 4)
    We assign a positive label to two kinds of anchors: 
        (i) the anchor/anchors with the highest Intersection-overUnion (IoU) overlap with a ground-truth box, or 
        (ii) an anchor that has an IoU overlap higher than 0.7 with any ground-truth box
    returns:
    -1 for negative
    0 for ignored
    1 for positive
'''
def classify_anchors(ground_truths, anchors, positive_iou_threshold = 0.7, negative_iou_threshold = 0.3):
    ious = IoUTensorial(ground_truths, anchors,format='XYXY')
    anchor_classes = np.zeros((anchors.shape[0]))
    anchor_classes[ious.T.max(axis=1) <= negative_iou_threshold] = -1
    # (i)
    anchor_classes[ious.argmax(axis=1)] = 1
    # (ii)
    anchor_classes[ious.T.max(axis=1) >= positive_iou_threshold] = 1
    return anchor_classes


def generate_minibatch_mask(anchors, ground_truths, batch_size=256, positives_ratio=0.5, positive_iou_threshold = 0.7, negative_iou_threshold = 0.3):
    anchor_classes = classify_anchors(ground_truths,anchors.reshape(-1,4), positive_iou_threshold, negative_iou_threshold)
    anchor_classes = anchor_classes.reshape((anchors.shape[0], anchors.shape[1], anchors.shape[2]))
    
    n_positives = int(min((anchor_classes == 1).sum(), batch_size * positives_ratio))
    n_negatives = batch_size - n_positives
    positives_indices = np.argwhere(anchor_classes == 1)
    np.random.shuffle(positives_indices)
    positives_indices_batch = positives_indices[:n_positives]

    negatives_indices = np.argwhere(anchor_classes == -1)
    np.random.shuffle(negatives_indices)
    negatives_indices_batch = negatives_indices[:n_negatives]

    anchors_batch_indices = np.zeros((anchors.shape[0], anchors.shape[1], anchors.shape[2]))
    # https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays
    anchors_batch_indices[list(positives_indices_batch.T)] = 1
    anchors_batch_indices[list(negatives_indices_batch.T)] = -1
    return anchors_batch_indices, positives_indices_batch, negatives_indices_batch