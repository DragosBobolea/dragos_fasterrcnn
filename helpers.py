import cv2
import numpy as np
import random 
import copy
import tensorflow as tf

'''
Computes vectorized IoU for two arrays of boxes
Adapted from https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
Args:
    boxes1: tensor of shape (-1, 4)
    boxes2: tensor of shape (-1, 4)
Returns:
    ious: tensor of shape (boxes1.shape[0], boxes2.shape[0])
'''
@tf.function
def intersection_over_union(boxes1, boxes2):
    x11, y11, x12, y12 = tf.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = tf.split(boxes2, 4, axis=1)

    xA = tf.maximum(x11, tf.transpose(x21))
    yA = tf.maximum(y11, tf.transpose(y21))
    xB = tf.minimum(x12, tf.transpose(x22))
    yB = tf.minimum(y12, tf.transpose(y22))

    interArea = tf.maximum((xB - xA), 0) * tf.maximum((yB - yA), 0)

    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)

    iou = interArea / (boxAArea + tf.transpose(boxBArea) - interArea)
    return iou

def IoUTensorial(A: np.ndarray, B: np.ndarray, format: str = 'XYWH') -> np.ndarray:
    '''
        Computes the Intersection over Union (IoU) of the rectangles in A vs those in B
        Rectangles are all in format (left, top, width, height) or all in format (left, top, right, bottom)
        A - tensor containing rectangles
        B - tensor containing rectangles

        Returns a tensor IoU of shape (|A|, |B|), containing the IoU of each rectangle pair (a,b), where a is in A and b is in B
    '''
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float)

    assert A.shape[1] == 4 and B.shape[1] == 4
    assert format in ['XYWH', 'XYXY']
    #don't alter original data
    A = np.copy(A)
    B = np.copy(B)

    nrA = A.shape[0] # number of rectangles in A
    nrB = B.shape[0] # number of rectangles in B

    #compute ares and then convert to (left, top, right, bottom) format
    if format == 'XYWH':
        #compute areas of recangles while we still have their width and height
        A_areas = A[:, 2] * A[:, 3]
        B_areas = B[:, 2] * B[:, 3]
        #convert to (left, top, right, bottom) format
        A[:, 2] = A[:, 0] + A[:, 2] - 1
        A[:, 3] = A[:, 1] + A[:, 3] - 1
        B[:, 2] = B[:, 0] + B[:, 2] - 1
        B[:, 3] = B[:, 1] + B[:, 3] - 1
    else:
        #compute areas of recangles
        A_areas = (A[:, 2] - A[:, 0] + 1) * (A[:, 3] - A[:, 1] + 1)
        B_areas = (B[:, 2] - B[:, 0] + 1) * (B[:, 3] - B[:, 1] + 1)


    #compute sum of areas of all the pairs of rectangles
    eA_areas = np.repeat(A_areas[:,          np.newaxis, np.newaxis], nrB, 1) # shape = (nrA, nrB, 1) contains the areas of rectangles in A
    eB_areas = np.repeat(B_areas[np.newaxis, :,          np.newaxis], nrA, 0) # shape = (nrA, nrB, 1) contains the areas of rectangles in B
    sum_area = np.sum(np.concatenate([eA_areas, eB_areas], axis=2), axis=2)

    # make two tensors eA and eB so that the first dimension chooses a box in A, the second dimension chooses a box in B, the third dimension chooses box attribute
    eA = np.repeat(A[:, None, :], nrB, 1) # shape = (nrA, nrB, 4) contains the rectangles in A
    eB = np.repeat(B[None, :, :], nrA, 0) # shape = (nrA, nrB, 4) contains the rectangles in B
    # split eA and eB into halfs and perform max and min
    half_shape = eA[:, :, 0:2].shape
    ul = np.maximum(eA[:, :, 0:2].ravel(), eB[:, :, 0:2].ravel()).reshape(half_shape) #upper left corner of intersection rectangle
    br = np.minimum(eA[:, :, 2:4].ravel(), eB[:, :, 2:4].ravel()).reshape(half_shape) #bottom right corner of intersection rectangle

    w = np.clip(br[:, :, 0] - ul[:, :, 0] + 1, 0, np.Infinity) #width of the intersection rectangle
    h = np.clip(br[:, :, 1] - ul[:, :, 1] + 1, 0, np.Infinity) #height of the intersection rectangle
    I = np.clip(w * h, 0, np.Infinity) # the intersection areas
    U = sum_area.reshape(I.shape) - I # the union areas
    IoU = I / U # the IoU scores of all the rectangle pairs in A and B
    return IoU.reshape((nrA, nrB))


def get_random_image(shape):
    image = np.zeros((shape[0],shape[1],3))
    boxes = []
    n_h = 5#np.random.randint(1,10)
    n_w = 5#np.random.randint(1,10)
    h_slice = shape[0] // n_h
    w_slice = shape[1] // n_w
    for i_h in range(n_h):
        for i_w in range(n_w):
            x = i_w * w_slice + np.random.randint(1,w_slice)
            y = i_h * h_slice + np.random.randint(1,h_slice)
            x2 = 40 + x + np.random.randint(1, x - i_w * w_slice + 1)
            y2 = 40 + y + np.random.randint(1, y - i_h * h_slice + 1)
            image[y:y2, x:x2] = 1
            boxes.append([x / float(shape[1]), y/ float(shape[1]), x2/ float(shape[1]), y2/ float(shape[1])]) 
    return image, np.array(boxes)