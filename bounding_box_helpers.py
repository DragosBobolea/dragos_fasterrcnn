import os
import numpy as np


def load_bounding_boxes(filepath):
    with open(filepath,'r') as f:
        text = f.read()
    lines = text.split('\n')
    boxes = np.array([line.split('\t')[:4] for line in lines if line],dtype=np.int32)
    
    return boxes