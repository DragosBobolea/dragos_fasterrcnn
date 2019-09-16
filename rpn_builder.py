from anchors import generate_anchors, generate_minibatch_mask
import numpy as np
import tensorflow as tf
keras = tf.keras

class RegionProposalNetwork(keras.Model):
    def __init__(self, backbone, scales, ratios):
        super(RegionProposalNetwork, self).__init__()
        # hard-coded parameters (for now)
        self.stride = 32
        self.base_anchor_size = 64
        self.positive_iou_threshold = 0.7
        self.negative_iou_threshold = 0.3
        self.batch_size = 256
        self.positives_ratio = 0.5
        self.max_number_of_predictions = 400
        # parameters
        self.backbone = backbone
        self.scales = scales
        self.ratios = ratios

        # layers
        self.image = tf.placeholder(dtype=float, shape=(None, None, None, 3), name='image')
        self.ground_truth_boxes = tf.placeholder(dtype=float, shape=(None, None, 4), name='ground_truth_boxes')
        
        self.conv1 = keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu')
        self.box_regression = keras.layers.Conv2D(4, 1)
        self.box_classification = keras.layers.Conv2D(1,1,activation='softmax')

    def call(self, input, training=False):
        x = self.backbone(input)
        x = self.conv1(x)
        output_regression = self.box_regression(x)
        output_classification = self.box_classification(x)

        return output_regression, output_classification

    def build_loss(self, ground_truths, predictions):
        pass
        # generate anchors
        # assign anchors to predictions
        # build minibatch
        # apply loss to minibatch

    def generate_minibatch(self):
        pass
        anchors = generate_anchors(image.shape)
        anchors_batch_indices, positive_anchors_indices, negative_anchors_indices = generate_minibatch_mask(anchors, ground_truths)
    
    


