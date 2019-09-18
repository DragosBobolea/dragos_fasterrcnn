import numpy as np
import tensorflow as tf
keras = tf.keras
from helpers import intersection_over_union
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
        self.anchor_templates = self.__get_anchor_templates()

        # layers
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
    

    '''
    Generates anchor templates, in XYXY format, centered at 0
    Returns:
        anchor_templates: numpy array of shape (n_scales * n_ratios, 4)
    '''
    def __get_anchor_templates(self):
        anchor_templates = np.zeros((len(self.ratios) * len(self.scales), 4))
        for ratio_idx in range(len(self.ratios)):
            for scale_idx in range(len(self.scales)):
                ratio = self.ratios[ratio_idx]
                scale = self.scales[scale_idx]
                area = scale * self.base_anchor_size ** 2
                width = np.sqrt(ratio * area)
                height = area / width
                anchor = [-width/2, -height/2, width/2, height/2]
                anchor_templates[ratio_idx * len(self.scales) + scale_idx] = anchor

        return anchor_templates.astype(np.float32)

    '''
    Generates anchors for feature map
    Anchors are in XYXY format, in absolute pixel values
    Args:
        feature_map: tensor of shape (1, height, width, channels)
    Returns:
        anchors: tensor of shape (1, height, width, num_anchors, 4)
    '''
    @tf.function
    def generate_anchors(self, feature_map):
        # TODO support minibatch by tiling anchors on first dimension
        assert feature_map.shape[0] == 1
        vertical_stride = tf.range(0,feature_map.shape[1])
        vertical_stride = tf.tile(vertical_stride,[feature_map.shape[2]])
        vertical_stride = tf.reshape(vertical_stride, (feature_map.shape[2], feature_map.shape[1]))
        vertical_stride = tf.transpose(vertical_stride)

        horizontal_stride = tf.range(0,feature_map.shape[2])
        horizontal_stride = tf.tile(horizontal_stride, [feature_map.shape[1]])
        horizontal_stride = tf.reshape(horizontal_stride, (feature_map.shape[1], feature_map.shape[2]))

        centers_xyxy = tf.stack([horizontal_stride, vertical_stride, horizontal_stride, vertical_stride], axis=2)

        centers_xyxy = self.stride * centers_xyxy
        centers_xyxy = tf.cast(centers_xyxy,tf.float32)

        centers_xyxy = tf.tile(centers_xyxy,[1,1,self.anchor_templates.shape[0]])
        centers_xyxy = tf.reshape(centers_xyxy, (feature_map.shape[1], feature_map.shape[2], self.anchor_templates.shape[0], 4))
        anchors = centers_xyxy + self.anchor_templates
        anchors = tf.expand_dims(anchors,axis=0)
        return anchors

    '''
    Creates masks for positive and negative anchors to be used for training
    Args:
        anchors: tensor of shape (1, height, width, num_anchors, 4)
        ground_truths: tensor of shape (1, None, 4)
    Returns:
        positive_minibatch: tensor of shape (1, -1, 6) where:
            first dimension is minibatch idx
            third dimension is (anchor idx) + (delta xywh)
        negative_minibatch: tensor of shape (1,-1) where:
            first dimension is minibatch idx
            second dimension is anchor flattened id
    '''
    def assign_anchors_to_ground_truths(self, anchors, ground_truths):
        anchors = tf.cast(anchors, tf.float32)
        ground_truths = tf.cast(ground_truths, tf.float32)
        flattened_ground_truths = tf.reshape(ground_truths, (-1,4))
        flattened_anchors = tf.reshape(anchors, (-1,4))
        ious = intersection_over_union(flattened_anchors, flattened_ground_truths)

        # positive_anchor_indices_i = tf.argmax(ious,axis=1)
        # ious = tf.reshape(ious,(anchors.shape[1], anchors.shape[2], anchors.shape[3], ground_truths.shape[1]))
        
        # (ii)
        max_iou_per_anchor = tf.reduce_max(ious,axis=1)
        positive_anchors = tf.greater_equal(max_iou_per_anchor, self.positive_iou_threshold)
        ground_truth_per_anchor = tf.argmax(ious,axis=1)
        positive_anchor_indices = tf.where(positive_anchors)[0]
        positive_gt_indices = tf.gather(ground_truth_per_anchor, positive_anchor_indices)

        positive_anchor_indices_unraveled = tf.unravel_index(positive_anchor_indices, (anchors.shape[0], anchors.shape[1], anchors.shape[2], anchors.shape[3]))
        return positive_anchor_indices_unraveled, positive_gt_indices
        # find indices of anchors with maxiou > 0.7
        # for these indices, find corresponding ground truth with ious.argmax()

        # # (i)
        # anchor_classes[ious.argmax(axis=1)] = 1
        # # (ii)
        # anchor_classes[ious.T.max(axis=1) >= positive_iou_threshold] = 1
    