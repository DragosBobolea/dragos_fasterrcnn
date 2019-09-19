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
        self.positive_iou_threshold = 0.5
        self.negative_iou_threshold = 0.3
        self.batch_size = 256
        self.positives_ratio = 0.5
        self.minibatch_positives_number = int(self.positives_ratio * self.batch_size)
        self.minibatch_negatives_number = self.batch_size - self.minibatch_positives_number
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

    '''
    Generates a random minibatch from positive and negative anchors
        Args:
        anchors: tensor of shape (1, height, width, num_anchors, 4)
        ground_truths: tensor of shape (1, None, 4)
    Returns:
        positive_anchor_indices: tensor of shape (1, 3, num_positive_anchors)
            Note: second dimension has indices for dimensions (height, width, num_anchors)
        positive_gt_indices: tensor of shape (1, num_positive_anchors)
        negative_anchor_indices: tensor of shape (1, 3, num_negative_anchors)
            Note: second dimension has indices for dimensions (height, width, num_anchors)
    '''
    @tf.function
    def generate_minibatch(self, anchors, ground_truths):
        positive_anchor_indices, positive_gt_indices, negative_anchor_indices = self.assign_anchors_to_ground_truths(anchors, ground_truths)
        n_positives = tf.minimum(tf.shape(positive_anchor_indices)[2], self.minibatch_positives_number)
        n_negatives = tf.minimum(tf.shape(negative_anchor_indices)[2], self.batch_size - n_positives)
        
        indices = tf.range(tf.shape(positive_anchor_indices)[2])
        indices = tf.random.shuffle(indices)
        indices = tf.slice(indices, [0], [n_positives])

        positive_anchor_indices = tf.gather(positive_anchor_indices, indices,axis=2)
        positive_gt_indices = tf.gather(positive_gt_indices, indices,axis=1)

        indices = tf.range(tf.shape(negative_anchor_indices)[2])
        indices = tf.random.shuffle(indices)
        indices = tf.slice(indices, [0], [n_negatives])
        negative_anchor_indices = tf.gather(negative_anchor_indices, indices,axis=2)

        return positive_anchor_indices, positive_gt_indices, negative_anchor_indices
        

    '''
    Computes targets for box regression
    ''' 
    def get_targets(self, anchors, ground_truths, positive_anchor_indices, positive_gt_indices, negative_anchor_indices):
        positive_anchors = tf.gather_nd(anchors[0],tf.transpose(positive_anchor_indices[0]))
        positive_anchors_coords = tf.unstack(positive_anchors, axis=1)
        positive_anchors_left = positive_anchors_coords[0]
        positive_anchors_top = positive_anchors_coords[1]
        positive_anchors_right = positive_anchors_coords[2]
        positive_anchors_bottom = positive_anchors_coords[3]
        positive_anchors_x = (positive_anchors_left + positive_anchors_right) / 2
        positive_anchors_y = (positive_anchors_top + positive_anchors_bottom) / 2
        positive_anchors_w = (positive_anchors_right - positive_anchors_left)
        positive_anchors_h = (positive_anchors_bottom - positive_anchors_top)

        ground_truths_coords = tf.gather(ground_truths[0], positive_gt_indices[0])
        ground_truths_coords = tf.unstack(ground_truths_coords, axis=1)
        ground_truths_left = ground_truths_coords[0]
        ground_truths_top = ground_truths_coords[1]
        ground_truths_right = ground_truths_coords[2]
        ground_truths_bottom = ground_truths_coords[3]
        ground_truths_x = (ground_truths_left + ground_truths_right) / 2
        ground_truths_y = (ground_truths_top + ground_truths_bottom) / 2
        ground_truths_w = (ground_truths_right - ground_truths_left)
        ground_truths_h = (ground_truths_bottom - ground_truths_top)

        target_x = (ground_truths_x - positive_anchors_x) / positive_anchors_w
        target_y = (ground_truths_y - positive_anchors_y) / positive_anchors_h
        target_w = tf.math.log(ground_truths_w / positive_anchors_w)
        target_h = tf.math.log(ground_truths_h / positive_anchors_h)
        target_boxes = tf.stack([target_x, target_y, target_w, target_h])
        

        return target_boxes
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
        feature_map_shape = tf.shape(feature_map)
        assert feature_map.shape[0] == 1
        vertical_stride = tf.range(0,feature_map_shape[1])
        vertical_stride = tf.tile(vertical_stride,[feature_map_shape[2]])
        vertical_stride = tf.reshape(vertical_stride, (feature_map_shape[2], feature_map_shape[1]))
        vertical_stride = tf.transpose(vertical_stride)

        horizontal_stride = tf.range(0,feature_map_shape[2])
        horizontal_stride = tf.tile(horizontal_stride, [feature_map_shape[1]])
        horizontal_stride = tf.reshape(horizontal_stride, (feature_map_shape[1], feature_map_shape[2]))

        centers_xyxy = tf.stack([horizontal_stride, vertical_stride, horizontal_stride, vertical_stride], axis=2)

        centers_xyxy = self.stride * centers_xyxy
        centers_xyxy = tf.cast(centers_xyxy,tf.float32)

        centers_xyxy = tf.tile(centers_xyxy,[1,1,self.anchor_templates.shape[0]])
        centers_xyxy = tf.reshape(centers_xyxy, (feature_map_shape[1], feature_map_shape[2], self.anchor_templates.shape[0], 4))
        anchors = centers_xyxy + self.anchor_templates
        anchors = tf.expand_dims(anchors,axis=0)
        return anchors

    '''
    Creates masks for positive and negative anchors to be used for training
    Args:
        anchors: tensor of shape (1, height, width, num_anchors, 4)
        ground_truths: tensor of shape (1, None, 4)
    Returns:
        positive_anchor_indices: tensor of shape (1, 3, num_positive_anchors)
            Note: second dimension has indices for dimensions (height, width, num_anchors)
        positive_gt_indices: tensor of shape (1, num_positive_anchors)
        negative_anchor_indices: tensor of shape (1, 3, num_negative_anchors)
            Note: second dimension has indices for dimensions (height, width, num_anchors)
    '''
    @tf.function
    def assign_anchors_to_ground_truths(self, anchors, ground_truths):
        anchors = tf.cast(anchors, tf.float32)
        ground_truths = tf.cast(ground_truths, tf.float32)
        flattened_ground_truths = tf.reshape(ground_truths, (-1,4))
        flattened_anchors = tf.reshape(anchors, (-1,4))
        ious = intersection_over_union(flattened_anchors, flattened_ground_truths)
        
        # (ii) anchors with IoU > threshold with any ground truth
        max_iou_per_anchor = tf.reduce_max(ious,axis=1)
        positive_anchors = tf.greater_equal(max_iou_per_anchor, self.positive_iou_threshold)
        ground_truth_per_anchor = tf.argmax(ious,axis=1)
        positive_anchor_indices_flattened = tf.where(positive_anchors)
        if positive_anchor_indices_flattened.shape[0] == 0:
            # (i) anchor with highest IoU, in case no anchors have ground truth IoU above threshold
            positive_anchor_indices_flattened = tf.argmax(max_iou_per_anchor)
        positive_anchor_indices_flattened = tf.reshape(positive_anchor_indices_flattened, [-1])
        positive_gt_indices = tf.gather(ground_truth_per_anchor, positive_anchor_indices_flattened)
        positive_gt_indices = tf.expand_dims(positive_gt_indices, axis=0)
        positive_anchor_indices = tf.unravel_index(positive_anchor_indices_flattened, (anchors.shape[1], anchors.shape[2], anchors.shape[3]))
        positive_anchor_indices = tf.expand_dims(positive_anchor_indices, axis=0)

        negative_anchors = tf.less_equal(max_iou_per_anchor, self.negative_iou_threshold)
        negative_anchor_indices_flattened = tf.where(negative_anchors)
        negative_anchor_indices_flattened = tf.reshape(negative_anchor_indices_flattened, [-1])
        negative_anchor_indices = tf.unravel_index(negative_anchor_indices_flattened, (anchors.shape[1], anchors.shape[2], anchors.shape[3]))
        negative_anchor_indices = tf.expand_dims(negative_anchor_indices, axis=0)
        return positive_anchor_indices, positive_gt_indices, negative_anchor_indices
    