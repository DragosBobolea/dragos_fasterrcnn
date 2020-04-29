import tensorflow as tf
keras = tf.keras
from helpers import intersection_over_union

class BoxPredictor(keras.Model):
    def __init__(self, num_classes):
        self.minibatch_size = 64
        self.positives_ratio = 0.25
        self.minibatch_positives_number = self.minibatch_size * self.positives_ratio
        self.positive_iou_threshold = 0.5
        self.negative_iou_threshold = 0.3
        self.loss_classification_weight = 1
        self.loss_regression_weight = 2
        self.num_classes = num_classes

        self.box_regression = keras.layers.Conv2D(filters = 4 * num_classes, kernel_size = 1, activation = 'relu', padding='valid')
        self.box_classification = keras.layers.Conv2D(filters = num_classes, kernel_size = 1, activation = 'relu', padding='valid')

    def call(self, box_features, training=False):
        '''
        input: (-1,4) tensor of boxes in yxyx coordinates
        '''
        output_regression = self.box_regression(box_features)
        output_classification = self.box_classification(box_features)
        output_classification = keras.activations.softmax(output_classification)
        output = tf.concat((output_classification, output_regression),axis=1)
        
        return output

    def loss(self, ground_truths, output, anchor_boxes):
        ground_truth_labels = ground_truths[:,:self.num_classes]
        ground_truth_boxes = ground_truths[:,self.num_classes:]
        output_classification = output[:,:self.num_classes]
        output_regression = output[:,self.num_classes:]
        
        positive_anchor_indices, positive_gt_indices, negative_anchor_indices = self.generate_minibatch(anchor_boxes, ground_truth_boxes)
        classification_indices = tf.concat(positive_anchor_indices, negative_anchor_indices)
        output_classification = tf.gather(output_classification, classification_indices)
        classification_targets = tf.gather(ground_truth_labels, positive_gt_indices)
        classification_targets = tf.concat(classification_targets, tf.zeros_like(negative_anchor_indices))
        classification_targets = tf.one_hot(classification_targets, self.num_classes)
        classification_loss = tf.losses.binary_crossentropy(output_classification, classification_targets)


        positive_gt_boxes = tf.gather(ground_truth_boxes, positive_gt_indices)
        output_regression = tf.gather(output_regression, positive_anchor_indices)
        positive_anchor_boxes = tf.gather(anchor_boxes, positive_anchor_indices)
        regression_targets = self.get_targets(positive_gt_boxes, positive_anchor_indices)

        huber_loss = tf.keras.losses.Huber()
        regression_loss = huber_loss(output_regression, regression_targets)

        return tf.reduce_mean(classification_loss) * self.loss_classification_weight + tf.reduce_mean(regression_loss) * self.loss_regression_weight
        
    def get_minibatch(self, ground_truths, anchor_boxes):
        positive_anchor_indices, positive_gt_indices, negative_anchor_indices = self.assign_anchors_to_ground_truths(anchor_boxes, ground_truths)
        n_positives = tf.minimum(tf.shape(positive_anchor_indices)[2], self.minibatch_positives_number)
        n_negatives = self.minibatch_size - n_positives

        indices = tf.range(tf.shape(positive_anchor_indices)[0])
        indices = tf.random.shuffle(indices)[:n_positives]
        positive_anchor_indices = tf.gather(positive_anchor_indices, indices)
        positive_gt_indices = tf.gather(positive_gt_indices, indices)

        negative_anchor_indices = tf.random.shuffle(negative_anchor_indices)[:n_negatives]

        return positive_anchor_indices, positive_gt_indices, negative_anchor_indices

    def assign_anchors_to_ground_truths(self, anchor_boxes, ground_truths):
        ious = intersection_over_union(anchor_boxes, ground_truths)
        max_iou_per_anchor = tf.reduce_max(ious,axis=1)
        positive_anchors = tf.greater_equal(max_iou_per_anchor, self.positive_iou_threshold)
        ground_truth_per_anchor = tf.argmax(ious,axis=1)
        negative_anchors = tf.lower_equal(max_iou_per_anchor, self.negative_iou_threshold)

        return positive_anchor_indices, positive_gt_indices, negative_anchor_indices

    def get_targets(self, ground_truths, anchor_boxes):
        positive_anchors = anchor_boxes
        positive_anchors_coords = tf.unstack(positive_anchors, axis=1)
        positive_anchors_left = positive_anchors_coords[0]
        positive_anchors_top = positive_anchors_coords[1]
        positive_anchors_right = positive_anchors_coords[2]
        positive_anchors_bottom = positive_anchors_coords[3]
        positive_anchors_x = (positive_anchors_left + positive_anchors_right) / 2
        positive_anchors_y = (positive_anchors_top + positive_anchors_bottom) / 2
        positive_anchors_w = (positive_anchors_right - positive_anchors_left)
        positive_anchors_h = (positive_anchors_bottom - positive_anchors_top)

        ground_truths_coords = tf.unstack(ground_truths, axis=1)
        ground_truths_left = ground_truths_coords[0]
        ground_truths_top = ground_truths_coords[1]
        ground_truths_right = ground_truths_coords[2]
        ground_truths_bottom = ground_truths_coords[3]
        ground_truths_x = (ground_truths_left + ground_truths_right) / tf.cast(2,tf.float32)
        ground_truths_y = (ground_truths_top + ground_truths_bottom) / tf.cast(2,tf.float32)
        ground_truths_w = (ground_truths_right - ground_truths_left)
        ground_truths_h = (ground_truths_bottom - ground_truths_top)

        target_x = (ground_truths_x - positive_anchors_x) / positive_anchors_w
        target_y = (ground_truths_y - positive_anchors_y) / positive_anchors_h
        target_w = tf.math.log(ground_truths_w / positive_anchors_w)
        target_h = tf.math.log(ground_truths_h / positive_anchors_h)
        target_boxes = tf.stack([target_x, target_y, target_w, target_h])
        target_boxes = tf.transpose(target_boxes)
        return target_boxes