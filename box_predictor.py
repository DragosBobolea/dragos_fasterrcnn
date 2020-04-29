import tensorflow as tf
keras = tf.keras

class BoxPredictor(keras.Model):
    def __init__(self, num_classes):

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

    def loss(self, ground_truths, output, image_shape):
        ground_truth_classification = ground_truths[:,:self.num_classes]
        ground_truth_regression = ground_truths[:,self.num_classes:]

        output_classification = output[:,:self.num_classes]

        classification_loss = tf.losses.binary_crossentropy(output_classification, ground_truth_classification)
        return tf.reduce_mean(classification_loss)
        
