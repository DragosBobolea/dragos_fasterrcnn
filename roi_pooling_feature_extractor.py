import tensorflow as tf
keras = tf.keras

class RoiPoolingFeatureExtractor(keras.Model):
    def __init__(self, feature_map, roi_pool_size):
        self.feature_map = feature_map
        self.roi_pool_size = roi_pool_size
        
        self.fc1 = keras.layers.Conv2D(filters=2048,kernel_size=roi_pool_size,activation='relu', padding='valid')
        self.fc2 = keras.layers.Conv2D(filters=2048,kernel_size=1,activation='relu', padding='valid')

    def call(self, input, training=False):
        '''
        input: (-1,4) tensor of boxes in yxyx coordinates
        '''
        cropped = tf.image.crop_and_resize(feature_map, input, self.roi_pool_size, method='bilinear')
        x = self.fc1(cropped)
        x = self.fc2(x)

        return x
