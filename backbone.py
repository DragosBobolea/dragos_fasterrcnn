import tensorflow as tf
keras = tf.keras
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPool2D


def _resnet_v1_50_block(input, base_depth, conv1stride=1):
    x = Conv2D(base_depth, kernel_size=1, strides=conv1stride, padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(base_depth, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(base_depth * 4, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    if input.shape[3] != x.shape[3]:
        residual = Conv2D(x.shape[3], kernel_size=1, strides=conv1stride, padding='same')(input)
        residual = BatchNormalization()(residual)
        residual = ReLU()(residual)
    else:
        residual = input
    x = residual + x
    return x

def _resnet_v1_50(input):
    #block 1
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    #block 2
    for i in range(3):
        x = _resnet_v1_50_block(x, 64, conv1stride=1)

    #block 3
    x = _resnet_v1_50_block(x, 128, conv1stride=2)
    for i in range(3):
        x = _resnet_v1_50_block(x, 128, conv1stride=1)
    
    #block 4
    x = _resnet_v1_50_block(x, 256, conv1stride=2)
    for i in range(5):
        x = _resnet_v1_50_block(x, 256, conv1stride=1)
    
    #block 5
    x = _resnet_v1_50_block(x, 512, conv1stride=2)
    for i in range(2):
        x = _resnet_v1_50_block(x, 512, conv1stride=1)
    
    return x

class ResNet50(Model):
    def call(self, input, training=False):
        result = _resnet_v1_50(input)
        return result
if __name__ == '__main__':
    from helpers import get_random_image

    input = Input(shape=(None, None, 3))
    output = _resnet_v1_50(input)
    backbone = Model(inputs=[input], outputs=[output])
    i = 0
    for layer in backbone.layers:
            if isinstance(layer, keras.layers.Conv2D) and i < 2:
                layer.strides = (1,1)
                i += 1
    image, boxes = get_random_image(shape=(224,224))
    image = np.expand_dims(image, axis=0)

    result = backbone(image)
    print(result)