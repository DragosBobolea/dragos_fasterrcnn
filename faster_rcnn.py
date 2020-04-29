import tensorflow as tf
keras = tf.keras
from box_predictor import BoxPredictor
from roi_pooling_feature_extractor import RoiPoolingFeatureExtractor

class FasterRCNN(keras.Model):
    def __init__(self, backbone, rpn, num_labels):
        self.num_labels = num_labels
        self.roi_pooling_size = 7
        self.backbone = backbone
        self.rpn = rpn
        
        self.feature_map = self.get_feature_map()        

        self.roi_pooling = RoiPoolingFeatureExtractor(self.feature_map, self.roi_pooling_size)
        self.box_predictor = BoxPredictor(self.num_labels)
        # hyper-parameters

        # layers

    def call(self, input, training = False):
        features = self.get_feature_map()(input)

        rpn_output = self.rpn(features)
        if training:
            roi_pooled = self.roi_pooling(rpn_output)
            output = self.box_predictor(roi_pooled)
        else:
            roi_pooled = self.roi_pooling(rpn_output)
            output = self.box_predictor(roi_pooled)



    def get_feature_map(self):
        return self.backbone.outputs[0]