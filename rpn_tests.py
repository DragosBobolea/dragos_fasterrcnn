import os
import numpy as np
from rpn_builder import RegionProposalNetwork
from helpers import intersection_over_union, get_random_image
from bounding_box_helpers import load_bounding_boxes
import unittest
import cv2
import tensorflow as tf
keras = tf.keras
import tensorflow_datasets as tfds

@tf.function
def train_one_step(model, optimizer, loss_object, x, y):
    print('Tracing train_one_step')
    with tf.GradientTape() as tape:
        predictions = model(x)
        image_shape = tf.shape(x[0])
        loss = loss_object(y, predictions, image_shape)

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return loss

# @tf.function
def train_epoch(model, optimizer, loss_object, epochs, steps_per_epoch, dataset):
    for step in range(epochs * steps_per_epoch):
        images, labels = dataset.get_next()
        images = tf.reshape(images, [1, tf.shape(images)[1], tf.shape(images)[2], 3])
        
        labels = tf.reshape(labels, [1, -1, 4])
        loss = train_one_step(model, optimizer, loss_object, images, labels)
        tf.print(f'Step {step}: loss {loss}')
        if (step + 1) % steps_per_epoch == 0:
            model.save_weights('model_weights.ckpt')

class RpnTest(unittest.TestCase):

    def get_backbone(self):
        backbone = keras.applications.MobileNet(include_top=False,weights='imagenet')
        # weights = backbone.get_weights()
        # i = 0
        # for layer in backbone.layers:
        #     if isinstance(layer, keras.layers.Conv2D) and layer.strides == (2,2) and i < 1:
        #         layer.strides = 1
        #         # layer.dilation_rate = 2
        #         i += 1

        # backbone = keras.models.model_from_json(backbone.to_json())
        # backbone.set_weights(weights)
        return backbone

    def get_dataset(self, image_shape=(1024,768)):
        def generator():
            while True:
                image, boxes = get_random_image(image_shape)
                image = np.expand_dims(image, axis=0)
                boxes = np.expand_dims(boxes, axis=0)
                yield image, boxes

        dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32)).prefetch(10)

        return dataset
        # ds, info = tfds.load('wider_face', split='train', shuffle_files=True, with_info=True, download_and_prepare_kwargs={'download_dir':'C:\\datasets\\wider_face'})
        # return ds

    def test_dataset(self):
        DEBUG = False
        if DEBUG:
            ds = self.get_dataset()
            image, boxes = iter(ds).get_next()
            print(boxes)

    def test_anchors(self):
        DEBUG = False
        if DEBUG:
            scales = [0.125, 0.25, 0.5]
            ratios = [0.5, 1, 2]

            dataset = self.get_dataset()
            image, boxes = iter(dataset).get_next()
            batch_image = tf.expand_dims(image,axis=0)
            output_image = (np.copy(image) + 0.5) * 64
            
            backbone = keras.applications.MobileNet(include_top=False,weights='imagenet')
            rpn = RegionProposalNetwork(backbone, scales, ratios)
            feature_map = backbone(batch_image)
            anchors = rpn.generate_anchors(feature_map, tf.shape(image))
            anchors = np.array(anchors).reshape(-1,4)
            anchors[:,[0,2]] *= output_image.shape[1]
            anchors[:,[1,3]] *= output_image.shape[0]
            for anchor in anchors:
                tl = (int(anchor[0]), int(anchor[1]))
                br = (int(anchor[2]), int(anchor[3]))
                cv2.rectangle(output_image, tl, br, (255,0,0),1)
            cv2.imshow('anchors', output_image)
            cv2.waitKey(0)

    def test_iou(self):
        if False:
            boxes1 = np.array([[0, 0, 10, 10], [50, 50, 60, 60]])      
            boxes2 = np.array([ [0, 0, 5, 5], [50, 50, 60, 60], [55, 55, 65, 65] ])
            ret = intersection_over_union(boxes1, boxes2)
            assert ret[0,1] == 0
            assert ret[1,1] == 1

    def test_assign_anchors(self):
        DEBUG = False
        if DEBUG:
            scales = [0.125, 1, 1.5]
            ratios = [0.5, 1, 2]

            dataset = self.get_dataset()
            while True:
                image, boxes = iter(dataset).get_next()
                batch_image = tf.expand_dims(image,axis=0)
                output_image = (np.copy(image) + 0.5) * 64
                
                backbone = self.get_backbone()
                rpn = RegionProposalNetwork(backbone, scales, ratios)
                feature_map = backbone(batch_image)
                anchors = rpn.generate_anchors(feature_map, tf.shape(image))
                positive_anchor_indices, positive_ground_truth_indices, negative_anchor_indices = rpn.assign_anchors_to_ground_truths(anchors, np.expand_dims(boxes,axis=0))
                positive_anchor_indices = np.array(positive_anchor_indices).reshape((3,-1))
                
                positive_ground_truth_indices = np.array(positive_ground_truth_indices).reshape((-1)).tolist()
                positive_anchors = np.squeeze(anchors)[positive_anchor_indices[0],positive_anchor_indices[1],positive_anchor_indices[2]]
                positive_ground_truths = np.take(boxes, positive_ground_truth_indices,axis=0)
                positive_ground_truths = np.array(boxes)

                negative_anchor_indices = np.array(negative_anchor_indices).reshape((3,-1)).tolist()
                negative_anchors = np.squeeze(anchors)[negative_anchor_indices[0],negative_anchor_indices[1],negative_anchor_indices[2]]
            
                # debug stuff
                positive_ground_truths[:,[0,2]] *= output_image.shape[1]
                positive_ground_truths[:,[1,3]] *= output_image.shape[0]
                positive_anchors[:,[0,2]] *= output_image.shape[1]
                positive_anchors[:,[1,3]] *= output_image.shape[0]
                negative_anchors[:,[0,2]] *= output_image.shape[1]
                negative_anchors[:,[1,3]] *= output_image.shape[0]
                
                for anchor in positive_anchors:
                    cv2.rectangle(output_image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (255,0,0),2)
                for anchor in positive_ground_truths:
                    cv2.rectangle(output_image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (0,255,0),1)
                for anchor in negative_anchors[:100]:
                    cv2.rectangle(output_image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (0,0,255),1)
                cv2.imshow('anchors', output_image)
                cv2.waitKey(0)

    def test_get_minibatch(self):
        DEBUG = False
        if DEBUG:
            scales = [0.125, 1, 1.5]
            ratios = [0.5, 1, 2]

            dataset = self.get_dataset()
            while True:
                image, boxes = iter(dataset).get_next()
                batch_image = tf.expand_dims(image,axis=0)
                output_image = (np.copy(image) + 0.5) * 64
                
                backbone = self.get_backbone()
                rpn = RegionProposalNetwork(backbone, scales, ratios)
                feature_map = backbone(batch_image)
                anchors = rpn.generate_anchors(feature_map, tf.shape(image))
                positive_anchor_indices, positive_ground_truth_indices, negative_anchor_indices = rpn.generate_minibatch(anchors, np.expand_dims(boxes,axis=0))
                positive_anchor_indices = np.array(positive_anchor_indices).reshape((3,-1))
                
                positive_ground_truth_indices = np.array(positive_ground_truth_indices).reshape((-1)).tolist()
                positive_anchors = np.squeeze(anchors)[positive_anchor_indices[0],positive_anchor_indices[1],positive_anchor_indices[2]]
                positive_ground_truths = np.take(boxes, positive_ground_truth_indices,axis=0)
                positive_ground_truths = np.array(boxes)

                negative_anchor_indices = np.array(negative_anchor_indices).reshape((3,-1)).tolist()
                negative_anchors = np.squeeze(anchors)[negative_anchor_indices[0],negative_anchor_indices[1],negative_anchor_indices[2]]
            
                # debug stuff
                positive_ground_truths[:,[0,2]] *= output_image.shape[1]
                positive_ground_truths[:,[1,3]] *= output_image.shape[0]
                positive_anchors[:,[0,2]] *= output_image.shape[1]
                positive_anchors[:,[1,3]] *= output_image.shape[0]
                negative_anchors[:,[0,2]] *= output_image.shape[1]
                negative_anchors[:,[1,3]] *= output_image.shape[0]
                
                for anchor in positive_anchors:
                    cv2.rectangle(output_image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (255,0,0),2)
                for anchor in positive_ground_truths:
                    cv2.rectangle(output_image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (0,255,0),1)
                for anchor in negative_anchors:
                    cv2.rectangle(output_image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (0,0,255),1)
                cv2.imshow('anchors', output_image)
                cv2.waitKey(0)
        

    def test_targets(self):
        DEBUG = False
        if DEBUG:
            image_shape = (1024,1024)
            scales = [0.5, 1, 2]
            ratios = [0.5, 1, 2]
            image, boxes = get_random_image(image_shape)
            image_batch = np.expand_dims(image.astype(np.float32),axis=0)
            backbone = keras.applications.ResNet50(include_top=False,weights='imagenet')
            rpn = RegionProposalNetwork(backbone, scales, ratios)
            image_feature_map = backbone(image_batch)
            rpn_output = rpn.call(image_feature_map)

    def test_rpn_loss(self):
        DEBUG = False
        if DEBUG:
            image = np.ones((500,500,3))
            box_size = 60
            bounding_boxes = np.array([[100,100,100+box_size,100+box_size],[300,300,300+box_size,300+box_size]])
            for box in bounding_boxes:
                image[box[1]:box[3],box[0]:box[2]] = 0

            scales = [0.5, 1, 2]
            ratios = [0.5, 1, 2]
            image_batch = image.reshape(1,500,500,3)
            backbone = keras.applications.ResNet50(include_top=False)
            rpn = RegionProposalNetwork(backbone, scales, ratios)
            image_feature_map = backbone(image_batch)
            rpn_output = rpn.call(image_feature_map)
            loss = rpn.rpn_loss(np.array([bounding_boxes]), rpn_output)

    def test_rpn(self):
        DEBUG = True
        if DEBUG:
            load_existing = True
            scales = [0.125, 1, 1.5]
            # ratios = [2]
            ratios = [0.5, 1, 2]

            dataset = iter(self.get_dataset())
            backbone = self.get_backbone()
            rpn = RegionProposalNetwork(scales, ratios)

            model = keras.Sequential([backbone, rpn])

            if True:
                try:
                    model.load_weights('model_weights.ckpt')
                except Exception as ex:
                    print('Cannot load weights')
            loss_object = rpn.rpn_loss
            optimizer = keras.optimizers.SGD(lr=0.003)
            EPOCHS = 10
            STEPS_PER_EPOCH = 200
            train_epoch(model, optimizer, loss_object, EPOCHS, STEPS_PER_EPOCH, dataset)

            while True:
                image, boxes = dataset.get_next()
                output_image = np.squeeze((np.copy(image) + 0.5) * 64)
                predicted_boxes = rpn.get_boxes(rpn.call(backbone(image)), tf.shape(image))
                predicted_boxes = np.array(predicted_boxes)
                predicted_boxes[:,[0,2]] *= output_image.shape[1]
                predicted_boxes[:,[1,3]] *= output_image.shape[0]
                
                for anchor in predicted_boxes:
                    cv2.rectangle(output_image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (255,0,0),2)
                cv2.imshow('anchors', output_image)
                cv2.waitKey(0)
    
    
if __name__ == '__main__':
    unittest.main()
