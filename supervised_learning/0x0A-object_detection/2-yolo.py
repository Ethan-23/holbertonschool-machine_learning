#!/usr/bin/env python3
"""0x0A. Object Detection"""

import tensorflow as tf
import numpy as np


class Yolo:
    """
    Yolo Class
    - model: the Darknet Keras model
    - class_names: a list of the class names for the model
    - class_t: the box score threshold for the initial filtering step
    - nms_t: the IOU threshold for non-max suppression
    - anchors: the anchor boxes
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path is the path to where a Darknet Keras model is stored
        classes_path is the path to where the list of class names used
            for the Darknet model, listed in order of index, can be found
        class_t is a float representing the box score threshold for the
            initial filtering step
        nms_t is a float representing the IOU threshold for non-max
            suppression
        anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes:
        - outputs is the number of outputs (predictions) made by the
            Darknet model
        - anchor_boxes is the number of anchor boxes used for each prediction
        - 2 => [anchor_box_width, anchor_box_height]
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as file:
            lines = file.readlines()
            self.class_names = []
            for cname in lines:
                self.class_names.append(cname[:-1])
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, z):
        """Sigmoid Function"""
        return 1/(1 + np.exp(-z))

    def process_outputs(self, outputs, image_size):
        """
        outputs is a list of numpy.ndarrays containing the predictions
            from the Darknet model for a single image:
        Each output will have the shape (grid_height, grid_width,
            anchor_boxes, 4 + 1 + classes)
        grid_height & grid_width => the height and width of the grid
            used for the output
        anchor_boxes => the number of anchor boxes used
        4 => (t_x, t_y, t_w, t_h)
        1 => box_confidence
        classes => class probabilities for all classes
        image_size is a numpy.ndarray containing the image’s original
            size [image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
        boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
            anchor_boxes, 4) containing the processed boundary boxes for
            each output, respectively:
        4 => (x1, y1, x2, y2)
        (x1, y1, x2, y2) should represent the boundary box relative to
            original image
        box_confidences: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 1) containing the box confidences for
            each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the box’s class
            probabilities for each output, respectively
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        for i, output in enumerate(outputs):
            anchor = self.anchors[i]
            grid_height, grid_width = output.shape[:2]
            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            box_confidence = output[..., 4]
            class_probabilities = output[..., 5:]

            sig_conf = self.sigmoid(box_confidence)
            sig_prob = self.sigmoid(class_probabilities)
            box_conf = np.expand_dims(sig_conf, axis=-1)

            box_confidences.append(box_conf)
            box_class_probs.append(sig_prob)

            b_wh = anchor * np.exp(t_wh)
            b_wh /= self.model.inputs[0].shape.as_list()[1:3]

            grid = np.tile(np.indices((grid_width, grid_height)).T,
                           anchor.shape[0]).reshape((grid_height, grid_width) +
                                                    anchor.shape)

            b_xy = (self.sigmoid(t_xy) + grid) / [grid_width, grid_height]

            b_xy1 = b_xy - (b_wh / 2)
            b_xy2 = b_xy + (b_wh / 2)

            box = np.concatenate((b_xy1, b_xy2), axis=-1)
            box *= np.tile(np.flip(image_size, axis=0), 2)

            boxes.append(box)
        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        boxes: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 4) containing the processed
            boundary boxes for
            each output, respectively
        box_confidences: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 1) containing the processed box
            confidences for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the processed
            box class probabilities for each output, respectively
        Returns a tuple of (filtered_boxes, box_classes, box_scores):
        - filtered_boxes: a numpy.ndarray of shape (?, 4) containing
            all of the filtered bounding boxes:
        - box_classes: a numpy.ndarray of shape (?,) containing the
            class number that each box in filtered_boxes predicts,
            respectively
        - box_scores: a numpy.ndarray of shape (?) containing the
            box scores for each box in filtered_boxes, respectively
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i, box in enumerate(boxes):
            scores = box_confidences[i] * box_class_probs[i]
            classes = np.argmax(scores, axis=-1)
            box_score = np.max(scores, axis=-1)
            box_class = np.argmax(scores, axis=-1)
            index = np.where(box_class > self.class_t)

            filtered_boxes.append(box[index])
            box_classes.append(box_class[index])
            box_scores.append(box_score[index])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_confidences = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return(filtered_boxes, box_confidences, box_scores)
