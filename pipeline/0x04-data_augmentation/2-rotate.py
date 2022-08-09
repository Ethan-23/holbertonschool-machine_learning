#!/usr/bin/env python3
"""Rotates an img 90 degrees"""

import tensorflow as tf


def rotate_image(image):
    """Rotates an img 90 degrees"""
    return tf.image.rot90(image)
