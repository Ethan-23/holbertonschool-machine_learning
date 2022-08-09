#!/usr/bin/env python3
"""Shears the image"""

import tensorflow as tf


def shear_image(image, intensity):
    """Shears the image"""
    return tf.keras.preprocessing.image.random_shear(
        x=image, intensity=intensity)
