#!/usr/bin/env python3
"""Changes image hue"""

import tensorflow as tf


def change_hue(image, delta):
    """Changes image hue"""
    return tf.image.adjust_hue(image, delta)
