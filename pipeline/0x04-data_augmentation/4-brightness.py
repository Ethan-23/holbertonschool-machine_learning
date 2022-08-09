#!/usr/bin/env python3
"""Changes image brightness"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """Changes image brightness"""
    return tf.image.adjust_brightness(image, max_delta)
