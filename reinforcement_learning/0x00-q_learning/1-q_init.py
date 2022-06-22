#!/usr/bin/env python3
"""load_env"""

import gym
import numpy as np


def q_init(env):
    """q_init"""
    Q_table = np.zeros((env.observation_space.n, env.action_space.n))
    return Q_table
