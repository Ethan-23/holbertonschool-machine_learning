#!/usr/bin/env python3
"""Trains agent to play atari breakout"""

from PIL import Image
import numpy as np
import gym
import tensorflow.keras as K
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Permute
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize((84, 84)).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == (84, 84)
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

def createModel(nb_actions):
    """creates model for training"""
    inputs = Input(shape=((4,) + (84, 84)))
    x = Permute((2, 3, 1))(inputs)
    x = Conv2D(32, 8, strides=4, activation='relu')(x)
    x = Conv2D(64, 4, strides=2, activation='relu')(x)
    x = Conv2D(64, 2, strides=3, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    action = Dense(nb_actions, activation='linear')(x)
    model = K.Model(inputs=inputs, outputs=action)

    return model

env = gym.make("Breakout-v4")
env.reset()
nb_actions = env.action_space.n
model = createModel(nb_actions)
model.summary()
memory = SequentialMemory(limit=1000000, window_length=4)
processor = AtariProcessor()

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                              value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
               memory=memory, processor=processor,
               nb_steps_warmup=50000, gamma=.99,
               target_model_update=10000, train_interval=4,
               delta_clip=1.)

dqn.compile(Adam(lr=.00025), metrics=['mae'])

dqn.fit(env,
        nb_steps=17500,
        log_interval=10000,
        visualize=False,
        verbose=2)

dqn.save_weights('policy.h5', overwrite=True)
