#!/usr/bin/env python3
from PIL import Image
import gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
import tensorflow.keras as K

createModel = __import__('train').createModel
AtariProcessor = __import__('train').AtariProcessor


if __name__ == '__main__':
    env = gym.make("Breakout-v4", render_mode='human')
    env.reset()
    num_actions = env.action_space.n
    model = createModel(num_actions)
    memory = SequentialMemory(limit=1000000, window_length=4)
    processor = AtariProcessor()
    dqn = DQNAgent(model=model, nb_actions=num_actions,
                   processor=processor, memory=memory)
    dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])

    dqn.load_weights('policy.h5')

    dqn.test(env, nb_episodes=10, visualize=True)