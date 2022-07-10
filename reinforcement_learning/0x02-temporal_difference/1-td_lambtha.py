#!/usr/bin/env python3
"""td_lambtha"""

import numpy as np

def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """td_lambtha"""
    
    for episode in range(episodes):
        state = env.reset()
        z = lambtha * gamma
        for step in range(max_steps):
            
            action = policy(state)

            next_state, reward, done, info = env.step(action)

            if env.desc.reshape(env.observation_space.n)[next_state] == b'H':
                reward = -1

            if env.desc.reshape(env.observation_space.n)[next_state] == b'G':
                reward = 1

            TD = reward + V[next_state]

            V[state] = (V[state] * z) + alpha * (TD - V[state])

            state = next_state
    return V