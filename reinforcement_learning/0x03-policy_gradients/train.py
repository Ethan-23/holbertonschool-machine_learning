#!/usr/bin/env python3
"""runs training for policy gradient"""

import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """implements a full training"""

    weight = np.random.rand(4, 2)
    all_rewards = []
    for episode in range(nb_episodes):
        ep_rewards = 0
        gradients = []
        rewards = []
        state = env.reset()[None, :]
        while True:
            action, gradient = policy_gradient(state, weight)
            next_state, reward, done, info = env.step(action)
            state = next_state[None, :]
            ep_rewards += reward
            if done:
                break
        print("Done")
        for i in range(len(gradients)):
            weight += (alpha * gradients[i] * sum([r * (gamma ** r) for t, r in enumerate(rewards[i:])]))
        all_rewards.append(ep_rewards)
        print("{}: {}".format(episode, ep_rewards), end="\r", flush=False)
    return(all_rewards)
