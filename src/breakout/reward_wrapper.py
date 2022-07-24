import gym
import numpy as np

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # Clip reward between 0 to 1
        return np.clip(reward, 0, 1)
