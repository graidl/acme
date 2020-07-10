"""Realizes Ranked Rewards calculation according to Laterre et al. (2018)."""

import numpy as np


class RankedRewardsBuffer:
    """Holds the most recent buffer_max_length total rewards and normalizes a new reward to -1 or +1 accordingly."""
    def __init__(self, buffer_max_length, percentile):
        self.buffer_max_length = buffer_max_length
        self.percentile = percentile
        self.buffer = []

    def add_reward(self, reward):
        if len(self.buffer) < self.buffer_max_length:
            self.buffer.append(reward)
        else:
            self.buffer = self.buffer[1:] + [reward]

    def normalize(self, reward):
        reward_threshold = np.percentile(self.buffer, self.percentile)
        if reward < reward_threshold:
            return -1.0
        else:
            return 1.0

    def get_state(self):
        return np.array(self.buffer)

    def set_state(self, state):
        if state is not None:
            self.buffer = list(state)
