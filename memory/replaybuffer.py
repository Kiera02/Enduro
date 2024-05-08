#!/bin/python3
# External
import numpy as np

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, memory_size, input_shape):

        # state, action, reward, next_state, done
        memory_shape = [
            ('state', np.float32, input_shape),
            ('action', np.int64),
            ('reward', np.float32),
            ('next_state', np.float32, input_shape),
            ('done', np.bool_)
        ]

        # Numpy record structure array allows, different data types
        # but with also batching ability
        self.memory = np.zeros(memory_size, dtype=memory_shape)
        self.memory_size = memory_size
        self.memory_counter = 0

    def save(self, state, action, reward, next_state, done):

        index = self.memory_counter % self.memory_size
        self.memory[index] = (state, action, reward, next_state, done)
        self.memory_counter += 1

    def sample(self, batch_size):

        maximum_current_memory = min(self.memory_counter, self.memory_size)
        indices = np.random.choice(maximum_current_memory, batch_size, replace=False)
        batch = self.memory[indices]

        return (
            np.array(batch['state']),
            np.array(batch['action']),
            np.array(batch['reward']),
            np.array(batch['next_state']),
            np.array(batch['done'])
        )
