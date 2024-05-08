# External
import gymnasium as gym
import numpy as np
import cv2

class NormalizeFrame(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)

        # Create the new observation space for the env
        # Convert to grayscale
        self.shape = shape

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self.shape, dtype=np.float32
        )

    def observation(self, observation):
        """Change from 255 grayscale to 0-1 scale
        """
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return (observation / 255.0).reshape(self.shape)
