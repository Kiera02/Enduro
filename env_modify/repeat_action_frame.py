# Internal
import collections

# External
import gymnasium as gym
import numpy as np

class RepeatActionInFrames(gym.Wrapper):
    def __init__(self, env: gym.Env, repeat=4):
        super().__init__(env)

        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frames = collections.deque(maxlen=2)

        if repeat <= 0:
            raise ValueError('Repeat value needs to be 1 or higher')

    def step(self, action):

        total_reward = 0
        done = False
        info = {}

        for i in range(self.repeat):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            self.frames.append(observation)

            if done:
                break

        # Open queue into arguments for np.maximum
        maximum_of_frames = np.maximum(*self.frames)
        return maximum_of_frames, total_reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.frames.clear()
        self.frames.append(observation)
        return observation
