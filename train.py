# External
import gymnasium as gym
import numpy as np
import torch

from agent import DQNAgent

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

if __name__=="__main__":
    env = gym.make("ALE/Enduro-v5")
    seed = 42
    np.random.seed(seed)
    seed_torch(seed)

    agent = DQNAgent(
        env=env,
        memory_size=100000,
        batch_size=32,
        target_update=100,
        epsilon_decay=1/2000,
        learning_rate=0.001,
        seed=seed
    )

    num_frames = 10000
    agent.train(num_frames=num_frames)
