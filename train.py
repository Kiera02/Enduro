import os
import sys
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import GrayScaleObservation, FrameStack
from tqdm import tqdm

# Path Append
sys.path.append(os.path.abspath(os.curdir))

# Internal
from env_modify.norm_frame import NormalizeFrame
from env_modify.repeat_action_frame import RepeatActionInFrames
from agent import DQNAgent

def prep_environment(env, shape, repeat):
    env = RepeatActionInFrames(env, repeat)
    env = GrayScaleObservation(env)
    env = NormalizeFrame(env, shape)
    return FrameStack(env, num_stack=repeat)

def check_convergence(rolling_means, threshold=0.01):
    if len(rolling_means) < 2:
        return False
    
    recent_means = rolling_means[-10:]
    return np.ptp(recent_means) < threshold

if __name__ == '__main__':
    repeat = 4
    frame_shape = (84, 84)
    gamma = 0.99
    epsilon = 1
    min_epsilon = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.0001
    games = 700
    rolling_average_n = 200
    directory = 'temp/finetune/attention_700_1/'
    convergence_threshold = 0.01


    if not os.path.exists(directory):
        raise NotADirectoryError('Folder specified to save plots and models does not exist')

    env = gym.make('ALE/Enduro-v5', max_episode_steps=5000)
    env = prep_environment(env, frame_shape, repeat)

    agent = DQNAgent(
        input_shape=env.observation_space.shape,
        action_shape=env.action_space.n,
        gamma=gamma,
        epsilon=epsilon,
        learning_rate=learning_rate,
        checkpoint_dir=directory,
    )

    best_score = 0

    plot_name = f'{directory}dqn_agent_enduro_plot.png'
    scores, steps, rolling_means, epsilons = [], [], [], []
    current_step = 0
    progress = tqdm(range(games), desc="Train")

    for episode in progress:
        done = False
        score = 0
        observation , _= env.reset()

        while not done:
            action = agent.choose_action(observation)
            new_observation, reward,_ ,done, info = env.step(action)
            score += reward

            agent.save_to_memory(observation, action, reward, new_observation, done)
            agent.learn()
            observation = new_observation
            current_step += 1

        scores.append(score)
        steps.append(current_step)
        epsilons.append(agent.epsilon)

        rolling_mean = np.mean(scores[-rolling_average_n:])
        rolling_means.append(rolling_mean)

        convergence_status = "Converged" if check_convergence(rolling_means, threshold=convergence_threshold) else "Not Converged"

        progress.set_postfix({"Episode": episode, "Score": score, "Avg": rolling_mean, "Best": best_score, "Status": convergence_status})

        if score > best_score:
            best_score = score
            agent.save_networks()

        # Decay epsilon
        if agent.epsilon > min_epsilon:
            agent.epsilon *= epsilon_decay
            agent.epsilon = max(agent.epsilon, min_epsilon)

        # # Check for convergence
        # if check_convergence(rolling_means, threshold = convergence_threshold):
        #     print(f"Convergence achieved at episode {episode}.")
        #     break


    fig, ax = plt.subplots()
    ax.plot(steps, rolling_means, color="red")
    ax.set_xlabel("steps", fontsize=12)
    ax.set_ylabel("Mean Score", color="red", fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(steps, epsilons, color="blue")
    ax2.set_ylabel("Epsilon", color="blue", fontsize=12)
    fig.savefig(plot_name)
