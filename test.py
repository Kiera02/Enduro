import gymnasium as gym
import cv2

from train import prep_environment
from agent import DQNAgent

def save_frames_as_video(frames, video_filename, fps=30):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    frame_shape = (84, 84)
    gamma = 0.99
    epsilon = 0.01
    learning_rate = 0.0001
    games = 10

    env = gym.make('EnduroNoFrameskip-v4', max_episode_steps=3000, render_mode='rgb_array')
    env = prep_environment(env, frame_shape, repeat=4)

    # Load the trained model

    agent = DQNAgent(
        input_shape=env.observation_space.shape,
        action_shape=env.action_space.n,
        gamma=gamma,
        epsilon=epsilon,
        learning_rate=learning_rate,
        checkpoint_dir='temp/finetune/attention_500/'
    )
    agent.load_networks()

    total_score = 0

    for episode in range(games):
        done = False
        observation, _ = env.reset()
        frames = []
        episode_score = 0

        while not done:
            frames.append(env.render())
            action = agent.choose_action(observation)
            observation, reward, _, done, info = env.step(action)
            episode_score += reward

        total_score += episode_score

        if episode == games - 1:
            video_filename = 'test_EnduroNoFrameskip-v4_finetune_attention_500.mp4'
            save_frames_as_video(frames, video_filename)
            print(f"Video saved as {video_filename}")

    average_score = total_score / games
    print(f"Average score version EnduroNoFrameskip-v4 over {games} games finetune: {average_score}")
