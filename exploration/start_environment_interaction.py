import numpy as np
from tqdm import tqdm

from remote_gym import RemoteEnvironment
from remote_gym.util import configure_logging

REMOTE_ENVIRONMENT = False
URL = "localhost"
PORT = 56789

N_EVALUATION_EPISODES = 100

configure_logging()

if __name__ == "__main__":
    # Create instance for managing communication with remotely running environment
    environment = RemoteEnvironment(url=URL, port=PORT, render_mode="human")

    # Print some environment information (observation and action space)
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space Shape", environment.observation_space.shape)
    print("Sample observation", environment.observation_space.sample())

    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape", environment.action_space.shape)
    print("Action Space Sample", environment.action_space.sample())

    print("\n _____REWARD RANGE_____ \n")
    print("Reward Range Interval", environment.reward_range)

    episode_rewards = []
    for episode in tqdm(range(N_EVALUATION_EPISODES)):
        episode_reward = 0
        prev_observation, _ = environment.reset()
        environment.render()
        prev_action = environment.action_space.sample()

        while True:
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = environment.step(prev_action)
            environment.render()
            done = terminated or truncated
            action = environment.action_space.sample()
            episode_reward += reward
            prev_action = action

            if done:
                episode_rewards.append(episode_reward)
                break

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    environment.__del__()
