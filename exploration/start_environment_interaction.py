import argparse
import logging
import sys

import numpy as np
from tqdm import tqdm

from remote_gym import RemoteEnvironment

N_INTERACTION_EPISODES = 100

root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        help="URL of the machine on which the environment should be hosted",
        default="localhost",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="Port of the environment hosting machine for gRPC communication",
        default=56789,
    )
    parser.add_argument(
        "--client_certificate",
        type=str,
        help="Path to TSL client certificate (optional, only for client authentication)",  # client.pem
        default=None,
    )
    parser.add_argument(
        "--client_private_key",
        type=str,
        help="Path to TLS client private key (optional, only for client authentication)",  # client-key.pem
        default=None,
    )
    parser.add_argument(
        "--root_certificate",
        type=str,
        help="Path to the root certificate (for TLS authentication)",  # ca.pem
        default=None,
    )
    parser.add_argument(
        "-r",
        "--render_mode",
        action="store_true",
        help="Render mode of the environment being interacted with",
        default=None,
    )

    args = parser.parse_args()

    url = args.url
    port = args.port
    client_credentials_paths = (args.root_certificate, args.client_certificate, args.client_private_key)

    # Create instance for managing communication with remotely running environment
    environment = RemoteEnvironment(
        url=url,
        port=port,
        client_credentials_paths=client_credentials_paths if any(client_credentials_paths) else None,
        render_mode=None,
    )

    # Print some environment information (observation and action space)
    print("Observation Space: ", environment.observation_space)
    print("Action Space: ", environment.action_space)
    print("Reward Range: ", environment.reward_range)

    episode_rewards = []
    for episode in tqdm(range(N_INTERACTION_EPISODES)):
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
