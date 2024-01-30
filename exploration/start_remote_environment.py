import logging

import gymnasium as gym
from remote_environment_management import start_as_remote_environment
from util import configure_logging

configure_logging()

if __name__ == "__main__":
    example_gym_env = gym.make("Acrobot-v1", render_mode="rgb_array")

    server = start_as_remote_environment(url="localhost", port=56789, local_environment=example_gym_env)

    try:
        server.wait_for_termination()
    except Exception as e:
        server.stop(None)
        logging.exception(e)
