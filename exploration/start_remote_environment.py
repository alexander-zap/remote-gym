import logging

import gymnasium as gym

from remote_gym import start_as_remote_environment
from remote_gym.util import configure_logging

URL = "localhost"
PORT = 56789
SERVER_CREDENTIALS_PATHS = ("..\\ssl\\server.pem", "..\\ssl\\server-key.pem", None)

configure_logging()

if __name__ == "__main__":
    example_gym_env = gym.make("Acrobot-v1", render_mode="rgb_array")

    server = start_as_remote_environment(
        local_environment=example_gym_env,
        url=URL,
        port=PORT,
        server_credentials_paths=SERVER_CREDENTIALS_PATHS,
        enable_rendering=True,
    )

    try:
        server.wait_for_termination()
    except Exception as e:
        server.stop(None)
        logging.exception(e)
