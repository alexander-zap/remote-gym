from remote_gym import RemoteEnvironment, create_remote_environment_server


def test_readme():
    # Launch server
    server = create_remote_environment_server(
        default_args={
            "entrypoint": "exploration/remote_environment_entrypoint.py",
            "entrypoint_kwargs": {
                "env": "Acrobot-v1",
                "render_mode": "rgb_array",
            },
        },
        url="localhost",
        port=12345,
    )

    # Start environment
    environment = RemoteEnvironment(
        url="localhost",
        port=12345,
        render_mode=None,
    )

    # Connect, reset, process an episode
    done = False
    episode_reward = 0
    environment.reset()
    while not done:
        action = environment.action_space.sample()
        _observation, reward, terminated, truncated, _info = environment.step(action)
        episode_reward += reward
        done = terminated or truncated

    # And stop the server
    server.stop(False)
