import gymnasium as gym


# noinspection PyUnusedLocal
def create_environment(env: str, render_mode: str, **kwargs):
    return gym.make(env, render_mode=render_mode)
