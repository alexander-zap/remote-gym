import logging
from typing import Optional, SupportsFloat, Text, Tuple

import cv2
import grpc
import numpy as np
from dm_env import StepType, specs
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_adaptor
from gymnasium import Env
from gymnasium.spaces import Box, Discrete, Space


class RemoteEnvironment(Env):
    """
    Wrapper implementation of remote environments,
    Implements central client-side management for the open communication interface between this wrapper Environment
    class and remotely running environments.
    """

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def reward_range(self):
        return self._reward_range

    @property
    def render_mode(self):
        return self._render_mode

    @action_space.setter
    def action_space(self, value):
        self._action_space = value

    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value

    @reward_range.setter
    def reward_range(self, value):
        self._reward_range = value

    @render_mode.setter
    def render_mode(self, value):
        self._render_mode = value

    def __init__(
        self,
        url: Text,
        port: int,
        client_credentials_paths: Optional[Tuple[Text, Optional[Text], Optional[Text]]] = None,
        render_mode: Text = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the remote environment connection.

        Requires credentials to connect to the secure server port in gRPC (needs to match the required server
            authentication on the server hosting the remotely running environment application).

        Args:
            url: URL to the machine where the remotely running environment application is hosted on.
            port: Open port on the remote machine (for communication with the remotely running environment application).
            client_credentials_paths (optional; local connection if not provided):
                Tuple of paths to TSL authentication files:
                - root_cert_path: Path to TSL root certificate
                - client_cert_path: Path to TSL client certificate (optional, only for client authentication)
                - client_private_key_path: Path to TLS client private key (optional, only for client authentication)
            render_mode: Specified mode of rendering, which is used by the .render method call.
                Available modes:
                - None: .render has no effect
                - “rgb_array” (default): return a single frame representing the current state of the environment.
                    A frame is a np.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.
                - “human”: .render displays the current frame on the current display using cv2
                NOTE: Rendering is only supported if the remote environment was initiated with `enable_rendering=True`
        """

        def convert_to_space(spec: specs.Array) -> Space:
            if isinstance(spec, specs.DiscreteArray):
                return Discrete(n=spec.num_values)
            # UInt-check required for detecting pixel-based RGB observations (these should be embedded in the Box space)
            if (
                isinstance(spec, specs.BoundedArray)
                and np.issubdtype(spec.dtype, np.integer)
                and not spec.dtype == np.uint8
            ):
                return Discrete(n=spec.maximum - spec.minimum + 1, start=spec.minimum)
            else:
                if isinstance(spec, specs.BoundedArray):
                    _min = spec.minimum
                    _max = spec.maximum

                    if np.isscalar(_min) and np.isscalar(_max):
                        # same min and max for every element
                        return Box(low=_min, high=_max, shape=spec.shape, dtype=spec.dtype)
                    else:
                        # different min and max for every element
                        return Box(
                            low=_min + np.zeros(shape=spec.shape, dtype=spec.dtype),
                            high=_max + np.zeros(shape=spec.shape, dtype=spec.dtype),
                            shape=spec.shape,
                            dtype=spec.dtype,
                        )
                elif isinstance(spec, specs.Array):
                    return Box(-np.inf, np.inf, shape=spec.shape, dtype=spec.dtype)
                else:
                    logging.error(
                        f"Unable to transform dm_env.spec {type(spec)} to Gym space."
                        f"Support for this dm_env.spec type can be added at the location of the raised ValueError."
                    )
                    raise ValueError

        if client_credentials_paths:
            root_cert_path, client_cert_path, client_private_key_path = client_credentials_paths
            root_cert = open(root_cert_path, "rb").read()
            client_authentication = True if client_private_key_path and client_cert_path else False

            client_private_key = open(client_private_key_path, "rb").read() if client_authentication else None
            client_cert_chain = open(client_cert_path, "rb").read() if client_authentication else None

            client_credentials = grpc.ssl_channel_credentials(
                root_certificates=root_cert, private_key=client_private_key, certificate_chain=client_cert_chain
            )
            logging.info(
                f"Connecting securely to port on {url}:{port}. "
                f"Client authentication is {'ATTEMPTED' if client_authentication else 'OMITTED'}."
            )
        else:
            client_credentials = grpc.local_channel_credentials()
            logging.info(
                f"Connecting securely to port on {url}:{port}. "
                f"SSL credentials were not provided, therefore can only connect to local servers."
            )

        self.connection = dm_env_rpc_connection.create_secure_channel_and_connect(f"{url}:{port}", client_credentials)

        self.remote_environment, self.world_name = dm_env_adaptor.create_and_join_world(
            self.connection, create_world_settings={}, join_world_settings={}
        )

        # Set local environment attributes retrieved from remote wrapper
        action_spec = self.remote_environment.action_spec()["action"]
        observation_spec = self.remote_environment.observation_spec()["observation"]
        reward_spec = self.remote_environment.reward_spec()

        self._action_space = convert_to_space(action_spec)
        self._observation_space = convert_to_space(observation_spec)
        self._reward_range = (
            (reward_spec.minimum, reward_spec.maximum)
            if isinstance(reward_spec, specs.BoundedArray)
            else (-float("inf"), float("inf"))
        )

        # Set local environment attributes for rendering
        self._render_mode = render_mode
        self.latest_image = None

    def step(self, action, *args, **kwargs) -> Tuple[object, SupportsFloat, bool, bool, dict]:
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        """
        timestep = self.remote_environment.step({"action": action})

        observation = timestep.observation.get("observation")
        self.latest_image = timestep.observation.get("rendering", None)
        reward = timestep.reward
        discount_factor = timestep.discount
        step_type = timestep.step_type

        terminated = False
        truncated = False

        if step_type is StepType.LAST:
            # NOTE: Assumption that discount_factor is 0.0 only for termination steps
            #   See https://github.com/google-deepmind/dm_env/blob/master/dm_env/_environment.py#L228
            if discount_factor == 0.0:
                terminated = True
            else:
                truncated = True

        return observation, reward, terminated, truncated, {}

    def reset(self, seed: Optional[int] = None, *args, **kwargs):
        """
        Resets the environment to an initial state.
        Returns the initial observation.
        """
        timestep = self.remote_environment.reset()

        observation = timestep.observation.get("observation")
        self.latest_image = timestep.observation.get("rendering", None)

        return observation, {}

    def render(self):
        """
        Renders the environment.
        """
        if not self._render_mode:
            pass
        elif self._render_mode == "human":
            if self.latest_image is not None:
                cv2.imshow("Remote Environment Rendering", self.latest_image)
                cv2.waitKey(1)
            else:
                logging.error("Rendering not possible, no image has been returned yet from the environment.")
        elif self._render_mode == "rgb_array":
            return self.latest_image
        else:
            raise NotImplementedError

    # FIXME: Does not trigger in main-file, investigate.
    def __del__(self):
        self.remote_environment.close()
        self.connection.close()
