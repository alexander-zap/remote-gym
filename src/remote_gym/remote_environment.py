import json
import logging
from typing import Dict, Optional, SupportsFloat, Text, Tuple, TypedDict

import cv2
import grpc
import numpy as np
from dm_env import StepType, specs
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_adaptor
from gymnasium import Env
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Space

from remote_gym.dm_env_rpc_connection_util import create_insecure_channel_and_connect


class RemoteArgs(TypedDict):
    """
    Configuration on how to start the remote environment.

    Attributes:
        repo (Optional[str]): The repository to clone. If set to None, cloning is not permitted.
        reference (Optional[str]): The tag, branch, or commit ID to check out.
        entrypoint (Optional[str]): The filename of the entrypoint, containing a create_environment function.
            Either the server (default args) or the remote environment (remote args) need to provide it.
        entrypoint_kwargs (Dict[str, any]): Additional parameters passed to the create_environment function.

    Note:
        The entrypoint file must define a function `create_environment`:
        ```py
        def create_environment(env_id: int, **kwargs) -> gym.Env:
            return gym.make(...)
        ```
        * `env_id` is a unique identifier for the environment, used for non-sharable resources.
        * `kwargs` are any additional kwargs, including entrypoint_kwargs passed from the RemoteEnvironment

        The server may add additional fields (such as `env_id`).
        To remain forward-compatible, use **kwargs to dismiss unused fields!


    Example:
        ```py
        args = RemoteArgs(
            repo='https://github.com/example/repo.git',
            reference='v1.0',
            entrypoint='main.py',
            entrypoint_kwargs={'param1': 'value1', 'param2': 2}
        )
        ```
    """

    repo: Optional[str]
    reference: Optional[str]
    entrypoint: Optional[str]
    entrypoint_kwargs: Dict[str, any]


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
        remote_args: RemoteArgs = {},
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
            remote_args: Options and kwargs sent to the remote environment server,
                overwriting "default" arguments which are already configured when initializing the server
                (in `remote_environment_management.create_remote_environment_server`).
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
        """

        def convert_to_space(spec: specs.Array) -> Space:
            if isinstance(spec, specs.DiscreteArray):
                return Discrete(n=spec.num_values)
            elif isinstance(spec, specs.BoundedArray):
                # UInt-check required for detecting pixel-based RGB observations (should be embedded in the Box space)
                if np.issubdtype(spec.dtype, np.integer) and not spec.dtype == np.uint8:
                    return MultiDiscrete(spec.maximum + 1)
                else:
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
            else:
                if isinstance(spec, specs.Array):
                    logging.error(f"Unable to transform dm_env.spec {type(spec)} to Gym space.")
                    # return Box(-np.inf, np.inf, shape=spec.shape, dtype=spec.dtype)
                else:
                    logging.error(f"{type(spec)} is not a known dm_env.spec.")
                raise ValueError(
                    "Unsupported spec. Support for new dm_env.specs can be added at the location of this raised error."
                )

        self.url = url
        self.port = port
        self.remote_args = remote_args
        self.client_credentials_paths = client_credentials_paths

        self.connection, self.remote_environment = self._connect_to_remote_environment()

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

        self._disconnect_from_remote_environment()

    def step(self, action, *args, **kwargs) -> Tuple[object, SupportsFloat, bool, bool, dict]:
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        """
        if self.remote_environment is None:
            raise RuntimeError("Environment not connected to remote environment, call reset first!.")

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

        NOTE: If no connection to remote environment is active, connection to remote environment is established here.
        """
        if not self.remote_environment:
            self.connection, self.remote_environment = self._connect_to_remote_environment()

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

    def close(self):
        self._disconnect_from_remote_environment()

    def _connect_to_remote_environment(
        self,
    ) -> Tuple[dm_env_adaptor.dm_env_rpc_connection.Connection, dm_env_adaptor.DmEnvAdaptor]:
        def create_channel_connection() -> dm_env_adaptor.dm_env_rpc_connection.Connection:
            """Create secure channel connection based on given paths in self.client_credentials_paths.
                Without credentials a secure channel connection is only possible on localhost.
                Otherwise, an insecure channel connection will be established.

            :return: connection (dm_env_adaptor.dm_env_rpc_connection.Connection):
                Channel connection to remote host, secure or insecure.
            """
            server_address = f"{self.url}:{self.port}"
            if self.client_credentials_paths:
                root_cert_path, client_cert_path, client_private_key_path = self.client_credentials_paths
                root_cert = open(root_cert_path, "rb").read()
                client_authentication = True if client_private_key_path and client_cert_path else False

                client_private_key = open(client_private_key_path, "rb").read() if client_authentication else None
                client_cert_chain = open(client_cert_path, "rb").read() if client_authentication else None

                client_credentials = grpc.ssl_channel_credentials(
                    root_certificates=root_cert, private_key=client_private_key, certificate_chain=client_cert_chain
                )
                logging.info(
                    f"Connecting securely to port on {self.url}:{self.port}. "
                    f"Client authentication is {'ATTEMPTED' if client_authentication else 'OMITTED'}."
                )
                connection = dm_env_rpc_connection.create_secure_channel_and_connect(
                    server_address,
                    client_credentials,
                )
            elif self.url in ["localhost", "127.0.0.1"]:
                client_credentials = grpc.local_channel_credentials()
                logging.info(
                    f"Connecting securely to port on {self.url}:{self.port}. "
                    f"SSL credentials were not provided, but attempting secure channel due to URL being localhost."
                )
                connection = dm_env_rpc_connection.create_secure_channel_and_connect(
                    server_address,
                    client_credentials,
                )
            else:
                logging.info(f"Connecting insecurely to port on {self.url}:{self.port}.")
                connection = create_insecure_channel_and_connect(server_address)

            return connection

        connection = create_channel_connection()
        remote_environment, _ = dm_env_adaptor.create_and_join_world(
            connection=connection,
            create_world_settings={"args": json.dumps(self.remote_args)},
            join_world_settings={},
        )
        return connection, remote_environment

    def _disconnect_from_remote_environment(self):
        if self.remote_environment:
            self.remote_environment.close()
            self.remote_environment = None
        if self.connection:
            self.connection.close()
            self.connection = None

    def __getstate__(self):
        """
        Return state values to be pickled.
        """
        state = self.__dict__.copy()
        # Remove the unpicklable entries
        del state["connection"]
        del state["remote_environment"]
        return state

    def __setstate__(self, state):
        """
        Restore state from the unpickled state values.

        This method also re-initializes the connection to the remotely running environment.

        WARNING: Instance is disconnected and reconnected after this method is called.
        """
        self.__dict__.update(state)
        self.remote_environment = None
        self.connection = None

    def __del__(self):
        self._disconnect_from_remote_environment()
