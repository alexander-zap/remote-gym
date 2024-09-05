import importlib.util
import json
import logging
import multiprocessing as mp
import os
import sys
import traceback
from concurrent import futures
from pathlib import Path
from typing import List, Optional, Text, Tuple, Union

import grpc
import gym
import gymnasium
import numpy as np
from dm_env_rpc.v1 import (
    dm_env_rpc_pb2,
    dm_env_rpc_pb2_grpc,
    spec_manager,
    tensor_spec_utils,
    tensor_utils,
)
from google.rpc import code_pb2, status_pb2

from remote_gym.repo_manager import RepoManager


def start_as_remote_environment(
    default_args: dict,
    url: Text,
    port: int,
    server_credentials_paths: Optional[Tuple[Text, Text, Optional[Text]]] = None,
    enable_rendering: bool = False,
) -> grpc.Server:
    """
    Method with which every environment can be transformed to a remote one.
    Starts the Catch gRPC server and passes the locally instantiated environment.
    Requires credentials to open a secure server port in gRPC. Needs to match the client authentication.

    NOTE: Use the RemoteEnvironment class to connect to a remotely started environment and provide a gym.Env interface.

    Args:
        default_args: The arguments passed to the environment's entrypoint/constructor.
        url: URL to the machine where the remote environment should be running on.
        port: Port to open (on the remote machine URL) for communication with the remote environment.
        server_credentials_paths (optional; local connection if not provided):
            Tuple of paths to TSL authentication files:
            - server_cert_path: Path to TSL server certificate
            - server_private_key_path: Path to TLS server private key
            - root_cert_path: Path to TSL root certificate (optional, only for client authentication)
        enable_rendering (bool; default False): Flag to enable rendering support for connecting RemoteEnvironments.
            NOTE: Only supported if the passed `local_environment` has its .render_mode attribute set to "rgb_array".

    Returns:
        server: Reference to the gRPC server (for later closing)

    """
    server = grpc.server(
        futures.ThreadPoolExecutor(),
    )
    servicer = RemoteEnvironmentService(default_args=default_args, enable_rendering=enable_rendering)
    dm_env_rpc_pb2_grpc.add_EnvironmentServicer_to_server(servicer, server)

    if server_credentials_paths:
        server_cert_path, server_private_key_path, root_cert_path = server_credentials_paths
        assert server_cert_path and server_private_key_path

        server_cert_chain = open(server_cert_path, "rb").read()
        server_private_key = open(server_private_key_path, "rb").read()
        root_cert = open(root_cert_path, "rb").read() if root_cert_path else None

        client_authentication_required = True if root_cert is not None else False
        server_credentials = grpc.ssl_server_credentials(
            private_key_certificate_chain_pairs=[(server_private_key, server_cert_chain)],
            root_certificates=root_cert,
            require_client_auth=client_authentication_required,
        )
        logging.info(
            f"Opening secure port on {url}:{port}. "
            f"Client authentication {'REQUIRED' if client_authentication_required else 'OPTIONAL'}."
        )
    else:
        server_credentials = grpc.local_server_credentials()
        logging.info(
            f"Opening secure port on {url}:{port}. "
            f"SSL credentials were not provided, therefore connection only accepts local connections."
        )

    assigned_port = server.add_secure_port(f"{url}:{port}", server_credentials)
    assert assigned_port == port
    server.start()

    logging.info(f"Remote environment running on {url}:{assigned_port}")

    return server


def space_to_dtype(space: Union[gym.Space, gymnasium.Space]) -> dm_env_rpc_pb2.DataType:
    """Extract the dm_env_rpc_pb2 data type from the Gym Space.

    Args:
        space: Gym or Gymnasium Space object for definition of observation spaces

    Returns:
        dtype of the TensorSpec
    """
    if space.dtype == np.int64:
        dtype = dm_env_rpc_pb2.INT64
    elif space.dtype == np.float32:
        dtype = dm_env_rpc_pb2.FLOAT
    elif space.dtype == np.uint8:
        dtype = dm_env_rpc_pb2.UINT8
    else:
        logging.error(
            f"Unexpected dtype {space.dtype} of space {space}, cannot convert to TensorSpec-dtype."
            f"Support for this dtype can be added at the location of the raised ValueError."
        )
        raise ValueError

    return dtype


def space_to_bounds(space: Union[gym.Space, gymnasium.Space]) -> Tuple:
    """Extract the upper and lower bounds of the Gym space.

    Args:
        space: Gym or Gymnasium Space object for definition of observation spaces

    Returns:
        Tuple (lower and upper and lower bounds of Gym space in the shape of the Gym shape)
    """
    if isinstance(space, gym.spaces.Discrete) or isinstance(space, gymnasium.spaces.Discrete):
        return space.start, space.start + space.n - 1
    elif isinstance(space, gym.spaces.Box) or isinstance(space, gymnasium.spaces.Box):
        return space.low, space.high
    elif isinstance(space, gym.spaces.MultiDiscrete) or isinstance(space, gymnasium.spaces.MultiDiscrete):
        low = [discrete_space.start for discrete_space in space]
        high = [discrete_space.start + discrete_space.n - 1 for discrete_space in space]
        return low, high
    else:
        logging.error(
            f"Unexpected space type {type(space)} of space {space}, cannot extract higher and lower bounds."
            f"Support for this space type can be added at the location of the raised ValueError."
        )
        raise ValueError


def create_gym_environment(args: dict, enable_rendering: bool) -> Union[gym.Env, gymnasium.Env]:
    # Clone the given repository
    repo = args.pop("repo", None)
    tag = args.pop("tag", None)
    entrypoint = args.pop("entrypoint", None)
    working_dir = Path("./") if repo is None else RepoManager().get(repo, tag)

    # Set to current directory
    os.chdir(working_dir.resolve())
    sys.path.insert(0, str((working_dir / "src").resolve()))

    # Load the entrypoint
    spec = importlib.util.spec_from_file_location("module.name", entrypoint)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Instantiate the environment
    create_environment = getattr(module, "create_environment", None)
    environment = create_environment(enable_rendering=enable_rendering, **args)
    return environment


def run_env_loop(
    args: List[any],
    enable_rendering: bool,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
):
    initialized = False
    try:
        env = create_gym_environment(args, enable_rendering)

        action_spec = {
            1: dm_env_rpc_pb2.TensorSpec(
                name="action",
                shape=env.action_space.shape,
                dtype=space_to_dtype(env.action_space),
            )
        }

        observation_spec = {
            1: dm_env_rpc_pb2.TensorSpec(
                name="observation",
                shape=env.observation_space.shape,
                dtype=space_to_dtype(env.observation_space),
            ),
            2: dm_env_rpc_pb2.TensorSpec(name="reward", dtype=dm_env_rpc_pb2.FLOAT),
        }

        action_space_bounds = space_to_bounds(env.action_space)
        observation_space_bounds = space_to_bounds(env.observation_space)

        tensor_spec_utils.set_bounds(action_spec[1], minimum=action_space_bounds[0], maximum=action_space_bounds[1])

        tensor_spec_utils.set_bounds(
            observation_spec[1], minimum=observation_space_bounds[0], maximum=observation_space_bounds[1]
        )

        tensor_spec_utils.set_bounds(
            observation_spec[2],
            minimum=env.reward_range[0],
            maximum=env.reward_range[1],
        )

        if enable_rendering:
            assert env.render_mode == "rgb_array", (
                "Rendering remote environments is only possible if the `render_mode` attribute "
                "of the passed environment is 'rgb_array'."
            )
            env.reset()
            render_shape = env.render().shape
            observation_spec.update(
                {3: dm_env_rpc_pb2.TensorSpec(name="rendering", shape=render_shape, dtype=dm_env_rpc_pb2.UINT8)}
            )
            tensor_spec_utils.set_bounds(
                observation_spec[3],
                minimum=0,
                maximum=255,
            )

        # Pass action and obs layout
        out_queue.put(action_spec)
        out_queue.put(observation_spec)
        initialized = True

        while True:
            action, reset = in_queue.get()
            if reset is None:
                break

            if reset:
                observation, info = env.reset()
                reward = 0.0
                terminated = False
                truncated = False
            else:
                observation, reward, terminated, truncated, info = env.step(action)

            rendering = env.render() if enable_rendering else None

            out_queue.put((observation, reward, terminated, truncated, rendering, info))

        env.close()
    except Exception as e:
        if not initialized:
            out_queue.put(None)
            out_queue.put(None)
        out_queue.put(e)

        logging.error("Stacktrace: %s", traceback.format_exc())


class ProcessedEnv:
    def __init__(self, args: dict, enable_rendering: bool, env_id: int):
        self.in_queue = mp.Queue()
        self.out_queue = mp.Queue()
        self.should_reset = True

        self.env_id = env_id
        args["env_id"] = self.env_id

        self.process = mp.Process(target=run_env_loop, args=(args, enable_rendering, self.in_queue, self.out_queue))
        self.process.start()

        self.action_spec, self.observation_spec = self.out_queue.get(), self.out_queue.get()

        if self.action_spec is None or self.observation_spec is None:
            raise self.out_queue.get()

        self.action_manager = spec_manager.SpecManager(self.action_spec)
        self.observation_manager = spec_manager.SpecManager(self.observation_spec)

    def close(self):
        self.in_queue.put((None, None))
        self.process.join()

    def step(self, action):
        self.in_queue.put((action, self.should_reset))
        self.should_reset = False
        result = self.out_queue.get()
        if isinstance(result, Exception):
            raise result
        return result

    def reset(self):
        self.should_reset = True


class RemoteEnvironmentService(dm_env_rpc_pb2_grpc.EnvironmentServicer):
    """Runs the environment as a gRPC EnvironmentServicer."""

    def __init__(self, default_args: dict, enable_rendering: bool):
        self.default_args = default_args
        self.enable_rendering = enable_rendering
        self.environments = {}

        self.env_ids = [i for i in range(1024)][::-1]  # TODO: Magic number

    def get_environment(self, user: str) -> ProcessedEnv:
        if user not in self.environments:
            raise ValueError(f"Environment for user {user} does not exist.")
        return self.environments[user]

    def new_environment(self, user: str, args: dict):
        # We do not permit custom repositories for security reasons
        # TODO: evaluate risk of custom entrypoints
        args.pop("repo", None)

        merged_args = {**self.default_args, **args}
        self.destroy_environment(user)
        if len(self.env_ids) == 0:
            raise ValueError("Max environment count exceeded.")
        env_id = self.env_ids.pop()
        self.environments[user] = ProcessedEnv(merged_args, self.enable_rendering, env_id)
        logging.info(f"Created new environment for user {user} ({len(self.environments)} total active)")

    def destroy_environment(self, user: str):
        if user in self.environments:
            self.environments[user].close()
            self.env_ids.append(self.environments[user].env_id)
            del self.environments[user]
            logging.info(f"Destroyed environment for user {user} ({len(self.environments)} total active)")

    def Process(self, request_iterator, context):
        """Processes incoming EnvironmentRequests.

        For each EnvironmentRequest the internal message is extracted and handled.
        The response for that message is then placed in a EnvironmentResponse which
        is returned to the client.

        An error status will be returned if an unknown message type is received or
        if the message is invalid for the current world state.


        Args:
          request_iterator: Message iterator provided by gRPC.
          context: Context provided by gRPC.

        Yields:
          EnvironmentResponse: Response for each incoming EnvironmentRequest.
        """

        for request in request_iterator:
            environment_response = dm_env_rpc_pb2.EnvironmentResponse()

            try:
                message_type = request.WhichOneof("payload")
                internal_request = getattr(request, message_type)
                logging.debug(f"Received message of type {message_type}.")

                if message_type == "create_world":
                    response = dm_env_rpc_pb2.CreateWorldResponse(world_name="world")

                    packed_args = internal_request.settings["args"]
                    args = (
                        {}
                        if packed_args.WhichOneof("payload") is None
                        else json.loads(tensor_utils.unpack_tensor(packed_args))
                    )
                    self.new_environment(context.peer(), args)

                elif message_type == "join_world":
                    # Make sure to shutdown when the client leaves
                    # todo not sure if this adds double callbacks
                    context.add_callback(lambda p=context.peer(): self.destroy_environment(p))

                    environment = self.get_environment(context.peer())
                    environment.reset()

                    response = dm_env_rpc_pb2.JoinWorldResponse()
                    for uid, action_space in environment.action_spec.items():
                        response.specs.actions[uid].CopyFrom(action_space)
                    for uid, observation_space in environment.observation_spec.items():
                        response.specs.observations[uid].CopyFrom(observation_space)

                elif message_type == "step":
                    environment = self.get_environment(context.peer())

                    unpacked_actions = environment.action_manager.unpack(internal_request.actions)
                    action = unpacked_actions.get("action")

                    observation, reward, terminated, truncated, _, rendering = environment.step(action)

                    response_observations = {"observation": observation, "reward": reward}

                    if self.enable_rendering:
                        response_observations.update({"rendering": rendering})

                    packed_response_observations = environment.observation_manager.pack(response_observations)

                    response = dm_env_rpc_pb2.StepResponse()

                    for requested_observation in internal_request.requested_observations:
                        response.observations[requested_observation].CopyFrom(
                            packed_response_observations[requested_observation]
                        )
                    if terminated or truncated:
                        response.state = dm_env_rpc_pb2.EnvironmentStateType.TERMINATED
                        environment.reset()
                    else:
                        response.state = dm_env_rpc_pb2.EnvironmentStateType.RUNNING

                elif message_type == "reset":
                    environment = self.get_environment(context.peer())
                    environment.reset()

                    response = dm_env_rpc_pb2.ResetResponse()
                    for uid, action_space in environment.action_spec.items():
                        response.specs.actions[uid].CopyFrom(action_space)
                    for uid, observation_space in environment.observation_spec.items():
                        response.specs.observations[uid].CopyFrom(observation_space)

                elif message_type == "reset_world":
                    self.new_environment(context.peer(), self.enable_rendering)

                    response = dm_env_rpc_pb2.ResetWorldResponse()

                elif message_type == "leave_world":
                    self.destroy_environment(context.peer())
                    response = dm_env_rpc_pb2.LeaveWorldResponse()

                elif message_type == "destroy_world":
                    self.destroy_environment(context.peer())
                    response = dm_env_rpc_pb2.DestroyWorldResponse()

                else:
                    raise RuntimeError("Unhandled message: {}".format(message_type))

                getattr(environment_response, message_type).CopyFrom(response)

            except Exception as e:  # pylint: disable=broad-except
                # noinspection PyUnresolvedReferences
                environment_response.error.CopyFrom(status_pb2.Status(code=code_pb2.INTERNAL, message=str(e)))
                raise e

            yield environment_response
