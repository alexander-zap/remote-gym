import logging
from concurrent import futures
from typing import Optional, Text, Tuple, Union

import grpc
import gym
import gymnasium
import numpy as np
from dm_env_rpc.v1 import (
    dm_env_rpc_pb2,
    dm_env_rpc_pb2_grpc,
    spec_manager,
    tensor_spec_utils,
)
from google.rpc import code_pb2, status_pb2


def start_as_remote_environment(
    local_environment: Union[gym.Env, gymnasium.Env],
    url: Text,
    port: int,
    server_credentials_paths: Optional[Tuple[Text, Text, Optional[Text]]] = None,
) -> grpc.Server:
    """
    Method with which every environment can be transformed to a remote one.
    Starts the Catch gRPC server and passes the locally instantiated environment.
    Requires credentials to open a secure server port in gRPC. Needs to match the client authentication.

    Args:
        local_environment: Environment which should be ran remotely on a server.
        url: URL to the machine where the remote environment should be running on.
        port: Port to open (on the remote machine URL) for communication with the remote environment.
        server_credentials_paths (optional; local connection if not provided):
        Tuple of paths to TSL authentication files
            - server_cert_path: Path to TSL server certificate
            - server_private_key_path: Path to TLS server private key
            - root_cert_path: Path to TSL root certificate (optional, only for client authentication)

    NOTE: Use the RemoteEnvironment class to connect to a remotely started environment
        and provide a gym.Env interface to a learning agent.

    Returns:
        server: Reference to the gRPC server (for later closing)

    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    servicer = RemoteEnvironmentService(environment=local_environment)
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


class RemoteEnvironmentService(dm_env_rpc_pb2_grpc.EnvironmentServicer):
    """Runs the environment as a gRPC EnvironmentServicer."""

    def __init__(self, environment: Union[gym.Env, gymnasium.Env]):
        self.environment = environment

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
            else:
                logging.error(
                    f"Unexpected space type {type(space)} of space {space}, cannot extract higher and lower bounds."
                    f"Support for this space type can be added at the location of the raised ValueError."
                )
                raise ValueError

        self.action_spec = {
            1: dm_env_rpc_pb2.TensorSpec(
                name="action",
                shape=self.environment.action_space.shape,
                dtype=space_to_dtype(self.environment.action_space),
            )
        }

        self.observation_spec = {
            1: dm_env_rpc_pb2.TensorSpec(
                name="observation",
                shape=self.environment.observation_space.shape,
                dtype=space_to_dtype(self.environment.observation_space),
            ),
            2: dm_env_rpc_pb2.TensorSpec(name="reward", dtype=dm_env_rpc_pb2.FLOAT),
        }

        action_space_bounds = space_to_bounds(self.environment.action_space)
        observation_space_bounds = space_to_bounds(self.environment.observation_space)

        tensor_spec_utils.set_bounds(
            self.action_spec[1], minimum=action_space_bounds[0], maximum=action_space_bounds[1]
        )

        tensor_spec_utils.set_bounds(
            self.observation_spec[1], minimum=observation_space_bounds[0], maximum=observation_space_bounds[1]
        )

        tensor_spec_utils.set_bounds(
            self.observation_spec[2],
            minimum=self.environment.reward_range[0],
            maximum=self.environment.reward_range[1],
        )

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

        action_manager = spec_manager.SpecManager(self.action_spec)
        observation_manager = spec_manager.SpecManager(self.observation_spec)
        environment_should_be_reset = False

        for request in request_iterator:
            environment_response = dm_env_rpc_pb2.EnvironmentResponse()

            try:
                message_type = request.WhichOneof("payload")
                internal_request = getattr(request, message_type)
                logging.debug(f"Received message of type {message_type}.")

                if message_type == "create_world":
                    environment_should_be_reset = True
                    response = dm_env_rpc_pb2.CreateWorldResponse(world_name="world")

                elif message_type == "join_world":
                    environment_should_be_reset = True
                    response = dm_env_rpc_pb2.JoinWorldResponse()
                    for uid, action_space in self.action_spec.items():
                        response.specs.actions[uid].CopyFrom(action_space)
                    for uid, observation_space in self.observation_spec.items():
                        response.specs.observations[uid].CopyFrom(observation_space)

                elif message_type == "step":
                    if environment_should_be_reset:
                        observation, info = self.environment.reset()
                        reward = 0.0
                        terminated = False
                        truncated = False
                        environment_should_be_reset = False
                    else:
                        unpacked_actions = action_manager.unpack(internal_request.actions)
                        action = unpacked_actions.get("action")
                        observation, reward, terminated, truncated, info = self.environment.step(action)

                    response = dm_env_rpc_pb2.StepResponse()
                    packed_observations = observation_manager.pack({"observation": observation, "reward": reward})

                    for requested_observation in internal_request.requested_observations:
                        response.observations[requested_observation].CopyFrom(
                            packed_observations[requested_observation]
                        )
                    if terminated or truncated:
                        response.state = dm_env_rpc_pb2.EnvironmentStateType.TERMINATED
                        environment_should_be_reset = True
                    else:
                        response.state = dm_env_rpc_pb2.EnvironmentStateType.RUNNING

                elif message_type == "reset":
                    environment_should_be_reset = True
                    response = dm_env_rpc_pb2.ResetResponse()
                    for uid, action_space in self.action_spec.items():
                        response.specs.actions[uid].CopyFrom(action_space)
                    for uid, observation_space in self.observation_spec.items():
                        response.specs.observations[uid].CopyFrom(observation_space)

                elif message_type == "reset_world":
                    environment_should_be_reset = True
                    response = dm_env_rpc_pb2.ResetWorldResponse()

                elif message_type == "leave_world":
                    response = dm_env_rpc_pb2.LeaveWorldResponse()

                elif message_type == "destroy_world":
                    response = dm_env_rpc_pb2.DestroyWorldResponse()

                else:
                    raise RuntimeError("Unhandled message: {}".format(message_type))

                getattr(environment_response, message_type).CopyFrom(response)

            except Exception as e:  # pylint: disable=broad-except
                environment_response.error.CopyFrom(status_pb2.Status(code=code_pb2.INTERNAL, message=str(e)))

            yield environment_response
