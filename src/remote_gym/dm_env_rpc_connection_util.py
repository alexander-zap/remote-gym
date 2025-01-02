# grpcio==1.68.1

from typing import Optional

from dm_env_rpc.v1 import connection
from grpc import channel_ready_future, insecure_channel


def create_insecure_channel_and_connect(
    server_address: str,
    timeout: Optional[float] = None,
    metadata: Optional[connection.Metadata] = None,
) -> connection.Connection:
    """Insecure version of `dm_env_rpc.v1.connection.create_secure_channel_and_connect`
    See: https://github.com/google-deepmind/dm_env_rpc/blob/master/dm_env_rpc/v1/connection.py#L151

    Creates an insecure channel from server address and connects.

    We allow the created channel to have un-bounded message lengths, to support
    large observations.

    Args:
      server_address: URI server address to connect to.
      timeout: Optional timeout in seconds to wait for channel to be ready.
        Default to waiting indefinitely.
      metadata: Optional sequence of 2-tuples, sent to the gRPC server as
          metadata.

    Returns:
      An instance of dm_env_rpc.Connection, where the channel is close upon the
      connection being closed.
    """
    options = [("grpc.max_send_message_length", -1), ("grpc.max_receive_message_length", -1)]
    channel = insecure_channel(server_address, options=options)
    channel_ready_future(channel).result(timeout)

    class _ConnectionWrapper(connection.Connection):
        """Utility to ensure channel is closed when the connection is closed."""

        def __init__(self, channel, metadata):
            super().__init__(channel=channel, metadata=metadata)
            self._channel = channel

        def __del__(self):
            self.close()

        def close(self):
            super().close()
            self._channel.close()

    return _ConnectionWrapper(channel=channel, metadata=metadata)
