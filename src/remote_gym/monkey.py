# grpcio==1.68.1

from grpc import channel_ready_future, insecure_channel
from typing import Optional

from dm_env_rpc.v1 import connection


def create_insecure_channel_and_connect(
    server_address: str,
    timeout: Optional[float] = None,
    metadata: Optional[connection.Metadata] = None,
) -> connection.Connection:

  channel = insecure_channel(
    server_address,
    options = [('grpc.max_send_message_length', -1),
               ('grpc.max_receive_message_length', -1)])
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


connection.create_insecure_channel_and_connect = create_insecure_channel_and_connect
