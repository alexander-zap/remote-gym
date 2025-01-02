# Hacky way of supporting Python 3.7, which does not have "Protocol" as part of the "typing" package.
# "Protocol" is required by "dm_env_rpc.v1.connection".
try:
    from typing import Protocol

except ImportError:
    import typing

    import typing_extensions

    typing.Protocol = typing_extensions.Protocol
    typing.runtime_checkable = typing_extensions.runtime_checkable

from .monkey import connection
from .remote_environment import RemoteArgs, RemoteEnvironment
from .remote_environment_management import create_remote_environment_server
