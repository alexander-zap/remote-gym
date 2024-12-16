# remote-gym: Hosting Gym-environments remotely

This is a module to run Gym environments remotely, to enable splitting environment hosting and agent training into separate processes (or even separate machines).
Communication between the two processes is executed by using TLS and the gRPC protocol.

Adapted `dm_env_rpc` for `Gym.env` environments.

## Usage

### Main Features

- Use the `create_remote_environment_server` method to start a `Gym.env` environment as a remotely running environment.
- Use the `RemoteEnvironment` class to manage the connection to a remotely running environment (from `create_remote_environment_server`) and provide the standardized `Gym.env` interface to your agents through a `RemoteEnvironment` object.
- Basically: `remote-gym` is to `Gym.env` as what `dm_env_rpc` is to `dm_env`.

### Getting Started

In [this example script](exploration/start_remote_environment.py) you can see how to start a remotely running environment.

In [this accompanying script](exploration/start_environment_interaction.py) you can see how to connect to and interact with the previously started environment from a separate process.

For a quick impression in this README, find a minimal environment hosting and environment interaction example below.

First process:

```py
import logging

from remote_gym import create_remote_environment_server

server = create_remote_environment_server(
    default_args={
        "entrypoint": "exploration/remote_environment_entrypoint.py",
    },
    # IP of the machine hosting the remote environment; can also be 0.0.0.0
    url=YOUR_SERVER_IP,
    # port the remote environment should use on the hosting machine
    port=PORT_FOR_REMOTE_ENVIRONMENT_TO_LISTEN,
    # not using a tuple but setting this completely to None is also possible in case only a local connection is required
    server_credentials_paths=("path/to/server.pem", "path/to/server-key.pem", "optional/path/to/ca.pem"),
)

try:
   server.wait_for_termination()
except Exception as e:
   server.stop(None)
   logging.exception(e)
```

With an `entrypoint.py` like this:

```py
import gymnasium as gym

def create_environment(enable_rendering: bool, env_id: int, **kwargs) -> gym.Env:
    return gym.make("Acrobot-v1")
```

Second process:

```py
from remote_gym import RemoteEnvironment

environment = RemoteEnvironment(
    url=YOUR_SERVER_IP,
    port=PORT_FOR_REMOTE_ENVIRONMENT_TO_RUN_ON,
    # not using a tuple but setting this completely to None is also possible in case only a local connection is required
    client_credentials_paths=("path/to/ca.pem", "optional/path/to/client.pem", "optional/path/to/client-key.pem"),
    # can be set to "human" or "rgb_array" if `enable_rendering` was set to True in remote environment hosting process
    render_mode=None,
)

done = False
episode_reward = 0
environment.reset()
while not done:
    action = environment.action_space.sample()
    _observation, reward, terminated, truncated, _info = environment.step(action)
    episode_reward += reward
    done = terminated or truncated
```

### Rendering

To preserve server resources and prevent network slowdowns, it is recommended to only enable rendering if required.
Renderings are automatically transferred from the remote management server to the RemoteEnvironment together with the
observation if the `render_mode` of the hosted environment is `rgb_array`.
The `render_mode` of the hosted environment should be controlled by the `entrypoint_kwargs` passed to the entrypoint.

## Set-Up

### Install all dependencies in your development environment

To set up your local development environment, please run:

```
poetry install
```

Behind the scenes, this creates a virtual environment and installs `remote_gym` along with its dependencies into a new virtualenv. Whenever you run `poetry run <command>`, that `<command>` is actually run inside the virtualenv managed by poetry.

You can now import functions and classes from the module with `import remote_gym`.

### Set-up for connecting the agent training process to remote environments running on a separate machine

Authenticating the communication channel via the connection of one machine to the other requires TLS (formerly SSL)
authentication.
This is achieved by using a [self-signed certificate](https://en.wikipedia.org/wiki/Self-signed_certificate),
meaning the certificate is not signed by a publicly trusted certificate authority (CA) but by a locally created CA.

> See https://github.com/joekottke/python-grpc-ssl for more details and a more in-depth tutorial on how to create the self-signed certificates.

All required configuration files to create a self-signed certificate chain can be found in the [ssl folder](/ssl).

1. The root certificate of the certificate authority (`ca.pem`) is created by following command:

   ```
   cfssl gencert -initca ca-csr.json | cfssljson -bare ca
   ```

1. The server certificate (`server.pem`) and respective private key (`server-key.pem`) is created by following command:

   ```
   cfssl gencert -ca="ca.pem" -ca-key="ca-key.pem" -config="ca-config.json" server-csr.json | cfssljson -bare server
   ```

Make sure to add all known hostnames of the machine hosting the remote environment. You can now test, whether the
client is able to connect to the server by running both example scripts.

- [`start_remote_environment`](/exploration/start_remote_environment.py) `-u SERVER.IP.HERE -p 56765  --server_certificate path\to\server.pem --server_private_key path\to\server-key.pem`
- [`start_environment_interaction`](/exploration/start_environment_interaction.py) `-u SERVER.IP.HERE -p 56765 --root_certificate path\to\ca.pem`

If the connection is not successful and the training is not starting, you can investigate on the server
(remote environment hosting machine) which IP is unsuccessfully attempting a TLS authentication to your IP by using
the [Wireshark tool](https://www.wireshark.org/download.html) with the filter `tcp.flags.reset==1 or tls.alert_message.level`.

Afterward you can add this IP to your hostnames to the [server SSL config file](/ssl/server-csr.json).

3. Optional for client authentication on the machine connecting to the remote environment:

   Create a client certificate (`client.pem`) and respective private key `client-key.pem` by running following command:

   ```
   cfssl gencert -ca="ca.pem" -ca-key="ca-key.pem" -config="ca-config.json" client-csr.json | cfssljson -bare client
   ```

Then you can use all certificates and keys:

- [`start_remote_environment`](/exploration/start_remote_environment.py) `-u SERVER.IP.HERE -p 56765  --root_certificate path\to\ca.pem --server_certificate path\to\server.pem --server_private_key path\to\server-key.pem`
- [`start_environment_interaction`](/exploration/start_environment_interaction.py) `-u SERVER.IP.HERE -p 56765 --root_certificate path\to\ca.pem --client_certificate path\to\client.pem --client_private_key path\to\client-key.pem`

## Development

### Notebooks

You can use your module code (`src/`) in Jupyter notebooks without running into import errors by running:

```
poetry run jupyter notebook
```

or

```
poetry run jupyter-lab
```

This starts the jupyter server inside the project's virtualenv.

Assuming you already have Jupyter installed, you can make your virtual environment available as a separate kernel by running:

```
poetry add ipykernel
poetry run python -m ipykernel install --user --name="remote-gym"
```

Note that we mainly use notebooks for experiments, visualizations and reports. Every piece of functionality that is meant to be reused should go into module code and be imported into notebooks.

### Contributions

Before contributing, please set up the pre-commit hooks to reduce errors and ensure consistency

```
pip install -U pre-commit
pre-commit install
```

If you run into any issues, you can remove the hooks again with `pre-commit uninstall`.

## License

© Alexander Zap
