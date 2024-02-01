# remote-gym: Hosting Gym-environments remotely

This is a module to run Gym environments remotely, to enable splitting environment hosting and agent training into separate processes (or even separate machines).
Communication between the two processes is executed by using TLS and the gRPC protocol.

Adapted `dm_env_rpc` for `Gym.env` environments.


## Usage

### Main Features
- Use the `start_as_remote_environment` method to convert a `Gym.env` environment into a remotely running environment.
- Use the `RemoteEnvironment` class to manage the connection to a remotely running environment and providing the standardized `Gym.env` interface to your agents.
- Basically: `remote-gym` is to `Gym.env` as what `dm_env_rpc` is to `dm_env`.

### Getting Started

In [this example script](exploration/start_remote_environment.py) you can see how to start a remotely running environment.

In [this accompanying script](exploration/start_environment_interaction.py) you can see how to connect to and interact with the previously started environment from a separate process.

For a quick impression in this README, find a minimal environment hosting and environment interaction example below.

First process:
```
server = start_as_remote_environment(
    url=URL,
    port=PORT,
    local_environment=YOUR_GYM_ENVIRONMENT_INSTANCE
)

server.wait_for_termination()
```

Second process:
```
environment = RemoteEnvironment(url=URL, port=PORT)
while not done:
    observation, reward, terminated, truncated, info = environment.step(prev_action)
    done = terminated or truncated
    action = environment.action_space.sample()
    episode_reward += reward
    prev_action = action
```

## Set-Up

### Install all dependencies in your development environment

To set up your local development environment, please run:

    poetry install

Behind the scenes, this creates a virtual environment and installs `remote_gym` along with its dependencies into a new virtualenv. Whenever you run `poetry run <command>`, that `<command>` is actually run inside the virtualenv managed by poetry.

You can now import functions and classes from the module with `import remote_gym`.


### Set-up for connecting the agent training process to remote environments running on a separate machine
Authenticating the communication channel via the connection of one machine to the other requires TLS (formerly SSL)
authentication.
This is achieved by using a [self-signed certificate](https://en.wikipedia.org/wiki/Self-signed_certificate),
meaning the certificate is not signed by a publicly trusted certificate authority (CA) but by a locally created CA.

See https://github.com/joekottke/python-grpc-ssl for more details.

All required configuration files to create a self-signed certificate chain can be found in the [ssl folder](/ssl).

1. The root certificate of the certificate authority (`ca.pem`) is created by following command:

       cfssl gencert -initca ca-csr.json | cfssljson -bare ca


2. The server certificate (`server.pem`) and respective private key (`server-key.pem`) is created by following command:

       cfssl gencert -ca=ca.pem -ca-key=ca-key.pem -config=ca-config.json server-csr.json | cfssljson -bare server

    Make sure to add all known hostnames of the machine hosting the remote environment. You can now test, whether the
    client is able to connect to the server by running both example scripts.
   - [`start_remote_environment`](/synthetic_player_experiment_space/start_remote_environment.py) `-u SERVER.IP.HERE -p 56765  --server_certificate path\to\server.pem --server_private_key path\to\server-key.pem`
   - [`start_agent_training`](/synthetic_player_experiment_space/start_agent_training.py) `-c path/to/config` with the config containing the environment config in following format:

          environment:
            framework: Remote
            params:
              url: SERVER.IP.HERE
              port: 56765
              client_credentials_paths:
              - ..\\ssl\\ca.pem
              - null
              - null

    If the connection is not successful and the training is not starting, you can investigate on the server
    (remote environment hosting machine) which IP is unsuccessfully attempting a TLS authentication to your IP by using
    the [Wireshark tool](https://www.wireshark.org/download.html) with the filter `tcp.flags.reset==1 or tls.alert_message.level`.

    Afterward you can add this IP to your hostnames to the [server SSL config file](/ssl/server-csr.json).


3. Optional for client authentication on the machine connecting to the remote environment:

    Create a client certificate (`client.pem`) and respective private key `client-key.pem` by running following command:

       cfssl gencert -ca=ca.pem -ca-key=ca-key.pem -config=ca-config.json client-csr.json | cfssljson -bare client





## Development

### Notebooks

You can use your module code (`src/`) in Jupyter notebooks without running into import errors by running:

    poetry run jupyter notebook

or

    poetry run jupyter-lab

This starts the jupyter server inside the project's virtualenv.

Assuming you already have Jupyter installed, you can make your virtual environment available as a separate kernel by running:

    poetry add ipykernel
    poetry run python -m ipykernel install --user --name="remote-gym"

Note that we mainly use notebooks for experiments, visualizations and reports. Every piece of functionality that is meant to be reused should go into module code and be imported into notebooks.

### Contributions

Before contributing, please set up the pre-commit hooks to reduce errors and ensure consistency

    pip install -U pre-commit
    pre-commit install

If you run into any issues, you can remove the hooks again with `pre-commit uninstall`.

## License

© Alexander Zap
