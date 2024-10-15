import argparse
import logging
import sys

from remote_gym import create_remote_environment_server

root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        help="URL of the machine on which the environment should be hosted",
        default="localhost",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="Port of the environment hosting machine for gRPC communication",
        default=56789,
    )
    parser.add_argument(
        "--server_certificate",
        type=str,
        help="Path to the self-signed server certificate (for TLS authentication)",  # server.pem
        default=None,
    )
    parser.add_argument(
        "--server_private_key",
        type=str,
        help="Path to the self-signed server private key (for TLS authentication)",  # server-key.pem
        default=None,
    )
    parser.add_argument(
        "--root_certificate",
        type=str,
        help="Path to the root certificate (for TLS authentication)",  # ca.pem
        default=None,
    )
    args = parser.parse_args()

    url = args.url
    port = args.port
    server_credentials_paths = (args.server_certificate, args.server_private_key, args.root_certificate)

    server = create_remote_environment_server(
        default_args={
            # Server options
            "repo": "git@github.com:Luke100000/remote-gym.git",
            "reference": "master",
            "entrypoint": "exploration/remote_environment_entrypoint.py",
            "entrypoint_kwargs": {
                "env": "Acrobot-v1",
                "render_mode": "rgb_array",
            },
        },
        url=url,
        port=port,
        server_credentials_paths=server_credentials_paths if any(server_credentials_paths) else None,
    )

    try:
        server.wait_for_termination()
    except Exception as e:
        server.stop(None)
        logging.exception(e)
