from pathlib import Path

from remote_gym.repo_manager import RepoManager

WORKING_DIR = Path(".repo_cache")
REPOSITORY = "ssh://git@stash.wargaming.net:2222/rp43/synthetic-player-environments.git"


def test_default():
    RepoManager(WORKING_DIR).get(REPOSITORY)


def test_update_default():
    RepoManager(WORKING_DIR).get(REPOSITORY)


def test_branch():
    RepoManager(WORKING_DIR).get(REPOSITORY, "master")


def test_commit():
    RepoManager(WORKING_DIR).get(REPOSITORY, "3486445022a73d43234bb32e19af66f161c03893")
