from pathlib import Path

from remote_gym.repo_manager import RepoManager

WORKING_DIR = Path(".repo_cache")
REPOSITORY = "https://github.com/Luke100000/remote-gym"


def test_default():
    RepoManager(WORKING_DIR).get(REPOSITORY)

    # And test update on the same repo
    RepoManager(WORKING_DIR).get(REPOSITORY)


def test_branch():
    RepoManager(WORKING_DIR).get(REPOSITORY, "master")


def test_commit():
    RepoManager(WORKING_DIR).get(REPOSITORY, "f08450496e6da433bf6551916868f9a32ce30691")


def test_tag():
    RepoManager(WORKING_DIR).get(REPOSITORY, "test_tag")
