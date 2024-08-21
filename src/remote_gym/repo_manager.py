import hashlib
from pathlib import Path

import git
from filelock import FileLock


def base64hash(string: str):
    return hashlib.sha256(string.encode("utf-8")).hexdigest()


def clone_and_checkout(directory: Path, repository: str, ref: str | None):
    # If the directory already exists, reuse the existing repo
    if directory.exists():
        repo = git.Repo(directory)
        repo.remotes.origin.fetch()
    else:
        # Clone the repository if it doesn't exist
        repo = git.Repo.clone_from(repository, directory)

    # Use the masters remote head by default
    if ref is None:
        ref = repo.remotes.origin.refs.HEAD.ref

    # Checkout the specific branch, tag, or commit
    repo.git.checkout(ref, force=True)


class RepoManager:
    """
    Clones and keeps repositories up to date.
    """

    def __init__(self, working_dir: Path = Path(".repo_cache")):
        self.working_dir = working_dir
        self.lock = FileLock(self.working_dir / "lock")

    def get(self, repository: str, tag: str = None):
        """
        Returns a path to the clones repository on the given tag.
        """
        self.working_dir.mkdir(exist_ok=True)

        target_dir = self.working_dir / f"{base64hash(repository + str(tag))}"

        with self.lock.acquire():
            clone_and_checkout(target_dir, repository, tag)

        return target_dir
