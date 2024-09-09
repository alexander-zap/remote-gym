import hashlib
from pathlib import Path
from typing import Optional

import git
from fasteners import InterProcessLock


def base64hash(string: str):
    return hashlib.sha256(string.encode("utf-8")).hexdigest()


def clone_and_checkout(directory: Path, repository: str, ref: Optional[str]):
    # If the directory already exists, reuse the existing repo
    if directory.exists():
        repo = git.Repo(directory)
        repo.remotes.origin.fetch(tags=True)
    else:
        # Clone the repository if it doesn't exist
        repo = git.Repo.clone_from(repository, directory)

    # Use the masters remote head by default
    if ref is None:
        ref = "HEAD"

    if ("origin/" + ref) in repo.references:
        commit = repo.commit("origin/" + ref)
    else:
        commit = repo.commit(ref)

    repo.git.reset("--hard", commit)


class RepoManager:
    """
    Clones and keeps repositories up to date.
    """

    def __init__(self, working_dir: Path = Path(".repo_cache")):
        """
        Args:
            working_dir: The cache directory for locally cloned repositories.
        """
        self.working_dir = working_dir
        self.lock = InterProcessLock(self.working_dir / "lock")

    def get(self, repository: str, tag: str = None) -> Path:
        """
        Returns a path to the clones repository on the given tag.

        Args:
            repository: Valid repository URL to clone from.
            tag: Valid reference (branch, tag, or commit hash) to checkout.
            None will default to HEAD.

        Returns:
            A path to the root of the checked out repository.
        """
        self.working_dir.mkdir(exist_ok=True)

        target_dir = self.working_dir / f"{base64hash(repository + str(tag))}"

        with self.lock:
            clone_and_checkout(target_dir, repository, tag)

        return target_dir
