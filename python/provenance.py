"""Provenance stored in every .coffea output (``output['provenance']``): the code
revision, the uncommitted Python changes, the run configuration and the hashes of
the correction payloads, so v1 / v1.1 / v1.2 outputs can be told apart later.
"""
import datetime
import hashlib
import os
import platform
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAYLOAD_DIRS = ("data/corrections", "data/toptag", "data/recomb")


def _git(*args):
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_hashes():
    hashes = {}
    for sub in PAYLOAD_DIRS:
        for root, dirs, files in os.walk(os.path.join(REPO, sub)):
            dirs.sort()
            for name in sorted(files):
                path = os.path.join(root, name)
                hashes[os.path.relpath(path, REPO)] = _sha256(path)
    return hashes


def collect(config):
    """Provenance dict for a run; ``config`` is the run configuration (e.g. vars(args))."""
    import awkward
    import coffea

    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_modified_files": status.splitlines() if status else [],
        "git_diff_py": _git("diff", "HEAD", "--", "*.py"),
        "config": {k: (v if isinstance(v, (str, int, float, bool, type(None))) else repr(v))
                   for k, v in dict(config).items()},
        "argv": list(sys.argv),
        "payload_sha256": payload_hashes(),
        "versions": {"python": platform.python_version(), "coffea": coffea.__version__,
                     "awkward": awkward.__version__},
        "host": platform.node(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
