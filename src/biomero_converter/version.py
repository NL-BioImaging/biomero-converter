from importlib.metadata import PackageNotFoundError, version as package_version
import os
import subprocess


REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_version():
    """
    Get the version from the GitHub release tag.

    Uses the BIOMERO_CONVERTER_VERSION environment variable if set (e.g. in the Docker image),
    otherwise the git tag of the source tree (e.g. 'v0.1.31', or 'v0.1.31-2-gabc1234' after the tag),
    otherwise the version of the installed package (taken from the tag when it was built).

    Returns:
        str: Version string, or 'unknown' if it cannot be determined.
    """
    version = os.environ.get('BIOMERO_CONVERTER_VERSION')
    if version:
        return version
    # only this repo's own checkout: an installed package may sit inside another git repo
    if os.path.exists(os.path.join(REPO_DIR, '.git')):
        try:
            return subprocess.run(['git', 'describe', '--tags', '--always'], cwd=REPO_DIR,
                                  capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        return 'v' + package_version('biomero-converter')
    except PackageNotFoundError:
        return 'unknown'
