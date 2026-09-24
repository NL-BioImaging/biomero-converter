import os
import subprocess


def get_version():
    """
    Get the version from the GitHub release tag.

    Uses the BIOMERO_CONVERTER_VERSION environment variable if set (e.g. in the Docker image),
    otherwise the git tag of the source tree (e.g. 'v0.1.31', or 'v0.1.31-2-gabc1234' after the tag).

    Returns:
        str: Version string, or 'unknown' if it cannot be determined.
    """
    version = os.environ.get('BIOMERO_CONVERTER_VERSION')
    if version:
        return version
    try:
        return subprocess.run(['git', 'describe', '--tags', '--always'],
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
