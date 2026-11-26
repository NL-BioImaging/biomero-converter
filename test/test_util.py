"""get_available_memory(): the free memory, limited by the (Slurm job) cgroup or Slurm memory limit."""
from types import SimpleNamespace

import psutil
import pytest

from biomero_converter.util import convert_to_um, get_available_memory


def test_convert_to_um():
    assert convert_to_um(1.5, 'mm') == 1500.0


def test_convert_to_um_angstrom():
    assert convert_to_um(10, 'Å') == 0.001


def test_convert_to_um_without_unit():
    assert convert_to_um(3, None) == 3


def test_convert_to_um_unknown_unit():
    assert convert_to_um(3, 'reference frame') == 3

GB = 1024 ** 3
FREE = 64 * GB


@pytest.fixture(autouse=True)
def free_memory(monkeypatch):
    # fixed free memory: the real one changes between calls (and can be below the limits, e.g. on CI runners)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=FREE))


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_cgroup_v2_job_limit(tmp_path):
    # limit on the job, not on its step: the smallest of the cgroup and its parents
    _write(tmp_path / 'proc_cgroup', '0::/slurm/uid_1/job_2/step_0\n')
    root = tmp_path / 'cgroup'
    _write(root / 'slurm/uid_1/job_2/memory.max', str(4 * GB))
    _write(root / 'slurm/uid_1/job_2/memory.current', str(1 * GB))
    _write(root / 'slurm/uid_1/job_2/step_0/memory.max', 'max')
    _write(root / 'slurm/uid_1/job_2/step_0/memory.current', str(1 * GB))
    available = get_available_memory(str(root), str(tmp_path / 'proc_cgroup'), environ={})
    assert available == 3 * GB


def test_cgroup_v1_limit(tmp_path):
    _write(tmp_path / 'proc_cgroup', '5:cpuset:/slurm/job_2\n4:memory:/slurm/job_2\n')
    root = tmp_path / 'cgroup'
    _write(root / 'memory/slurm/job_2/memory.limit_in_bytes', str(2 * GB))
    _write(root / 'memory/slurm/job_2/memory.usage_in_bytes', str(GB // 2))
    _write(root / 'memory/memory.limit_in_bytes', str(2 ** 63 - 4096))   # no limit
    _write(root / 'memory/memory.usage_in_bytes', str(GB))
    available = get_available_memory(str(root), str(tmp_path / 'proc_cgroup'), environ={})
    assert available == GB * 3 // 2


def test_slurm_environment_without_cgroup(tmp_path):
    no_cgroup = str(tmp_path / 'missing')
    rss = psutil.Process().memory_info().rss
    available = get_available_memory(no_cgroup, no_cgroup, environ={'SLURM_MEM_PER_CPU': '1024',
                                                                    'SLURM_CPUS_ON_NODE': '4'})
    assert available <= max(4 * GB - rss, 0)
