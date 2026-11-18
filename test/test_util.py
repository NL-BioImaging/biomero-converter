"""get_available_memory(): the free memory, limited by the (Slurm job) cgroup or Slurm memory limit."""
import psutil

from biomero_converter.util import get_available_memory

GB = 1024 ** 3


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
    assert available == min(3 * GB, psutil.virtual_memory().available)


def test_cgroup_v1_limit(tmp_path):
    _write(tmp_path / 'proc_cgroup', '5:cpuset:/slurm/job_2\n4:memory:/slurm/job_2\n')
    root = tmp_path / 'cgroup'
    _write(root / 'memory/slurm/job_2/memory.limit_in_bytes', str(2 * GB))
    _write(root / 'memory/slurm/job_2/memory.usage_in_bytes', str(GB // 2))
    _write(root / 'memory/memory.limit_in_bytes', str(2 ** 63 - 4096))   # no limit
    _write(root / 'memory/memory.usage_in_bytes', str(GB))
    available = get_available_memory(str(root), str(tmp_path / 'proc_cgroup'), environ={})
    assert available == min(GB * 3 // 2, psutil.virtual_memory().available)


def test_slurm_environment_without_cgroup(tmp_path):
    no_cgroup = str(tmp_path / 'missing')
    rss = psutil.Process().memory_info().rss
    available = get_available_memory(no_cgroup, no_cgroup, environ={'SLURM_MEM_PER_CPU': '1024',
                                                                    'SLURM_CPUS_ON_NODE': '4'})
    assert available <= max(4 * GB - rss, 0)
