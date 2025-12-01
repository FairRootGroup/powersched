# sanity test for workload generator
import numpy as np
from workloadgen import WorkloadGenerator, WorkloadGenConfig

def assert_job_valid(j, cfg):
    assert cfg.min_duration <= j.duration <= cfg.max_duration
    assert cfg.min_nodes <= j.nodes <= cfg.max_nodes
    assert cfg.min_cores <= j.cores_per_node <= cfg.max_cores

def test_determinism():
    cfg = WorkloadGenConfig(arrivals="poisson", poisson_lambda=100.0, max_new_jobs_per_hour=1500)
    gen = WorkloadGenerator(cfg)

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    for hour in range(50):
        a = gen.sample(hour, rng1)
        b = gen.sample(hour, rng2)
        assert [(j.duration, j.nodes, j.cores_per_node) for j in a] == [(j.duration, j.nodes, j.cores_per_node) for j in b]

def test_constraints():
    cfg = WorkloadGenConfig(arrivals="flat", flat_jobs_per_hour=200, max_new_jobs_per_hour=200)
    gen = WorkloadGenerator(cfg)
    rng = np.random.default_rng(7)
    for hour in range(200):
        jobs = gen.sample(hour, rng)
        for j in jobs:
            assert_job_valid(j, cfg)

def test_poisson_mean_sanity():
    cfg = WorkloadGenConfig(arrivals="poisson", poisson_lambda=50.0, max_new_jobs_per_hour=1500)
    gen = WorkloadGenerator(cfg)
    rng = np.random.default_rng(1)
    counts = [len(gen.sample(h, rng)) for h in range(2000)]
    mean = float(np.mean(counts))
    assert 48.0 < mean < 52.0, mean  # tighter band, still reliable for 2000 samples

if __name__ == "__main__":
    test_determinism()
    test_constraints()
    test_poisson_mean_sanity()
    print("[OK] workloadgen sanity checks passed")
