# sanity test for workload generator
import numpy as np
from src.workloadgen import WorkloadGenerator, WorkloadGenConfig

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


def test_flat_attribute_targets():
    cfg = WorkloadGenConfig(
        arrivals="flat",
        flat_jobs_per_hour=64,
        flat_jitter=0,
        flat_duration_target=12,
        flat_nodes_target=3,
        flat_cores_target=8,
        flat_duration_jitter=0,
        flat_nodes_jitter=0,
        flat_cores_jitter=0,
        min_duration=1,
        max_duration=170,
        min_nodes=1,
        max_nodes=16,
        min_cores=1,
        max_cores=96,
    )
    gen = WorkloadGenerator(cfg)
    rng = np.random.default_rng(2)
    jobs = gen.sample(0, rng)

    assert len(jobs) == 64
    assert all(j.duration == 12 for j in jobs)
    assert all(j.nodes == 3 for j in jobs)
    assert all(j.cores_per_node == 8 for j in jobs)


def test_poisson_attribute_lambdas_are_used():
    cfg = WorkloadGenConfig(
        arrivals="poisson",
        poisson_lambda=200.0,
        poisson_lambda_duration=2.0,
        poisson_lambda_nodes=2.0,
        poisson_lambda_cores=2.0,
        min_duration=1,
        max_duration=170,
        min_nodes=1,
        max_nodes=16,
        min_cores=1,
        max_cores=96,
    )
    gen = WorkloadGenerator(cfg)
    rng = np.random.default_rng(3)

    jobs = gen.sample(0, rng)
    assert len(jobs) > 0
    mean_duration = float(np.mean([j.duration for j in jobs]))
    mean_nodes = float(np.mean([j.nodes for j in jobs]))
    mean_cores = float(np.mean([j.cores_per_node for j in jobs]))

    # With lambda=2 and lower bound clipping at 1, means should remain low.
    assert mean_duration < 6.0, mean_duration
    assert mean_nodes < 6.0, mean_nodes
    assert mean_cores < 6.0, mean_cores

if __name__ == "__main__":
    test_determinism()
    test_constraints()
    test_poisson_mean_sanity()
    test_flat_attribute_targets()
    test_poisson_attribute_lambdas_are_used()
    print("[OK] workloadgen sanity checks passed")
