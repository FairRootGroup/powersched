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


def test_bursts_are_additive_to_base_distribution():
    cfg = WorkloadGenConfig(
        arrivals="flat",
        flat_jobs_per_hour=10,
        flat_jitter=0,
        flat_duration_target=50,
        flat_nodes_target=5,
        flat_cores_target=20,
        flat_duration_jitter=0,
        flat_nodes_jitter=0,
        flat_cores_jitter=0,
        burst_small_prob=1.0,
        burst_small_jobs_min=3,
        burst_small_jobs_max=3,
        burst_small_duration_min=1,
        burst_small_duration_max=1,
        burst_small_nodes_min=1,
        burst_small_nodes_max=1,
        burst_small_cores_min=1,
        burst_small_cores_max=1,
        burst_heavy_prob=1.0,
        burst_heavy_jobs_min=2,
        burst_heavy_jobs_max=2,
        burst_heavy_duration_min=170,
        burst_heavy_duration_max=170,
        burst_heavy_nodes_min=16,
        burst_heavy_nodes_max=16,
        burst_heavy_cores_min=96,
        burst_heavy_cores_max=96,
        min_duration=1,
        max_duration=170,
        min_nodes=1,
        max_nodes=16,
        min_cores=1,
        max_cores=96,
    )
    gen = WorkloadGenerator(cfg)
    rng = np.random.default_rng(10)
    jobs = gen.sample(0, rng)
    tuples = [(j.duration, j.nodes, j.cores_per_node) for j in jobs]

    assert len(jobs) == 15
    assert tuples.count((50, 5, 20)) == 10
    assert tuples.count((1, 1, 1)) == 3
    assert tuples.count((170, 16, 96)) == 2


def test_zero_burst_prob_keeps_base_only():
    cfg = WorkloadGenConfig(
        arrivals="flat",
        flat_jobs_per_hour=12,
        flat_jitter=0,
        flat_duration_target=40,
        flat_nodes_target=4,
        flat_cores_target=12,
        flat_duration_jitter=0,
        flat_nodes_jitter=0,
        flat_cores_jitter=0,
        burst_small_prob=0.0,
        burst_heavy_prob=0.0,
        burst_small_jobs_min=10,
        burst_small_jobs_max=10,
        burst_heavy_jobs_min=10,
        burst_heavy_jobs_max=10,
        min_duration=1,
        max_duration=170,
        min_nodes=1,
        max_nodes=16,
        min_cores=1,
        max_cores=96,
    )
    gen = WorkloadGenerator(cfg)
    rng = np.random.default_rng(11)
    jobs = gen.sample(0, rng)

    assert len(jobs) == 12
    assert all((j.duration, j.nodes, j.cores_per_node) == (40, 4, 12) for j in jobs)


def test_burst_determinism_with_fixed_seed():
    cfg = WorkloadGenConfig(
        arrivals="poisson",
        poisson_lambda=30.0,
        burst_small_prob=0.35,
        burst_small_jobs_min=10,
        burst_small_jobs_max=20,
        burst_heavy_prob=0.15,
        burst_heavy_jobs_min=1,
        burst_heavy_jobs_max=4,
        min_duration=1,
        max_duration=170,
        min_nodes=1,
        max_nodes=16,
        min_cores=1,
        max_cores=96,
    )
    g1 = WorkloadGenerator(cfg)
    g2 = WorkloadGenerator(cfg)
    r1 = np.random.default_rng(1234)
    r2 = np.random.default_rng(1234)

    for h in range(100):
        a = [(j.duration, j.nodes, j.cores_per_node) for j in g1.sample(h, r1)]
        b = [(j.duration, j.nodes, j.cores_per_node) for j in g2.sample(h, r2)]
        assert a == b

if __name__ == "__main__":
    test_determinism()
    test_constraints()
    test_poisson_mean_sanity()
    test_flat_attribute_targets()
    test_poisson_attribute_lambdas_are_used()
    test_bursts_are_additive_to_base_distribution()
    test_zero_burst_prob_keeps_base_only()
    test_burst_determinism_with_fixed_seed()
    print("[OK] workloadgen sanity checks passed")
