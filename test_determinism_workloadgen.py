# determinism test for workload generator
import numpy as np
from workloadgen import WorkloadGenerator, WorkloadGenConfig

def jobs_to_tuples(jobs):
    return [(j.duration, j.nodes, j.cores_per_node) for j in jobs]

def check_determinism(cfg, seed=12345, hours=50):
    g1 = WorkloadGenerator(cfg)
    g2 = WorkloadGenerator(cfg)

    r1 = np.random.default_rng(seed)
    r2 = np.random.default_rng(seed)

    for h in range(hours):
        a = jobs_to_tuples(g1.sample(h, r1))
        b = jobs_to_tuples(g2.sample(h, r2))
        assert a == b, f"Determinism failed at hour={h}: {a[:3]} vs {b[:3]}"
    print("[OK] determinism")

def check_bounds(cfg, seed=1, hours=50):
    gen = WorkloadGenerator(cfg)
    rng = np.random.default_rng(seed)

    for h in range(hours):
        jobs = gen.sample(h, rng)
        assert len(jobs) <= cfg.max_new_jobs_per_hour
        if cfg.hard_cap_jobs is not None:
            assert len(jobs) <= cfg.hard_cap_jobs

        for j in jobs:
            assert cfg.min_duration <= j.duration <= cfg.max_duration
            assert cfg.min_nodes <= j.nodes <= cfg.max_nodes
            assert cfg.min_cores <= j.cores_per_node <= cfg.max_cores

    print("[OK] bounds & caps")

def check_distribution_smoke(cfg, seed=7, hours=2000):
    gen = WorkloadGenerator(cfg)
    rng = np.random.default_rng(seed)

    counts = np.array([len(gen.sample(h, rng)) for h in range(hours)], dtype=np.int32)
    mean = float(counts.mean())
    p0 = float((counts == 0).mean())

    print(f"[INFO] arrivals={cfg.arrivals} mean={mean:.2f} p0={p0:.3f} max={counts.max()}")

    #if cfg.arrivals == "flat":
    if cfg.arrivals == "uniform":
        # Uniform integers 0..M => mean ~ M/2 (close-ish for long runs)
        expected = cfg.max_new_jobs_per_hour / 2.0
        assert abs(mean - expected) / expected < 0.10, "flat mean looks off (smoke check)"
    elif cfg.arrivals == "poisson":
        # Only a loose sanity check: mean should not be wildly off lambda unless capped heavily
        target = min(cfg.poisson_lambda, cfg.max_new_jobs_per_hour)
        if target > 0:
            assert abs(mean - target) / target < 0.20, "poisson mean looks off (smoke check)"

    print("[OK] distribution smoke")

def main():
    # Example configs to test
    flat_cfg = WorkloadGenConfig(arrivals="flat", max_new_jobs_per_hour=1000)
    pois_cfg = WorkloadGenConfig(arrivals="poisson", poisson_lambda=200.0, max_new_jobs_per_hour=1000)

    for cfg in (flat_cfg, pois_cfg):
        check_determinism(cfg)
        check_bounds(cfg)
        check_distribution_smoke(cfg)

if __name__ == "__main__":
    main()
