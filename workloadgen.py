# workloadgen.py
from __future__ import annotations


'''A deterministic, configurable workload generator that can produce realistic and pathological 
job streams (arrivals + job shapes), without relying on historic scheduler logs.'''


'''Requirements:
Hard:
- Deterministic under env.reset(seed=...): same seed + same config => identical job stream.
- Controllable: one can dial job rate, duration mix, node/cores mix, correlation strength, and “stress modes”.
- Composable: easy to plug in multiple “components” (baseline traffic + bursts + maintenance window, etc.).
- Future-proof for "wrong time estimates": job specs must be easy to extend with estimated_duration (and later extra fields).

Soft (nice to have):
- Realistic correlations: e.g. longer jobs tend to request more nodes, daily arrival patterns, etc.
- Replaying canned “scenarios” (regression tests) with fixed seeds.
'''


from dataclasses import dataclass, replace
from typing import List, Optional
import numpy as np


@dataclass(frozen=True)
class JobSpec:
    duration: int
    nodes: int
    cores_per_node: int


@dataclass(frozen=True)
class WorkloadGenConfig:
    # arrivals: "flat" or "poisson"
    arrivals: str = "poisson"
    max_new_jobs_per_hour: int = 1500
    poisson_lambda: float = 200.0
    flat_jobs_per_hour: int = 200   # target arrivals for flat mode
    flat_jitter: int = 0           # +/- jitter; 0 => perfectly flat


    # resource ranges (v1: just uniform ranges; later we add mixtures/correlations)
    min_duration: int = 1
    max_duration: int = 170
    min_nodes: int = 1
    max_nodes: int = 16
    min_cores: int = 1
    max_cores: int = 96

    # optional hard cap safety (useful if someone sets poisson_lambda insane)
    hard_cap_jobs: Optional[int] = None


class WorkloadGenerator:
    def __init__(self, cfg: WorkloadGenConfig):
        arrivals = cfg.arrivals.lower().strip()
        if arrivals not in ("flat", "poisson", "uniform"):
            raise ValueError(f"arrivals must be 'flat', 'uniform' or 'poisson', got: {cfg.arrivals}")
        self.cfg = replace(cfg, arrivals=arrivals)

    def _sample_job_count(self, rng: np.random.Generator) -> int:
        """
        Arrival modes:
          - flat: constant arrivals around a target, optional +/- jitter (0 => perfectly constant)
          - poisson: Poisson(lambda)
          - uniform: discrete-uniform in [0, max_new_jobs_per_hour] (very noisy hour-to-hour)
        """
        mode = self.cfg.arrivals

        if mode == "flat":
            target = int(getattr(self.cfg, "flat_jobs_per_hour", self.cfg.max_new_jobs_per_hour))
            jitter = int(getattr(self.cfg, "flat_jitter", 0))

            if jitter <= 0:
                k = target
            else:
                k = int(rng.integers(target - jitter, target + jitter + 1))

        elif mode == "poisson":
            k = int(rng.poisson(self.cfg.poisson_lambda))

        elif mode == "uniform":
            # This is the old "flat".
            k = int(rng.integers(0, self.cfg.max_new_jobs_per_hour + 1))

        else:
            raise ValueError(f"Unknown arrivals mode: {mode}")

        # clamp + safety
        k = min(k, int(self.cfg.max_new_jobs_per_hour))
        if self.cfg.hard_cap_jobs is not None:
            k = min(k, int(self.cfg.hard_cap_jobs))
        if k < 0:
            k = 0
        return k

    def sample(self, hour_idx: int, rng: np.random.Generator) -> List[JobSpec]:
        # hour_idx currently unused, but we keep it to enable daily patterns later.
        n = self._sample_job_count(rng)

        if n == 0:
            return []

        durations = rng.integers(self.cfg.min_duration, self.cfg.max_duration + 1, size=n, dtype=np.int32)
        nodes = rng.integers(self.cfg.min_nodes, self.cfg.max_nodes + 1, size=n, dtype=np.int32)
        cores = rng.integers(self.cfg.min_cores, self.cfg.max_cores + 1, size=n, dtype=np.int32)

        return [JobSpec(int(durations[i]), int(nodes[i]), int(cores[i])) for i in range(n)]
