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
    # arrivals mode shared across count + job attributes: "flat", "poisson", "uniform"
    arrivals: str = "poisson"
    uniform_min_new_jobs_per_hour: int = 0
    max_new_jobs_per_hour: int = 1500
    poisson_lambda: float = 200.0
    poisson_lambda_duration: Optional[float] = None
    poisson_lambda_nodes: Optional[float] = None
    poisson_lambda_cores: Optional[float] = None
    flat_jobs_per_hour: int = 200   # target arrivals for flat mode
    flat_jitter: int = 0           # +/- jitter; 0 => perfectly flat
    flat_duration_target: Optional[int] = None
    flat_nodes_target: Optional[int] = None
    flat_cores_target: Optional[int] = None
    flat_duration_jitter: int = 0
    flat_nodes_jitter: int = 0
    flat_cores_jitter: int = 0


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

        duration_mid = int(round((int(cfg.min_duration) + int(cfg.max_duration)) / 2.0))
        nodes_mid = int(round((int(cfg.min_nodes) + int(cfg.max_nodes)) / 2.0))
        cores_mid = int(round((int(cfg.min_cores) + int(cfg.max_cores)) / 2.0))

        if int(cfg.min_duration) > int(cfg.max_duration):
            raise ValueError("min_duration must be <= max_duration")
        if int(cfg.min_nodes) > int(cfg.max_nodes):
            raise ValueError("min_nodes must be <= max_nodes")
        if int(cfg.min_cores) > int(cfg.max_cores):
            raise ValueError("min_cores must be <= max_cores")
        if int(cfg.uniform_min_new_jobs_per_hour) > int(cfg.max_new_jobs_per_hour):
            raise ValueError("uniform_min_new_jobs_per_hour must be <= max_new_jobs_per_hour")

        self.cfg = replace(
            cfg,
            arrivals=arrivals,
            poisson_lambda_duration=(
                float(cfg.poisson_lambda_duration)
                if cfg.poisson_lambda_duration is not None
                else float(duration_mid)
            ),
            poisson_lambda_nodes=(
                float(cfg.poisson_lambda_nodes)
                if cfg.poisson_lambda_nodes is not None
                else float(nodes_mid)
            ),
            poisson_lambda_cores=(
                float(cfg.poisson_lambda_cores)
                if cfg.poisson_lambda_cores is not None
                else float(cores_mid)
            ),
            flat_duration_target=(
                int(cfg.flat_duration_target)
                if cfg.flat_duration_target is not None
                else int(duration_mid)
            ),
            flat_nodes_target=(
                int(cfg.flat_nodes_target)
                if cfg.flat_nodes_target is not None
                else int(nodes_mid)
            ),
            flat_cores_target=(
                int(cfg.flat_cores_target)
                if cfg.flat_cores_target is not None
                else int(cores_mid)
            ),
        )

    def _sample_attr_array(
        self,
        rng: np.random.Generator,
        size: int,
        mode: str,
        min_value: int,
        max_value: int,
        poisson_lambda: float,
        flat_target: int,
        flat_jitter: int,
    ) -> np.ndarray:
        if size <= 0:
            return np.array([], dtype=np.int32)

        if mode == "flat":
            if flat_jitter <= 0:
                values = np.full(size, int(flat_target), dtype=np.int64)
            else:
                values = rng.integers(
                    int(flat_target) - int(flat_jitter),
                    int(flat_target) + int(flat_jitter) + 1,
                    size=size,
                )
        elif mode == "poisson":
            values = rng.poisson(float(poisson_lambda), size=size)
        elif mode == "uniform":
            values = rng.integers(int(min_value), int(max_value) + 1, size=size)
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")

        return np.clip(values, int(min_value), int(max_value)).astype(np.int32)

    def _sample_job_count(self, rng: np.random.Generator) -> int:
        """
        Arrival modes:
          - flat: constant arrivals around a target, optional +/- jitter (0 => perfectly constant)
          - poisson: Poisson(lambda)
          - uniform: discrete-uniform in [uniform_min_new_jobs_per_hour, max_new_jobs_per_hour]
        """
        mode = self.cfg.arrivals

        if mode == "flat":
            target = int(self.cfg.flat_jobs_per_hour)
            jitter = int(self.cfg.flat_jitter)

            if jitter <= 0:
                k = target
            else:
                k = int(rng.integers(target - jitter, target + jitter + 1))

        elif mode == "poisson":
            k = int(rng.poisson(self.cfg.poisson_lambda))

        elif mode == "uniform":
            k = int(
                rng.integers(
                    int(self.cfg.uniform_min_new_jobs_per_hour),
                    int(self.cfg.max_new_jobs_per_hour) + 1,
                )
            )

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

        mode = self.cfg.arrivals
        durations = self._sample_attr_array(
            rng=rng,
            size=n,
            mode=mode,
            min_value=int(self.cfg.min_duration),
            max_value=int(self.cfg.max_duration),
            poisson_lambda=float(self.cfg.poisson_lambda_duration),
            flat_target=int(self.cfg.flat_duration_target),
            flat_jitter=int(self.cfg.flat_duration_jitter),
        )
        nodes = self._sample_attr_array(
            rng=rng,
            size=n,
            mode=mode,
            min_value=int(self.cfg.min_nodes),
            max_value=int(self.cfg.max_nodes),
            poisson_lambda=float(self.cfg.poisson_lambda_nodes),
            flat_target=int(self.cfg.flat_nodes_target),
            flat_jitter=int(self.cfg.flat_nodes_jitter),
        )
        cores = self._sample_attr_array(
            rng=rng,
            size=n,
            mode=mode,
            min_value=int(self.cfg.min_cores),
            max_value=int(self.cfg.max_cores),
            poisson_lambda=float(self.cfg.poisson_lambda_cores),
            flat_target=int(self.cfg.flat_cores_target),
            flat_jitter=int(self.cfg.flat_cores_jitter),
        )

        return [JobSpec(int(durations[i]), int(nodes[i]), int(cores[i])) for i in range(n)]
