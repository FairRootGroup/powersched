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
    poisson_lambda_duration: float | None = None
    poisson_lambda_nodes: float | None = None
    poisson_lambda_cores: float | None = None
    flat_jobs_per_hour: int = 200   # target arrivals for flat mode
    flat_jitter: int = 0           # +/- jitter; 0 => perfectly flat
    flat_duration_target: int | None = None
    flat_nodes_target: int | None = None
    flat_cores_target: int | None = None
    flat_duration_jitter: int = 0
    flat_nodes_jitter: int = 0
    flat_cores_jitter: int = 0

    # Optional burst injectors (additive on top of base arrivals)
    # Burst 1: many small-ish jobs at once
    burst_small_prob: float = 0.0
    burst_small_jobs_min: int = 50
    burst_small_jobs_max: int = 1500
    burst_small_duration_min: int = 1
    burst_small_duration_max: int = 2
    burst_small_nodes_min: int = 1
    burst_small_nodes_max: int = 1
    burst_small_cores_min: int = 1
    burst_small_cores_max: int = 4

    # Burst 2: heavy jobs (high duration + high resource demand)
    burst_heavy_prob: float = 0.0
    burst_heavy_jobs_min: int = 1
    burst_heavy_jobs_max: int = 12
    burst_heavy_duration_min: int = 72
    burst_heavy_duration_max: int = 170
    burst_heavy_nodes_min: int = 4
    burst_heavy_nodes_max: int = 16
    burst_heavy_cores_min: int = 32
    burst_heavy_cores_max: int = 96


    # resource ranges (v1: just uniform ranges; later we add mixtures/correlations)
    min_duration: int = 1
    max_duration: int = 170
    min_nodes: int = 1
    max_nodes: int = 16
    min_cores: int = 1
    max_cores: int = 96

    # optional hard cap safety (useful if someone sets poisson_lambda insane)
    hard_cap_jobs: int | None = None


class WorkloadGenerator:
    def __init__(self, cfg: WorkloadGenConfig) -> None:
        arrivals = cfg.arrivals.lower().strip()
        if arrivals not in ("flat", "poisson", "uniform"):
            raise ValueError(f"arrivals must be 'flat', 'uniform' or 'poisson', got: {cfg.arrivals}")

        duration_mid = (cfg.min_duration + cfg.max_duration) // 2
        nodes_mid = (cfg.min_nodes + cfg.max_nodes) // 2
        cores_mid = (cfg.min_cores + cfg.max_cores) // 2

        if cfg.min_duration > cfg.max_duration:
            raise ValueError("min_duration must be <= max_duration")
        if cfg.min_nodes > cfg.max_nodes:
            raise ValueError("min_nodes must be <= max_nodes")
        if cfg.min_cores > cfg.max_cores:
            raise ValueError("min_cores must be <= max_cores")
        if cfg.uniform_min_new_jobs_per_hour > cfg.max_new_jobs_per_hour:
            raise ValueError("uniform_min_new_jobs_per_hour must be <= max_new_jobs_per_hour")
        if not (0.0 <= cfg.burst_small_prob <= 1.0):
            raise ValueError("burst_small_prob must be in [0, 1]")
        if not (0.0 <= cfg.burst_heavy_prob <= 1.0):
            raise ValueError("burst_heavy_prob must be in [0, 1]")
        if cfg.burst_small_jobs_min > cfg.burst_small_jobs_max:
            raise ValueError("burst_small_jobs_min must be <= burst_small_jobs_max")
        if cfg.burst_heavy_jobs_min > cfg.burst_heavy_jobs_max:
            raise ValueError("burst_heavy_jobs_min must be <= burst_heavy_jobs_max")

        def _bound(value: int, low: int, high: int) -> int:
            return min(max(value, low), high)

        burst_small_duration_min = _bound(cfg.burst_small_duration_min, cfg.min_duration, cfg.max_duration)
        burst_small_duration_max = _bound(cfg.burst_small_duration_max, cfg.min_duration, cfg.max_duration)
        burst_small_nodes_min = _bound(cfg.burst_small_nodes_min, cfg.min_nodes, cfg.max_nodes)
        burst_small_nodes_max = _bound(cfg.burst_small_nodes_max, cfg.min_nodes, cfg.max_nodes)
        burst_small_cores_min = _bound(cfg.burst_small_cores_min, cfg.min_cores, cfg.max_cores)
        burst_small_cores_max = _bound(cfg.burst_small_cores_max, cfg.min_cores, cfg.max_cores)

        burst_heavy_duration_min = _bound(cfg.burst_heavy_duration_min, cfg.min_duration, cfg.max_duration)
        burst_heavy_duration_max = _bound(cfg.burst_heavy_duration_max, cfg.min_duration, cfg.max_duration)
        burst_heavy_nodes_min = _bound(cfg.burst_heavy_nodes_min, cfg.min_nodes, cfg.max_nodes)
        burst_heavy_nodes_max = _bound(cfg.burst_heavy_nodes_max, cfg.min_nodes, cfg.max_nodes)
        burst_heavy_cores_min = _bound(cfg.burst_heavy_cores_min, cfg.min_cores, cfg.max_cores)
        burst_heavy_cores_max = _bound(cfg.burst_heavy_cores_max, cfg.min_cores, cfg.max_cores)

        self.cfg = replace(
            cfg,
            arrivals=arrivals,
            poisson_lambda_duration=(
                cfg.poisson_lambda_duration
                if cfg.poisson_lambda_duration is not None
                else float(duration_mid)
            ),
            poisson_lambda_nodes=(
                cfg.poisson_lambda_nodes
                if cfg.poisson_lambda_nodes is not None
                else float(nodes_mid)
            ),
            poisson_lambda_cores=(
                cfg.poisson_lambda_cores
                if cfg.poisson_lambda_cores is not None
                else float(cores_mid)
            ),
            flat_duration_target=(
                cfg.flat_duration_target
                if cfg.flat_duration_target is not None
                else duration_mid
            ),
            flat_nodes_target=(
                cfg.flat_nodes_target
                if cfg.flat_nodes_target is not None
                else nodes_mid
            ),
            flat_cores_target=(
                cfg.flat_cores_target
                if cfg.flat_cores_target is not None
                else cores_mid
            ),
            burst_small_duration_min=min(burst_small_duration_min, burst_small_duration_max),
            burst_small_duration_max=max(burst_small_duration_min, burst_small_duration_max),
            burst_small_nodes_min=min(burst_small_nodes_min, burst_small_nodes_max),
            burst_small_nodes_max=max(burst_small_nodes_min, burst_small_nodes_max),
            burst_small_cores_min=min(burst_small_cores_min, burst_small_cores_max),
            burst_small_cores_max=max(burst_small_cores_min, burst_small_cores_max),
            burst_heavy_duration_min=min(burst_heavy_duration_min, burst_heavy_duration_max),
            burst_heavy_duration_max=max(burst_heavy_duration_min, burst_heavy_duration_max),
            burst_heavy_nodes_min=min(burst_heavy_nodes_min, burst_heavy_nodes_max),
            burst_heavy_nodes_max=max(burst_heavy_nodes_min, burst_heavy_nodes_max),
            burst_heavy_cores_min=min(burst_heavy_cores_min, burst_heavy_cores_max),
            burst_heavy_cores_max=max(burst_heavy_cores_min, burst_heavy_cores_max),
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
                values = np.full(size, flat_target, dtype=np.int64)
            else:
                values = rng.integers(
                    flat_target - flat_jitter,
                    flat_target + flat_jitter + 1,
                    size=size,
                )
        elif mode == "poisson":
            values = rng.poisson(poisson_lambda, size=size)
        elif mode == "uniform":
            values = rng.integers(min_value, max_value + 1, size=size)
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")

        return np.clip(values, min_value, max_value).astype(np.int32)

    def _sample_job_count(self, rng: np.random.Generator) -> int:
        """
        Arrival modes:
          - flat: constant arrivals around a target, optional +/- jitter (0 => perfectly constant)
          - poisson: Poisson(lambda)
          - uniform: discrete-uniform in [uniform_min_new_jobs_per_hour, max_new_jobs_per_hour]
        """
        mode = self.cfg.arrivals

        if mode == "flat":
            target = self.cfg.flat_jobs_per_hour
            jitter = self.cfg.flat_jitter

            if jitter <= 0:
                k = target
            else:
                k = int(rng.integers(target - jitter, target + jitter + 1))

        elif mode == "poisson":
            k = int(rng.poisson(self.cfg.poisson_lambda))

        elif mode == "uniform":
            k = int(
                rng.integers(
                    self.cfg.uniform_min_new_jobs_per_hour,
                    self.cfg.max_new_jobs_per_hour + 1,
                )
            )

        else:
            raise ValueError(f"Unknown arrivals mode: {mode}")

        # clamp + safety
        k = min(k, self.cfg.max_new_jobs_per_hour)
        if self.cfg.hard_cap_jobs is not None:
            k = min(k, self.cfg.hard_cap_jobs)
        if k < 0:
            k = 0
        return k

    def sample(self, hour_idx: int, rng: np.random.Generator) -> list[JobSpec]:
        # hour_idx currently unused, but we keep it to enable daily patterns later.
        base_n = self._sample_job_count(rng)
        mode = self.cfg.arrivals
        if base_n > 0:
            durations = self._sample_attr_array(
                rng=rng,
                size=base_n,
                mode=mode,
                min_value=self.cfg.min_duration,
                max_value=self.cfg.max_duration,
                poisson_lambda=self.cfg.poisson_lambda_duration,
                flat_target=self.cfg.flat_duration_target,
                flat_jitter=self.cfg.flat_duration_jitter,
            )
            nodes = self._sample_attr_array(
                rng=rng,
                size=base_n,
                mode=mode,
                min_value=self.cfg.min_nodes,
                max_value=self.cfg.max_nodes,
                poisson_lambda=self.cfg.poisson_lambda_nodes,
                flat_target=self.cfg.flat_nodes_target,
                flat_jitter=self.cfg.flat_nodes_jitter,
            )
            cores = self._sample_attr_array(
                rng=rng,
                size=base_n,
                mode=mode,
                min_value=self.cfg.min_cores,
                max_value=self.cfg.max_cores,
                poisson_lambda=self.cfg.poisson_lambda_cores,
                flat_target=self.cfg.flat_cores_target,
                flat_jitter=self.cfg.flat_cores_jitter,
            )
        else:
            durations = np.array([], dtype=np.int32)
            nodes = np.array([], dtype=np.int32)
            cores = np.array([], dtype=np.int32)

        def _sample_burst_count(prob: float, min_jobs: int, max_jobs: int) -> int:
            if prob <= 0.0 or max_jobs <= 0:
                return 0
            if rng.random() >= prob:
                return 0
            return int(rng.integers(min_jobs, max_jobs + 1))

        small_n = _sample_burst_count(
            self.cfg.burst_small_prob,
            self.cfg.burst_small_jobs_min,
            self.cfg.burst_small_jobs_max,
        )
        if small_n > 0:
            durations = np.concatenate(
                [
                    durations,
                    rng.integers(
                        self.cfg.burst_small_duration_min,
                        self.cfg.burst_small_duration_max + 1,
                        size=small_n,
                    ).astype(np.int32),
                ]
            )
            nodes = np.concatenate(
                [
                    nodes,
                    rng.integers(
                        self.cfg.burst_small_nodes_min,
                        self.cfg.burst_small_nodes_max + 1,
                        size=small_n,
                    ).astype(np.int32),
                ]
            )
            cores = np.concatenate(
                [
                    cores,
                    rng.integers(
                        self.cfg.burst_small_cores_min,
                        self.cfg.burst_small_cores_max + 1,
                        size=small_n,
                    ).astype(np.int32),
                ]
            )

        heavy_n = _sample_burst_count(
            self.cfg.burst_heavy_prob,
            self.cfg.burst_heavy_jobs_min,
            self.cfg.burst_heavy_jobs_max,
        )
        if heavy_n > 0:
            durations = np.concatenate(
                [
                    durations,
                    rng.integers(
                        self.cfg.burst_heavy_duration_min,
                        self.cfg.burst_heavy_duration_max + 1,
                        size=heavy_n,
                    ).astype(np.int32),
                ]
            )
            nodes = np.concatenate(
                [
                    nodes,
                    rng.integers(
                        self.cfg.burst_heavy_nodes_min,
                        self.cfg.burst_heavy_nodes_max + 1,
                        size=heavy_n,
                    ).astype(np.int32),
                ]
            )
            cores = np.concatenate(
                [
                    cores,
                    rng.integers(
                        self.cfg.burst_heavy_cores_min,
                        self.cfg.burst_heavy_cores_max + 1,
                        size=heavy_n,
                    ).astype(np.int32),
                ]
            )

        total_n = len(durations)
        if self.cfg.hard_cap_jobs is not None and total_n > self.cfg.hard_cap_jobs:
            hard_cap = self.cfg.hard_cap_jobs
            durations = durations[:hard_cap]
            nodes = nodes[:hard_cap]
            cores = cores[:hard_cap]
            total_n = hard_cap

        if total_n == 0:
            return []

        return [JobSpec(int(durations[i]), int(nodes[i]), int(cores[i])) for i in range(total_n)]
