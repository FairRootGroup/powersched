#!/usr/bin/env python3

'''Standalone sanity check for Env, that instantiates the env the same way "train.py" does, but runs a battery of check
like API compliance, invariants, determinism.'''

'''Use it as:
- Quick invariant run: python -m test.test_sanity_env.py --steps 200
- Full belt-and-braces: python -m test.test_sanity_env.py --check-gym --check-determinism --steps 300
- With external data: python -m test.test_sanity_env.py --prices <<data/prices.csv>> --hourly-jobs <<data/slurm_hourly.log>> --steps 300 --print-job-every 50 --print-job-kind both
'''

import argparse
import numpy as np

from gymnasium.utils.env_checker import check_env

from src.environment import ComputeClusterEnv, Weights
from src.plot_config import PlotConfig
import pandas as pd
from src.workloadgen import WorkloadGenerator, WorkloadGenConfig

# Import environment variables:
from src.config import (
    MAX_JOB_DURATION,
    MIN_NODES_PER_JOB, MAX_NODES_PER_JOB,
    MIN_CORES_PER_JOB,
    CORES_PER_NODE, EPISODE_HOURS
)


def load_prices(prices_file_path: str | None):
    if not prices_file_path:
        return None
    df = pd.read_csv(prices_file_path, parse_dates=["Date"])
    prices = df["Price"].astype(float).tolist()
    print(f"Loaded {len(prices)} prices from CSV: {prices_file_path}")
    return prices

# -----------------------------
# Invariants / sanity checks
# -----------------------------
def _extract(obs):
    nodes = obs["nodes"]
    q = obs["job_queue"].reshape(-1, 4)
    prices = obs["predicted_prices"]
    return nodes, q, prices

def check_invariants(env, obs):
    nodes, q, prices = _extract(obs)

    # ---- Shapes ----
    assert nodes.ndim == 1, nodes.shape
    assert q.ndim == 2 and q.shape[1] == 4, q.shape
    assert prices.shape == (24,), prices.shape

    # ---- Node bounds ----
    assert np.all(nodes >= -1), f"nodes < -1 exists, min={nodes.min()}"
    assert np.all(nodes <= MAX_JOB_DURATION), f"nodes > MAX_JOB_DURATION exists, max={nodes.max()}"

    # ---- Predicted prices must be finite ----
    assert np.all(np.isfinite(prices)), "predicted_prices contains NaN/inf"

    # ---- cores_available invariants (from env, not obs) ----
    cores_available = env.cores_available
    assert cores_available.shape == nodes.shape
    assert np.all((cores_available >= 0) & (cores_available <= CORES_PER_NODE)), f"cores_available out of bounds min={cores_available.min()} max={cores_available.max()}"

    off = (nodes == -1)
    idle = (nodes == 0)

    assert np.all(cores_available[off] == 0), "off nodes must have 0 cores_available"
    assert np.all(cores_available[idle] == CORES_PER_NODE), "idle nodes must have full cores_available"

    # ---- Queue invariants ----
    dur = q[:, 0]
    age = q[:, 1]
    nn  = q[:, 2]
    cpn = q[:, 3]

    # slots either "all zeros" or "duration > 0"
    all_zero = np.all(q == 0, axis=1)
    assert np.all((dur == 0) == all_zero), "queue has partially-zero slots (corruption / holes)"

    # bounds (based on your constants; keep hard-coded here to avoid importing env constants)
    assert np.all((dur >= 0) & (dur <= 170)), f"duration out of bounds min={dur.min()} max={dur.max()}"
    assert np.all((age >= 0) & (age <= 168)), f"age out of bounds min={age.min()} max={age.max()}"

    # if a job exists, its nodes/cores should be positive and within limits
    active = (dur > 0)
    if np.any(active):
        assert np.all((nn[active] >= 1) & (nn[active] <= 16)), f"job nnodes out of bounds nn={nn[active]}"
        assert np.all((cpn[active] >= 1) & (cpn[active] <= 96)), f"cores_per_node out of bounds cpn={cpn[active]}"

    # ---- next_empty_slot sanity (optional) ----
    if hasattr(env, "next_empty_slot"):
        ne = env.next_empty_slot
        if ne < len(q):
            assert q[ne, 0] == 0, "next_empty_slot does not point at an empty slot"
        if ne > 0:
            assert np.all(q[:ne, 0] != 0), "hole exists before next_empty_slot"


def check_obs_is_copy(env, obs):
    # ensure obs doesn't alias env.state buffers
    before = int(env.state["nodes"][0])
    obs["nodes"][0] = 12345
    after = int(env.state["nodes"][0])
    assert after == before, "obs['nodes'] aliases env.state['nodes'] (not a copy)"


def determinism_test(make_env, seed, n_steps=200):
    # fixed action sequence
    env0 = make_env()
    env0.reset(seed=seed, options={"price_start_index": 0})
    actions = [env0.action_space.sample() for _ in range(n_steps)]
    env0.close()

    def rollout():
        env = make_env()
        # Pin external price window so determinism doesn't vary by episode.
        obs, _ = env.reset(seed=seed, options={"price_start_index": 0})
        traj = []
        done = False
        i = 0
        while not done and i < n_steps:
            obs, r, term, trunc, info = env.step(actions[i])
            # record a small fingerprint
            traj.append((
                float(r),
                float(info.get("step_cost", 0.0)),
                int(info.get("num_unprocessed_jobs", -1)),
                int(info.get("num_on_nodes", -1)),
                int(info.get("dropped_this_episode", -1)),
            ))
            done = term or trunc
            i += 1
        env.close()
        return traj

    a = rollout()
    b = rollout()
    assert a == b, "Determinism failed: same seed + same actions produced different trajectories"


# -----------------------------
# CLI + env construction
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Sanity checks for ComputeClusterEnv (no training).")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--check-gym", action="store_true", help="Run gymnasium check_env")
    p.add_argument("--check-determinism", action="store_true")
    # mirror train.py-ish knobs (mostly optional)
    p.add_argument("--session", default="sanity")
    p.add_argument("--render", type=str, default="none", choices=["human", "none"])
    p.add_argument("--prices", default="")
    p.add_argument("--job-durations", default="")
    p.add_argument("--jobs", default="")
    p.add_argument("--hourly-jobs", default="")
    p.add_argument("--efficiency-weight", type=float, default=0.5)
    p.add_argument("--price-weight", type=float, default=0.2)
    p.add_argument("--idle-weight", type=float, default=0.1)
    p.add_argument("--job-age-weight", type=float, default=0.1)
    p.add_argument("--drop-weight", type=float, default=0.1)
    p.add_argument("--workload-gen",type=str,default="",choices=["", "flat", "poisson"],help="Enable workload generator (default: disabled).",)
    p.add_argument("--wg-poisson-lambda", type=float, default=200.0, help="Poisson lambda for jobs/hour.")
    p.add_argument("--wg-max-jobs-hour", type=int, default=1500, help="Cap jobs/hour for generator.")
    p.add_argument("--print-job-every", type=int, default=0,
              help="Print one sample job every N steps (0 disables).")
    p.add_argument("--print-job-kind", choices=["queue", "running", "both"], default="queue",
              help="Where to sample the job from.")
    p.add_argument("--print-job-index", type=int, default=-1,
              help="Queue index to print (>=0), or -1 to print first active job.")


    return p.parse_args()


def make_env_from_args(args, env_cls=ComputeClusterEnv):
    weights = Weights(
        efficiency_weight=args.efficiency_weight,
        price_weight=args.price_weight,
        idle_weight=args.idle_weight,
        job_age_weight=args.job_age_weight,
        drop_weight=args.drop_weight
    )

    workload_gen = None
    if args.workload_gen:
        cfg = WorkloadGenConfig(
            arrivals=args.workload_gen,
            poisson_lambda=args.wg_poisson_lambda,
            max_new_jobs_per_hour=args.wg_max_jobs_hour,
            min_duration=1,
            max_duration=MAX_JOB_DURATION,
            min_nodes=MIN_NODES_PER_JOB,
            max_nodes=MAX_NODES_PER_JOB,
            min_cores=MIN_CORES_PER_JOB,
            max_cores=CORES_PER_NODE,
        )
        workload_gen = WorkloadGenerator(cfg)


     # Train.py passes strings; the env treats "" as falsy in some places and truthy in others.
    # To be safe: normalize "" -> None here.
    def norm_path(x):
        return None if (x is None or str(x).strip() == "") else x

    return env_cls(
        weights=weights,
        session=args.session,
        render_mode=args.render,
        external_prices=load_prices(args.prices),
        external_durations=norm_path(args.job_durations),
        external_jobs=norm_path(args.jobs),
        external_hourly_jobs=norm_path(args.hourly_jobs),
        plot_config=PlotConfig(
            skip_plot_price=True,
            skip_plot_online_nodes=True,
            skip_plot_used_nodes=True,
            skip_plot_job_queue=True,
        ),
        steps_per_iteration=EPISODE_HOURS,  # prevent plot cadence surprises
        evaluation_mode=False,
        workload_gen=workload_gen
    )

def maybe_print_job(env, obs, step_idx, every, kind="queue", job_index=-1):
    if not every or every <= 0:
        return
    if step_idx % every != 0:
        return

    nodes, q, prices = _extract(obs)

    def print_queue_job():
        active = np.flatnonzero(q[:, 0] > 0)
        if active.size == 0:
            print(f"[job@step {step_idx}] queue empty")
            return
        idx = int(active[0]) if job_index < 0 else int(job_index)
        if idx < 0 or idx >= q.shape[0]:
            print(f"[job@step {step_idx}] invalid queue index {idx}")
            return
        d, a, nn, cpn = map(int, q[idx])
        print(f"[job@step {step_idx}] QUEUE idx={idx}: dur_h={d} age_h={a} nodes={nn} cores_per_node={cpn}")

    def print_running_job():
        if not env.running_jobs:
            print(f"[job@step {step_idx}] running_jobs empty")
            return
        # deterministic-ish: smallest job_id
        job_id = sorted(env.running_jobs.keys())[0]
        jd = env.running_jobs[job_id]
        dur = int(jd["duration"])
        alloc = jd.get("allocation", [])
        nn = len(alloc)
        cpn = int(alloc[0][1]) if alloc else 0
        node_ids = [int(x[0]) for x in alloc[:8]]
        more = "" if len(alloc) <= 8 else f" (+{len(alloc)-8} more)"
        print(f"[job@step {step_idx}] RUNNING job_id={job_id}: rem_h={dur} nodes={nn} cores_per_node={cpn} node_idxs={node_ids}{more}")

    if kind in ("queue", "both"):
        print_queue_job()
    if kind in ("running", "both"):
        print_running_job()



def main():
    args = parse_args()

    class DeterministicPriceEnv(ComputeClusterEnv):
        def reset(self, seed=None, options=None):
            if options is None:
                options = {}
            if seed is not None and "price_start_index" not in options:
                options = dict(options)
                options["price_start_index"] = 0
            return super().reset(seed=seed, options=options)


# -------------------------------------

    seed = 123
    action = np.array([1, 0], dtype=np.int64)  # "maintain, magnitude 1" effectively

    env = make_env_from_args(args, env_cls=DeterministicPriceEnv)

    o1, _ = env.reset(seed=seed)
    o1s, r1, t1, tr1, i1 = env.step(action)

    o2, _ = env.reset(seed=seed)
    o2s, r2, t2, tr2, i2 = env.step(action)

    def cmp(name, a, b):
        eq = np.array_equal(a, b)
        print(name, "equal:", eq)
        if not eq:
            # show first mismatch
            idx = np.flatnonzero(a.flatten() != b.flatten())[0]
            print(" first mismatch idx:", idx, "a:", a.flatten()[idx], "b:", b.flatten()[idx])

    cmp("nodes", o1s["nodes"], o2s["nodes"])
    cmp("job_queue", o1s["job_queue"], o2s["job_queue"])
    cmp("predicted_prices", o1s["predicted_prices"], o2s["predicted_prices"])
    print("reward", r1, r2)
    print("info.current_price", i1.get("current_price"), i2.get("current_price"))

#----------------------------------------

    # 1) Gym API compliance (optional)
    if args.check_gym:
        # Pin external price window so gym's determinism check is meaningful.
        env = make_env_from_args(args, env_cls=DeterministicPriceEnv)
        check_env(env, skip_render_check=True)
        env.close()
        print("[OK] gymnasium check_env passed")

    # 2) Invariants + copy checks during random rollout
    env = make_env_from_args(args)
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        env.action_space.seed(args.seed + ep)  # IMPORTANT: deterministic action sampling
        check_invariants(env, obs)
        #check_obs_is_copy(env, obs)

        done = False
        steps = 0
        while not done and steps < args.steps:
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            maybe_print_job(
                env, obs,
                step_idx=steps,
                every=args.print_job_every,
                kind=args.print_job_kind,
                job_index=args.print_job_index,
            )
            check_invariants(env, obs)
            done = term or trunc
            steps += 1

        print(f"[OK] episode={ep} steps={steps} done={done}")
    env.close()

    # 3) Determinism (optional)
    if args.check_determinism:
        determinism_test(lambda: make_env_from_args(args), seed=args.seed, n_steps=min(args.steps, 500))
        print("[OK] determinism test passed")

    print("done")


if __name__ == "__main__":
    main()
