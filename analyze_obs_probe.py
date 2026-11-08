"""Observation-masking probe: which parts of the observation do trained policies use?

Runs a trained model through the same evaluation as train.py --evaluate-savings,
with one part of the observation altered before the policy sees it.

  --mode closed-loop : the altered observation drives the policy; report savings/service
  --mode agreement   : the unaltered policy drives the env; at every step, also ask the
                       policy what it would do under each alteration and count agreement

Variants: full, aggregates_zero (the pending_* statistics), queue_summary_zero (the queue_*
age/size summaries), time_const (time features fixed at midnight on a Monday), prices_const
(forecast flattened to its mean), prices_roll12 (forecast shifted by 12 hours).

Example:
  python analyze_obs_probe.py --model sessions/x/models/e0.4_p0.4_i0.0_a0.2_d0.0/7500000.zip \
      --weights 0.4,0.4,0.0,0.2,0.0 --prices data-external/prices_2024.csv \
      --variant prices_roll12 --seed 301 --out probe.json
"""
import argparse, json, os, time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from src.environment import ComputeClusterEnv, Weights
from src.prices import load_price_csv

AGGREGATES = ["pending_job_count", "pending_core_hours", "pending_avg_duration", "pending_max_nodes"]
QUEUE_SUMMARIES = ["queue_ages", "queue_age_core_hours", "queue_sizes", "queue_oldest"]
VARIANTS = ["full", "aggregates_zero", "queue_summary_zero", "time_const", "prices_const", "prices_roll12"]


def alter(obs, variant):
    o = {k: v.copy() for k, v in obs.items()}
    if variant == "aggregates_zero":
        for k in AGGREGATES:
            o[k][:] = 0
    elif variant == "queue_summary_zero":
        for k in QUEUE_SUMMARIES:
            o[k][:] = 0
    elif variant == "time_const":
        o["time_features"][:] = (0.0, 1.0, 0.0, 1.0)
    elif variant == "prices_const":
        o["predicted_prices"][:] = o["predicted_prices"].mean()
    elif variant == "prices_roll12":
        # Realistic shape and levels, wrong timing: the forecast shifted by 12 hours.
        o["predicted_prices"] = np.roll(o["predicted_prices"], 12)
    return o


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--weights", required=True, help="e,p,i,a,d")
    p.add_argument("--variant", default="full", choices=VARIANTS)
    p.add_argument("--mode", default="closed-loop", choices=["closed-loop", "agreement"])
    p.add_argument("--seed", type=int, default=301)
    p.add_argument("--eval-months", type=int, default=13)
    p.add_argument("--prices", default="data-external/prices_2023.csv")
    p.add_argument("--hourly-jobs", default="data-internal/allusers-main-30.log")
    p.add_argument("--job-arrival-scale", type=float, default=1.0)
    p.add_argument("--stochastic", action="store_true", help="Sample actions instead of taking the most likely one.")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    set_random_seed(a.seed)
    e, pr, i, ag, d = (float(x) for x in a.weights.split(","))
    prices, price_times = load_price_csv(a.prices)
    scratch = os.path.dirname(os.path.abspath(a.out))
    env = ComputeClusterEnv(weights=Weights(e, pr, i, ag, d), session="probe", render_mode="none",
                            external_prices=prices, external_price_times=price_times,
                            external_durations=None, external_jobs=None,
                            external_hourly_jobs=a.hourly_jobs, plot_config=None,
                            steps_per_iteration=100000, evaluation_mode=True, workload_gen=None,
                            job_arrival_scale=a.job_arrival_scale, jobs_exact_replay=False, output_dir=scratch)
    env.session_dir = env.plots_dir = scratch + "/"
    env.reset(seed=a.seed)
    model = PPO.load(a.model, device="cpu")
    env.set_progress(int(os.path.basename(a.model).split(".")[0]) // 100000)

    agree = {v: {"all": 0, "type": 0, "mag": 0} for v in VARIANTS[1:]}
    steps = 0
    t0 = time.time()
    for _ in range(a.eval_months * 2):
        obs, _ = env.reset()
        done = False
        while not done:
            if a.mode == "agreement":
                action, _ = model.predict(obs, deterministic=True)
                for v in VARIANTS[1:]:
                    alt, _ = model.predict(alter(obs, v), deterministic=True)
                    agree[v]["all"] += int(np.array_equal(alt, action))
                    agree[v]["type"] += int(alt[0] == action[0])
                    agree[v]["mag"] += int(alt[1] == action[1])
            else:
                action, _ = model.predict(alter(obs, a.variant), deterministic=not a.stochastic)
            obs, _, term, trunc, _ = env.step(action)
            steps += 1
            done = term or trunc

    eps = env.metrics.episode_costs
    agent = sum(float(x["agent_cost"]) for x in eps)
    base_off = sum(float(x["baseline_cost_off"]) for x in eps)
    submitted = sum(int(x["jobs_submitted"]) for x in eps)
    completed = sum(int(x["jobs_completed"]) for x in eps)
    launched = sum(int(x.get("jobs_launched", x["jobs_completed"])) for x in eps)
    wait = sum(float(x["avg_wait_time"]) * int(x.get("jobs_launched", x["jobs_completed"])) for x in eps) / max(launched, 1)
    result = {
        "model": a.model, "weights": a.weights, "variant": a.variant, "mode": a.mode, "seed": a.seed,
        "job_arrival_scale": a.job_arrival_scale, "prices": a.prices, "stochastic": a.stochastic,
        "episodes": len(eps), "steps": steps, "seconds": round(time.time() - t0),
        "savings_pct_vs_idle_free_total": 100 * (base_off - agent) / base_off,
        "savings_pct_vs_idle_free_mean_episode": float(np.mean([x["savings_pct_baseline_off"] for x in eps])),
        "completion_pct": 100 * completed / max(submitted, 1),
        "avg_wait_h": wait,
        "jobs_lost": sum(int(x.get("jobs_lost_total", x.get("jobs_dropped", 0))) for x in eps),
    }
    if a.mode == "agreement":
        result["agreement_pct"] = {v: {k: 100 * n / steps for k, n in c.items()} for v, c in agree.items()}
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
