"""
Run with:
python -m test.test_inspect_workloadgen --arrivals poisson --poisson-lambdas4 200,10,6,24 --max-jobs-hour 1500 --hours 336 --plot --burst-small-prob 0.2 --burst-heavy-prob 0.02
"""

# inspect_workloadgen.py
import argparse
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from src.workloadgen import WorkloadGenConfig, WorkloadGenerator
from train import _parse_quad_floats, _parse_quad_ints, _parse_quad_ranges


def digest_jobs_triplets(triplets):
    """
    Stable digest to verify determinism.

    We digest (hour_idx, duration, nodes, cores_per_node) so the hash is robust against
    future refactors that might change how jobs are flattened/stored.
    """
    arr = np.array(triplets, dtype=np.int32)  # (hour, duration, nodes, cores)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def summarize(name, x):
    x = np.asarray(x)
    if x.size == 0:
        print(f"{name}: (empty)")
        return
    qs = np.percentile(x, [0, 1, 10, 50, 90, 99, 100])
    print(
        f"{name}: n={x.size} mean={x.mean():.3f} std={x.std():.3f} "
        f"min/p1/p10/p50/p90/p99/max={qs[0]:.3f}/{qs[1]:.3f}/{qs[2]:.3f}/{qs[3]:.3f}/{qs[4]:.3f}/{qs[5]:.3f}/{qs[6]:.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--hours", type=int, default=24 * 14)
    ap.add_argument("--arrivals", choices=["flat", "poisson", "uniform"], default="poisson")
    ap.add_argument("--poisson-lambda", type=float, default=200.0, help="Legacy: arrivals-only poisson lambda.")
    ap.add_argument("--poisson-lambdas4", type=_parse_quad_floats, default=None, help="arrivals,duration,nodes,cores")
    ap.add_argument("--max-jobs-hour", type=int, default=1500)
    ap.add_argument("--plot", action="store_true")
    # Flat params (true flat with optional jitter)
    ap.add_argument("--flat-jobs-hour", type=int, default=200, help="Legacy: arrivals-only flat target.")
    ap.add_argument("--flat-jitter", type=int, default=0, help="Legacy: arrivals-only flat jitter.")
    ap.add_argument("--flat-targets4", type=_parse_quad_ints, default=None, help="arrivals,duration,nodes,cores")
    ap.add_argument("--flat-jitters4", type=_parse_quad_ints, default=None, help="arrivals,duration,nodes,cores")
    ap.add_argument(
        "--uniform-ranges4",
        type=_parse_quad_ranges,
        default=None,
        help="a_min:a_max,d_min:d_max,n_min:n_max,c_min:c_max",
    )
    ap.add_argument("--burst-small-prob", type=float, default=0.0, help="Probability of additive small-job burst per hour.")
    ap.add_argument("--burst-heavy-prob", type=float, default=0.0, help="Probability of additive heavy-job burst per hour.")
    args = ap.parse_args()

    # Default ranges (used for uniform and for clipping in all modes).
    uniform_min_jobs = 0
    max_jobs_hour = int(args.max_jobs_hour)
    min_duration, max_duration = 1, 170
    min_nodes, max_nodes = 1, 16
    min_cores, max_cores = 1, 96
    if args.uniform_ranges4 is not None:
        (uniform_min_jobs, max_jobs_hour), (min_duration, max_duration), (min_nodes, max_nodes), (min_cores, max_cores) = args.uniform_ranges4

    default_duration_mid = (min_duration + max_duration) // 2
    default_nodes_mid = (min_nodes + max_nodes) // 2
    default_cores_mid = (min_cores + max_cores) // 2

    if args.poisson_lambdas4 is not None:
        poisson_lambda_arrivals, poisson_lambda_duration, poisson_lambda_nodes, poisson_lambda_cores = args.poisson_lambdas4
    else:
        poisson_lambda_arrivals = float(args.poisson_lambda)
        poisson_lambda_duration = float(default_duration_mid)
        poisson_lambda_nodes = float(default_nodes_mid)
        poisson_lambda_cores = float(default_cores_mid)

    if args.flat_targets4 is not None:
        flat_jobs_per_hour, flat_duration_target, flat_nodes_target, flat_cores_target = args.flat_targets4
    else:
        flat_jobs_per_hour = int(args.flat_jobs_hour)
        flat_duration_target = default_duration_mid
        flat_nodes_target = default_nodes_mid
        flat_cores_target = default_cores_mid

    if args.flat_jitters4 is not None:
        flat_jitter_arrivals, flat_duration_jitter, flat_nodes_jitter, flat_cores_jitter = args.flat_jitters4
    else:
        flat_jitter_arrivals = int(args.flat_jitter)
        flat_duration_jitter = 0
        flat_nodes_jitter = 0
        flat_cores_jitter = 0

    cfg = WorkloadGenConfig(
        arrivals=args.arrivals,
        uniform_min_new_jobs_per_hour=uniform_min_jobs,
        max_new_jobs_per_hour=max_jobs_hour,
        poisson_lambda=poisson_lambda_arrivals,
        poisson_lambda_duration=poisson_lambda_duration,
        poisson_lambda_nodes=poisson_lambda_nodes,
        poisson_lambda_cores=poisson_lambda_cores,
        flat_jobs_per_hour=flat_jobs_per_hour,
        flat_jitter=flat_jitter_arrivals,
        flat_duration_target=flat_duration_target,
        flat_nodes_target=flat_nodes_target,
        flat_cores_target=flat_cores_target,
        flat_duration_jitter=flat_duration_jitter,
        flat_nodes_jitter=flat_nodes_jitter,
        flat_cores_jitter=flat_cores_jitter,
        burst_small_prob=float(args.burst_small_prob),
        burst_heavy_prob=float(args.burst_heavy_prob),
        min_duration=min_duration,
        max_duration=max_duration,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        min_cores=min_cores,
        max_cores=max_cores,
    )
    gen = WorkloadGenerator(cfg)

    rng = np.random.default_rng(args.seed)

    jobs_per_hour = []
    node_hours_per_hour = []
    core_hours_per_hour = []

    # NEW: robust digest input includes hour_idx
    all_jobs_triplets = []

    for h in range(args.hours):
        jobs = gen.sample(h, rng)
        jobs_per_hour.append(len(jobs))

        # Track jobs with hour info for digest + future debugging
        for j in jobs:
            all_jobs_triplets.append((h, int(j.duration), int(j.nodes), int(j.cores_per_node)))

        nh = sum(int(j.duration) * int(j.nodes) for j in jobs)
        ch = sum(int(j.duration) * int(j.nodes) * int(j.cores_per_node) for j in jobs)
        node_hours_per_hour.append(nh)
        core_hours_per_hour.append(ch)

    print(f"hours: {args.hours}")
    summarize("jobs/hour", jobs_per_hour)
    summarize("node-hours/hour", node_hours_per_hour)
    summarize("core-hours/hour", core_hours_per_hour)

    if all_jobs_triplets:
        # Unpack triplets for summarize (keeps your previous summaries unchanged)
        durations = [t[1] for t in all_jobs_triplets]
        nodes = [t[2] for t in all_jobs_triplets]
        cpn = [t[3] for t in all_jobs_triplets]
        summarize("duration[h]", durations)
        summarize("nodes", nodes)
        summarize("cores/node", cpn)

    print("digest:", digest_jobs_triplets(all_jobs_triplets))

    # Optional determinism self-check (same seed => identical digest)
    rng2 = np.random.default_rng(args.seed)
    all_jobs_triplets_2 = []
    for h in range(args.hours):
        for j in gen.sample(h, rng2):
            all_jobs_triplets_2.append((h, int(j.duration), int(j.nodes), int(j.cores_per_node)))

    assert (
        digest_jobs_triplets(all_jobs_triplets) == digest_jobs_triplets(all_jobs_triplets_2)
    ), "Generator not deterministic under same seed!"

    if args.plot:
        # 4x2 grid so we can keep everything in one figure
        fig, axs = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)

        # time-series
        axs[0, 0].plot(np.arange(args.hours), jobs_per_hour)
        axs[0, 0].set_title("Jobs per hour over time")
        axs[0, 0].set_xlabel("hour index")
        axs[0, 0].set_ylabel("jobs")

        # histogram jobs/hour
        axs[0, 1].hist(jobs_per_hour, bins=50)
        axs[0, 1].set_title("Jobs per hour (hist)")
        axs[0, 1].set_xlabel("jobs/hour")
        axs[0, 1].set_ylabel("count")

        axs[1, 0].hist(node_hours_per_hour, bins=50)
        axs[1, 0].set_title("Node-hours per hour (hourly workload volume)")
        axs[1, 0].set_xlabel("node-hours/hour")
        axs[1, 0].set_ylabel("count")

        axs[1, 1].hist(core_hours_per_hour, bins=50)
        axs[1, 1].set_title("Core-hours per hour (total compute demand per hour)")
        axs[1, 1].set_xlabel("core-hours/hour")
        axs[1, 1].set_ylabel("count")

        # Unpack for plotting histograms
        durations = [t[1] for t in all_jobs_triplets] if all_jobs_triplets else []
        nodes = [t[2] for t in all_jobs_triplets] if all_jobs_triplets else []
        cpn = [t[3] for t in all_jobs_triplets] if all_jobs_triplets else []

        axs[2, 0].hist(durations, bins=50)
        axs[2, 0].set_title("Durations (hours)")
        axs[2, 0].set_xlabel("duration [h]")
        axs[2, 0].set_ylabel("count")

        axs[2, 1].hist(nodes, bins=16)
        axs[2, 1].set_title("Nodes (Jobs shape/Volume)")
        axs[2, 1].set_xlabel("nodes")
        axs[2, 1].set_ylabel("count")

        axs[3, 0].hist(cpn, bins=32)
        axs[3, 0].set_title("Cores per node (Jobs shape/Volume)")
        axs[3, 0].set_xlabel("cores/node")
        axs[3, 0].set_ylabel("count")

        # jobs by hour-of-day
        hod = np.arange(args.hours) % 24
        jobs_by_hod = np.zeros(24, dtype=np.int64)
        for h, k in enumerate(jobs_per_hour):
            jobs_by_hod[hod[h]] += int(k)

        axs[3, 1].bar(np.arange(24), jobs_by_hod)
        axs[3, 1].set_title("Total jobs by hour-of-day")
        axs[3, 1].set_xlabel("hour of day")
        axs[3, 1].set_ylabel("jobs")

        #plt.show()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{args.arrivals}_lambda{poisson_lambda_arrivals}" if args.arrivals == "poisson" else args.arrivals
        fname = f"{prefix}_{timestamp}.png" if prefix else f"Workload-Gen_{timestamp}.png"
        save_path = os.path.join("", fname)
        plt.savefig(save_path, dpi=250, bbox_inches="tight")


if __name__ == "__main__":
    main()
