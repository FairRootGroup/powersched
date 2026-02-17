"""
Run with:
python -m test.test_inspect_workloadgen --workload-gen poisson --wg-poisson-lambdas4 200,10,6,24 --wg-max-jobs-hour 1500 --hours 336 --plot --wg-burst-small-prob 0.2 --wg-burst-heavy-prob 0.02
"""

# inspect_workloadgen.py
import argparse
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from src.workloadgen import WorkloadGenerator
from src.workloadgen_cli import add_workloadgen_args, build_workloadgen_config


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
    add_workloadgen_args(ap)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    cfg = build_workloadgen_config(args)
    if cfg is None:
        ap.error("--workload-gen is required (e.g. --workload-gen poisson)")
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
        prefix = f"{cfg.arrivals}_lambda{cfg.poisson_lambda}" if cfg.arrivals == "poisson" else cfg.arrivals
        fname = f"{prefix}_{timestamp}.png" if prefix else f"Workload-Gen_{timestamp}.png"
        out_dir = os.path.join(os.path.dirname(__file__), "test_output")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, fname)
        plt.savefig(save_path, dpi=250, bbox_inches="tight")


if __name__ == "__main__":
    main()
