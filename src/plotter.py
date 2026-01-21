import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import numpy as np
import os


def _as_series(x, n):
    if x is None:
        return None
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size >= n:
        return a[:n]
    out = np.full(n, np.nan, dtype=float)
    out[:a.size] = a
    return out


def _compute_cumulative_savings(episode_costs):
    """
    episode_costs: list of dicts with keys:
      agent_cost, baseline_cost, baseline_cost_off
    Returns arrays for plotting.
    """
    if not episode_costs:
        return None

    cum_s = []
    cum_s_off = []
    monthly_pct = []
    monthly_pct_off = []

    total = 0.0
    total_off = 0.0

    for i, ep in enumerate(episode_costs):
        agent = float(ep["agent_cost"])
        base = float(ep["baseline_cost"])
        base_off = float(ep["baseline_cost_off"])

        total += (base - agent)
        total_off += (base_off - agent)
        cum_s.append(total)
        cum_s_off.append(total_off)

        # monthly % every 2 episodes (episode = 2 weeks assumption)
        if i % 2 == 1:
            prev = episode_costs[i - 1]
            month_base = float(prev["baseline_cost"]) + base
            month_base_off = float(prev["baseline_cost_off"]) + base_off
            month_agent = float(prev["agent_cost"]) + agent

            pct = ((month_base - month_agent) / month_base * 100.0) if month_base > 0 else 0.0
            pct_off = ((month_base_off - month_agent) / month_base_off * 100.0) if month_base_off > 0 else 0.0

            # duplicate for step-like visualization
            monthly_pct.extend([pct, pct])
            monthly_pct_off.extend([pct_off, pct_off])

    # x-axis in "months" (2-week steps)
    n_eps = len(episode_costs)
    weeks = (np.arange(1, n_eps + 1) * 2.0)
    months = weeks / 4.33

    # monthly arrays are shorter (only defined at month boundaries) -> pad/align
    if len(monthly_pct) < n_eps:
        last = monthly_pct[-1] if monthly_pct else 0.0
        monthly_pct = monthly_pct + [last] * (n_eps - len(monthly_pct))
        last_off = monthly_pct_off[-1] if monthly_pct_off else 0.0
        monthly_pct_off = monthly_pct_off + [last_off] * (n_eps - len(monthly_pct_off))

    return {
        "months": months,
        "cum_s": np.asarray(cum_s, dtype=float),
        "cum_s_off": np.asarray(cum_s_off, dtype=float),
        "monthly_pct": np.asarray(monthly_pct[:n_eps], dtype=float),
        "monthly_pct_off": np.asarray(monthly_pct_off[:n_eps], dtype=float),
    }


def plot_dashboard(env, num_hours, max_nodes, episode_costs=None, save=True, show=True, suffix=""):
    """
    Per-hour dashboard: price, nodes, queue, reward components, etc.
    NOTE: episode_costs is accepted for backwards compatibility but NOT used here anymore.
          Cumulative savings now lives in plot_cumulative_savings().
    """
    hours = np.arange(num_hours)

    # ----- header text -----
    completion_rate = (env.metrics.jobs_completed / env.metrics.jobs_submitted * 100) if env.metrics.jobs_submitted > 0 else 0.0
    baseline_completion_rate = (env.metrics.baseline_jobs_completed / env.metrics.baseline_jobs_submitted * 100) if env.metrics.baseline_jobs_submitted > 0 else 0.0
    avg_wait = (env.metrics.total_job_wait_time / env.metrics.jobs_completed) if env.metrics.jobs_completed > 0 else 0.0
    baseline_avg_wait = (env.baseline_total_job_wait_time / env.metrics.baseline_jobs_completed) if env.metrics.baseline_jobs_completed > 0 else 0.0

    base_cost = float(env.metrics.baseline_cost)
    base_cost_off = float(env.metrics.baseline_cost_off)
    agent_cost = float(env.metrics.total_cost)

    pct_vs_base = ((base_cost - agent_cost) / base_cost * 100.0) if base_cost > 0 else 0.0
    pct_vs_base_off = ((base_cost_off - agent_cost) / base_cost_off * 100.0) if base_cost_off > 0 else 0.0

    header = (
        f"{env.session} | ep:{env.current_episode} step:{env.current_step} | {env.weights}\n"
        f"Cost: €{agent_cost:.0f}, Base: €{base_cost:.0f} (+{base_cost - agent_cost:.0f}, {pct_vs_base:.1f}%), "
        f"Base_Off: €{base_cost_off:.0f} (+{base_cost_off - agent_cost:.0f}, {pct_vs_base_off:.1f}%)\n"
        f"Jobs: {env.metrics.jobs_completed}/{env.metrics.jobs_submitted} ({completion_rate:.0f}%, wait={avg_wait:.1f}h, Q={env.metrics.max_queue_size_reached}) | "
        f"Base: {env.metrics.baseline_jobs_completed}/{env.metrics.baseline_jobs_submitted} ({baseline_completion_rate:.0f}%, wait={baseline_avg_wait:.1f}h, Q={env.metrics.baseline_max_queue_size_reached})"
    )

    # ----- collect per-hour panels (one / panel, optional overlay) -----
    panels = []

    def add_panel(title, series, ylabel, ylim=None, overlay=None):
        """
        overlay: optional (label, series2)
        """
        s = _as_series(series, num_hours)
        if s is None:
            return

        ov = None
        if overlay is not None:
            ov_label, ov_series = overlay
            s2 = _as_series(ov_series, num_hours)
            if s2 is not None:
                ov = (ov_label, s2)

        panels.append((title, s, ylabel, ylim, ov))

    # Price
    if not env.skip_plot_price:
        add_panel("Electricity price", env.metrics.price_stats, "€/MWh", None)

    # Nodes
    if not env.skip_plot_online_nodes:
        add_panel("Online nodes", env.metrics.on_nodes, "count", (0, max_nodes * 1.1))
    if not env.skip_plot_used_nodes:
        add_panel("Used nodes", env.metrics.used_nodes, "count", (0, max_nodes))

    # Queue + running jobs (same plot)
    if not env.skip_plot_job_queue:
        running_series = getattr(env.metrics, "running_jobs_counts", None)  # optional, may not exist
        add_panel(
            "Job queue & running jobs",
            env.metrics.job_queue_sizes,
            "jobs",
            None,
            overlay=("Running jobs", running_series),
        )

    # Reward components
    if env.plot_eff_reward:
        add_panel("Efficiency reward (%)", env.metrics.eff_rewards, "score", None)
    if env.plot_price_reward:
        add_panel("Price reward (%)", env.metrics.price_rewards, "score", None)
    if env.plot_idle_penalty:
        add_panel("Idle penalty (%)", env.metrics.idle_penalties, "score", None)
    if env.plot_job_age_penalty:
        add_panel("Job-age penalty (%)", env.metrics.job_age_penalties, "score", None)
    if env.plot_total_reward:
        add_panel("Total reward", getattr(env.metrics, "rewards", None), "reward", None)  # optional, may not exist

    if not panels:
        print("plot_dashboard(): nothing to plot.")
        return

    n_pan = len(panels)
    ncols = 2 if n_pan <= 6 else 3
    nrows = int(np.ceil(n_pan / ncols))

    fig = plt.figure(figsize=(14, 3.2 * nrows))
    gs = GridSpec(nrows, ncols, figure=fig)

    # Place panel axes
    axs = []
    for i in range(nrows * ncols):
        r = i // ncols
        c = i % ncols
        axs.append(fig.add_subplot(gs[r, c]))

    # Plot per-hour panels
    for idx, (title, s, ylabel, ylim, overlay) in enumerate(panels):
        ax = axs[idx]
        # main series
        ax.plot(hours, s, label=title)
        # overlay series (e.g. running jobs)
        if overlay is not None:
            ov_label, s2 = overlay
            ax.plot(hours, s2, label=ov_label, linestyle="--")
            ax.legend(fontsize=7)

        ax.set_title(title, fontsize=9, pad=2)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
        if ylim is not None:
            ax.set_ylim(*ylim)

    # Hide unused axes
    for j in range(n_pan, nrows * ncols):
        axs[j].axis("off")

    # Shared x-label
    for ax in axs[(nrows - 1) * ncols : nrows * ncols]:
        if ax.has_data():
            ax.set_xlabel("Hours", fontsize=9)

    # Header text
    fig.subplots_adjust(top=0.82, left=0.06, right=0.98, bottom=0.06, hspace=0.45, wspace=0.25)
    fig.text(0.01, 0.99, header, ha="left", va="top", fontsize=9, family="monospace")

    # Save/show
    prefix = f"e{env.weights.efficiency_weight}_p{env.weights.price_weight}_i{env.weights.idle_weight}_d{env.weights.job_age_weight}"
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{prefix}_{suffix}_{timestamp}.png"
        save_path = os.path.join(env.plots_dir, fname)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Dashboard figure saved as: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_cumulative_savings(env, episode_costs, session_dir=None, save=True, show=True, suffix=""):
    """
    Separate canvas for long-term cumulative savings & monthly % savings.
    """
    data = _compute_cumulative_savings(episode_costs)
    if data is None:
        print("plot_cumulative_savings(): no episode_costs, skipping.")
        return None

    months = data["months"]
    cum_s = data["cum_s"]
    cum_s_off = data["cum_s_off"]
    monthly_pct = data["monthly_pct"]
    monthly_pct_off = data["monthly_pct_off"]

    # Basic stats
    final_savings = float(cum_s[-1])
    final_savings_off = float(cum_s_off[-1])
    avg_monthly_savings = float(np.mean(monthly_pct)) if monthly_pct.size > 0 else 0.0
    avg_monthly_savings_off = float(np.mean(monthly_pct_off)) if monthly_pct_off.size > 0 else 0.0

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Primary axis - cumulative savings (€)
    ax1.set_xlabel("Time (months)", fontsize=12)
    ax1.set_ylabel("Cumulative savings (€)", fontsize=12)
    line1 = ax1.plot(months, cum_s, linewidth=3, label="Savings vs baseline (with idle)")
    line1b = ax1.plot(months, cum_s_off, linewidth=3, linestyle="--", label="Savings vs baseline_off (no idle)")
    ax1.tick_params(axis="y")
    ax1.grid(True, alpha=0.3)

    # Secondary axis - monthly savings %
    ax2 = ax1.twinx()
    ax2.set_ylabel("Monthly savings (%)", fontsize=12)
    line2 = ax2.plot(months, monthly_pct, linewidth=2, linestyle=":", alpha=0.7, label="Monthly % (vs baseline)")
    line2b = ax2.plot(months, monthly_pct_off, linewidth=2, linestyle=":", alpha=0.7, label="Monthly % (vs baseline_off)")
    ax2.tick_params(axis="y")

    max_pct = max(
        float(np.max(monthly_pct)) if monthly_pct.size > 0 else 0.0,
        float(np.max(monthly_pct_off)) if monthly_pct_off.size > 0 else 0.0,
    )
    ax2.set_ylim(0, max_pct * 1.1 if max_pct > 0 else 100)

    # Title and summary box
    weights_str = str(env.weights)
    plt.title(
        f"PowerSched Long-Term Cost Savings Analysis\n{weights_str}\n"
        f"Savings vs Baseline: €{final_savings:,.0f} ({avg_monthly_savings:.1f}% avg) | "
        f"Savings vs Baseline_off: €{final_savings_off:,.0f} ({avg_monthly_savings_off:.1f}% avg)",
        fontsize=14,
        pad=20,
    )

    textstr = (
        f"Vs Baseline (with idle):\n"
        f"  €{final_savings:,.0f} | {avg_monthly_savings:.1f}%\n"
        f"Vs Baseline_off (no idle):\n"
        f"  €{final_savings_off:,.0f} | {avg_monthly_savings_off:.1f}%"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment="top", bbox=props)

    # Combine legends
    lines = line1 + line1b + line2 + line2b
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=9)

    plt.tight_layout()

    # Save/show
    prefix = f"e{env.weights.efficiency_weight}_p{env.weights.price_weight}_i{env.weights.idle_weight}_d{env.weights.job_age_weight}"
    if session_dir is None:
        session_dir = env.plots_dir
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"cumulative_savings_{prefix}_{suffix}_{timestamp}.png"
        save_path = os.path.join(session_dir, fname)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Cumulative savings figure saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)

    return {
        "total_savings": final_savings,
        "avg_monthly_savings_pct": avg_monthly_savings,
        "total_savings_off": final_savings_off,
        "avg_monthly_savings_pct_off": avg_monthly_savings_off,
    }
