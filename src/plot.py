import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os

def plot(env, num_hours, max_nodes, save=True, show=True, suffix=""):
    hours = np.arange(num_hours)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left y-axis for electricity price
    color = 'tab:blue'
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Electricity Price (€/MWh)', color=color)
    if not env.plot_config.skip_plot_price:
        ax1.plot(hours, env.metrics.price_stats, color=color, label='Electricity Price (€/MWh)')
    ax1.tick_params(axis='y', labelcolor=color)

    # Right y-axis for counts and rewards
    ax2 = ax1.twinx()
    ax2.set_ylabel('Count / Rewards', color='tab:orange')

    # Original metrics
    if not env.plot_config.skip_plot_online_nodes:
        ax2.plot(hours, env.metrics.on_nodes, color='orange', label='Online Nodes')
    if not env.plot_config.skip_plot_used_nodes:
        ax2.plot(hours, env.metrics.used_nodes, color='green', label='Used Nodes')
    if not env.plot_config.skip_plot_job_queue:
        ax2.plot(hours, env.metrics.job_queue_sizes, color='red', label='Job Queue Size')

    # New metrics with dashed lines
    if env.plot_config.plot_eff_reward:
        ax2.plot(hours, env.metrics.eff_rewards, color='brown', linestyle='--', label='Efficiency Rewards')
    if env.plot_config.plot_price_reward:
        ax2.plot(hours, env.metrics.price_rewards, color='blue', linestyle='--', label='Price Rewards')
    if env.plot_config.plot_idle_penalty:
        ax2.plot(hours, env.metrics.idle_penalties, color='green', linestyle='--', label='Idle Penalties')
    if env.plot_config.plot_job_age_penalty:
        ax2.plot(hours, env.metrics.job_age_penalties, color='yellow', linestyle='--', label='Job Age Penalties')

    ax2.tick_params(axis='y')
    if env.plot_config.plot_idle_penalty or env.plot_config.plot_job_age_penalty:
        ax2.set_ylim(-100, max_nodes)
    else:
        ax2.set_ylim(0, max_nodes)

    # Calculate job metrics
    completion_rate = (env.metrics.jobs_completed / env.metrics.jobs_submitted * 100) if env.metrics.jobs_submitted > 0 else 0
    baseline_completion_rate = (env.metrics.baseline_jobs_completed / env.metrics.baseline_jobs_submitted * 100) if env.metrics.baseline_jobs_submitted > 0 else 0
    avg_wait = env.metrics.total_job_wait_time / env.metrics.jobs_completed if env.metrics.jobs_completed > 0 else 0
    baseline_avg_wait = env.metrics.baseline_total_job_wait_time / env.metrics.baseline_jobs_completed if env.metrics.baseline_jobs_completed > 0 else 0

    plt.title(f"{env.session} | ep:{env.current_episode} step:{env.current_step} | {env.weights}\n"
              f"Cost: €{env.metrics.total_cost:.0f}, Base: €{env.metrics.baseline_cost:.0f} "
              f"(+{env.metrics.baseline_cost - env.metrics.total_cost:.0f}, {((env.metrics.baseline_cost - env.metrics.total_cost) / env.metrics.baseline_cost) * 100:.1f}%), "
              f"Base_Off: €{env.metrics.baseline_cost_off:.0f} "
              f"(+{env.metrics.baseline_cost_off - env.metrics.total_cost:.0f}, {((env.metrics.baseline_cost_off - env.metrics.total_cost) / env.metrics.baseline_cost_off) * 100:.1f}%)\n"
              f"Jobs: {env.metrics.jobs_completed}/{env.metrics.jobs_submitted} ({completion_rate:.0f}%, "
              f"wait={avg_wait:.1f}h, Q={env.metrics.max_queue_size_reached}) | "
              f"Base: {env.metrics.baseline_jobs_completed}/{env.metrics.baseline_jobs_submitted} ({baseline_completion_rate:.0f}%, "
              f"wait={baseline_avg_wait:.1f}h, Q={env.metrics.baseline_max_queue_size_reached})",
              fontsize=9)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    prefix = f"e{env.weights.efficiency_weight}_p{env.weights.price_weight}_i{env.weights.idle_weight}_d{env.weights.job_age_weight}"

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{env.plots_dir}{prefix}_{suffix:09d}_{timestamp}.png")
        print(f"Figure saved as: {env.plots_dir}{prefix}_{suffix:09d}_{timestamp}.png\nExpecting next save after {env.next_plot_save + env.steps_per_iteration}")
    if show:
        plt.show()

    plt.close(fig)

def plot_reward(env, num_used_nodes, num_idle_nodes, current_price, num_off_nodes, average_future_price, num_processed_jobs, num_node_changes, job_queue_2d, max_nodes):
    used_nodes, idle_nodes, rewards = [], [], []
    noop_print = lambda *args: None

    num_unprocessed_jobs = np.sum(job_queue_2d[:, 0] > 0)

    for i in range(max_nodes + 1):
        for j in range(max_nodes + 1 - i):
            reward, _, _, _, _, _ = env.reward_calculator.calculate(
                i, j, current_price, average_future_price, num_off_nodes,
                num_processed_jobs, num_node_changes, job_queue_2d,
                num_unprocessed_jobs, env.weights, 0, noop_print
            )
            used_nodes.append(i)
            idle_nodes.append(j)
            rewards.append(reward)

    plt.figure(figsize=(14, 12))

    scatter = plt.scatter(used_nodes, idle_nodes, c=rewards, cmap='viridis', s=50)
    plt.colorbar(scatter, label='Reward')

    plt.xlabel('Number of Used Nodes')
    plt.ylabel('Number of Idle Nodes')

    title = f"session: {env.session}, step: {env.current_step}, episode: {env.current_episode}\ncurrent_price: {current_price:.2f}, average_future_price: {average_future_price:.2f}\nnum_processed_jobs: {num_processed_jobs}, num_node_changes: {num_node_changes}, num_off_nodes: {num_off_nodes}"
    plt.title(title, fontsize=10)

    plt.plot([0, max_nodes], [max_nodes, 0], 'r--', linewidth=2, label='Max Nodes Constraint')
    plt.plot([0, max_nodes - num_off_nodes], [max_nodes - num_off_nodes, 0], 'b--', linewidth=2, label='Online/Offline Separator')

    current_reward, _, _, _, _, _ = env.reward_calculator.calculate(
        num_used_nodes, num_idle_nodes, current_price, average_future_price,
        max_nodes - num_used_nodes - num_idle_nodes, num_processed_jobs,
        num_node_changes, job_queue_2d, num_unprocessed_jobs,
        env.weights, 0, noop_print
    )
    plt.scatter(num_used_nodes, num_idle_nodes, color='red', s=100, zorder=5, label=f'Current Reward: {current_reward:.2f}')

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cumulative_savings(env, episode_costs, session_dir, months=12, save=True, show=True):
    """
    Plot cumulative cost savings over time from multiple episodes.

    Args:
        episode_costs: List of episode cost data
        session_dir: Path to session directory for saving plots
        months: Number of months to simulate (each episode = 2 weeks)
        save: Whether to save the plot
        show: Whether to display the plot
    """

    if not episode_costs:
        print("No episode cost data available.")
        print("Run training with cost tracking enabled first.")
        return

    episodes_needed = months * 2  # 2 episodes per month (each episode = 2 weeks)
    episodes_available = len(episode_costs)

    if episodes_available < episodes_needed:
        print(f"Warning: Only {episodes_available} episodes available, requested {episodes_needed}")
        episodes_needed = episodes_available

    # Calculate cumulative savings for both baselines
    cumulative_savings = []
    cumulative_savings_off = []
    monthly_savings_pct = []
    monthly_savings_pct_off = []
    total_saved = 0
    total_saved_off = 0

    for i in range(episodes_needed):
        episode_data = episode_costs[i]
        agent_cost = episode_data['agent_cost']
        baseline_cost = episode_data['baseline_cost']
        baseline_cost_off = episode_data['baseline_cost_off']

        # Cumulative savings vs baseline (with idle nodes)
        episode_savings = baseline_cost - agent_cost
        total_saved += episode_savings
        cumulative_savings.append(total_saved)

        # Cumulative savings vs baseline_off (no idle nodes)
        episode_savings_off = baseline_cost_off - agent_cost
        total_saved_off += episode_savings_off
        cumulative_savings_off.append(total_saved_off)

        # Calculate monthly savings percentage (every 2 episodes)
        if i % 2 == 1:  # End of month
            month_baseline = episode_costs[i-1]['baseline_cost'] + baseline_cost
            month_baseline_off = episode_costs[i-1]['baseline_cost_off'] + baseline_cost_off
            month_agent = episode_costs[i-1]['agent_cost'] + agent_cost

            month_savings_pct = ((month_baseline - month_agent) / month_baseline) * 100 if month_baseline > 0 else 0
            monthly_savings_pct.append(month_savings_pct)
            monthly_savings_pct.append(month_savings_pct)  # Duplicate for visualization

            month_savings_pct_off = ((month_baseline_off - month_agent) / month_baseline_off) * 100 if month_baseline_off > 0 else 0
            monthly_savings_pct_off.append(month_savings_pct_off)
            monthly_savings_pct_off.append(month_savings_pct_off)  # Duplicate for visualization

    # Create time axis (2-week intervals)
    time_periods = np.arange(1, episodes_needed + 1) * 2  # Convert to weeks
    time_months = time_periods / 4.33  # Convert to months (approximately)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Primary axis - Cumulative savings (€)
    color1 = 'tab:blue'
    color1b = 'tab:cyan'
    ax1.set_xlabel('Time (Months)', fontsize=12)
    ax1.set_ylabel('Cumulative Savings (€)', fontsize=12)
    line1 = ax1.plot(time_months, cumulative_savings, color=color1, linewidth=3, label='Savings vs Baseline (with idle)')
    line1b = ax1.plot(time_months, cumulative_savings_off, color=color1b, linewidth=3, linestyle='--', label='Savings vs Baseline_off (no idle)')
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3)

    # Secondary axis - Monthly savings percentage
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    color2b = 'tab:red'
    ax2.set_ylabel('Monthly Savings (%)', fontsize=12)
    line2 = ax2.plot(time_months, monthly_savings_pct, color=color2, linewidth=2, linestyle=':', alpha=0.7, label='Monthly % (vs baseline)')
    line2b = ax2.plot(time_months, monthly_savings_pct_off, color=color2b, linewidth=2, linestyle=':', alpha=0.7, label='Monthly % (vs baseline_off)')
    ax2.tick_params(axis='y')
    max_pct = max(max(monthly_savings_pct) if monthly_savings_pct else 0, max(monthly_savings_pct_off) if monthly_savings_pct_off else 0)
    ax2.set_ylim(0, max_pct * 1.1 if max_pct > 0 else 100)

    # Add seasonal shading (optional)
    for i in range(0, int(months), 3):
        if i + 3 <= months:
            ax1.axvspan(i, i + 3, alpha=0.1, color='gray')

    # Title and statistics
    final_savings = cumulative_savings[-1]
    final_savings_off = cumulative_savings_off[-1]
    avg_monthly_savings = np.mean(monthly_savings_pct) if monthly_savings_pct else 0
    avg_monthly_savings_off = np.mean(monthly_savings_pct_off) if monthly_savings_pct_off else 0

    plt.title(f'PowerSched Long-Term Cost Savings Analysis\n'
              f'{env.weights}\n'
              f'Savings vs Baseline: €{final_savings:,.0f} ({avg_monthly_savings:.1f}% avg) | '
              f'Savings vs Baseline_off: €{final_savings_off:,.0f} ({avg_monthly_savings_off:.1f}% avg)',
              fontsize=14, pad=20)

    # Add inset box with key metrics
    textstr = (f'Vs Baseline (with idle):\n'
               f'  €{final_savings:,.0f} | {avg_monthly_savings:.1f}%\n'
               f'Vs Baseline_off (no idle):\n'
               f'  €{final_savings_off:,.0f} | {avg_monthly_savings_off:.1f}%')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    # Combine legends
    lines = line1 + line1b + line2 + line2b
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=9)

    plt.tight_layout()

    if save:
        prefix = f"e{env.weights.efficiency_weight}_p{env.weights.price_weight}_i{env.weights.idle_weight}_d{env.weights.job_age_weight}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(session_dir, f"cumulative_savings_{prefix}_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cumulative savings plot saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)

    return {
        'total_savings': final_savings,
        'avg_monthly_savings_pct': avg_monthly_savings,
        'total_savings_off': final_savings_off,
        'avg_monthly_savings_pct_off': avg_monthly_savings_off
    }