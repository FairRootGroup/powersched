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
    if not env.skip_plot_price:
        ax1.plot(hours, env.price_stats, color=color, label='Electricity Price (€/MWh)')
    ax1.tick_params(axis='y', labelcolor=color)

    # Right y-axis for counts and rewards
    ax2 = ax1.twinx()
    ax2.set_ylabel('Count / Rewards', color='tab:orange')

    # Original metrics
    if not env.skip_plot_online_nodes:
        ax2.plot(hours, env.on_nodes, color='orange', label='Online Nodes')
    if not env.skip_plot_used_nodes:
        ax2.plot(hours, env.used_nodes, color='green', label='Used Nodes')
    if not env.skip_plot_job_queue:
        ax2.plot(hours, env.job_queue_sizes, color='red', label='Job Queue Size')

    # New metrics with dashed lines
    if env.plot_eff_reward:
        ax2.plot(hours, env.eff_rewards, color='brown', linestyle='--', label='Efficiency Rewards')
    if env.plot_price_reward:
        ax2.plot(hours, env.price_rewards, color='blue', linestyle='--', label='Price Rewards')
    if env.plot_idle_penalty:
        ax2.plot(hours, env.idle_penalties, color='green', linestyle='--', label='Idle Penalties')
    if env.plot_job_age_penalty:
        ax2.plot(hours, env.job_age_penalties, color='yellow', linestyle='--', label='Job Age Penalties Penalties')

    ax2.tick_params(axis='y')
    if env.plot_idle_penalty or env.plot_job_age_penalty:
        ax2.set_ylim(-100, max_nodes)
    else:
        ax2.set_ylim(0, max_nodes)

    plt.title(f"session: {env.session}, "
              f"episode: {env.current_episode}, step: {env.current_step}\n"
              f"{env.weights}\n"
              f"Cost: €{env.total_cost:.2f}, "
              f"Base_Cost: €{env.baseline_cost:.2f} "
              f"({'+' if env.baseline_cost - env.total_cost >= 0 else '-'}"
              f"{abs(env.baseline_cost - env.total_cost):.2f}) {((env.baseline_cost - env.total_cost) / env.baseline_cost) * 100:.2f}%, "
              f"Base_Cost_Off: €{env.baseline_cost_off:.2f} "
              f"({'+' if env.baseline_cost_off - env.total_cost >= 0 else '-'}"
              f"{abs(env.baseline_cost_off - env.total_cost):.2f}) {((env.baseline_cost_off - env.total_cost) / env.baseline_cost_off) * 100:.2f}%")

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

    for i in range(max_nodes + 1):
        for j in range(max_nodes + 1 - i):
            reward, _ = env.calculate_reward(i, j, current_price, average_future_price, num_off_nodes, num_processed_jobs, num_node_changes, job_queue_2d)
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

    current_reward, _ = env.calculate_reward(num_used_nodes, num_idle_nodes, current_price, average_future_price, max_nodes - num_used_nodes - num_idle_nodes, num_processed_jobs, num_node_changes, job_queue_2d)
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

    # Calculate cumulative savings
    cumulative_savings = []
    monthly_savings_pct = []
    total_saved = 0

    for i in range(episodes_needed):
        episode_data = episode_costs[i]
        agent_cost = episode_data['agent_cost']
        baseline_cost = episode_data['baseline_cost']

        episode_savings = baseline_cost - agent_cost
        total_saved += episode_savings
        cumulative_savings.append(total_saved)

        # Calculate monthly savings percentage (every 2 episodes)
        if i % 2 == 1:  # End of month
            month_baseline = episode_costs[i-1]['baseline_cost'] + baseline_cost
            month_agent = episode_costs[i-1]['agent_cost'] + agent_cost
            month_savings_pct = ((month_baseline - month_agent) / month_baseline) * 100
            monthly_savings_pct.append(month_savings_pct)
            monthly_savings_pct.append(month_savings_pct)  # Duplicate for visualization

    # Create time axis (2-week intervals)
    time_periods = np.arange(1, episodes_needed + 1) * 2  # Convert to weeks
    time_months = time_periods / 4.33  # Convert to months (approximately)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Primary axis - Cumulative savings (€)
    color1 = 'tab:blue'
    ax1.set_xlabel('Time (Months)', fontsize=12)
    ax1.set_ylabel('Cumulative Savings (€)', color=color1, fontsize=12)
    line1 = ax1.plot(time_months, cumulative_savings, color=color1, linewidth=3, label='Cumulative Savings')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Secondary axis - Monthly savings percentage
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Monthly Savings (%)', color=color2, fontsize=12)
    line2 = ax2.plot(time_months, monthly_savings_pct, color=color2, linewidth=2, linestyle='--', alpha=0.7, label='Monthly Savings %')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(monthly_savings_pct) * 1.1)

    # Add break-even line if implementation cost is known
    # For now, use a placeholder of €20,000
    implementation_cost = 20000
    ax1.axhline(y=implementation_cost, color='red', linestyle=':', linewidth=2, label=f'Break-even (€{implementation_cost:,})')

    # Add seasonal shading (optional)
    for i in range(0, int(months), 3):
        if i + 3 <= months:
            ax1.axvspan(i, i + 3, alpha=0.1, color='gray')

    # Title and statistics
    final_savings = cumulative_savings[-1]
    avg_monthly_savings = np.mean(monthly_savings_pct)
    roi_months = implementation_cost / (final_savings / months) if final_savings > 0 else float('inf')

    plt.title(f'PowerSched Long-Term Cost Savings Analysis\n'
              f'{env.weights}\n'
              f'Total Savings: €{final_savings:,.0f} | '
              f'Avg Monthly Reduction: {avg_monthly_savings:.1f}% | '
              f'ROI Period: {roi_months:.1f} months',
              fontsize=14, pad=20)

    # Add inset box with key metrics
    textstr = f'Total Savings: €{final_savings:,.0f}\nAverage Monthly: {avg_monthly_savings:.1f}% reduction\nROI: {roi_months:.1f} months'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    # Combine legends
    lines = line1 + line2 + [ax1.lines[-1]]  # Include break-even line
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

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
        'roi_months': roi_months
    }