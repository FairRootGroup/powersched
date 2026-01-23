from dataclasses import dataclass


@dataclass
class PlotConfig:
    quick_plot: bool = False
    plot_rewards: bool = False
    plot_once: bool = False
    plot_eff_reward: bool = True
    plot_price_reward: bool = True
    plot_idle_penalty: bool = True
    plot_job_age_penalty: bool = True
    plot_total_reward: bool = True
    plot_price: bool = True
    plot_online_nodes: bool = True
    plot_used_nodes: bool = True
    plot_job_queue: bool = True
