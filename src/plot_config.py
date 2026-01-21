from dataclasses import dataclass


@dataclass
class PlotConfig:
    quick_plot: bool = False
    plot_rewards: bool = False
    plot_once: bool = False
    plot_eff_reward: bool = False
    plot_price_reward: bool = False
    plot_idle_penalty: bool = False
    plot_job_age_penalty: bool = False
    plot_total_reward: bool = False
    skip_plot_price: bool = False
    skip_plot_online_nodes: bool = False
    skip_plot_used_nodes: bool = False
    skip_plot_job_queue: bool = False
