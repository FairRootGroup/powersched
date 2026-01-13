from stable_baselines3.common.env_checker import check_env
from src.environment import ComputeClusterEnv
from src.weights import Weights

weights = Weights(
    efficiency_weight=0.7,
    price_weight=0.2,
    idle_weight=0.1,
    job_age_weight=0.0,
    drop_weight=0.0
)

env = ComputeClusterEnv(
    weights=weights,
    session='check',
    render_mode='none',
    quick_plot=False,
    external_prices=None,
    external_durations=None,
    external_jobs=None,
    external_hourly_jobs=None,
    plot_rewards=False,
    plots_dir='sessions/check/plots',
    plot_once=False,
    plot_eff_reward=False,
    plot_price_reward=False,
    plot_idle_penalty=False,
    plot_job_age_penalty=False,
    skip_plot_price=True,
    skip_plot_online_nodes=True,
    skip_plot_used_nodes=True,
    skip_plot_job_queue=True,
    steps_per_iteration=100000,
    evaluation_mode=False
)

check_env(env)

print('done')
