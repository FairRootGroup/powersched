from stable_baselines3.common.env_checker import check_env
from src.environment import ComputeClusterEnv
from src.weights import Weights
from src.plot_config import PlotConfig

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
    external_prices=None,
    external_durations=None,
    external_jobs=None,
    external_hourly_jobs=None,
    plot_config=PlotConfig(
        plot_price=False,
        plot_online_nodes=False,
        plot_used_nodes=False,
        plot_job_queue=False,
    ),
    steps_per_iteration=100000,
    evaluation_mode=False
)

check_env(env)

print('done')
