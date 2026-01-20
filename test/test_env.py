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
    session='test',
    render_mode='',
    quick_plot=False,
    external_prices=None,
    external_durations=None,
    external_jobs=None,
    external_hourly_jobs=None,
    plot_rewards=False,
    plots_dir='sessions/test/plots',
    plot_once=False,
    plot_eff_reward=False,
    plot_price_reward=False,
    plot_idle_penalty=False,
    plot_job_age_penalty=False,
    skip_plot_price=False,
    skip_plot_online_nodes=False,
    skip_plot_used_nodes=False,
    skip_plot_job_queue=False,
    steps_per_iteration=100000,
    evaluation_mode=False
)
episodes = 1

for episode in range(episodes):
    print("episode: ", episode)
    done = False
    obs, info = env.reset()
    while not done:
        random_action = env.action_space.sample()
        # print("  action: ", random_action)
        obs, reward, terminated, truncated, info = env.step(random_action)
        done = terminated or truncated
        # print("  reward: [", reward, "]")

print('done')
