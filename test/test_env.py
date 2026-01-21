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
    session='test',
    render_mode='',
    external_prices=None,
    external_durations=None,
    external_jobs=None,
    external_hourly_jobs=None,
    plot_config=PlotConfig(),
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
