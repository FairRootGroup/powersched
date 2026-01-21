import time

from gymnasium import spaces
import gymnasium as gym
import numpy as np
from colorama import init, Fore

from src.prices_deterministic import Prices
from src.weights import Weights
from src.plot import plot, plot_reward
from src.sampler_duration import durations_sampler
from src.sampler_jobs import DurationSampler
from src.sampler_hourly import hourly_sampler

# Import refactored modules
from src.config import (
    MAX_NODES, MAX_QUEUE_SIZE, MAX_CHANGE, MAX_JOB_DURATION,
    MAX_JOB_AGE, CORES_PER_NODE, MAX_CORES_PER_JOB,
    MAX_NODES_PER_JOB, EPISODE_HOURS
)
from src.job_management import (
    process_ongoing_jobs, add_new_jobs,
    assign_jobs_to_available_nodes
)
from src.node_management import adjust_nodes
from src.reward_calculation import RewardCalculator
from src.baseline import baseline_step
from src.workload_generator import generate_jobs
from src.metrics_tracker import MetricsTracker

# For deterministic RNG
from gymnasium.utils import seeding

init()  # Initialize colorama


class PlottingComplete(Exception):
    """Raised when plotting is complete and the application should terminate."""
    pass


class ComputeClusterEnv(gym.Env):
    """An environment for scheduling compute jobs based on electricity price predictions."""

    metadata = {'render.modes': ['human', 'none']}

    def render(self, mode='human'):
        self.render_mode = mode

    def set_progress(self, iterations):
        self.current_step = iterations * self.steps_per_iteration
        self.current_episode = self.current_step // EPISODE_HOURS
        print(f"Resuming training... step: {self.current_step}, episode: {self.current_episode}, hour: {self.metrics.current_hour}")
        self.next_plot_save = iterations * self.steps_per_iteration + EPISODE_HOURS

    def env_print(self, *args):
        """Prints only if the render mode is 'human'."""
        if self.render_mode == 'human':
            print(*args)

    def __init__(self,
                 weights: Weights,
                 session,
                 render_mode,
                 quick_plot,
                 external_prices,
                 external_durations,
                 external_jobs,
                 external_hourly_jobs,
                 plot_rewards,
                 plots_dir,
                 plot_once,
                 plot_eff_reward,
                 plot_price_reward,
                 plot_idle_penalty,
                 plot_job_age_penalty,
                 skip_plot_price,
                 skip_plot_online_nodes,
                 skip_plot_used_nodes,
                 skip_plot_job_queue,
                 steps_per_iteration,
                 evaluation_mode=False,
                 workload_gen=None):
        super().__init__()

        self.weights = weights
        self.session = session
        self.render_mode = render_mode
        self.quick_plot = quick_plot
        self.external_prices = external_prices
        self.external_durations = external_durations
        self.external_jobs = external_jobs
        self.external_hourly_jobs = external_hourly_jobs
        self.plot_rewards = plot_rewards
        self.plots_dir = plots_dir
        self.plot_once = plot_once
        self.plot_eff_reward = plot_eff_reward
        self.plot_price_reward = plot_price_reward
        self.plot_idle_penalty = plot_idle_penalty
        self.plot_job_age_penalty = plot_job_age_penalty
        self.skip_plot_price = skip_plot_price
        self.skip_plot_online_nodes = skip_plot_online_nodes
        self.skip_plot_used_nodes = skip_plot_used_nodes
        self.skip_plot_job_queue = skip_plot_job_queue
        self.steps_per_iteration = steps_per_iteration
        self.evaluation_mode = evaluation_mode

        self.next_plot_save = self.steps_per_iteration

        # Initialize metrics tracker
        self.metrics = MetricsTracker()

        # Initialize cost tracking for long-term analysis
        self.session_dir = f"sessions/{session}"

        self.prices = Prices(self.external_prices)

        # Initialize deterministic RNG, instead of global RNG
        self.np_random = None
        self._seed = None
        self.workload_gen = workload_gen

        if self.external_durations:
            durations_sampler.init(self.external_durations)

        if self.external_jobs and not self.workload_gen:
            self.jobs_sampler = DurationSampler()
            print(f"Loading jobs from {self.external_jobs}")
            self.jobs_sampler.parse_jobs(self.external_jobs, 60)
            print(f"Parsed jobs for {len(self.jobs_sampler.jobs)} hours")
            print(f"Parsed aggregated jobs for {len(self.jobs_sampler.aggregated_jobs)} hours")
            self.jobs_sampler.precalculate_hourly_jobs(CORES_PER_NODE, MAX_NODES_PER_JOB)
            print(f"Max jobs per hour: {self.jobs_sampler.max_new_jobs_per_hour}")
            print(f"Max job duration: {self.jobs_sampler.max_job_duration}")
            print(f"Parsed hourly jobs for {len(self.jobs_sampler.hourly_jobs)} hours")

        if self.external_hourly_jobs:
            print(f"Loading hourly jobs from {self.external_hourly_jobs}")
            hourly_sampler.parse_jobs(self.external_hourly_jobs)
            print(f"Hourly sampler initialized with 24-hour distributions")

        self.current_step = 0
        self.current_episode = 0

        # Initialize to -1, so that first reset() sets it to 0
        self.episode_idx = -1

        print(f"{self.weights}")
        print(f"prices.MAX_PRICE: {self.prices.MAX_PRICE:.2f}, prices.MIN_PRICE: {self.prices.MIN_PRICE:.2f}")
        print(f"Price Statistics: {self.prices.get_price_stats()}")

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(self.prices)

        # actions: - change number of available nodes:
        #   action_type:      0: decrease, 1: maintain, 2: increase
        #   action_magnitude: 0-MAX_CHANGE (+1ed in the action)
        self.action_space = spaces.MultiDiscrete([3, MAX_CHANGE])

        self.observation_space = spaces.Dict({
            # nodes: [-1: off, 0: idle, >0: booked for n hours]
            'nodes': spaces.Box(
                low=-1,
                high=MAX_JOB_DURATION,
                shape=(MAX_NODES,),
                dtype=np.int32
            ),
            # job queue: [job duration, job age, job nodes, job cores per node, ...]
            'job_queue': spaces.Box(
                low=0,
                high=max(MAX_JOB_DURATION, MAX_JOB_AGE, MAX_NODES_PER_JOB, MAX_CORES_PER_JOB),
                shape=(MAX_QUEUE_SIZE * 4,),
                dtype=np.int32
            ),
            # predicted prices for the next 24h
            'predicted_prices': spaces.Box(
                low=-1000,
                high=1000,
                shape=(24,),
                dtype=np.float32
            ),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, self._seed = seeding.np_random(seed)

        # Track which episode this env instance is on
        if not hasattr(self, "episode_idx"):
            self.episode_idx = 0
        else:
            self.episode_idx += 1

        # Reset metrics
        self.metrics.reset_state_metrics()

        # Choose starting index in the external price series
        if self.prices is not None and self.prices.external_prices is not None:
            n_prices = len(self.prices.external_prices)
            episode_span = EPISODE_HOURS

            # Episode k starts at hour k * episode_span (wrapping around the year)
            start_index = (self.episode_idx * episode_span) % n_prices
            if options and "price_start_index" in options: # For testing Purposes. Leave out 'options' to advance episode.
                start_index = int(options["price_start_index"]) % n_prices
            self.prices.reset(start_index=start_index)
        else:
            # Synthetic prices or no external prices
            self.prices.reset(start_index=0)

        self.state = {
            # Initialize all nodes to be 'online but free' (0)
            'nodes': np.zeros(MAX_NODES, dtype=np.int32),
            # Initialize job queue to be empty
            'job_queue': np.zeros((MAX_QUEUE_SIZE * 4), dtype=np.int32),
            # Initialize predicted prices array
            'predicted_prices': self.prices.predicted_prices.copy(),
        }

        self.baseline_state = {
            'nodes': np.zeros(MAX_NODES, dtype=np.int32),
            'job_queue': np.zeros((MAX_QUEUE_SIZE * 4), dtype=np.int32),
        }

        self.cores_available = np.full(MAX_NODES, CORES_PER_NODE, dtype=np.int32)
        self.baseline_cores_available = np.full(MAX_NODES, CORES_PER_NODE, dtype=np.int32)

        # Job tracking: { job_id: {'duration': remaining_hours, 'allocation': [(node_idx1, cores1), ...]}, ... }
        self.running_jobs = {}
        self.baseline_running_jobs = {}

        self.next_job_id = 0  # shared between baseline and normal jobs

        # Track next empty slot in job queue for O(1) insertion
        self.next_empty_slot = 0
        self.baseline_next_empty_slot = 0

        return self.state, {}

    def step(self, action):
        self.current_step += 1
        self.metrics.current_hour += 1
        if self.metrics.current_hour == 1:
            self.current_episode += 1
        self.env_print(Fore.GREEN + f"\n[[[ Starting episode: {self.current_episode}, step: {self.current_step}, hour: {self.metrics.current_hour}" + Fore.RESET)

        self.state['predicted_prices'] = self.prices.advance_and_get_predicted_prices()
        current_price = self.state['predicted_prices'][0]
        self.env_print("predicted_prices: ", np.array2string(self.state['predicted_prices'], separator=" ", max_line_width=np.inf, formatter={'float_kind': lambda x: "{:05.2f}".format(x)}))

        # reshape the 1d job_queue array into 2d for cleaner code
        job_queue_2d = self.state['job_queue'].reshape(-1, 4)

        # Decrement booked time for nodes and complete running jobs
        self.env_print("[1] Processing ongoing jobs...")
        completed_jobs = process_ongoing_jobs(self.state['nodes'], self.cores_available, self.running_jobs)
        self.env_print(f"{len(completed_jobs)} jobs completed: [{' '.join(['#' + str(job_id) for job_id in completed_jobs]) if len(completed_jobs) > 0 else ''}]")

        # Generate new jobs
        self.env_print(f"[2] Generating new jobs...")
        new_jobs_count, new_jobs_durations, new_jobs_nodes, new_jobs_cores = generate_jobs(
            self.metrics.current_hour, job_queue_2d,
            self.external_jobs, self.external_hourly_jobs, self.external_durations,
            self.workload_gen, self.jobs_sampler if hasattr(self, 'jobs_sampler') else None,
            hourly_sampler, durations_sampler, self.np_random
        )

        # Add new jobs to queue
        self.env_print(f"[2] Adding {new_jobs_count} new jobs to the queue...")
        new_jobs, self.next_empty_slot = add_new_jobs(
            job_queue_2d, new_jobs_count, new_jobs_durations,
            new_jobs_nodes, new_jobs_cores, self.next_empty_slot
        )
        self.metrics.jobs_submitted += len(new_jobs)
        self.metrics.jobs_rejected_queue_full += (new_jobs_count - len(new_jobs))

        self.env_print("nodes: ", np.array2string(self.state['nodes'], separator=' ', max_line_width=np.inf))
        self.env_print(f"cores_available: {np.array2string(self.cores_available, separator=' ', max_line_width=np.inf)} ({np.sum(self.cores_available)})")
        self.env_print(f">>> adding {len(new_jobs)} new jobs to the queue: {' '.join(['[{}h {} {}x{}]'.format(d, a, n, c) for d, a, n, c in new_jobs])}")
        self.env_print("job_queue: ", ' '.join(['[{} {} {} {}]'.format(d, a, n, c) for d, a, n, c in job_queue_2d if d > 0]))

        action_type, action_magnitude = action
        action_magnitude += 1

        self.env_print(f"[3] Adjusting nodes based on action: type={action_type}, magnitude={action_magnitude}...")
        num_node_changes = adjust_nodes(action_type, action_magnitude, self.state['nodes'], self.cores_available, self.env_print)

        # Assign jobs to available nodes
        self.env_print(f"[4] Assigning jobs to available nodes...")

        num_launched_jobs, self.next_empty_slot, num_dropped_this_step, self.next_job_id = assign_jobs_to_available_nodes(
            job_queue_2d, self.state['nodes'], self.cores_available, self.running_jobs,
            self.next_empty_slot, self.next_job_id, self.metrics, is_baseline=False
        )

        self.env_print(f"   {num_launched_jobs} jobs launched")

        # Calculate node utilization stats
        num_used_nodes = np.sum(self.state['nodes'] > 0)
        num_on_nodes = np.sum(self.state['nodes'] >= 0)
        num_off_nodes = np.sum(self.state['nodes'] == -1)
        num_idle_nodes = num_on_nodes - num_used_nodes
        num_unprocessed_jobs = np.sum(job_queue_2d[:, 0] > 0)
        average_future_price = np.mean(self.state['predicted_prices'])
        num_used_cores = num_on_nodes * CORES_PER_NODE - np.sum(self.cores_available)

        # update stats
        self.metrics.on_nodes.append(num_on_nodes)
        self.metrics.used_nodes.append(num_used_nodes)
        self.metrics.job_queue_sizes.append(num_unprocessed_jobs)
        self.metrics.price_stats.append(current_price)

        # Track max queue size
        if num_unprocessed_jobs > self.metrics.max_queue_size_reached:
            self.metrics.max_queue_size_reached = num_unprocessed_jobs

        self.env_print(f"[5] Calculating reward...")

        # Baseline step
        baseline_cost, baseline_cost_off, self.baseline_next_empty_slot, self.next_job_id = baseline_step(
            self.baseline_state, self.baseline_cores_available, self.baseline_running_jobs,
            current_price, new_jobs_count, new_jobs_durations, new_jobs_nodes, new_jobs_cores,
            self.baseline_next_empty_slot, self.next_job_id, self.metrics, self.env_print
        )

        self.metrics.baseline_cost += baseline_cost
        self.metrics.baseline_cost_off += baseline_cost_off

        step_reward, step_cost, eff_reward_norm, price_reward_norm, idle_penalty_norm, job_age_penalty_norm = self.reward_calculator.calculate(
            num_used_nodes, num_idle_nodes, current_price, average_future_price,
            num_off_nodes, num_launched_jobs, num_node_changes, job_queue_2d,
            num_unprocessed_jobs, self.weights, num_dropped_this_step, self.env_print
        )

        self.metrics.episode_reward += step_reward
        self.metrics.total_cost += step_cost

        # Store normalized reward components for plotting
        self.metrics.eff_rewards.append(eff_reward_norm * 100)
        self.metrics.price_rewards.append(price_reward_norm * 100)
        self.metrics.job_age_penalties.append(job_age_penalty_norm * 100)
        self.metrics.idle_penalties.append(idle_penalty_norm * 100)

        # print stats
        self.env_print(f"[6] End of step stats...")
        self.env_print("job queue: ", ' '.join(['[{} {} {} {}]'.format(d, a, n, c) for d, a, n, c in job_queue_2d if d > 0]))
        self.env_print(f"{len(self.running_jobs)} running jobs: {' '.join(['[#{}: {}h, {}x{}]'.format(job_id, job_data['duration'], len(job_data['allocation']), int(job_data['allocation'][0][1])) for job_id, job_data in self.running_jobs.items()]) if len(self.running_jobs) > 0 else '[]'}")
        self.env_print(f"launched jobs: {num_launched_jobs}, unprocessed jobs: {num_unprocessed_jobs}")
        self.env_print(f"nodes: ON: {num_on_nodes}, OFF: {num_off_nodes}, used: {num_used_nodes}, IDLE: {num_idle_nodes}. node changes: {num_node_changes}")
        self.env_print("nodes: ", np.array2string(self.state['nodes'], separator=" ", max_line_width=np.inf))
        self.env_print(f"cores used: {num_used_cores} out of {num_on_nodes * CORES_PER_NODE} available cores")
        self.env_print(f"cores_available: {np.array2string(self.cores_available, separator=' ', max_line_width=np.inf)} ({np.sum(self.cores_available)})")
        self.env_print(f"price: current: {current_price}, average future: {average_future_price:.4f}")
        self.env_print(f"step reward: {step_reward:.4f}, episode reward: {self.metrics.episode_reward:.4f}")

        if self.plot_rewards:
            plot_reward(self, num_used_nodes, num_idle_nodes, current_price, num_off_nodes, average_future_price, num_launched_jobs, num_node_changes, job_queue_2d, MAX_NODES)

        truncated = False
        terminated = False
        if self.metrics.current_hour == EPISODE_HOURS:
            if self.render_mode == 'human':
                plot(self, EPISODE_HOURS, MAX_NODES, False, True, self.current_step)
                if self.plot_once:
                    raise PlottingComplete
            else:
                # Only do training plots in training mode
                if not self.evaluation_mode and self.current_step > self.next_plot_save:
                    plot(self, EPISODE_HOURS, MAX_NODES, True, False, self.current_step)
                    self.next_plot_save += self.steps_per_iteration
                    print(self.next_plot_save)
            truncated = True
            terminated = False

            # Record episode costs for long-term analysis
            self.metrics.record_episode_completion(self.current_episode)

        # flatten job_queue again
        self.state['job_queue'] = job_queue_2d.flatten()

        if self.render_mode == 'human':
            # go slow to be able to read stuff in human mode
            if not self.quick_plot:
                time.sleep(1)

        self.env_print(Fore.GREEN + f"]]]" + Fore.RESET)

        return self.state, step_reward, terminated, truncated, {}
