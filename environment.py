import time

from gymnasium import spaces
import gymnasium as gym
import numpy as np
from colorama import init, Fore

from prices import Prices
from weights import Weights
from plot import plot, plot_reward
from sampler_duration import durations_sampler
from sampler_jobs import jobs_sampler

init()  # Initialize colorama

WEEK_HOURS = 168

MAX_NODES = 335  # Maximum number of nodes
MAX_QUEUE_SIZE = 1000  # Maximum number of jobs in the queue
MAX_CHANGE = MAX_NODES
MAX_JOB_DURATION = 170 # maximum job runtime in hours
MAX_JOB_AGE = WEEK_HOURS # job waits maximum a week
MAX_NEW_JOBS_PER_HOUR = 1500

COST_IDLE = 150 # Watts
COST_USED = 450 # Watts

CORES_PER_NODE = 96
MIN_CORES_PER_JOB = 1
MAX_CORES_PER_JOB = 96
MIN_NODES_PER_JOB = 1
MAX_NODES_PER_JOB = 16

COST_IDLE_MW = COST_IDLE / 1000000 # MW
COST_USED_MW = COST_USED / 1000000 # MW

EPISODE_HOURS = WEEK_HOURS * 2

# possible rewards:
# - cost savings (due to disabled nodes)
# - reduced conventional energy usage
# - cost of systems doing nothing (should not waste available resources)
# - job queue advancement
# Reward components
# REWARD_TURN_OFF_NODE = 0.1 # Reward for each node turned off
# REWARD_PROCESSED_JOB = 1   # Reward for processing jobs under favorable prices
# PENALTY_NODE_CHANGE = -0.05 # Penalty for changing node state
PENALTY_IDLE_NODE = -0.1 # Penalty for idling nodes
PENALTY_WAITING_JOB = -0.1  # Penalty for each hour a job is delayed

# TODO:
# - should the observation space be normalized too?

class PlottingComplete(Exception):
    """Raised when plotting is complete and the application should terminate."""
    pass

class ComputeClusterEnv(gym.Env):
    """An environment for scheduling compute jobs based on electricity price predictions."""

    metadata = { 'render.modes': ['human', 'none'] }

    def render(self, mode='human'):
        self.render_mode = mode

    def set_progress(self, iterations):
        self.current_step = iterations * self.steps_per_iteration
        self.current_episode = self.current_step // EPISODE_HOURS
        print(f"Resuming training... step: {self.current_step}, episode: {self.current_episode}, hour: {self.current_hour}")
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
                 steps_per_iteration):
        super().__init__()

        self.weights = weights
        self.session = session
        self.render_mode = render_mode
        self.quick_plot = quick_plot
        self.external_prices = external_prices
        self.external_durations = external_durations
        self.external_jobs = external_jobs
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

        self.next_plot_save = self.steps_per_iteration

        self.prices = Prices(self.external_prices)

        if self.external_durations:
            durations_sampler.init(self.external_durations)

        if self.external_jobs:
            print(f"Loading jobs from {self.external_jobs}")
            jobs_sampler.parse_jobs(self.external_jobs, 60)
            print(f"Parsed jobs for {len(jobs_sampler.jobs)} hours")
            print(f"Parsed aggregated jobs for {len(jobs_sampler.aggregated_jobs)} hours")
            jobs_sampler.precalculate_hourly_jobs(CORES_PER_NODE, MAX_NODES_PER_JOB)
            print(f"Max jobs per hour: {jobs_sampler.max_new_jobs_per_hour}")
            print(f"Max job duration: {jobs_sampler.max_job_duration}")
            print(f"Parsed hourly jobs for {len(jobs_sampler.hourly_jobs)} hours")

        self.current_step = 0
        self.current_episode = 0
        self.current_hour = 0

        self.total_cost = 0
        self.baseline_cost = 0
        self.baseline_cost_off = 0

        self.reset_state()

        print(f"{self.weights}")
        print(f"prices.MAX_PRICE: {self.prices.MAX_PRICE:.2f}, prices.MIN_PRICE: {self.prices.MIN_PRICE:.2f}")
        print(f"Price Statistics: {self.prices.get_price_stats()}")
        # self.prices.plot_price_histogram(use_original=False)

        cost_for_min_efficiency = self.power_cost(0, MAX_NODES, self.prices.MAX_PRICE)
        cost_for_max_efficiency = self.power_cost(MAX_NODES, 0, self.prices.MIN_PRICE)

        self.min_efficiency_reward = self.reward_efficiency(0, cost_for_min_efficiency) # Worst case: nodes running but no work being done
        self.max_efficiency_reward = max(1.0, self.reward_efficiency(MAX_NODES, cost_for_max_efficiency)) # Best case: all nodes running and doing work

        self.min_price_reward = 0
        # self.min_price_reward = self.reward_price(self.prices.MAX_PRICE, self.prices.MIN_PRICE, MAX_NEW_JOBS_PER_HOUR)  # Worst case: highest current price, lowest future price, max jobs processed
        self.max_price_reward = self.reward_price(self.prices.MIN_PRICE, self.prices.MAX_PRICE, MAX_NEW_JOBS_PER_HOUR)  # Best case: lowest current price, highest future price, max jobs processed

        self.min_idle_penalty = self.penalty_idle(0)
        self.max_idle_penalty = self.penalty_idle(MAX_NODES)

        self.min_job_age_penalty = -0.0
        self.max_job_age_penalty = PENALTY_WAITING_JOB * MAX_JOB_AGE * MAX_QUEUE_SIZE

        # actions: - change number of available nodes:
        #   direction: 0: decrease, 1: maintain, 2: increase
        #   num nodes: 0-9 (+1ed in the action)
        self.action_space = spaces.MultiDiscrete([3, MAX_CHANGE])

        # - predicted allocation
        # - predicted green/conventional ratio
        # - predicted usage/load
        self.observation_space = spaces.Dict({
            # nodes: [-1: off, 0: idle, >0: booked for n hours]
            'nodes': spaces.Box(
                low=-1,
                high=MAX_JOB_DURATION,
                shape=(MAX_NODES,),  # Correct shape to (100,)
                dtype=np.int32
            ),
            # job queue: [job duration, job age, job nodes, job cores per node, job duration, job age, job nodes, job cores per node, ...]
            'job_queue': spaces.Box(
                low=0,
                high=max(MAX_JOB_DURATION, MAX_JOB_AGE, MAX_NODES_PER_JOB, MAX_CORES_PER_JOB),
                shape=(MAX_QUEUE_SIZE * 4,),  # Each job has 4 values now: duration, age, nodes, cores per node
                dtype=np.int32
            ),
            # predicted prices for the next 24h
            'predicted_prices': spaces.Box(
                low=-1000,
                high=1000, # Assuming there's no maximum price
                shape=(24,), # Prices for the next 24 hours
                dtype=np.float32
            ),
        })

    def reset(self, seed = None, options = None):
        self.reset_state()

        self.state = {
            # Initialize all nodes to be 'online but free' (0)
            'nodes': np.zeros(MAX_NODES, dtype=np.int32),
            # Initialize job queue to be empty
            'job_queue': np.zeros((MAX_QUEUE_SIZE * 4), dtype=np.int32),
            # Initialize predicted prices array
            'predicted_prices': self.prices.get_predicted_prices(),
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

        self.next_job_id = 0 # shared between baseline and normal jobs

        return self.state, {}

    def reset_state(self):
        self.current_hour = 0
        self.episode_reward = 0

        self.on_nodes = []
        self.used_nodes = []
        self.job_queue_sizes = []
        self.price_stats = []

        self.eff_rewards = []
        self.price_rewards = []
        self.idle_penalties = []
        self.job_age_penalties = []

        self.total_cost = 0
        self.baseline_cost = 0
        self.baseline_cost_off = 0

    def step(self, action):
        self.current_step += 1
        self.current_hour += 1
        if self.current_hour == 1:
            self.current_episode += 1
        self.env_print(Fore.GREEN + f"\n[[[ Starting episode: {self.current_episode}, step: {self.current_step}, hour: {self.current_hour}" + Fore.RESET)

        self.state['predicted_prices'] = self.prices.get_predicted_prices()
        current_price = self.state['predicted_prices'][0]
        self.env_print("predicted_prices: ", np.array2string(self.state['predicted_prices'], separator=" ", max_line_width=np.inf, formatter={'float_kind': lambda x: "{:05.2f}".format(x)}))

        # reshape the 1d job_queue array into 2d for cleaner code
        job_queue_2d = self.state['job_queue'].reshape(-1, 4)  # Now a (MAX_QUEUE_SIZE, 4) array

        # Decrement booked time for nodes and complete running jobs
        self.env_print("[1] Processing ongoing jobs...")
        completed_jobs = self.process_ongoing_jobs(self.state['nodes'], self.cores_available, self.running_jobs)
        self.env_print(f"{len(completed_jobs)} jobs completed: [{' '.join(['#' + str(job_id) for job_id in completed_jobs]) if len(completed_jobs) > 0 else ''}]")

        # Update job queue with new jobs. If queue is full, do nothing
        new_jobs_durations = []
        new_jobs_nodes = []
        new_jobs_cores = []
        new_jobs_count = 0

        if self.external_jobs:
            jobs = jobs_sampler.sample_one_hourly(wrap=True)['hourly_jobs']
            if len(jobs) > 0:
                for job in jobs:
                    new_jobs_count += 1
                    new_jobs_durations.append(job['duration_hours'])
                    new_jobs_nodes.append(job['nnodes'])
                    new_jobs_cores.append(job['cores_per_node'])
        else:
            new_jobs_count = np.random.randint(0, MAX_NEW_JOBS_PER_HOUR + 1)
            if self.external_durations:
                new_jobs_durations = durations_sampler.sample(new_jobs_count)
            else:
                new_jobs_durations = np.random.randint(1, MAX_JOB_DURATION + 1, size=new_jobs_count)
            # Generate random node and core requirements
            for _ in range(new_jobs_count):
                new_jobs_nodes.append(np.random.randint(MIN_NODES_PER_JOB, MAX_NODES_PER_JOB + 1))
                new_jobs_cores.append(np.random.randint(MIN_CORES_PER_JOB, CORES_PER_NODE + 1))

        self.env_print(f"[2] Adding {new_jobs_count} new jobs to the queue...")
        new_jobs = self.add_new_jobs(job_queue_2d, new_jobs_count, new_jobs_durations, new_jobs_nodes, new_jobs_cores)

        self.env_print("nodes: ", np.array2string(self.state['nodes'], separator=' ', max_line_width=np.inf))
        self.env_print(f"cores_available: {np.array2string(self.cores_available, separator=' ', max_line_width=np.inf)} ({np.sum(self.cores_available)})")
        self.env_print(f">>> adding {len(new_jobs)} new jobs to the queue: {' '.join(['[{}h {} {}x{}]'.format(d, a, n, c) for d, a, n, c in new_jobs])}")
        self.env_print("job_queue: ", ' '.join(['[{} {} {} {}]'.format(d, a, n, c) for d, a, n, c in job_queue_2d if d > 0]))

        action_type, action_magnitude = action  # Unpack the action array
        action_magnitude += 1

        self.env_print(f"[3] Adjusting nodes based on action: type={action_type}, magnitude={action_magnitude}...")
        num_node_changes = self.adjust_nodes(action_type, action_magnitude, self.state['nodes'], self.cores_available)

        # assign jobs to available nodes
        self.env_print(f"[4] Assigning jobs to available nodes...")
        num_launched_jobs = self.assign_jobs_to_available_nodes(job_queue_2d, self.state['nodes'], self.cores_available, self.running_jobs)
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
        self.on_nodes.append(num_on_nodes)
        self.used_nodes.append(num_used_nodes)
        self.job_queue_sizes.append(num_unprocessed_jobs)
        self.price_stats.append(current_price)

        self.env_print(f"[5] Calculating reward...")
        # baseline
        baseline_cost, baseline_cost_off = self.baseline_step(current_price, new_jobs_count, new_jobs_durations, new_jobs_nodes, new_jobs_cores)
        self.baseline_cost += baseline_cost
        self.baseline_cost_off += baseline_cost_off

        # calculate reward
        step_reward, step_cost = self.calculate_reward(num_used_nodes, num_idle_nodes, current_price, average_future_price, num_off_nodes, num_launched_jobs, num_node_changes, job_queue_2d, num_unprocessed_jobs)
        self.episode_reward = self.episode_reward + step_reward
        self.total_cost += step_cost

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
        self.env_print(f"step reward: {step_reward:.4f}, episode reward: {self.episode_reward:.4f}")

        if self.plot_rewards:
            plot_reward(self, num_used_nodes, num_idle_nodes, current_price, num_off_nodes, average_future_price, num_launched_jobs, num_node_changes, job_queue_2d, MAX_NODES)

        truncated = False
        terminated = False
        if self.current_hour == EPISODE_HOURS:
            # # sparse reward
            # if self.total_cost < self.baseline_cost_off:
            #     cost_improvement = self.baseline_cost_off - self.total_cost
            #     # Scale the reward to be roughly 10% of total episode reward when cost savings are significant
            #     baseline_reward = 0.1 * (cost_improvement / self.baseline_cost_off) * EPISODE_HOURS
            #     self.env_print(f"$$$BASELINE: {baseline_reward:.4f} (cost savings: €{cost_improvement:.2f})")
            #     reward += baseline_reward
            #     self.env_print(f"TOTAL (dense + sparse) reward: {reward:.4f}")

            if self.render_mode == 'human':
                plot(self, EPISODE_HOURS, MAX_NODES, False, True, self.current_step)
                if self.plot_once:
                    raise PlottingComplete
            else:
                if self.current_step > self.next_plot_save:
                    plot(self, EPISODE_HOURS, MAX_NODES, True, False, self.current_step)
                    self.next_plot_save += self.steps_per_iteration
                    print(self.next_plot_save)

            terminated = True

        # flatten job_queue again
        self.state['job_queue'] = job_queue_2d.flatten()

        if self.render_mode == 'human':
            # go slow to be able to read stuff in human mode
            if not self.quick_plot:
                time.sleep(1)

        self.env_print(Fore.GREEN + f"]]]" + Fore.RESET)

        return self.state, step_reward, terminated, truncated, {}

    def process_ongoing_jobs(self, nodes, cores_available, running_jobs):
        completed_jobs = []

        for job_id, job_data in running_jobs.items():
            job_data['duration'] -= 1

            # Check if job is completed
            if job_data['duration'] <= 0:
                completed_jobs.append(job_id)
                # Release resources
                for node_idx, cores_used in job_data['allocation']:
                    cores_available[node_idx] += cores_used

        # Remove completed jobs
        for job_id in completed_jobs:
            del running_jobs[job_id]

        # Update node times based on remaining jobs
        # Reset all nodes first
        for i in range(MAX_NODES):
            if nodes[i] > 0:  # Don't touch turned-off nodes
                nodes[i] = 0

        # Set node times based on jobs
        for job_id, job_data in running_jobs.items():
            remaining_time = job_data['duration']
            for node_idx, _ in job_data['allocation']:
                nodes[node_idx] = max(nodes[node_idx], remaining_time)

        return completed_jobs

    def add_new_jobs(self, job_queue_2d, new_jobs_count, new_jobs_durations, new_jobs_nodes, new_jobs_cores):
        new_jobs = []
        if new_jobs_count > 0:
            for i in range(new_jobs_count):
                # Find empty slot in job queue
                for j in range(len(job_queue_2d)):
                    if job_queue_2d[j][0] == 0:  # Empty slot
                        job_queue_2d[j] = [
                            new_jobs_durations[i],
                            0,  # Age starts at 0
                            new_jobs_nodes[i],  # Number of nodes required
                            new_jobs_cores[i]   # Cores per node required
                        ]
                        new_jobs.append(job_queue_2d[j])
                        break
        return new_jobs

    def adjust_nodes(self, action_type, action_magnitude, nodes, cores_available):
        num_node_changes = 0

        # Adjust nodes based on action
        if action_type == 0: # Decrease number of available nodes
            self.env_print(f"   >>> turning OFF up to {action_magnitude} nodes")
            nodes_modified = 0
            for i in range(len(nodes)):
                # Find idle nodes (no jobs running)
                if nodes[i] == 0 and cores_available[i] == CORES_PER_NODE:
                    nodes[i] = -1 # Turn off
                    cores_available[i] = 0 # No cores available on off nodes
                    nodes_modified += 1
                    num_node_changes += 1
                    if nodes_modified == action_magnitude:
                        break
        elif action_type == 1:
            self.env_print(f"   >>> Not touching any nodes")
            pass # maintain node count = do nothing
        elif action_type == 2: # Increase number of available nodes
            self.env_print(f"   >>> turning ON up to {action_magnitude} nodes")
            nodes_modified = 0
            for i in range(len(nodes)):
                if nodes[i] == -1:  # Find off node
                    nodes[i] = 0  # Turn on
                    cores_available[i] = CORES_PER_NODE  # Reset cores to full availability
                    nodes_modified += 1
                    num_node_changes += 1
                    if nodes_modified == action_magnitude:
                        break

        return num_node_changes

    def assign_jobs_to_available_nodes(self, job_queue_2d, nodes, cores_available, running_jobs):
        num_processed_jobs = 0

        for job_idx, job in enumerate(job_queue_2d):
            job_duration, job_age, job_nodes, job_cores_per_node = job

            if job_duration <= 0:
                continue

            candidate_nodes = []

            for node_idx in range(len(nodes)):
                if nodes[node_idx] >= 0 and cores_available[node_idx] >= job_cores_per_node:
                    candidate_nodes.append(node_idx)

            # Check if we have enough nodes for this job
            if len(candidate_nodes) >= job_nodes:
                # We found enough nodes, assign the job
                job_allocation = []

                # Use the first 'job_nodes' candidates
                for i in range(job_nodes):
                    node_idx = candidate_nodes[i]

                    # Update node resources
                    cores_available[node_idx] -= job_cores_per_node

                    # If this node now has a longer job duration, update it
                    if nodes[node_idx] < job_duration:
                        nodes[node_idx] = job_duration

                    # Add to job allocation
                    job_allocation.append((node_idx, job_cores_per_node))

                # Add job to running_jobs
                running_jobs[self.next_job_id] = {
                    'duration': job_duration,
                    'allocation': job_allocation
                }
                self.next_job_id += 1

                # Clear job from queue
                job_queue_2d[job_idx] = [0, 0, 0, 0]

                num_processed_jobs += 1
            else:
                # Not enough resources for this job, increment its age
                job_queue_2d[job_idx][1] += 1

        return num_processed_jobs

    def baseline_step(self, current_price, new_jobs_count, new_jobs_durations, new_jobs_nodes, new_jobs_cores):
        job_queue_2d = self.baseline_state['job_queue'].reshape(-1, 4)

        self.process_ongoing_jobs(self.baseline_state['nodes'], self.baseline_cores_available, self.baseline_running_jobs)

        self.add_new_jobs(job_queue_2d, new_jobs_count, new_jobs_durations, new_jobs_nodes, new_jobs_cores)

        self.assign_jobs_to_available_nodes(job_queue_2d, self.baseline_state['nodes'], self.baseline_cores_available, self.baseline_running_jobs)

        num_used_nodes = np.sum(self.baseline_state['nodes'] > 0)
        num_on_nodes = np.sum(self.baseline_state['nodes'] > -1)
        num_idle_nodes = num_on_nodes - num_used_nodes

        self.baseline_state['job_queue'] = job_queue_2d.flatten()

        baseline_cost = self.power_cost(num_used_nodes, num_idle_nodes, current_price)
        self.env_print(f"    > baseline_cost: €{baseline_cost:.4f} | used nodes: {num_used_nodes}, idle nodes: {num_idle_nodes}")
        baseline_cost_off = self.power_cost(num_used_nodes, 0, current_price)
        self.env_print(f"    > baseline_cost_off: €{baseline_cost_off:.4f} | used nodes: {num_used_nodes}, idle nodes: 0")
        return baseline_cost, baseline_cost_off

    def calculate_reward(self, num_used_nodes, num_idle_nodes, current_price, average_future_price, num_off_nodes, num_processed_jobs, num_node_changes, job_queue_2d, num_unprocessed_jobs):
        # 0. Efficiency. Reward calculation based on Workload (used nodes) (W) / Cost (C)
        total_cost = self.power_cost(num_used_nodes, num_idle_nodes, current_price)
        efficiency_reward_norm = self.reward_efficiency_normalized(num_used_nodes, num_idle_nodes, num_unprocessed_jobs, total_cost)
        efficiency_reward_weighted = self.weights.efficiency_weight * efficiency_reward_norm
        # self.env_print(f"$$$EFF: {efficiency_reward_weighted:.4f} = {efficiency_reward_norm:.4f} x {self.weights.efficiency_weight}")

        # 1. increase reward for each turned off node, more if the current price is higher than average
        # turned_off_reward = self.reward_turned_off(num_off_nodes, average_future_price, current_price)

        # 2. increase reward if jobs were scheduled in this step and the current price is below average
        price_reward_norm = self.reward_price_normalized(current_price, average_future_price, num_processed_jobs)
        price_reward_weighted = self.weights.price_weight * price_reward_norm
        # self.env_print(f"$$$PRICE: {price_reward_weighted:.4f} = {price_reward_norm:.4f} x {self.weights.price_weight}")

        # 3. penalize delayed jobs, more if they are older. but only if there are turned off nodes
        job_age_penalty_norm = self.penalty_job_age_normalized(num_off_nodes, job_queue_2d)
        job_age_penalty_weighted = self.weights.job_age_weight * job_age_penalty_norm
        # self.env_print(f"$$$AGE: {job_age_penalty_weighted:.4f} = {job_age_penalty_norm:.4f} x {self.weights.job_age_weight}")

        # 4. penalty to avoid too frequent node state changes
        # node_change_penalty = self.penalty_node_changes(num_node_changes)

        # 5. penalty for idling nodes
        idle_penalty_norm = self.penalty_idle_normalized(num_idle_nodes)
        idle_penalty_weighted = self.weights.idle_weight * idle_penalty_norm
        # self.env_print(f"$$$IDLE: {idle_penalty_weighted:.4f} = {idle_penalty_norm:.4f} x {self.weights.idle_weight}")

        self.eff_rewards.append(efficiency_reward_norm * 100)
        self.price_rewards.append(price_reward_norm * 100)
        self.job_age_penalties.append(job_age_penalty_norm * 100)
        self.idle_penalties.append(idle_penalty_norm * 100)

        reward = (
            efficiency_reward_weighted
            # + 0.0 * turned_off_reward
            + price_reward_weighted
            + job_age_penalty_weighted
            + idle_penalty_weighted
        )

        self.env_print(f"    > $$$TOTAL: {reward:.4f} = {efficiency_reward_weighted:.4f} + {price_reward_weighted:.4f} + {idle_penalty_weighted:.4f} + {job_age_penalty_weighted:.4f}")
        self.env_print(f"    > step cost: €{total_cost:.4f}")

        return reward, total_cost

    def power_cost(self, num_used_nodes, num_idle_nodes, current_price):
        idle_cost = COST_IDLE_MW * current_price * num_idle_nodes
        usage_cost = COST_USED_MW * current_price * num_used_nodes
        total_cost = idle_cost + usage_cost
        # self.env_print(f"$$EFF total_cost: {total_cost} = idle_cost: {idle_cost} + usage_cost: {usage_cost}")
        return total_cost

    def reward_efficiency(self, num_used_nodes, total_cost):
        return num_used_nodes / (total_cost + 1e-6)

    def reward_efficiency_normalized(self, num_used_nodes, num_idle_nodes, num_unprocessed_jobs, total_cost):
        current_reward = 0
        if num_used_nodes + num_idle_nodes == 0:
            if num_unprocessed_jobs == 0:
                current_reward = 1
                # self.env_print(f"$$E efficiency_reward: {current_reward:.4f} (nothing is used and no outstanding jobs)")
            else:
                current_reward = np.clip(1.0 / np.log1p(num_unprocessed_jobs), a_min=None, a_max=1.0)
                # self.env_print(f"$$E efficiency_reward: {current_reward:.4f} (nothing is used and {num_unprocessed_jobs} outstanding jobs)")
        else:
            current_reward = self.reward_efficiency(num_used_nodes, total_cost)
            # self.env_print(f"$$E efficiency_reward (w/c): {current_reward:.4f} (= {num_used_nodes} / (€{total_cost:.2f} + 1e-6)), num_used_nodes: {num_used_nodes}")
            current_reward = normalize(current_reward, self.min_efficiency_reward, self.max_efficiency_reward)
            # self.env_print(f"$E normalized_reward: {current_reward:.4f} | min_efficiency_reward: {self.min_efficiency_reward:.4f}, max_efficiency_reward: {self.max_efficiency_reward:.4f}")
        return current_reward

    # def reward_turned_off(self, num_off_nodes, average_future_price, current_price):
    #     turned_off_reward = REWARD_TURN_OFF_NODE * num_off_nodes * (1 / average_future_price * current_price)
    #     # self.env_print(f"$$ turned_off_reward: {turned_off_reward} ({REWARD_TURN_OFF_NODE} * {num_off_nodes} * (1 / {average_future_price} * {current_price}))")
    #     return turned_off_reward

    def reward_price(self, current_price, average_future_price, num_processed_jobs):
        history_avg, future_avg = self.prices.get_price_context()

        if history_avg is not None:
            # We have some history - use both past and future
            context_avg = (history_avg + future_avg) / 2
            price_diff = context_avg - current_price
        else:
            # No history yet - fall back to just using future prices
            price_diff = average_future_price - current_price

        price_reward = price_diff * num_processed_jobs
        # if current_price < average_future_price:
            # price_reward = REWARD_PROCESSED_JOB * num_processed_jobs
        return price_reward

    def reward_price_normalized(self, current_price, average_future_price, num_processed_jobs):
        current_reward = self.reward_price(current_price, average_future_price, num_processed_jobs)
        # self.env_print(f"$$P: price_reward: {current_reward:.4f} | current_price: €{current_price:.2f}, average_future_price: €{average_future_price:.2f}, num_processed_jobs: {num_processed_jobs}")
        if num_processed_jobs == 0:
            normalized_reward = 0
        else:
            normalized_reward = normalize(current_reward, self.min_price_reward, self.max_price_reward)
        # self.env_print(f"$P: normalized_price_reward: {normalized_reward:.4f} | min_price_reward: {self.min_price_reward:.2f}, max_price_reward: {self.max_price_reward:.2f}")
        # Clip the value to ensure it's between 0 and 1
        # normalized_reward = np.clip(normalized_reward, 0, 1)
        return normalized_reward

    # def penalty_node_changes(self, num_node_changes):
    #     node_change_penalty = PENALTY_NODE_CHANGE * num_node_changes
    #     # self.env_print(f"$$ node change penalty: {node_change_penalty}")
    #     return node_change_penalty

    def penalty_idle(self, num_idle_nodes):
        idle_penalty = PENALTY_IDLE_NODE * num_idle_nodes
        return idle_penalty

    def penalty_idle_normalized(self, num_idle_nodes):
        current_penalty = self.penalty_idle(num_idle_nodes)
        # self.env_print(f"$$I current_penalty: {current_penalty:.4f} (num_idle_nodes: {num_idle_nodes})")
        normalized_penalty = - normalize(current_penalty, self.min_idle_penalty, self.max_idle_penalty)
        # self.env_print(f"$I normalized_penalty: {normalized_penalty:.4f} | min_penalty: {self.min_idle_penalty}, max_penalty: {self.max_idle_penalty}")
        # Clip the value to ensure it's between 0 and 1
        normalized_penalty = np.clip(normalized_penalty, -1, 0)
        # self.env_print(f"$I CLIPPED normalized_penalty: {normalized_penalty}")
        return normalized_penalty

    def penalty_job_age(self, num_off_nodes, job_queue_2d):
        job_age_penalty = 0
        if num_off_nodes > 0:
            for job in job_queue_2d:
                job_duration, job_age, _, _ = job
                if job_duration > 0:
                    job_age_penalty += PENALTY_WAITING_JOB * job_age
        return job_age_penalty

    def penalty_job_age_normalized(self, num_off_nodes, job_queue_2d):
        current_penalty = self.penalty_job_age(num_off_nodes, job_queue_2d)
        # self.env_print(f"$$D current_penalty: {current_penalty:.4f}")
        normalized_penalty = - normalize(current_penalty, self.min_job_age_penalty, self.max_job_age_penalty)
        # self.env_print(f"$D normalized_penalty: {normalized_penalty:.4f} | min_penalty: {self.min_job_age_penalty}, max_penalty: {self.max_job_age_penalty}")
        normalized_penalty = np.clip(normalized_penalty, -1, 0)
        # self.env_print(f"$D CLIPPED normalized_penalty: {normalized_penalty}")
        return normalized_penalty

def normalize(current, minimum, maximum):
    if maximum == minimum:
        return 0.5  # Avoid division by zero
    return (current - minimum) / (maximum - minimum)
