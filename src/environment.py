import time
from collections import deque
from typing import Any

from gymnasium import spaces
import gymnasium as gym
import numpy as np
from colorama import init, Fore

from src.prices import Prices
from src.weights import Weights
from src.plot_config import PlotConfig
from src.plotter import plot_episode
from src.sampler_duration import durations_sampler
from src.sampler_jobs import DurationSampler
from src.sampler_hourly import hourly_sampler
from src.workloadgen import WorkloadGenerator

# Import refactored modules
from src.config import (
    MAX_NODES, MAX_QUEUE_SIZE, MAX_CHANGE, MAX_JOB_DURATION,
    CORES_PER_NODE, MAX_CORES_PER_JOB, MAX_JOB_AGE_OBS,
    MAX_NODES_PER_JOB, EPISODE_HOURS
)
from src.job_management import (
    process_ongoing_jobs, add_new_jobs,
    assign_jobs_to_available_nodes, fill_queue_from_backlog, age_backlog_queue,
    age_job_queue,
)
from src.node_management import adjust_nodes
from src.reward_calculation import RewardCalculator, power_consumption_mwh
from src.baseline import baseline_step
from src.workload_generator import generate_jobs
from src.metrics_tracker import MetricsTracker


init()  # Initialize colorama


class PlottingComplete(Exception):
    """Raised when plotting is complete and the application should terminate."""
    pass


class ComputeClusterEnv(gym.Env):
    """An environment for scheduling compute jobs based on electricity price predictions."""

    metadata = {'render.modes': ['human', 'none']}
    DROP_STREAK_TERMINATION_PENALTY = -200.0

    def render(self, mode: str = 'human') -> None:
        self.render_mode = mode

    def set_progress(self, iterations: int) -> None:
        self.current_step = iterations * self.steps_per_iteration
        self.current_episode = self.current_step // EPISODE_HOURS
        print(f"Resuming training... step: {self.current_step}, episode: {self.current_episode}, hour: {self.metrics.current_hour}")
        self.next_plot_save = iterations * self.steps_per_iteration + EPISODE_HOURS

    def env_print(self, *args: Any) -> None:
        """Prints only if the render mode is 'human'."""
        if self.render_mode == 'human':
            print(*args)

    def __init__(self,
                 weights: Weights,
                 session: str,
                 render_mode: str,
                 external_prices: list[float] | np.ndarray | None,
                 external_durations: str | None,
                 external_jobs: str | None,
                 external_hourly_jobs: str | None,
                 plot_config: PlotConfig,
                 steps_per_iteration: int,
                 evaluation_mode: bool = False,
                 workload_gen: WorkloadGenerator | None = None,
                 job_arrival_scale: float = 1.0,
                 jobs_exact_replay: bool = False,
                 output_dir: str = "sessions",
                 jobs_exact_replay_aggregate: bool = False,
                 flush_after_drop_streak: int = 0) -> None:
        super().__init__()

        self.weights = weights
        self.session = session
        self.render_mode = render_mode
        self.external_prices = external_prices
        self.external_durations = external_durations
        self.external_jobs = external_jobs
        self.external_hourly_jobs = external_hourly_jobs
        self.plot_config = plot_config
        self.steps_per_iteration = steps_per_iteration
        self.evaluation_mode = evaluation_mode
        self.job_arrival_scale = float(job_arrival_scale)
        self.jobs_exact_replay = bool(jobs_exact_replay)
        self.jobs_exact_replay_aggregate = bool(jobs_exact_replay_aggregate)
        self.flush_after_drop_streak = max(0, int(flush_after_drop_streak))
        self.consecutive_drop_steps = 0

        self.next_plot_save = self.steps_per_iteration

        # Initialize metrics tracker
        self.metrics = MetricsTracker()

        # Initialize cost tracking for long-term analysis
        self.session_dir = f"{output_dir}/{session}"
        self.plots_dir = f"{output_dir}/{session}/plots/"

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
            if self.jobs_exact_replay:
                max_raw_jobs = max((len(v) for v in self.jobs_sampler.jobs.values()), default=0)
                if self.jobs_exact_replay_aggregate:
                    print("Jobs replay mode: exact timeline (aggregated per step)")
                else:
                    print("Jobs replay mode: exact timeline (raw jobs per hour)")
                print(f"Max raw jobs per hour: {max_raw_jobs}")
            else:
                self.jobs_sampler.precalculate_hourly_jobs(CORES_PER_NODE, MAX_NODES_PER_JOB)
                print("Jobs replay mode: aggregated hourly templates")
                print(f"Max jobs per hour: {self.jobs_sampler.max_new_jobs_per_hour}")
                print(f"Max job duration: {self.jobs_sampler.max_job_duration}")
                print(f"Parsed hourly jobs for {len(self.jobs_sampler.hourly_jobs)} hours")

        if self.external_hourly_jobs:
            print(f"Loading hourly jobs from {self.external_hourly_jobs}")
            hourly_sampler.parse_jobs(self.external_hourly_jobs)
            hourly_sampler.precalculate_hourly_templates(CORES_PER_NODE, MAX_NODES_PER_JOB)
            print(f"Hourly sampler initialized with 24-hour distributions")
        print(f"Job arrival scale: {self.job_arrival_scale:.3f}x")

        self.current_step = 0
        self.current_episode = 0

        # Initialize to -1, so that first reset() sets it to 0
        self.episode_idx = -1

        print(f"{self.weights}")
        print(f"prices.MAX_PRICE: {self.prices.MAX_PRICE:.2f}, prices.MIN_PRICE: {self.prices.MIN_PRICE:.2f}")
        print(f"Price Statistics: {self.prices.get_price_stats()}")

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(self.prices)

        self.metrics.reset_timeline_metrics()
        self.metrics.reset_episode_metrics()
        self._reset_timeline_state(start_index=0)
        # A fresh env still needs one real reset() call before rollout starts. After that,
        # episode boundaries should only roll metrics, not wipe the live simulation state.
        self._timeline_initialized = False

        # actions: - change number of available nodes:
        #   action_type:      0: decrease, 1: maintain, 2: increase
        #   action_magnitude: 0-MAX_CHANGE (+1ed in the action)
        #   do_refill:        0: don't refill from backlog, 1: refill from backlog
        self.action_space = spaces.MultiDiscrete([3, MAX_CHANGE, 2])

        self.observation_space = spaces.Dict({
            # nodes: [-1: off, 0: idle, >0: booked for n hours]
            'nodes': spaces.Box(low=-1, high=MAX_JOB_DURATION, shape=(MAX_NODES,), dtype=np.int32),
            # job queue: [job duration, job age, job nodes, job cores per node, ...]
            'job_queue': spaces.Box(
                low=0,
                high=max(MAX_JOB_DURATION, MAX_JOB_AGE_OBS, MAX_NODES_PER_JOB, MAX_CORES_PER_JOB),
                shape=(MAX_QUEUE_SIZE * 4,),
                dtype=np.int32
            ),
            # predicted prices for the next 24h
            'predicted_prices': spaces.Box(low=-1000, high=1000, shape=(24,), dtype=np.float32),
            # Summary statistics for all outstanding jobs (queue + backlog)
            'pending_job_count': spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32),
            'pending_core_hours': spaces.Box(low=0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32),
            'pending_avg_duration': spaces.Box(low=0, high=MAX_JOB_DURATION, shape=(1,), dtype=np.float32),
            'pending_max_nodes': spaces.Box(low=0, high=MAX_NODES_PER_JOB, shape=(1,), dtype=np.int32),
            'backlog_size': spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32),
        })

    def _reset_timeline_state(self, start_index: int) -> None:
        self.prices.reset(start_index=start_index)
        self.consecutive_drop_steps = 0

        self.state = {
            # Initialize all nodes to be 'online but free' (0)
            'nodes': np.zeros(MAX_NODES, dtype=np.int32),
            # Initialize job queue to be empty
            'job_queue': np.zeros((MAX_QUEUE_SIZE * 4), dtype=np.int32),
            # Initialize predicted prices array
            'predicted_prices': self.prices.predicted_prices.copy(),
            # Summary statistics for all outstanding jobs (queue + backlog)
            'pending_job_count': np.array([0], dtype=np.int32),
            'pending_core_hours': np.array([0.0], dtype=np.float32),
            'pending_avg_duration': np.array([0.0], dtype=np.float32),
            'pending_max_nodes': np.array([0], dtype=np.int32),
            'backlog_size': np.array([0], dtype=np.int32),
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

        self.backlog_queue = deque()
        self.baseline_backlog_queue = deque()

        self.next_job_id = 0  # shared between baseline and normal jobs

        # Track next empty slot in job queue for O(1) insertion
        self.next_empty_slot = 0
        self.baseline_next_empty_slot = 0

        # Versioned cache invalidation for pending job stats.
        self._queue_backlog_version = 0
        self._cached_queue_backlog_version = -1

    def _mark_queue_backlog_mutation(self) -> None:
        """Invalidate pending-job stats cache after queue/backlog content changes."""
        self._queue_backlog_version += 1

    def _pending_work_summary(self, job_queue_2d: np.ndarray, backlog_queue: deque | None = None) -> dict[str, float | int]:
        """
        Summarize currently pending work across the visible queue and the overflow backlog.

        The reward only needs a few dense signals:
        - instantaneous runnable demand (nodes * cores)
        - longer-horizon remaining work (core-hours)
        - overdue mass after the intentional deferral grace period
        """
        if backlog_queue is None:
            backlog_queue = self.backlog_queue

        active_jobs_mask = job_queue_2d[:, 0] > 0
        queue_rows = job_queue_2d[active_jobs_mask].astype(np.float32, copy=False)

        if backlog_queue:
            backlog_rows = np.asarray(list(backlog_queue), dtype=np.float32)
        else:
            backlog_rows = np.empty((0, 4), dtype=np.float32)

        if queue_rows.size and backlog_rows.size:
            combined_rows = np.vstack((queue_rows, backlog_rows))
        elif queue_rows.size:
            combined_rows = queue_rows
        elif backlog_rows.size:
            combined_rows = backlog_rows
        else:
            combined_rows = np.empty((0, 4), dtype=np.float32)

        if combined_rows.size == 0:
            return {
                "pending_job_count": 0,
                "pending_core_demand": 0.0,
                "pending_core_hours": 0.0,
                "pending_avg_duration": 0.0,
                "pending_max_nodes": 0,
                "backlog_size": len(backlog_queue),
                "oldest_age": 0.0,
                "overdue_jobs": 0,
                "overdue_age_core_hours": 0.0,
            }

        durations = combined_rows[:, 0]
        ages = combined_rows[:, 1]
        nodes = combined_rows[:, 2]
        cores = combined_rows[:, 3]
        core_demand = nodes * cores

        grace = float(self.reward_calculator.DEFERRAL_GRACE_HOURS)
        overdue_age = np.clip(ages - grace, a_min=0.0, a_max=None)
        overdue_mask = overdue_age > 0.0

        return {
            "pending_job_count": int(combined_rows.shape[0]),
            "pending_core_demand": float(np.sum(core_demand)),
            "pending_core_hours": float(np.sum(durations * core_demand)),
            "pending_avg_duration": float(np.mean(durations)),
            "pending_max_nodes": int(np.max(nodes)),
            "backlog_size": len(backlog_queue),
            "oldest_age": float(np.max(ages)),
            "overdue_jobs": int(np.count_nonzero(overdue_mask)),
            "overdue_age_core_hours": float(np.sum(overdue_age * core_demand)),
        }

    def _update_pending_job_stats(self, job_queue_2d: np.ndarray) -> None:
        """Update summary statistics for all outstanding jobs (queue + backlog)."""
        # Fast path: skip recalculation if queue/backlog version is unchanged.
        if self._cached_queue_backlog_version == self._queue_backlog_version:
            return  # Stats unchanged from last step

        summary = self._pending_work_summary(job_queue_2d)
        # Update state
        self.state['pending_job_count'][0] = int(summary["pending_job_count"])
        self.state['pending_core_hours'][0] = float(summary["pending_core_hours"])
        self.state['pending_avg_duration'][0] = float(summary["pending_avg_duration"])
        self.state['pending_max_nodes'][0] = int(summary["pending_max_nodes"])
        self.state['backlog_size'][0] = int(summary["backlog_size"])

        # Cache the queue/backlog version for next step.
        self._cached_queue_backlog_version = self._queue_backlog_version

    @staticmethod
    def _count_queued_jobs(job_queue_2d: np.ndarray) -> int:
        """Count active jobs represented in a dense queue array."""
        return int(np.count_nonzero(job_queue_2d[:, 0] > 0))

    def _flush_workload_side(
        self,
        job_queue_2d: np.ndarray,
        nodes: np.ndarray,
        cores_available: np.ndarray,
        running_jobs: dict[int, dict[str, Any]],
        backlog_queue: deque,
    ) -> int:
        """
        Drop all outstanding work for one side of the simulation and reset its cluster.

        This is intentionally stronger than a normal episode rollover: queued jobs,
        backlog jobs, and still-running jobs are all considered lost so the next
        episode can start from a clean slate.
        """
        flushed_jobs = (
            self._count_queued_jobs(job_queue_2d)
            + len(backlog_queue)
            + len(running_jobs)
        )

        job_queue_2d.fill(0)
        nodes.fill(0)
        cores_available.fill(CORES_PER_NODE)
        running_jobs.clear()
        backlog_queue.clear()
        return int(flushed_jobs)

    def _flush_workload_state(self) -> dict[str, int | float | bool]:
        """Flush both agent and baseline states immediately after a trigger step."""
        agent_job_queue_2d = self.state['job_queue'].reshape(-1, 4)
        baseline_job_queue_2d = self.baseline_state['job_queue'].reshape(-1, 4)

        agent_jobs_flushed = self._flush_workload_side(
            agent_job_queue_2d,
            self.state['nodes'],
            self.cores_available,
            self.running_jobs,
            self.backlog_queue,
        )
        baseline_jobs_flushed = self._flush_workload_side(
            baseline_job_queue_2d,
            self.baseline_state['nodes'],
            self.baseline_cores_available,
            self.baseline_running_jobs,
            self.baseline_backlog_queue,
        )

        self.next_empty_slot = 0
        self.baseline_next_empty_slot = 0
        self.metrics.current_running_jobs = 0

        if agent_jobs_flushed > 0:
            self.metrics.jobs_flushed += agent_jobs_flushed
            self.metrics.episode_jobs_flushed += agent_jobs_flushed
        if baseline_jobs_flushed > 0:
            self.metrics.baseline_jobs_flushed += baseline_jobs_flushed
            self.metrics.episode_baseline_jobs_flushed += baseline_jobs_flushed

        flush_penalty = self.reward_calculator.loss_penalty(agent_jobs_flushed)

        if agent_jobs_flushed > 0 or baseline_jobs_flushed > 0:
            self._mark_queue_backlog_mutation()
            self._update_pending_job_stats(agent_job_queue_2d)
        self.consecutive_drop_steps = 0

        self.state['job_queue'] = agent_job_queue_2d.flatten()
        self.baseline_state['job_queue'] = baseline_job_queue_2d.flatten()

        return {
            "flush_applied": bool(agent_jobs_flushed > 0 or baseline_jobs_flushed > 0),
            "agent_jobs_flushed": agent_jobs_flushed,
            "baseline_jobs_flushed": baseline_jobs_flushed,
            "flush_penalty": float(flush_penalty),
        }

    def _resolve_reset_start_index(self, options: dict[str, Any]) -> int:
        """
        Choose the initial price position for a true timeline reset.

        Normal episode rollovers are continuous and should not call this helper. It is
        only used when the whole simulation timeline is intentionally restarted.
        """
        if "price_start_index" in options:
            if self.prices is not None and self.prices.external_prices is not None:
                n_prices = len(self.prices.external_prices)
                return int(options["price_start_index"]) % n_prices
            return int(options["price_start_index"])

        if self.prices.external_prices is not None:
            if self.evaluation_mode:
                return (self.episode_idx * EPISODE_HOURS) % len(self.prices.external_prices)
            return int(self.np_random.integers(0, len(self.prices.external_prices)))

        # Even the synthetic logic prices benefit from varying the initial phase during training.
        return int(self.np_random.integers(0, self.prices.PREDICTION_WINDOW))

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict]:
        if options is None:
            options = {}

        super().reset(seed=seed)

        # Track which episode this env instance is on
        if not hasattr(self, "episode_idx"):
            self.episode_idx = 0
        else:
            self.episode_idx += 1

        self.metrics.reset_episode_metrics()

        hard_reset = bool(options.get("hard_reset", False))
        if not self._timeline_initialized or hard_reset:
            # Hard reset: restart prices, queues, nodes, backlog, and running jobs.
            # This is only for the very first env reset or for explicit ablations/debug runs.
            start_index = self._resolve_reset_start_index(options)
            self._reset_timeline_state(start_index=start_index)
            self._timeline_initialized = True
        else:
            # Soft reset: keep the ongoing simulation exactly as-is and just start a new
            # reporting window. Any price_start_index request is ignored unless a hard reset
            # is explicitly requested, because jumping the price stream mid-timeline would
            # break continuity.
            job_queue_2d = self.state['job_queue'].reshape(-1, 4)
            self._mark_queue_backlog_mutation()
            self._update_pending_job_stats(job_queue_2d)
            self.state['predicted_prices'] = self.prices.predicted_prices.copy()

        return self.state, {}

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        self.metrics.current_hour += 1
        self.metrics.total_time_hours += 1
        if self.metrics.current_hour == 1:
            self.current_episode += 1
        self.env_print(Fore.GREEN + f"\n[[[ Starting episode: {self.current_episode}, step: {self.current_step}, hour: {self.metrics.current_hour}" + Fore.RESET)

        self.state['predicted_prices'] = self.prices.advance_and_get_predicted_prices()
        current_price = float(self.state['predicted_prices'][0])
        if self.render_mode == 'human':
            self.env_print("predicted_prices: ", np.array2string(self.state['predicted_prices'], separator=" ", max_line_width=np.inf, formatter={'float_kind': lambda x: "{:05.2f}".format(x)}))

        # reshape the 1d job_queue array into 2d for cleaner code
        job_queue_2d = self.state['job_queue'].reshape(-1, 4)
        queue_backlog_mutated = False

        # Decrement booked time for nodes and complete running jobs
        self.env_print("[1] Processing ongoing jobs...")
        completed_jobs = process_ongoing_jobs(
            self.state['nodes'],
            self.cores_available,
            self.running_jobs,
            self.metrics,
            is_baseline=False,
        )
        self.env_print(f"{len(completed_jobs)} jobs completed: [{' '.join(['#' + str(job_id) for job_id in completed_jobs]) if len(completed_jobs) > 0 else ''}]")

        # Age jobs already waiting before admitting this step's new arrivals.
        self.next_empty_slot, queue_aged_dropped = age_job_queue(job_queue_2d, self.next_empty_slot)
        if queue_aged_dropped > 0:
            queue_backlog_mutated = True

        # Age helper queues (jobs waiting outside the fixed queue)
        backlog_aged_dropped = age_backlog_queue(self.backlog_queue, self.metrics, _is_baseline=False)
        if backlog_aged_dropped > 0:
            queue_backlog_mutated = True

        # Fill real queue from helper before accepting new jobs
        self.next_empty_slot, moved_from_backlog = fill_queue_from_backlog(job_queue_2d, self.backlog_queue, self.next_empty_slot)
        if moved_from_backlog > 0:
            queue_backlog_mutated = True

        # Generate new jobs
        self.env_print(f"[2] Generating new jobs...")
        new_jobs_count, new_jobs_durations, new_jobs_nodes, new_jobs_cores = generate_jobs(
            self.metrics.current_hour,
            self.external_jobs, self.external_hourly_jobs, self.external_durations,
            self.workload_gen, self.jobs_sampler if hasattr(self, 'jobs_sampler') else None,
            hourly_sampler, durations_sampler, self.np_random,
            job_arrival_scale=self.job_arrival_scale,
            jobs_exact_replay=self.jobs_exact_replay,
        )

        # Add new jobs to queue (overflow goes to helper)
        self.env_print(f"[2] Adding {new_jobs_count} new jobs to the queue...")
        new_jobs, self.next_empty_slot, backlog_dropped = add_new_jobs(
            job_queue_2d, new_jobs_count, new_jobs_durations,
            new_jobs_nodes, new_jobs_cores, self.next_empty_slot, self.backlog_queue
        )
        if len(new_jobs) > 0:
            queue_backlog_mutated = True
        if backlog_dropped > 0:
            self.metrics.jobs_rejected_queue_full += backlog_dropped
            self.metrics.episode_jobs_rejected_queue_full += backlog_dropped
        self.metrics.jobs_submitted += new_jobs_count
        self.metrics.episode_jobs_submitted += new_jobs_count

        if self.render_mode == 'human':
            self.env_print("nodes: ", np.array2string(self.state['nodes'], separator=' ', max_line_width=np.inf))
            self.env_print(f"cores_available: {np.array2string(self.cores_available, separator=' ', max_line_width=np.inf)} ({np.sum(self.cores_available)})")
        self.env_print(f">>> adding {len(new_jobs)} new jobs to the queue: {' '.join(['[{}h {} {}x{}]'.format(d, a, n, c) for d, a, n, c in new_jobs])}")
        self.env_print("job_queue: ", ' '.join(['[{} {} {} {}]'.format(d, a, n, c) for d, a, n, c in job_queue_2d if d > 0]))

        # Snapshot the pending queue the agent is deciding about *before* launching jobs.
        decision_pending_summary = self._pending_work_summary(job_queue_2d)

        action_type, action_magnitude, do_refill = action
        action_magnitude += 1

        self.env_print(f"[3] Adjusting nodes based on action: type={action_type}, magnitude={action_magnitude}, refill={do_refill}...")
        num_node_changes = adjust_nodes(action_type, action_magnitude, self.state['nodes'], self.cores_available, self.env_print)

        # Assign jobs to available nodes
        self.env_print(f"[4] Assigning jobs to available nodes...")

        num_dropped_this_step = queue_aged_dropped + backlog_aged_dropped + backlog_dropped
        num_launched_jobs, self.next_empty_slot, queue_dropped, self.next_job_id = assign_jobs_to_available_nodes(
            job_queue_2d, self.state['nodes'], self.cores_available, self.running_jobs,
            self.next_empty_slot, self.next_job_id, self.metrics, is_baseline=False
        )
        if num_launched_jobs > 0 or queue_dropped > 0:
            queue_backlog_mutated = True
        num_dropped_this_step += queue_dropped

        self.env_print(f"   {num_launched_jobs} jobs launched")

        # Refill queue from backlog if agent chose to do so
        if do_refill == 1 and len(self.backlog_queue) > 0:
            self.next_empty_slot, moved = fill_queue_from_backlog(job_queue_2d, self.backlog_queue, self.next_empty_slot)
            if moved > 0:
                queue_backlog_mutated = True
                self.env_print(f"   {moved} jobs moved from backlog to queue")
                # Try to assign the newly queued jobs
                extra_launched, self.next_empty_slot, extra_dropped, self.next_job_id = assign_jobs_to_available_nodes(
                    job_queue_2d, self.state['nodes'], self.cores_available, self.running_jobs,
                    self.next_empty_slot, self.next_job_id, self.metrics, is_baseline=False
                )
                if extra_launched > 0 or extra_dropped > 0:
                    queue_backlog_mutated = True
                num_launched_jobs += extra_launched
                num_dropped_this_step += extra_dropped
                if extra_launched > 0:
                    self.env_print(f"   {extra_launched} additional jobs launched from backlog")

        remaining_pending_summary = self._pending_work_summary(job_queue_2d)

        # Update summary statistics for all outstanding jobs (queue + backlog)
        if queue_backlog_mutated:
            self._mark_queue_backlog_mutation()
        self._update_pending_job_stats(job_queue_2d)

        # Calculate node utilization stats
        num_used_nodes = np.sum(self.state['nodes'] > 0)
        num_on_nodes = np.sum(self.state['nodes'] >= 0)
        num_off_nodes = np.sum(self.state['nodes'] == -1)
        num_idle_nodes = num_on_nodes - num_used_nodes
        num_unprocessed_jobs = np.sum(job_queue_2d[:, 0] > 0)
        combined_queue_size = num_unprocessed_jobs + len(self.backlog_queue)
        num_unprocessed_jobs = combined_queue_size
        average_future_price = float(np.mean(self.state['predicted_prices']))
        num_used_cores = int(num_on_nodes * CORES_PER_NODE - np.sum(self.cores_available))
        num_running_jobs = len(self.running_jobs)

        # update stats
        self.metrics.on_nodes.append(num_on_nodes)
        self.metrics.used_nodes.append(num_used_nodes)
        self.metrics.job_queue_sizes.append(num_unprocessed_jobs)
        self.metrics.price_stats.append(current_price)
        self.metrics.current_running_jobs = num_running_jobs
        self.metrics.episode_running_jobs_counts.append(num_running_jobs)
        self.metrics.episode_on_nodes.append(num_on_nodes)
        self.metrics.episode_used_nodes.append(num_used_nodes)
        self.metrics.episode_used_cores.append(num_used_cores)
        self.metrics.episode_job_queue_sizes.append(num_unprocessed_jobs)
        self.metrics.episode_price_stats.append(current_price)
        self.metrics.episode_pending_jobs_end = int(remaining_pending_summary["pending_job_count"])
        self.metrics.episode_pending_core_demand_end = float(remaining_pending_summary["pending_core_demand"])
        self.metrics.episode_pending_core_hours_end = float(remaining_pending_summary["pending_core_hours"])
        self.metrics.episode_overdue_jobs_end = int(remaining_pending_summary["overdue_jobs"])
        self.metrics.episode_overdue_age_core_hours_end = float(remaining_pending_summary["overdue_age_core_hours"])

        # Track max queue size (queue only, without backlog)
        queue_only_size = np.sum(job_queue_2d[:, 0] > 0)
        if queue_only_size > self.metrics.max_queue_size_reached:
            self.metrics.max_queue_size_reached = queue_only_size
        if queue_only_size > self.metrics.episode_max_queue_size_reached:
            self.metrics.episode_max_queue_size_reached = queue_only_size

        # Track max backlog size
        backlog_size = len(self.backlog_queue)
        if backlog_size > self.metrics.max_backlog_size_reached:
            self.metrics.max_backlog_size_reached = backlog_size
        if backlog_size > self.metrics.episode_max_backlog_size_reached:
            self.metrics.episode_max_backlog_size_reached = backlog_size

        self.env_print(f"[5] Calculating reward...")

        # Baseline step
        baseline_cost, baseline_cost_off, baseline_power_mwh, baseline_power_off_mwh, self.baseline_next_empty_slot, self.next_job_id, baseline_num_used_nodes, baseline_num_used_cores = baseline_step(
            self.baseline_state, self.baseline_cores_available, self.baseline_running_jobs,
            current_price, new_jobs_count, new_jobs_durations, new_jobs_nodes, new_jobs_cores,
            self.baseline_next_empty_slot, self.next_job_id, self.metrics, self.env_print,
            self.baseline_backlog_queue
        )

        self.metrics.baseline_cost += baseline_cost
        self.metrics.baseline_cost_off += baseline_cost_off
        self.metrics.episode_baseline_cost += baseline_cost
        self.metrics.episode_baseline_cost_off += baseline_cost_off
        self.metrics.baseline_power_consumption_mwh += baseline_power_mwh
        self.metrics.baseline_power_consumption_off_mwh += baseline_power_off_mwh
        self.metrics.episode_baseline_power_consumption_mwh += baseline_power_mwh
        self.metrics.episode_baseline_power_consumption_off_mwh += baseline_power_off_mwh

        self.metrics.episode_baseline_used_nodes.append(baseline_num_used_nodes)
        self.metrics.episode_baseline_used_cores.append(baseline_num_used_cores)

        step_reward, step_cost, eff_reward_norm, price_reward, idle_penalty_norm, job_age_penalty_norm = self.reward_calculator.calculate(
            num_used_nodes, num_idle_nodes, current_price, average_future_price,
            num_off_nodes, job_queue_2d, num_unprocessed_jobs, self.weights,
            num_dropped_this_step, self.env_print, num_on_nodes, num_used_cores,
            decision_pending_core_demand=float(decision_pending_summary["pending_core_demand"]),
            remaining_overdue_age_core_hours=float(remaining_pending_summary["overdue_age_core_hours"]),
        )

        self.metrics.episode_reward += step_reward
        step_power_mwh = power_consumption_mwh(num_on_nodes, num_used_cores)
        self.metrics.total_cost += step_cost
        self.metrics.episode_total_cost += step_cost
        self.metrics.total_power_consumption_mwh += step_power_mwh
        self.metrics.episode_total_power_consumption_mwh += step_power_mwh

        # Store normalized reward components for plotting
        self.metrics.eff_rewards.append(eff_reward_norm * 100)
        self.metrics.price_rewards.append(price_reward * 100)
        self.metrics.job_age_penalties.append(job_age_penalty_norm * 100)
        self.metrics.idle_penalties.append(idle_penalty_norm * 100)
        self.metrics.rewards.append(step_reward)
        self.metrics.episode_eff_rewards.append(eff_reward_norm * 100)
        self.metrics.episode_price_rewards.append(price_reward * 100)
        self.metrics.episode_job_age_penalties.append(job_age_penalty_norm * 100)
        self.metrics.episode_idle_penalties.append(idle_penalty_norm * 100)
        self.metrics.episode_rewards.append(step_reward)
        self.metrics.jobs_dropped += num_dropped_this_step
        self.metrics.episode_jobs_dropped += num_dropped_this_step
        if num_dropped_this_step > 0:
            self.consecutive_drop_steps += 1
        else:
            self.consecutive_drop_steps = 0
        if self.consecutive_drop_steps > self.metrics.episode_max_drop_streak:
            self.metrics.episode_max_drop_streak = self.consecutive_drop_steps
        flush_applied = False
        flush_penalty = 0.0
        agent_jobs_flushed = 0
        baseline_jobs_flushed = 0
        terminal_penalty_applied = 0.0
        drop_streak_steps = self.consecutive_drop_steps
        flush_triggered_by_drop_streak = (
            self.flush_after_drop_streak > 0
            and self.consecutive_drop_steps >= self.flush_after_drop_streak
        )
        if flush_triggered_by_drop_streak:
            flush_result = self._flush_workload_state()
            flush_applied = bool(flush_result["flush_applied"])
            flush_penalty = float(flush_result["flush_penalty"])
            agent_jobs_flushed = int(flush_result["agent_jobs_flushed"])
            baseline_jobs_flushed = int(flush_result["baseline_jobs_flushed"])
            previous_step_reward = step_reward
            terminal_penalty_applied = float(self.DROP_STREAK_TERMINATION_PENALTY)
            step_reward = terminal_penalty_applied
            self.metrics.episode_reward += terminal_penalty_applied - previous_step_reward
            if self.metrics.rewards:
                self.metrics.rewards[-1] = terminal_penalty_applied
            if self.metrics.episode_rewards:
                self.metrics.episode_rewards[-1] = terminal_penalty_applied

            post_flush_pending_summary = self._pending_work_summary(self.state['job_queue'].reshape(-1, 4))
            self.metrics.episode_pending_jobs_end = int(post_flush_pending_summary["pending_job_count"])
            self.metrics.episode_pending_core_demand_end = float(post_flush_pending_summary["pending_core_demand"])
            self.metrics.episode_pending_core_hours_end = float(post_flush_pending_summary["pending_core_hours"])
            self.metrics.episode_overdue_jobs_end = int(post_flush_pending_summary["overdue_jobs"])
            self.metrics.episode_overdue_age_core_hours_end = float(post_flush_pending_summary["overdue_age_core_hours"])

            if flush_applied:
                self.env_print(
                    f"[flush] Drop streak {self.flush_after_drop_streak}+ reached "
                    f"({drop_streak_steps} steps). "
                    f"Dropped outstanding work immediately: "
                    f"agent={agent_jobs_flushed}, baseline={baseline_jobs_flushed}, "
                    f"loss_penalty={flush_penalty:.4f}, terminal_penalty={terminal_penalty_applied:.4f}"
                )
        
        # print stats
        self.env_print(f"[6] End of step stats...")
        self.env_print("job queue: ", ' '.join(['[{} {} {} {}]'.format(d, a, n, c) for d, a, n, c in job_queue_2d if d > 0]))
        self.env_print(f"{len(self.running_jobs)} running jobs: {' '.join(['[#{}: {}h, {}x{}]'.format(job_id, job_data['duration'], len(job_data['allocation']), int(job_data['allocation'][0][1])) for job_id, job_data in self.running_jobs.items()]) if len(self.running_jobs) > 0 else '[]'}")
        self.env_print(f"launched jobs: {num_launched_jobs}, unprocessed jobs: {num_unprocessed_jobs}")
        self.env_print(f"nodes: ON: {num_on_nodes}, OFF: {num_off_nodes}, used: {num_used_nodes}, IDLE: {num_idle_nodes}. node changes: {num_node_changes}")
        if self.render_mode == 'human':
            self.env_print("nodes: ", np.array2string(self.state['nodes'], separator=" ", max_line_width=np.inf))
            self.env_print(f"cores used: {num_used_cores} out of {num_on_nodes * CORES_PER_NODE} available cores")
            self.env_print(f"cores_available: {np.array2string(self.cores_available, separator=' ', max_line_width=np.inf)} ({np.sum(self.cores_available)})")
        self.env_print(f"price: current: {current_price}, average future: {average_future_price:.4f}")
        self.env_print(f"step reward: {step_reward:.4f}, episode reward: {self.metrics.episode_reward:.4f}")

        truncated = False
        terminated = False
        if flush_triggered_by_drop_streak:
            terminated = True
            truncated = False
        elif self.metrics.current_hour == EPISODE_HOURS:
            if self.render_mode == 'human':
                plot_episode(self, EPISODE_HOURS, MAX_NODES, False, True, self.current_step)
                if self.plot_config.plot_once:
                    raise PlottingComplete
            else:
                # Only do training plots in training mode
                if not self.evaluation_mode and self.current_step > self.next_plot_save:
                    plot_episode(self, EPISODE_HOURS, MAX_NODES, True, False, self.current_step)
                    self.next_plot_save += self.steps_per_iteration
                    print(self.next_plot_save)
            truncated = True
            terminated = False

        if terminated or truncated:
            # Record episode costs before reset so callbacks/evaluation can read the finished episode.
            self.metrics.record_episode_completion(self.current_episode)

        # flatten job_queue again
        self.state['job_queue'] = job_queue_2d.flatten()

        if self.render_mode == 'human':
            # go slow to be able to read stuff in human mode
            if not self.plot_config.quick_plot:
                time.sleep(1)

        self.env_print(Fore.GREEN + f"]]]" + Fore.RESET)

        info = {
            "step_cost": step_cost,
            "num_unprocessed_jobs": num_unprocessed_jobs,
            "num_on_nodes": num_on_nodes,
            "episode_jobs_dropped": self.metrics.episode_jobs_dropped,
            "episode_jobs_flushed": self.metrics.episode_jobs_flushed,
            "episode_jobs_lost_total": self.metrics.episode_jobs_dropped + self.metrics.episode_jobs_flushed,
            "episode_flush_applied": flush_applied,
            "episode_flush_triggered_by_drop_streak": flush_triggered_by_drop_streak,
            "drop_streak_steps": drop_streak_steps,
            "drop_streak_flush_armed": flush_triggered_by_drop_streak,
            "step_flush_penalty": flush_penalty,
            "step_terminal_penalty": terminal_penalty_applied,
            "step_jobs_flushed": agent_jobs_flushed,
            "step_baseline_jobs_flushed": baseline_jobs_flushed,
        }

        return self.state, step_reward, terminated, truncated, info
