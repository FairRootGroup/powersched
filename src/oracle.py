"""Liquid oracle: theoretical lower bound on electricity cost for a given workload.

The liquid oracle answers: given perfect knowledge of all arriving jobs and prices
for an episode, and the freedom to run any job at any hour (no arrival-time or
deadline constraints), what is the minimum achievable electricity cost?

It treats the workload as a divisible fluid -- total core-hours -- and redistributes
it greedily to the cheapest hours first, up to cluster capacity.  This is an
optimistic lower bound; real schedulers face arrival times, deadlines, and discrete
job sizes that the oracle ignores.

Use it as a benchmark: the gap between baseline_off and oracle is the maximum
workload-shifting savings available in principle.  The fraction of that gap captured
by the agent is how close to optimal the agent is.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.config import CORES_PER_NODE, MAX_NODES, COST_USED_MW, MAX_JOB_AGE

# Carried-over job tuple: (age, duration, nodes, cores_per_node)
CarriedJob = tuple[int, int, int, int]


class LiquidOracle:
    """
    Accumulates job work (core-hours) and prices during an episode, then solves
    for the minimum cost schedule at episode end.

    Algorithm:
      1. Total work W = sum over all arriving jobs of (duration * nodes * cores_per_node).
      2. Sort episode hours by price, cheapest first.
      3. Fill each hour up to MAX_CORES_PER_HOUR until W is exhausted.
      4. Cost = sum_t (core_hours_t / CORES_PER_NODE) * COST_USED_MW * price_t.

    No idle-node overhead is charged: the oracle powers on exactly the nodes it needs.
    Negative prices are handled correctly -- those hours are filled first and earn revenue.
    """

    MAX_CORES_PER_HOUR: int = MAX_NODES * CORES_PER_NODE  # 335 * 96 = 32,160

    def __init__(self) -> None:
        self._prices: list[float] = []
        self._total_core_hours: float = 0.0

    def reset(self, carried_jobs: list[CarriedJob] | None = None) -> None:
        """Reset at the start of a new episode.

        carried_jobs: jobs left in the queue/backlog at the end of the previous
            episode, as (age, duration, nodes, cores_per_node) tuples.  Their
            work is pre-loaded so the oracle accounts for the same workload the
            agent inherits.
        """
        self._prices = []
        self._total_core_hours = 0.0
        if carried_jobs:
            for _, duration, nodes, cores in carried_jobs:
                self._total_core_hours += float(duration * nodes * cores)

    def record(
        self,
        price: float,
        new_jobs_durations: list[int],
        new_jobs_nodes: list[int],
        new_jobs_cores: list[int],
    ) -> None:
        """Record one hour's price and the core-hours of work that arrived."""
        if len(new_jobs_durations) != len(new_jobs_nodes) or len(new_jobs_durations) != len(new_jobs_cores):
            raise ValueError(
                f"Job list lengths must match: durations={len(new_jobs_durations)}, "
                f"nodes={len(new_jobs_nodes)}, cores={len(new_jobs_cores)}"
            )
        self._prices.append(price)
        for d, n, c in zip(new_jobs_durations, new_jobs_nodes, new_jobs_cores):
            self._total_core_hours += float(d * n * c)

    def solve(self) -> float:
        """
        Compute the minimum cost to execute all recorded work given the episode's
        price series and cluster capacity.

        Returns:
            Minimum achievable electricity cost (euros) for this episode's workload.
            Returns 0.0 if no work or no prices were recorded.
        """
        if not self._prices or self._total_core_hours <= 0.0:
            return 0.0

        sorted_prices = np.sort(np.asarray(self._prices, dtype=float))  # cheapest first

        remaining = self._total_core_hours
        cost = 0.0
        max_cores = float(self.MAX_CORES_PER_HOUR)

        for price in sorted_prices:
            if remaining <= 0.0:
                break
            core_hours = min(remaining, max_cores)
            # No idle overhead: nodes powered = core_hours / CORES_PER_NODE (fractional)
            power_mwh = (core_hours / CORES_PER_NODE) * COST_USED_MW
            cost += power_mwh * price
            remaining -= core_hours

        return cost

    @property
    def total_core_hours(self) -> float:
        """Total core-hours of work recorded so far this episode."""
        return self._total_core_hours


class ContiguousOracle:
    """
    Theoretical lower bound on electricity cost that respects all real job
    constraints except perfect foresight:

      - Jobs cannot start before they arrive.
      - Each job runs continuously for its full duration (no splitting/preemption).
      - Fixed node/core requirements must be met simultaneously at every hour.
      - Jobs expire if not started within MAX_JOB_AGE hours of arrival.

    Algorithm (least-slack-first greedy):
      For each job, sorted by fewest feasible start positions relative to duration:
        1. Find every candidate start time t in [arrival, arrival+MAX_JOB_AGE-duration].
        2. Check feasibility: max capacity in [t, t+duration) + job_cores <= cluster_capacity.
        3. Pick the cheapest feasible window (window price sum via prefix sums, O(1)/candidate).
        4. Commit that window to the capacity array.

    The greedy does not guarantee the global optimum, so the result is an upper
    bound on what a perfect-foresight scheduler could achieve, not a lower bound.
    It is always >= LiquidOracle cost (fewer scheduling freedoms).

    Drop-in replacement for LiquidOracle: same record()/reset()/solve() interface.
    """

    MAX_CORES_PER_HOUR: int = MAX_NODES * CORES_PER_NODE  # 335 * 96 = 32,160

    def __init__(self) -> None:
        self._prices: list[float] = []
        # (arrival_hour, duration, total_cores) per job.
        # arrival_hour is negative for carried-over jobs: -age encodes remaining
        # lifetime as MAX_JOB_AGE - age hours from episode start.
        self._jobs: list[tuple[int, int, float]] = []
        self.unscheduled_count: int = 0
        self.spillover_count: int = 0

    def reset(self, carried_jobs: list[CarriedJob] | None = None) -> None:
        """Reset at the start of a new episode.

        carried_jobs: jobs left in the queue/backlog at the end of the previous
            episode, as (age, duration, nodes, cores_per_node) tuples.  Encoded
            with arrival = -age so that their deadline (arrival + MAX_JOB_AGE)
            equals the remaining lifetime from episode start.
        """
        self._prices = []
        self._jobs = []
        self.unscheduled_count = 0
        self.spillover_count = 0
        if carried_jobs:
            for age, duration, nodes, cores in carried_jobs:
                self._jobs.append((-age, duration, float(nodes * cores)))

    def record(
        self,
        price: float,
        new_jobs_durations: list[int],
        new_jobs_nodes: list[int],
        new_jobs_cores: list[int],
    ) -> None:
        """Record one hour's price and all jobs that arrived this step."""
        if len(new_jobs_durations) != len(new_jobs_nodes) or len(new_jobs_durations) != len(new_jobs_cores):
            raise ValueError(
                f"Job list lengths must match: durations={len(new_jobs_durations)}, "
                f"nodes={len(new_jobs_nodes)}, cores={len(new_jobs_cores)}"
            )
        arrival = len(self._prices)  # 0-based index of this hour in the episode
        self._prices.append(price)
        for d, n, c in zip(new_jobs_durations, new_jobs_nodes, new_jobs_cores):
            self._jobs.append((arrival, d, float(n * c)))

    def solve(self) -> float:
        """
        Compute the minimum cost schedule that honors job continuity, arrival
        times, deadlines, and capacity constraints.

        Returns:
            Minimum achievable cost (euros).  0.0 if no jobs recorded.
            Jobs that cannot be placed (expired window or fully capacity-blocked)
            are skipped; check unscheduled_count after calling solve().
        """
        if not self._prices or not self._jobs:
            self.unscheduled_count = 0
            self.spillover_count = 0
            return 0.0

        T = len(self._prices)
        prices = np.asarray(self._prices, dtype=float)

        # Prefix sums: sum(prices[t : t+d]) = price_prefix[t+d] - price_prefix[t]
        price_prefix = np.empty(T + 1, dtype=float)
        price_prefix[0] = 0.0
        np.cumsum(prices, out=price_prefix[1:])

        capacity = np.zeros(T, dtype=float)
        max_cores = float(self.MAX_CORES_PER_HOUR)
        cost_factor = COST_USED_MW / CORES_PER_NODE  # cost per core-hour per unit price

        # Least-slack-first: jobs with fewest candidate start positions (relative
        # to their duration) are harder to place and get first pick of cheap windows.
        # For carried-over jobs arrival is negative; clamp t_min to 0 for slack calc.
        def _slack(job: tuple[int, int, float]) -> float:
            arrival, duration, _ = job
            t_min = max(0, arrival)
            t_max = min(arrival + MAX_JOB_AGE, T) - duration
            return max(t_max - t_min + 1, 0) / max(duration, 1)

        self.unscheduled_count = 0
        self.spillover_count = 0
        total_cost = 0.0

        for arrival, duration, total_cores in sorted(self._jobs, key=_slack):
            t_min = max(0, arrival)  # carried-over jobs have negative arrival
            t_max = min(arrival + MAX_JOB_AGE, T) - duration

            if t_max < t_min:
                # Cross-episode spillover: job arrived too late to finish within
                # this episode but hasn't expired — it will be carried into the
                # next episode's oracle via the reset() carryover mechanism.
                latest_start_deadline = arrival + MAX_JOB_AGE - duration
                # If deadline already makes the window impossible, this job is
                # genuinely unschedulable (not spillover).
                if latest_start_deadline < t_min:
                    self.unscheduled_count += 1
                else:
                    # Window is only impossible because the episode horizon ends.
                    self.spillover_count += 1
                continue

            n_candidates = t_max - t_min + 1

            # Sliding max of the capacity array over windows of size `duration`.
            # Vectorised across all candidate start times at once.
            cap_slice = capacity[t_min : t_max + duration]
            if duration == 1:
                max_in_window = cap_slice[:n_candidates].copy()
            else:
                max_in_window = sliding_window_view(cap_slice, duration).max(axis=1)
            feasible = max_in_window + total_cores <= max_cores

            if not feasible.any():
                self.unscheduled_count += 1
                continue  # fully capacity-blocked; job unschedulable

            # O(1) per candidate: window price sum from prefix array
            t_starts = np.arange(t_min, t_max + 1, dtype=np.intp)
            window_costs = (
                (price_prefix[t_starts + duration] - price_prefix[t_starts])
                * total_cores
                * cost_factor
            )
            window_costs[~feasible] = np.inf

            best_idx = int(window_costs.argmin())
            capacity[t_min + best_idx : t_min + best_idx + duration] += total_cores
            total_cost += float(window_costs[best_idx])

        return total_cost
