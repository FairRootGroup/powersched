"""
Rule-based node controllers, evaluated in place of a trained policy (train.py --evaluate-savings --heuristic).

Both decide, like the policy, once per hour before that hour's arrivals, and act through the same
action space (switch off / keep / switch on one of NODE_CHANGE_STEPS nodes per hour; switching off
only affects idle nodes). They read the waiting queue and node states from the environment itself,
which is more detail than the policy's observation summarises.

  demand:k=v,...  price-blind demand following. Target online nodes = nodes running jobs + the nodes
                  the waiting jobs' core demand fills + `headroom` idle nodes.
                  headroom (default 0): extra idle nodes kept online for the next hour's arrivals.
  price:k=v,...   price-threshold deferral on top of demand following. While the current price is above
                  the `quantile` of the 24 h price window (current hour and the next 23), only jobs
                  that have waited at least `max_wait` hours count as demand; the rest are deferred.
                  quantile (default 0.5), max_wait (default 12), headroom (default 0).
"""

from __future__ import annotations

import numpy as np

from src.config import CORES_PER_NODE, NODE_CHANGE_STEPS

_STEPS = np.asarray(NODE_CHANGE_STEPS)


class HeuristicPolicy:
    def __init__(self, env, spec: str) -> None:
        name, _, params = spec.partition(":")
        if name not in ("demand", "price"):
            raise ValueError(f"unknown heuristic {name!r}, expected demand or price")
        values = dict(p.split("=", 1) for p in params.split(",") if p.strip())
        defaults = {"headroom": 0.0} if name == "demand" else {"headroom": 0.0, "quantile": 0.5, "max_wait": 12.0}
        unknown = set(values) - set(defaults)
        if unknown:
            raise ValueError(f"unknown parameters for {name}: {sorted(unknown)}")
        self.env = env
        self.name = name
        self.params = {k: float(values.get(k, v)) for k, v in defaults.items()}

    @property
    def label(self) -> str:
        """Short name for directories and summaries, e.g. price_h0_q0.5_w12."""
        p = self.params
        label = f"{self.name}_h{p['headroom']:g}"
        if self.name == "price":
            label += f"_q{p['quantile']:g}_w{p['max_wait']:g}"
        return label

    def target_online(self) -> int:
        env = self.env
        nodes = env.state["nodes"]
        rows = env.queue.rows()
        if self.name == "price":
            prices = env.state["predicted_prices"]  # [0] is the hour this action applies to
            if prices[0] > np.quantile(prices, self.params["quantile"]):
                # Ages grow by one before this hour's dispatch.
                rows = rows[rows[:, 1] + 1 >= self.params["max_wait"]]
        demand_cores = int(np.sum(rows[:, 2].astype(np.int64) * rows[:, 3]))
        used = int(np.sum(nodes > 0))
        free_on_used = int(np.sum(env.cores_available[nodes > 0]))
        demand_nodes = int(np.ceil(max(0, demand_cores - free_on_used) / CORES_PER_NODE))
        return used + demand_nodes + int(self.params["headroom"])

    def predict(self, obs, deterministic: bool = True):
        nodes = self.env.state["nodes"]
        online = int(np.sum(nodes >= 0))
        delta = min(self.target_online(), len(nodes)) - online
        if delta == 0:
            return np.array([1, 0]), None
        # Largest step that does not overshoot the target.
        index = int(np.searchsorted(_STEPS, abs(delta), side="right") - 1)
        return np.array([2 if delta > 0 else 0, index]), None
