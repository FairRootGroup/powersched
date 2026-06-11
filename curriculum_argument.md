# Curriculum Rationale for Deferral-Oriented Training

## Abstract

We do not use a curriculum because the agent is incapable of learning the final task directly. We use a curriculum because the desired policy is a structured, long-horizon behavior that is difficult to discover reliably when realistic workload noise, burstiness, and price irregularity are introduced from the start. The goal is to learn a specific control law: defer flexible work out of expensive hours, then clear backlog aggressively in cheap hours, while keeping overdue work and job loss near zero. The curriculum should help because each stage preserves that same control law while gradually increasing nuisance complexity. In contrast, simply training the full problem for far more steps mainly increases exposure to locally adequate but strategically wrong behaviors, such as immediate service, low-level trickle service, or unstable over-deferral.

## 1. Problem Setting

The scheduling problem is not a one-step price-selection task. It is a sequential decision problem in which the agent must coordinate several coupled choices across many hours:

- whether to keep nodes online now or wait,
- whether queued work should be launched now or deferred,
- whether backlog should be refilled into the visible queue,
- and how aggressively pending work should be cleared once a cheap window arrives.

The intended behavior is therefore not "minimize cost" in the abstract. It is the more constrained policy:

1. avoid unnecessary execution in expensive hours,
2. use the forecast to recognize that cheaper hours are imminent,
3. preserve pending work during a deliberate deferral window,
4. then convert backlog into high throughput during cheap hours,
5. without allowing backlog to become overdue or unstable.

This structure is directly supported by the environment. The agent observes a 24-hour forecast window in [src/prices.py](../src/prices.py) and [src/environment.py](../src/environment.py), as well as pending-work summaries such as outstanding job count, pending core-hours, and backlog size in [src/environment.py](../src/environment.py).

## 2. Why the Task Is Hard to Learn Directly

The main difficulty is not the absence of signal. The main difficulty is delayed and entangled credit assignment.

If the agent serves work immediately during expensive hours, it may avoid future queue pressure and avoid any risk of overdue jobs. That behavior is strategically wrong, but it can still look locally safe. If the agent defers too aggressively, the policy becomes exposed to drops, overdue backlog, and terminal instability. The desired policy sits between those two extremes and requires a coordinated sequence of actions whose benefit may only become clear many hours later.

This creates three learning obstacles.

### 2.1 Long-horizon credit assignment

The key beneficial action is often "do less now so that more can be done later under better prices." The reward consequence of that choice is distributed across multiple future hours rather than being concentrated in the current step. The agent must therefore learn a multi-step motif rather than a simple reactive rule.

### 2.2 Competing local optima

Several behaviors are easier to discover than the target behavior:

- immediate service: process jobs as soon as possible and largely ignore price timing,
- trickle service: keep a small continuous flow through both expensive and cheap periods,
- over-deferral: wait too long, then fail to clear backlog before it becomes dangerous.

All three can produce locally reasonable rewards for substantial portions of training. As a result, the optimizer can spend large amounts of time refining the wrong policy family instead of discovering the intended one.

### 2.3 Variance from realistic data

Real arrivals and real prices contain burst structure, phase irregularity, and changing load intensity. Those sources of variation matter for final performance, but they obscure the underlying causal rule we want the policy to acquire first. When introduced too early, they reduce the signal-to-noise ratio of the deferral objective.

## 3. Why the Current Reward Design Makes the Curriculum Plausible

The reward was redesigned to teach "defer into cheap, then serve hard in cheap" rather than to punish waiting indiscriminately.

### 3.1 Price timing is explicit

Useful work in cheap hours is rewarded, and useful work in expensive hours is penalized in [src/reward_calculation.py](../src/reward_calculation.py). This term shapes when work should happen, not merely whether work happened.

### 3.2 Deferral is no longer structurally contradicted

The blackout term only rewards turning everything off when there is truly no work to do. It no longer immediately treats a queued backlog as evidence that waiting is wrong. This matters because an expensive-hour deferral policy would otherwise be fighting an internal contradiction in the objective.

### 3.3 Cheap windows now carry service pressure

The former job-age slot was repurposed into a cheap-hour service-pressure term. If the system is in a cheap phase and meaningful pending demand exists, under-serving that cheap phase is penalized. This directly discourages the degenerate "wait now, but also fail to ramp later" strategy.

### 3.4 Overdue starvation is structural, not optional

Once work survives beyond the 24-hour grace window, near-zero service becomes intrinsically bad through a post-grace starvation term. This is important because it removes the possibility that the agent can succeed merely by learning to delay forever.

### 3.5 Losses are visible

Dropped or flushed jobs are penalized, and episode summaries expose pending backlog, overdue backlog, launch-based wait time, and drop-related metrics. This prevents "cheap-looking" policies from succeeding by silently accumulating hidden service debt.

## 4. Why "Just Train for 50M Steps" Is Not a Sufficient Answer

The claim that enough steps should eventually solve the full task assumes that exploration will routinely reach the target behavior and that optimization will then prefer it over the easier alternatives. In this setting, neither assumption is reliable.

### 4.1 More data does not fix a bad basin

On-policy PPO improves the policy it is already sampling from. If training falls into an immediate-service basin or a trickle-service basin, additional steps mostly refine that basin. More experience is helpful only if the agent already encounters enough promising defer-then-clear trajectories for the optimizer to reinforce them.

### 4.2 Intermediate forms of the right policy can look worse before they look better

A partially learned defer policy may initially increase queue pressure before it learns the matching cheap-hour clear behavior. During that period, the policy can receive worse returns than a simpler always-serve heuristic. This makes discovery of the full behavior fragile: the optimizer may move away from the promising region before the entire action sequence is assembled.

### 4.3 Realistic variability buries the core signal

When bursty arrivals and irregular real prices are present from the start, the deferral motif is statistically diluted by many unrelated sources of variance. In that regime, 50 million steps are not 50 million clean lessons about phase behavior. They are 50 million samples from a mixture of effects, many of which do not help identify the intended control rule.

### 4.4 The failure modes are not symmetric

Serving too early is often safe but suboptimal. Deferring too hard is risky and can trigger drops or instability. Because one wrong direction is safer than the other, unguided optimization is biased toward conservative behaviors that avoid catastrophic backlog dynamics even if they sacrifice price arbitrage.

For these reasons, "more steps" and "better discoverability" are not interchangeable. The curriculum is meant to improve discoverability.

## 5. Why the Stage Sequence Should Help

Each stage keeps the same behavioral invariant but changes the amount of nuisance complexity.

### Stage A: flat arrivals + logic prices

This stage isolates the phase rule itself. With deterministic 12-hour expensive and 12-hour cheap periods, the policy can learn the basic timing pattern without needing to infer it from noisy price trajectories.

### Stage B: high-load flat arrivals + logic prices

This removes excess slack. If the cheap half-day can clear everything trivially, the agent does not need to learn disciplined backlog management. Increasing load forces the price-timing decision to matter.

### Stage C: bursty arrivals + logic prices

This stage tests queue-spike robustness. A small per-hour probability of a burst injects sudden load spikes on top of base arrivals. Because the burst check is price-blind - applied with equal probability across all 336 hours - spikes are distributed roughly evenly between expensive and cheap periods. The agent must absorb unpredictable surges without destabilizing, while still maintaining the defer-then-clear pattern it learned in Stage B.

### Stage D: main arrivals + logic prices

At this point the policy transfers to realistic workload structure while price phases remain simple. The agent is now solving a more realistic queueing problem without having to relearn what counts as cheap and expensive.

### Stage E: main arrivals + noisy logic prices

This stage adds robustness. Cheap and expensive windows are no longer perfectly regular, but the defer-then-clear strategy still applies. The agent must now generalize the learned motif across moderate price perturbations.
NOTE: In practice this stage has not shown as much impact, as it was theorized. Execution can be developed further. Currently, this stage is skipped, and Stage F is focused more strongly.

### Stage F: main arrivals + real prices

Only after the core behavior is stable under realistic jobs and noisy price phases do we move to real prices. This final stage should behave as fine-tuning rather than first-principles discovery.

## 6. What Would Count as Evidence That the Curriculum Is Working

The curriculum is not justified by intuition alone. It makes clear empirical predictions.

During successful progression through the stages, we expect:

- lower execution during expensive hours,
- higher cheap-hour share of executed core-hours,
- launch-time wait to rise modestly in the logic-price stages,
- pending jobs at episode end to remain controlled,
- overdue jobs at episode end to remain near zero,
- and drop or flush rates to remain near zero.

These are the right diagnostics because they separate intentional deferral from service failure. A policy that merely delays work should show rising overdue backlog or losses. A policy that learned the intended behavior should shift execution into cheap hours while keeping those failure indicators low.

## 7. Conclusion

The curriculum should function because it does not change the target policy across stages. Instead, it changes how easy that policy is to discover. Early stages expose the underlying defer-then-clear structure with high signal and low ambiguity. Later stages reintroduce the realistic sources of complexity that matter for deployment. In this problem, that is preferable to relying on brute-force training time alone, because more steps on the unreduced task mostly provide more opportunities to strengthen locally adequate but globally wrong behaviors.

The central claim is therefore modest but important: the curriculum is not a shortcut around learning the real task. It is a way of making the real task legible enough for the optimizer to find the correct behavioral basin in the first place.
