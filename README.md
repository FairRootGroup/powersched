#### powersched

`python ./checkenv.py` - Check the environment.

`python ./testenv.py` - Short training run.

`python ./train.py` - Infinite training run with tensorboard logs and intermediate models save.

`python ./train_iter.py` - Sequentially launch `./train.py` with different weights.

`./train.py` accepts `--render` argument with "human" or "none" ("none" is the default). "none" trains silently, while "human" runs intentionally slower, adds some debug output and graph output after each episode.

### Curriculum Training

The current training setup is intentionally curriculum-based. The target behavior is not merely "use fewer nodes" or "wait longer"; it is the more specific policy:

- execute little work during expensive hours,
- defer safely while cheap hours are still ahead,
- then clear backlog aggressively during cheap hours,
- while keeping overdue backlog and job loss near zero.

This is now encoded directly in the environment and reward design:

- the agent sees a 24h price forecast window,
- cheap-hour execution is rewarded and expensive-hour execution is penalized,
- cheap hours penalize under-service when backlog exists,
- overdue backlog after the 24h grace period becomes intrinsically bad,
- and end-of-episode pending and overdue metrics make "saving money by not serving work" visible.

The practical reason for using a curriculum instead of only training longer is that the full problem has several easy but wrong local optima:

- serve immediately and ignore price timing,
- trickle a small amount of work continuously,
- or over-defer until backlog becomes unstable.

Those behaviors can produce tolerable short-horizon rewards, so simply running PPO for more steps does not guarantee discovery of the desired defer-then-clear policy. The curriculum reduces variance and improves credit assignment by first teaching the core phase behavior under deterministic logic prices and only then adding load, burstiness, realistic arrivals, price noise, and finally real prices.

Current intended sequence:

1. Stage A: flat arrivals + logic prices.
2. Stage B: high-load flat arrivals + logic prices.
3. Stage C: expensive-half-heavy or bursty arrivals + logic prices.
4. Stage D: main arrivals + logic prices.
5. Stage E: main arrivals + noisy logic prices.
6. Stage F: main arrivals + real prices.

In short: more steps on the full problem mostly improve whatever basin the optimizer already occupies; the curriculum is meant to make the correct basin discoverable first.

For a more formal write-up, see [analysis/curriculum_argument.md](analysis/curriculum_argument.md).
