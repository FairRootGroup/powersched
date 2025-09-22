The reward function consists of four weighted components that are normalized and combined:

Each component is normalized to [0,1] range (penalties to [-1,0])
Components are weighted using configurable coefficients
Final reward = efficiency_reward + price_reward + idle_penalty + job_age_penalty

### 1. Efficiency Reward

Purpose: Maximizes computational work relative to energy cost

Formula: num_used_nodes / (power_cost + ε) when nodes are active
Special cases:
- Returns 1.0 if no nodes used and no pending jobs
- Returns 1/log(pending_jobs) if no nodes used but jobs waiting

### 2. Price Reward

Purpose: Encourages job processing during low electricity price periods

Formula: (average_price - current_price) × num_jobs_processed
Uses historical and future price context to determine if current pricing is favorable

### 3. Idle Penalty

Purpose: Discourages keeping nodes online but unused, promoting energy efficiency

Formula: -0.1 × num_idle_nodes

### 4. Job Age Penalty

Purpose: Penalizes delaying jobs when compute capacity is available

Formula: -0.1 × sum(-0.1 * job_age) for all queued jobs, but only when nodes are turned off

#### Normalization Details

Each component is normalized using (current - min) / (max - min) with bounds calculated as:

### Efficiency Reward Bounds

Min: 0 / (cost_at_max_price + ε) - No work done at highest electricity cost
Max: max(1.0, MAX_NODES / cost_at_min_price) - All nodes working at lowest electricity cost

### Price Reward Bounds

Min: 0 (no reward for unfavorable pricing)
Max: (MAX_PRICE - MIN_PRICE) × MAX_NEW_JOBS_PER_HOUR - Best price differential with maximum job throughput

### Idle Penalty Bounds

Min: 0 (no idle nodes)
Max: -0.1 × MAX_NODES (all nodes idle)

### Job Age Penalty Bounds

Min: 0 (no delayed jobs)
Max: -0.1 × MAX_JOB_AGE × MAX_QUEUE_SIZE - Maximum queue with oldest jobs
