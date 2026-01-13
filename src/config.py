"""Configuration constants for the PowerSched environment."""

WEEK_HOURS = 168

MAX_NODES = 335  # Maximum number of nodes
MAX_QUEUE_SIZE = 1000  # Maximum number of jobs in the queue
MAX_CHANGE = MAX_NODES
MAX_JOB_DURATION = 170  # maximum job runtime in hours
MAX_JOB_AGE = WEEK_HOURS  # job waits maximum a week
MAX_NEW_JOBS_PER_HOUR = 1500

COST_IDLE = 150  # Watts
COST_USED = 450  # Watts

CORES_PER_NODE = 96
MIN_CORES_PER_JOB = 1
MAX_CORES_PER_JOB = 96
MIN_NODES_PER_JOB = 1
MAX_NODES_PER_JOB = 16

COST_IDLE_MW = COST_IDLE / 1000000  # MW
COST_USED_MW = COST_USED / 1000000  # MW

EPISODE_HOURS = WEEK_HOURS * 2

PENALTY_DROPPED_JOB = -5.0  # explicit penalty for each job dropped due to exceeding MAX_JOB_AGE

# Reward/penalty constants
PENALTY_IDLE_NODE = -0.1  # Penalty for idling nodes
PENALTY_WAITING_JOB = -0.1  # Penalty for each hour a job is delayed
