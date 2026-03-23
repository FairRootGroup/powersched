"""Configuration constants for the PowerSched environment."""

WEEK_HOURS: int = 168

MAX_NODES: int = 335  # Maximum number of nodes
MAX_QUEUE_SIZE: int = 2500  # Maximum number of jobs in the queue
MAX_BACKLOG_SIZE: int = 50000  # Maximum number of jobs in the backlog (overflow) queue
MAX_CHANGE: int = MAX_NODES
MAX_JOB_DURATION: int = 170  # maximum job runtime in hours
# Drop jobs that have waited longer than this in queue/backlog.
MAX_JOB_AGE = WEEK_HOURS * 2
MAX_NEW_JOBS_PER_HOUR: int = 1500

COST_IDLE: int = 150  # Watts
COST_USED: int = 450  # Watts

CORES_PER_NODE: int = 96
MIN_CORES_PER_JOB: int = 1
MAX_CORES_PER_JOB: int = 96
MIN_NODES_PER_JOB: int = 1
MAX_NODES_PER_JOB: int = 16

COST_IDLE_MW: float = COST_IDLE / 1000000  # MW
COST_USED_MW: float = COST_USED / 1000000  # MW

EPISODE_HOURS: int = WEEK_HOURS * 2
MAX_JOB_AGE_OBS: int = EPISODE_HOURS * 13  # maximum job age observable in the state, here set to ~6 Months

PENALTY_DROPPED_JOB: float = -5.0  # legacy per-job lost-job penalty constant

# Reward/penalty constants
PENALTY_IDLE_NODE: float = -0.1  # Penalty for idling nodes
PENALTY_WAITING_JOB: float = -0.1  # Penalty for each hour a job is delayed
