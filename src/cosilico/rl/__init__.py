"""RL Training for Cosilico DSL Generation.

This package implements an outer RL loop that learns from trajectories
to improve the system prompt for future runs.
"""

from .trainer import RLTrainer, run_rl_experiment
from .state import LearningState, TrajectoryExample, FailurePattern
from .prompt_evolver import PromptEvolver

__all__ = [
    "RLTrainer",
    "run_rl_experiment",
    "LearningState",
    "TrajectoryExample",
    "FailurePattern",
    "PromptEvolver",
]
