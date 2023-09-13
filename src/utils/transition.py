from torch import Tensor
from dataclasses import dataclass


@dataclass
class Transition:
    observation: Tensor | None
    action: Tensor | None
    next_observation: Tensor | None
    reward: Tensor | None
    terminated: Tensor | None
    truncated: Tensor | None


@dataclass
class TransitionBatch:
    observations: Tensor | None
    actions: Tensor | None
    next_observations: Tensor | None
    rewards: Tensor | None
    terminateds: Tensor | None
    truncateds: Tensor | None
