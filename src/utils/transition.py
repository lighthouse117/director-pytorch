from torch import Tensor
from dataclasses import dataclass


@dataclass
class Transition:
    """
    A transition data at a single timestep.
    - observation: (*observation_shape)
    - action: (action_size,)
    - next_observation: (*observation_shape)
    - reward: (1,)
    - terminated: (1,)
    - truncated: (1,)
    """

    observation: Tensor | None
    action: Tensor | None
    next_observation: Tensor | None
    reward: Tensor | None
    terminated: Tensor | None
    truncated: Tensor | None


@dataclass
class TransitionBatch:
    """
    A batch of transitions.
    - observations: (batch_size, *observation_shape)
    - actions: (batch_size, action_size)
    - next_observations: (batch_size, *observation_shape)
    - rewards: (batch_size, 1)
    - terminateds: (batch_size, 1)
    - truncateds: (batch_size, 1)
    """

    observations: Tensor | None
    actions: Tensor | None
    next_observations: Tensor | None
    rewards: Tensor | None
    terminateds: Tensor | None
    truncateds: Tensor | None
