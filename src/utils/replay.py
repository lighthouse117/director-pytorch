import torch
import numpy as np

from omegaconf import DictConfig
from utils.transition import Transition, TransitionBatch


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_size: int,
        device: str,
        config: DictConfig,
    ):
        self.device = device
        self.config = config

        # Length of the buffer
        self.capacity: int = config.capacity

        # Batch size for sampling
        self.batch_size: int = config.batch_size

        # Elements in the buffer
        self.observations = np.empty((self.capacity, *observation_shape))
        self.actions = np.empty((self.capacity, action_size))  # One-hot encoded
        self.next_observations = np.empty((self.capacity, *observation_shape))
        self.rewards = np.empty((self.capacity, 1))
        self.terminateds = np.empty((self.capacity, 1))
        self.truncateds = np.empty((self.capacity, 1))

        # Current position in the buffer
        self.current_index = 0

        self.is_full = False

    def __len__(self):
        return self.capacity if self.is_full else self.current_index

    def add(self, transition: Transition):
        """Add a transition to the buffer."""
        self.observations[self.current_index] = transition.observation
        self.actions[self.current_index] = transition.action
        self.next_observations[self.current_index] = transition.next_observation
        self.rewards[self.current_index] = transition.reward
        self.terminateds[self.current_index] = transition.terminated
        self.truncateds[self.current_index] = transition.truncated

        self.current_index = (self.current_index + 1) % self.capacity

        if self.current_index == 0:
            self.is_full = True

    def sample(self) -> TransitionBatch:
        """Sample a batch of transitions from the buffer."""
        indices = np.random.randint(
            low=0,
            high=self.capacity if self.is_full else self.current_index,
            size=self.batch_size,
        )

        # Convert to tensors and return the batch
        return TransitionBatch(
            observations=torch.as_tensor(
                self.observations[indices],
                dtype=torch.float,
                device=self.device,
            ),
            actions=torch.as_tensor(
                self.actions[indices],
                dtype=torch.float,
                device=self.device,
            ),
            next_observations=torch.as_tensor(
                self.next_observations[indices],
                dtype=torch.float,
                device=self.device,
            ),
            rewards=torch.as_tensor(
                self.rewards[indices],
                dtype=torch.float,
                device=self.device,
            ),
            terminateds=torch.as_tensor(
                self.terminateds[indices],
                dtype=torch.float,
                device=self.device,
            ),
            truncateds=torch.as_tensor(
                self.truncateds[indices],
                dtype=torch.float,
                device=self.device,
            ),
        )
