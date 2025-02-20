import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple
from typing import List
import torch


def select_action(policy: nn.Module, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
    """
    Selects an action from the policy network based on the current state using stochastic sampling.

    Instead of always choosing the action with the highest probability, this function samples
    an action from the probability distribution output by the policy network. This introduces
    randomness and encourages exploration in Reinforcement Learning.

    Args:
        policy (nn.Module): The policy network, which takes the state as input and returns
                            a probability distribution over possible actions.
        state (torch.Tensor): The current state of the environment, represented as a tensor.

    Returns:
        Tuple[int, torch.Tensor]: A tuple containing:
            - The selected action (int).
            - The log-probability of the selected action (torch.Tensor),
              which is used to update the policy in the REINFORCE algorithm.
    """
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(
        0
    )  # Ensure batch size = 1
    probs = policy(state)  # Get probability distribution from the policy network
    action_dist = Categorical(probs)  # Create a categorical distribution
    action = action_dist.sample()  # Sample an action
    return action.item(), action_dist.log_prob(  # type: ignore
        action
    )  # Return the action and its log-probability


def compute_discounted_rewards(
    rewards: List[float], gamma: float = 0.99
) -> torch.Tensor:
    """
    Computes the discounted rewards for a sequence of rewards in a reinforcement learning episode.

    The discounted reward at each time step `t` is calculated using the formula:

        G_t = R_t + γ * G_{t+1}

    where:
        - `G_t` is the discounted return at time `t`.
        - `R_t` is the immediate reward at time `t`.
        - `γ` (gamma) is the discount factor, controlling the importance of future rewards.

    This function iterates over the rewards in reverse order to efficiently compute the discounted
    returns for an entire episode.

    Args:
        rewards (List[float]): A list of rewards obtained at each time step in an episode.
        gamma (float, optional): The discount factor (default is 0.99).

    Returns:
        torch.Tensor: A tensor of discounted rewards, with the same length as `rewards`.

    Example:
        >>> rewards = [1, 0, -1, 2]
        >>> compute_discounted_rewards(rewards, gamma=0.9)
        tensor([ 0.4390, -0.49,  1.1,  2. ])
    """
    discounted_rewards = []
    total_reward = 0
    for r in reversed(rewards):  # Iterate from the last reward to the first
        total_reward = r + gamma * total_reward  # Compute discounted return
        discounted_rewards.insert(0, total_reward)  # Insert at the beginning
    return torch.tensor(discounted_rewards, dtype=torch.float32)  # Convert to tensor
