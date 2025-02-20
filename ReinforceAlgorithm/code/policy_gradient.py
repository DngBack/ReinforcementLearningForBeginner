import torch
import torch.nn as nn
import torch.optim as optim
import gym
from typing import List
from .utils import select_action
from .utils import compute_discounted_rewards


def train_policy_gradient(
    env: gym.Env,
    policy: nn.Module,
    optimizer: optim.Optimizer,
    num_episodes: int = 500,
    gamma: float = 0.99,
) -> List[float]:
    """
    Trains a policy network using the REINFORCE algorithm with Monte Carlo policy updates.

    Args:
        env (gym.Env): The reinforcement learning environment.
        policy (nn.Module): The policy network that outputs action probabilities.
        optimizer (optim.Optimizer): The optimizer for updating the policy network.
        num_episodes (int, optional): The number of episodes to train. Defaults to 500.
        gamma (float, optional): The discount factor for future rewards. Defaults to 0.99.

    Returns:
        List[float]: A list containing the total rewards for each episode.

    Workflow:
        1. Reset the environment at the start of each episode.
        2. Collect log probabilities and rewards for each step until the episode ends.
        3. Compute discounted rewards to properly weigh rewards over time.
        4. Normalize rewards to stabilize training.
        5. Compute the policy loss using the log probabilities and discounted rewards.
        6. Perform a gradient update to improve the policy.
        7. Store and return the total rewards per episode.

    Notes:
        - The function handles different versions of OpenAI Gym's `reset()` and `step()` API.
        - Rewards are normalized to reduce variance and improve stability.
    """
    reward_history = []

    # loop through episodes to train
    for episode in range(num_episodes):
        # reset the environment when the episode is completed
        state = env.reset()
        if isinstance(state, tuple):  # Handle different Gym versions
            state = state[0]

        log_probs = []
        rewards = []

        done = False
        while not done:
            action, log_prob = select_action(policy, state)
            step_result = env.step(action)

            # Handle Gym's different step API versions
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = (
                    terminated or truncated
                )  # Handle different termination conditions
            else:
                next_state, reward, done, _ = (
                    step_result  # Older versions # type: ignore
                )

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

        # Compute discounted rewards
        discounted_rewards = compute_discounted_rewards(rewards, gamma)

        # Normalize rewards to reduce variance
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-9
        )

        # Compute policy loss
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.stack(policy_loss).sum()

        # Update policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        total_episode_reward = sum(rewards)
        reward_history.append(total_episode_reward)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_episode_reward}")

    return reward_history
