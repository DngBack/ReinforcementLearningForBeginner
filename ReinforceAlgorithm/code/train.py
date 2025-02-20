import gym
from .policy import PolicyNetwork
import torch.optim as optim
from .policy_gradient import train_policy_gradient

env = gym.make("CartPole-v1")
policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)  # type: ignore
optimizer = optim.Adam(policy.parameters(), lr=0.01)

reward_history = train_policy_gradient(env, policy, optimizer, num_episodes=500)
