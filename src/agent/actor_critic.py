import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("Project root directory:", project_root)
sys.path.append(project_root)
from src.grid_world import GridWorld

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
import matplotlib.pyplot as plt

class ActorNetwork(nn.Module):
    """
    Actor network that outputs action probabilities.
    Uses the same architecture as PPO for consistency.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.network(x)
        return torch.distributions.Categorical(logits=logits)


class CriticNetwork(nn.Module):
    """
    Critic network that estimates state values.
    Uses the same architecture as PPO for consistency.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
class ActorCriticAgent:
    def __init__(self, env: GridWorld, gamma: float = 0.99, learning_rate_actor: float = 0.001, 
                 learning_rate_critic: float = 0.001, max_step: int = 1000):
        
        self.env = env
        self.gamma = gamma
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.state_dim = 1
        self.action_dim = env.num_actions
        self.actor = ActorNetwork(self.state_dim, self.action_dim) # 生成策略
        self.critic = CriticNetwork(self.state_dim) # 计算action值
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic)
        self.log_probs = None
        self.max_step = max_step
        self.actor_loss_episode = []
        self.critic_loss_episode = []
        self.actor_loss = []
        self.critic_loss = []

    def select_action(self, state_index):
        with torch.no_grad():
            state_tensor = torch.tensor(state_index, dtype=torch.float32).unsqueeze(0)
            action_dist = self.actor(state_tensor)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            self.log_probs = log_prob

        return self.env.action_space[action.item()]
    
    def update(self, action_index, state_index, next_state_index, rewards, done):
        # Convert state indices to tensors
        state_tensor = torch.tensor(state_index, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state_index, dtype=torch.float32).unsqueeze(0)

        # Compute value estimates
        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor)

        # Compute returns
        TD_target = rewards + (1 - done) * self.gamma * next_value.detach()
        TD_error = TD_target - value

        # Update critic
        critic_loss = nn.MSELoss()(value, TD_target)
        self.critic_loss_episode.append(critic_loss.item())
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Update actor
        dist = self.actor(state_tensor)
        action_tensor = torch.tensor([action_index], dtype=torch.long)  # 修复点1
        log_prob = dist.log_prob(action_tensor)                         # 修复点2
        actor_loss = -log_prob * TD_error.detach()
        self.actor_loss_episode.append(actor_loss.item())
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
    

    def train(self, num_episodes: int = 1000):
        for episode in range(num_episodes):
            state, info = self.env.reset()
            state_index = self.env.pos_2_index(state)
            done = False
            total_reward = 0
            steps = 0
            while not done and steps < self.max_step:
                action = self.select_action(state_index)
                action_index = self.env.action_space.index(action)
                next_state, reward, done, info = self.env.step(action)
                next_state_index = self.env.pos_2_index(next_state)

                # Update the agent
                self.update(action_index, state_index, next_state_index, reward, done)

                state_index = next_state_index
                total_reward += reward
                steps += 1
            self.actor_loss.append(np.mean(self.actor_loss_episode))
            self.critic_loss.append(np.mean(self.critic_loss_episode))
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Steps: {steps}, Actor Loss: {self.actor_loss[-1]:.4f}, Critic Loss: {self.critic_loss[-1]:.4f}")

        print("Training complete.")

if __name__ == "__main__":
    env = GridWorld()
    agent = ActorCriticAgent(env)
    agent.train(num_episodes=1000)

    # Plot the loss history in two subplots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(agent.actor_loss, label='Actor Loss')
    plt.title('Actor Loss Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(agent.critic_loss, label='Critic Loss', color='orange')
    plt.title('Critic Loss Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    




    

