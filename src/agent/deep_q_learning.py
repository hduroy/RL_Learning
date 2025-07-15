import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt


# 获取当前文件所在目录的上三层目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("Project root directory:", project_root)
sys.path.append(project_root)

from src.grid_world import GridWorld
from examples.arguments import args  # 再上一层目录
from q_learning import QLearningAgent as QL

# 增加迭代进度条
from tqdm import tqdm

# 导入深度学习库
import torch
import torch.nn as nn


class DQNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        """
        A simple replay buffer to store experiences for DQN training.
        Args:
            capacity (int): Maximum number of experiences to store.
            obs_shape (tuple): Shape of the observation space.
            action_dim (int): Dimension of the action space.
        """
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        # Pre-allocate memory
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.states[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size


class DeepQLearningAgent(QL):
    """
    Deep Q-Learning Agent using a neural network for function approximation.
    """

    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        num_episodes: int = 1000,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        replay_buffer_size: int = 10000,
        iteration: int = 100,
    ):
        super().__init__(env, gamma, epsilon, epsilon_decay, min_epsilon, num_episodes, learning_rate)
        
        self.state_dim = 1  # 暂时只有索引，没有其他特征
        self.action_index_dim = 1
        self.action_dim = env.num_actions

        # Initialize the neural network
        self.q_network = DQNet(self.state_dim, self.action_dim)
        self.target_network = DQNet(self.state_dim, self.action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            replay_buffer_size, self.state_dim, self.action_index_dim
        )

        # Hyperparameters
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.loss_fn = nn.MSELoss()  # Loss function for Q-learning
        self.iteration = iteration

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space[np.random.randint(self.action_dim)]
        else:
            with torch.no_grad():
                batch_state =  torch.tensor(state, dtype=torch.float32).reshape(1, -1)
                q_values = self.q_network(batch_state)
                return self.env.action_space[torch.argmax(q_values).item()]

    def generate_experience(self,  max_step, start_state=None):
        """
        Generate an experience tuple (state, action, reward, next_state, done) using the policy.
        Args:
            policy (np.ndarray): The policy to follow during the episode.
            max_step (int): Maximum number of steps in the episode.
            start_state (int, optional): The initial state to start the episode. Defaults to None.

        Returns:
            tuple: A tuple containing (state, action, reward, next_state, done).
        """
        if start_state is None:
            state, info = self.env.reset()
        else:
            state, info = self.env.set_state(start_state)
            
        done = False
        steps = 0

        while not done or steps < max_step:
            state_index = self.env.pos_2_index(state)
            # 根据self.policy实现均匀采样
            action = self.select_action(state_index)
            next_state, reward, done, info = self.env.only_step(action)
            next_state_index = self.env.pos_2_index(next_state)
            action_index = self.env.action_space.index(action)
            self.replay_buffer.add(state_index, action_index, reward, next_state_index, done)
            state = next_state
            steps += 1
        return self.replay_buffer

    def train_step(self, loss_func, iteration):
        """
        Perform a single training step.
        Args:
            loss (torch.Tensor): The loss value to backpropagate.
        """
        if self.replay_buffer.size < self.batch_size:
            print("buffer中样本不足")
        total_loss = 0
        for i in range(iteration):
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = batch
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            q_values = self.q_network(states).gather(dim=1, index=actions)
        

            # compute target Q values
            with torch.no_grad():
                target_q_values = self.target_network(next_states)
                target_q_values = (
                    rewards + (1 - dones) * self.gamma * target_q_values.max(dim=1)[0]
                )  # 完成态没有下一个状态的Q值
                
        
    
            loss = loss_func(q_values.squeeze(), target_q_values)
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return total_loss / iteration
    
    def get_optimal_policy(self):
        
        with torch.no_grad():
            for state_index in range(self.env.num_states):
               state = self.env.index_2_pos(state_index)
               if state not in self.env.forbidden_states or state in self.env.target_state:
                   q_values = self.q_network(torch.tensor( state_index, dtype=torch.float32).reshape(1, -1))
                   self.Q[state_index] = q_values
            self.policy = np.argmax(self.Q, axis=1)  # 贪婪策略
            self.policy = np.eye(self.env.num_actions)[self.policy]
        return self.policy, self.Q
                

                   
                   


    def train(self):
        # state, _ = self.env.reset()
        # policy = self.generate_random_policy()
        # self.generate_experience(1000, start_state=state)
        state, _ = self.env.reset()
        self.generate_experience(1000, start_state=state)
        loss_history = []
        for episode in range(self.num_episodes):
            
            loss = self.train_step(self.loss_fn, self.iteration)
            loss_history.append(loss)
            # update the target network
            self.target_network.load_state_dict(self.q_network.state_dict()) 

            # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if episode % 100 == 0:
                print(f"Episode {episode }: Loss = {loss:.4f}: Epsilon = {self.epsilon:.4f}")
        return loss_history
   
if __name__ == "__main__":
    env = GridWorld()
    state = env.reset()
    env.render(animation_interval=2)
    # Create the Q-learning agent
    agent = DeepQLearningAgent(env, gamma=0.9, epsilon=1.0, num_episodes=1000, learning_rate=0.001, batch_size= 128, iteration= 100)
    
    # Train the agent
    history = agent.train()

    print("Training completed.")
    policy, Q = agent.get_optimal_policy()
    env.add_policy(policy)
    env.add_state_values(agent.Q.max(axis=1))  # Add state values for visualization
    env.render(animation_interval=100)  # Render the environment with the learned policy
    