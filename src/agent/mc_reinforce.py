from monte_carlo import MonteCarloAgent
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

class MCRNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MCRNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.softmax = nn.Softmax(dim=-1)

        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

class MonteCarloAgent_Reinforce(MonteCarloAgent):
    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        num_episodes: int = 5000,
        learning_rate: float = 0.001,
        max_step: int = 500,
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.995 # 不需要使用e-greedy，因为softmax本身具有一定探索性，此处为个人实验留痕
    ):
        super().__init__(env, gamma, epsilon, num_episodes)
        self.state_dim = 1
        self.action_dim = env.num_actions
        self.max_step = max_step
        self.policy_network = MCRNet(self.state_dim, env.num_actions)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.log_probs = []
        self.loss_history = []
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
    def select_action(self, state_index):
        state_tensor = torch.tensor(state_index, dtype=torch.float32).unsqueeze(0)  # [1, ...]
        action_probs = self.policy_network(state_tensor)

        dist = torch.distributions.Categorical(action_probs)

        # if np.random.rand() < self.epsilon:
        #     action = torch.tensor(np.random.randint(self.action_dim))  # 随机动作索引
        # else:
        #     action = torch.argmax(action_probs, dim=-1)  # 最大概率动作索引

        # 根据动作概率分布，生成动作
        action = dist.sample()

        # 获取 log_prob（自动计算 log(prob[action])）
        log_prob = dist.log_prob(action)

        # 存入 list（注意 detach，防止保存整个计算图）
        self.log_probs.append(log_prob)

        return self.env.action_space[action.item()]
    def compute_returns(self, rewards: List[float], gamma: float) -> torch.Tensor:
        """计算每个时间步的折扣回报 G_t"""
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        # 归一化（可选）
        returns = torch.tensor(returns, dtype=torch.float32)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    # def update_policy(self):
    #     """更新策略模型"""
    #     # print(f"log_probs:{self.log_probs}")
    #     log_probs = torch.stack(self.log_probs)
    #     # print(log_probs.shape)
    #     returns = self.return_list
    #     # returns = torch.cat(self.return_list)

    #     # 定义损失函数
    #     loss = -(log_probs * returns).sum()
    #     self.loss_history.append(loss.item())

    #     # 更新策略
        
    #     loss.backward()
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()
    def update_policy(self):
        all_log_probs = torch.stack(self.episode_log_probs)  # shape: [T_total]
        all_returns = torch.tensor(self.episode_returns, dtype=torch.float32)  # shape: [T_total]
        # print(all_log_probs.shape)
        # print(all_returns.shape)
        loss = - (all_log_probs * all_returns).mean()  # 用 mean 替代 sum
        self.loss_history.append(loss.item())

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    
    def train(self):
        """训练模型"""
        # 使用mini-batch
        self.episode_log_probs = []
        self.episode_returns = []
        for episode in range(self.num_episodes):
            state ,_= self.env.reset()
            done = False
            rewards = []
            self.log_probs = []
            # 生成一个 mini batch
            step = 0
            while not done and step < self.max_step:
                state_index = self.env.pos_2_index(state)
                action = self.select_action(state_index)
                next_state, reward, done, _ = self.env.step(action)
                # print(f"State: {state}, Next_state{next_state},Action: {action}, Reward: {reward}, Done: {done}")
                rewards.append(reward)
                state = next_state
                step += 1
                

            returns = self.compute_returns(rewards, self.gamma)
            self.episode_log_probs.extend(self.log_probs)
            self.episode_returns.extend(returns.tolist())   # 注意：如果是 Tensor，也要处理成 Tensor 列表
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            if (episode + 1) % 10 == 0:  # 每 10 个 episode 更新一次
                self.update_policy()
                self.episode_log_probs.clear()
                self.episode_returns.clear()
                total_reward = sum(rewards)
        
                print(f"Episode {episode+1}/{self.num_episodes}, Total Reward: {total_reward} Loss: {self.loss_history[-1]:.4f}, Steps: {step}, epsilon: {self.epsilon:.4f}")
            

         # 输出训练结果
        for state_index in range(self.env.num_states):
            action_probs = self.policy_network(torch.tensor(state_index, dtype=torch.float32).reshape(1, -1))
            self.Q[state_index] = action_probs.detach().numpy()
            index = np.argmax(action_probs.detach().numpy())
            self.policy[state_index] = np.eye(self.env.num_actions)[index]
            print(f"State {state_index}: Action Probabilities: {action_probs.detach().numpy()}")

        print("Policy:")
        print(self.policy)
        print("Q-values:")
        print(self.Q)

if __name__ == "__main__":
    env = GridWorld()
    agent = MonteCarloAgent_Reinforce(env)
    agent.train()




        