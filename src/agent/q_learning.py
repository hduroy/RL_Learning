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
from monte_carlo import MonteCarloAgent

#增加迭代进度条
from tqdm import tqdm


class QLearningAgent(MonteCarloAgent):
    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        num_episodes: int = 1000,
        learning_rate: float = 0.01,
        max_steps: int = 1000
    ):
        super().__init__(env, gamma, epsilon, num_episodes)
        self.Q = np.zeros((env.num_states, env.num_actions))  # 初始化Q值
        self.policy = self.generate_random_policy()  # 初始化策略
        self.state_action_counts = np.zeros((env.num_states, env.num_actions))
        self.returns = np.zeros((env.num_states, env.num_actions))
        self.alpha = learning_rate  # 学习率
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.max_steps = max_steps  # 最大步数限制
        # self.policy2 = np.zeros((env.num_states, env.num_actions))  # 初始化第二个策略 体现off-policy特性

    def update_policy(self, epsilon: float = 0.1):
        """
        Update the policy based on the current Q values.
        """
        self.epsilon = epsilon
        if self.epsilon == 0:
            self.policy = np.argmax(self.Q, axis=1)  # 贪婪策略
            self.policy = np.eye(self.env.num_actions)[self.policy]
        else:
            self.epsilon = max(self.min_epsilon, self.epsilon)  # 确保epsilon不为0
            self.policy = np.zeros((self.env.num_states, self.env.num_actions))
            for state in range(self.env.num_states):
                action = np.argmax(self.Q[state])
                self.policy[state, action] = 1 - self.epsilon + (self.epsilon / self.env.num_actions)
                for a in range(self.env.num_actions):
                    if a != action:
                        self.policy[state, a] = self.epsilon / self.env.num_actions
        return self.policy
    
    def update(self, state, action, reward, next_state):
        """
        Update the Q-value based on the action taken and the reward received.
        """
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action] 
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
    
    def select_action(self, state):
        """
        Select an action based on the epsilon-greedy policy with state_index
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space[np.random.randint(self.env.num_actions)]
        else:
            max_action = np.argmax(self.Q[state])
            return self.env.action_space[max_action]

    def train(self):
        """
        Train the agent using Q-learning.
        """
        
        reward_history = 0
        episode_lengths = 0
        for episode in tqdm(range(self.num_episodes), desc="Training"):
            state, info = self.env.reset()
            done = False
            step = 0
            
            while not done:
    
                state_index = self.env.pos_2_index(state)  #这个模型index和pos之间的转化关系总是需要注意的
                action = self.select_action(state_index)
                
                next_state, reward, done, info = self.env.step(action)
                
                
                next_state_index = self.env.pos_2_index(next_state)
                state_index = self.env.pos_2_index(state)
                action_index = self.env.action_space.index(action)  # 获取动作的索引
                # update Q-value
                self.update(state_index, action_index, reward, next_state_index)

                # Update the state
                state = next_state
                step += 1
                reward_history += reward
                if done or step >= self.max_steps:
                    break

            # Decay epsilon after each episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            episode_lengths += step
            # print the process
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{self.num_episodes}, batch mean reward: {reward_history / 100:.2f}, batch mean length: {episode_lengths / 100:.2f} epsilon: {self.epsilon:.4f}")
                reward_history = 0
                episode_lengths = 0
                current_policy = np.argmax(self.Q, axis=1)
                current_policy = np.eye(self.env.num_actions)[current_policy]  # Convert to one
                self.env.add_policy(current_policy)
                self.env.add_state_values(self.Q.max(axis=1))  # Add state values for visualization
                self.env.render(animation_interval=2)  # Render the environment with the learned policy
        print("Training completed.")


if __name__ == "__main__":
    # Create the environment
    env = GridWorld()
    state = env.reset()
    env.render(animation_interval=2)
    
    # Create the Q-learning agent
    agent = QLearningAgent(env, gamma=0.9, epsilon=1.0, num_episodes=1000, learning_rate=0.1)
    
    # Train the agent
    agent.train()
    
    # Save the trained Q-values
    np.save("q_values.npy", agent.Q)
    
    # Optionally, you can visualize the learned policy or Q-values here
    print("Q-values:", agent.Q)
    policy = np.argmax(agent.Q, axis=1)
    policy = np.eye(env.num_actions)[policy]  # Convert to one-hot encoding
    env.add_policy(policy)
    env.add_state_values(agent.Q.max(axis=1))  # Add state values for visualization
    env.render(animation_interval=100)  # Render the environment with the learned policy