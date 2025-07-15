import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print("Project root directory:", project_root)
sys.path.append(project_root)

from src.grid_world import GridWorld



import numpy as np
from itertools import product



class TDLinear:
    def __init__(self, env: GridWorld, policy, alpha=0.001, gamma=0.9, epsilon=1, num_episodes=2000, q=3):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.q = q  # 傅里叶阶数

        # 状态和动作数量
        self.num_states = env.num_states
        self.num_actions = env.num_actions

        # 构建傅里叶基的所有组合 (c1, c2)
        self.coefficients = list(product(range(q+1), repeat=2))  # 所有可能的 (c1, c2)
        self.feature_dim = len(self.coefficients)  # 特征维度 = (q+1)^2

        # 每个动作对应一个权重向量
        self.weights = np.zeros((self.num_actions, self.feature_dim))

        # 初始化策略
        self.policy = policy  # policy(s) → 返回动作分布或ε-greedy选择的动作

    def _normalize_state(self, s):
        """
        将状态编号转换为归一化的 (x, y) 坐标，范围 [0, 1]
        """
        x, y = s
        return x/4, y/4
    def _fourier_basis(self, s):
        """
        构造傅里叶特征向量 φ(s)
        """
        
    
        x, y = self._normalize_state(s)
        features = []
        for c1, c2 in self.coefficients:
            val = np.cos(np.pi * (c1 * x + c2 * y))
            features.append(val)
        return np.array(features)

    def _q_value(self, s, a):
        """
        计算 q(s, a) = φ(s)^T · w[a]
        """
        phi = self._fourier_basis(s)
        return np.dot(phi, self.weights[a])

    def _update_weights(self, s, a, r, s_next):
        """
        使用 TD(0) 更新权重：
        w ← w + α [r + γ q(s', a') - q(s, a)] ∇w q(s, a)
        """
        phi = self._fourier_basis(s)
        a_next = self.select_action(s_next)
        a_next_index = self.env.action_space.index(a_next)
        target_q = self._q_value(s_next, a_next_index)  # 下一步使用当前策略选择动作
        td_error = r + self.gamma * target_q - self._q_value(s, a)
        self.weights[a] += self.alpha * td_error * phi

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
        训练过程：多轮 episodes 进行 TD 学习
        """
        for episode in range(self.num_episodes):
            print(f"Episode {episode}")
            state, _ = self.env.reset()
            state_index = self.env.pos_2_index(state)
            done = False

            while not done:
                # 根据当前策略选择动作（如 ε-greedy）
                action = self.select_action(state_index)
                action_index = self.env.action_space.index(action)
                # 环境交互
                next_state, reward, done, _ = self.env.step(action)
                # 更新权重
                self._update_weights(state, action_index, reward, next_state)
                # 状态转移
                state = next_state

    def get_q_values(self):
        """
        返回所有状态-动作对的 Q 值
        """
        Q = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                state = self.env.index_2_pos(s)
                Q[s, a] = self._q_value(state, a)
        return Q
    def get_state_values(self):
        """
        返回所有状态的 V 值
        """
        V = np.zeros(self.num_states)
        Q = self.get_q_values()
        for s in range(self.num_states):
            V[s] = np.sum(self.policy[s, :] * Q[s, :])
        return V
    

if __name__ == "__main__":
    env = GridWorld()

    policy_matrix = np.random.rand(env.num_states, len(env.action_space))
    policy = policy_matrix / np.sum(policy_matrix, axis=1, keepdims=True)

    td = TDLinear(env, policy, q=3)
    td.train()
    Q = td.get_q_values()
    print(Q)
    # 通过 Q 计算state value


    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def plot_V_surface(td : TDLinear):
        rows, cols = 5, 5
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        Z = np.zeros_like(X)
        V = td.get_state_values()
        for i in range(rows):
            for j in range(cols):
                s = i * cols + j
                Z[i, j] = V[s]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        plt.title(f"State-value Surface ")
        plt.show()

    plot_V_surface(td)
