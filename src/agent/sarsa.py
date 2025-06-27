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



class SarsaAgent(MonteCarloAgent):
    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        num_episodes: int = 1000,
    ):
        super().__init__(env, gamma, epsilon, num_episodes)
        self.Q = np.zeros((env.num_states, env.num_actions))  # 初始化Q值
        self.policy = self.generate_random_policy()  # 初始化策略
        self.state_action_counts = np.zeros((env.num_states, env.num_actions))
        self.returns = np.zeros((env.num_states, env.num_actions))
        self.alpha = 0.1  # 学习率


    def update_policy(self, epsilon: float = 0.1):
        """
        Update the policy based on the current Q values.
        """
        self.epsilon = epsilon
        if self.epsilon == 0:
            self.policy = np.argmax(self.Q, axis=1) # 贪婪策略
            self.policy = np.eye(self.env.num_actions)[self.policy]
        else:
            self.epsilon = max(0.01, self.epsilon)  # 确保epsilon不为0
            self.policy = np.zeros((self.env.num_states, self.env.num_actions))
            for state in range(self.env.num_states):
                action = np.argmax(self.Q[state])
                self.policy[state, action] = 1 - self.epsilon + (self.epsilon / self.env.num_actions)
                for a in range(self.env.num_actions):
                    if a != action:
                        self.policy[state, a] = self.epsilon / self.env.num_actions
        return self.policy
    
    
    def policy_evaluation(self, steps=None):
        """
        Evaluate the current policy by calculating the Q values.
        """
        if steps is None: # 如果没有指定步数,表示使用蒙特卡洛方法
            for episode in tqdm(range(self.num_episodes), desc="Policy Evaluation"):
                episode_data = self.generate_random_episode(self.policy)
                G = 0
                # visited = set()
                for t in reversed(range(len(episode_data))):
                    state, action, reward= episode_data[t]
                    G = reward + self.gamma * G
                    # if (state, action) not in visited:
                    #     visited.add((state, action))
                    self.returns[state, action] += G
                    self.state_action_counts[state, action] += 1
                    self.Q[state, action] = self.returns[state, action] / self.state_action_counts[state, action]
                    # 更新策略
                    self.update_policy(self.epsilon)
                    # print(f"Episode {episode + 1}: state={state}, action={action}, G={G}, Q={self.Q[state]}")
        else:  # 如果指定了步数,表示使用Sarsa方法
            self.n_step = steps  # 设置 n-step 的步数
            for episode in tqdm(range(self.num_episodes), desc="Policy Evaluation"):
                state, _ = self.env.reset()
                state_index = self.env.pos_2_index(state)
                action = self.env.sample_action(self.policy[state_index])
                # print(action)

                # states = [state]
                # actions = [action]
                # rewards = [0]  # dummy 第一个 reward 是 0，方便索引对齐

                done = state == self.env.target_state

                while not done:
                    #此处如果路径不更新，会导致路径非常长，影响最终的收敛
                    states = [state]
                    actions = [action]
                    rewards = [0]  # dummy 第一个 reward 是 0，方便索引对齐
                    # Step 1: 生成固定步数的轨迹
                    for step in range(steps):
                        next_state, reward, done, _ = self.env.only_step(action)
                        next_action = self.env.sample_action(self.policy[next_state])

                        # 添加到轨迹
                        states.append(next_state)
                        actions.append(next_action)
                        rewards.append(reward)

                        state = next_state
                        action = next_action

                        if done:
                            break

                    # Step 2: 倒序更新 Q 值（兼容任意 n_step） 只用step步长的路径更新state
                
                    for t in reversed(range(len(states) - 1)):  # t 从最后一步往前推
                        # 设置最大回溯步数
                        max_t_plus_n = min(t + self.n_step, len(states) - 1)
                        G = 0.0

                        # 累加中间奖励（t 到 max_t_plus_n - 1）
                        for k in range(t, max_t_plus_n):
                            G += (self.gamma ** (k - t)) * rewards[k + 1]

                        # 加上最终状态的 Q 值（如果还没终止）
                        if max_t_plus_n < len(states):
                            actual_steps = max_t_plus_n - t
                            next_q = self.Q[self.env.pos_2_index(states[max_t_plus_n]),
                                            self.env.action_space.index(actions[max_t_plus_n])]
                            G += (self.gamma ** actual_steps) * next_q

                        # 更新 Q 值
                        s_idx = self.env.pos_2_index(states[t])
                        a_idx = self.env.action_space.index(actions[t])
                        current_q = self.Q[s_idx, a_idx]
                        self.Q[s_idx, a_idx] += self.alpha * (G - current_q)
                        

                    
                    

                    # Step 3: 重置起点为最后的状态/动作
                    state = states[-1]
                    action = actions[-1]

                    # 如果已经完成，跳出循环
                    if done:
                        break

                    # 更新策略
                    self.epslion = max(0.01, episode / self.num_episodes * self.epsilon)
                    self.update_policy(self.epslion)  # 更新策略
                    # env.add_policy(self.policy)  # 更新策略到环境中
                    # env.render()  # 渲染环境

        self.V = np.zeros(self.env.num_states)
        for state in range(self.env.num_states):
            self.V[state] = np.max(self.Q[state])


        return self.V, self.policy
if __name__ == "__main__":
    
    # Example usage
    env = GridWorld()
    agent = SarsaAgent(env, gamma=0.9, epsilon=0.1, num_episodes=100)

    state = env.reset()
    env.render(animation_interval=2)
    policy = agent.generate_random_policy()
    state_values, optimal_policy= agent.policy_evaluation(steps=10)  # 使用5步Sarsa方法进行策略评估
    #将最大值作为最优策略，即最大的action的p为1，其余为0
    optimal_policy = np.argmax(optimal_policy, axis=1)  # Convert policy to action indices
    optimal_policy = np.eye(env.num_actions)[optimal_policy]  # Convert to one-hot encoding
    print("State Value Function:", state_values)
    print("Final Policy:", agent.policy)
    env.add_state_values(state_values)
    env.add_policy(optimal_policy)
    env.render(animation_interval=100)


    #统计不同step数的收敛速度
    steps = [1, 2, 3, 5, 10, 20]
    time_results = []
    for step in steps:
        agent = SarsaAgent(env, gamma=0.9, epsilon=1, num_episodes=100)
        state_values, optimal_policy = agent.policy_evaluation(steps=step)
        start_time = time.time()
        agent.policy_evaluation(steps=step)
        end_time = time.time()
        time_results.append(end_time - start_time)
    plt.plot(steps, time_results, marker='o')
    plt.xlabel('Steps')
    plt.ylabel('Time (seconds)')
    plt.title('Convergence Speed vs Steps')
    plt.grid()
    plt.savefig('sarsa_convergence_speed.png')
    plt.show()
