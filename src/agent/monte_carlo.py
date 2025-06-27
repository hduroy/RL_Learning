import numpy as np
import sys
import os

# 获取当前文件所在目录的上三层目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("Project root directory:", project_root)
sys.path.append(project_root)

from src.grid_world import GridWorld
from examples.arguments import args  # 再上一层目录

#增加迭代进度条
from tqdm import tqdm


class MonteCarloAgent:
    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        num_episodes: int = 1000,
    ):
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.policy = self.generate_random_policy()
        self.returns = np.zeros(
            (env.num_states, env.num_actions)
        )  # 用于存储每个状态-动作对的总回报
        self.state_action_counts = np.zeros(
            (env.num_states, env.num_actions)
        )  # 用于存储每个状态-动作对的访问次数 用于计算平均回报
        # 初始化Q值
        self.Q = np.zeros((env.num_states, env.num_actions))

    def generate_random_episode(self, policy, start_state=None, length=500):
        """
        Generate a random episode from the environment.
        Args:
            start_state (int, optional): The initial state to start the episode. Defaults to None.
            length (int, optional): The maximum length of the episode. Defaults to None.
            policy (np.ndarray): The policy to follow during the episode. If None, a random policy is used.

        Returns:
            episode (list): A list of tuples (state, action, reward) representing the episode.
        """
        episode = []

        if start_state is None:
            state, info = self.env.reset()
            #随机选择一个状态作为初始状态
            # start_state = np.random.choice(self.env.num_states)
            # state = self.env.index_2_pos(start_state)  # Convert index to position
            # while state in self.env.forbidden_states or state == self.env.target_state:
            #     start_state = np.random.choice(self.env.num_states)
            #     state = self.env.index_2_pos(start_state)
        else:
            state, info = self.env.set_state(start_state)
        
        done = False
        steps = 0

        while not done:
            state = self.env.pos_2_index(state)  # Convert state to index
            action_prob = self.policy[state] if policy is not None else None
            if action_prob is not None: # If a policy is provided, sample an action based on the policy
                # print(f"Action probabilities for state {state}: {action_prob}")
                action_list = np.arange(self.env.num_actions)
                # print(f"Action list: {action_list}")
                action_index = np.random.choice(
                    action_list, p=action_prob
                )
            else:
                # If no policy is provided, select a random action
                action_index = np.random.choice(
                    self.env.num_actions
                )  # Select a random action index
            action = self.env.action_space[action_index]  # Map index to actual action
            # print(f"Selected action: {action} for state {state}")
            next_state, reward, done, info = self.env.step(action)
            episode.append((state, action_index, reward)) # 这里将action_index作为动作的索引,检查了半年
            state = next_state
            # print(f"Step {steps}: next State {state}, Action {action}, Reward {reward}")
            steps += 1
            # env.render()  # Render the environment
            if length is not None and steps >= length:
                break
            
        return episode

    def generate_random_policy(self):
        """
        Generate a random policy.
        Returns:
            policy (np.ndarray): A 2D array representing the policy.
        """
        num_actions = len(self.env.action_space)
        policy = (
            np.ones((self.env.num_states, num_actions)) / num_actions
        )  # Uniform random policy
        return policy
    
    def evaluate_policy(self, policy, greedy=False):
        """
        Evaluate the current policy using Monte Carlo method.
        几个易错点：
        state and action should be indices, not positions,这两种模式的转变需要注意，建议使用_idex和_pos来区分
        
        """
        self.policy = policy  # Update the agent's policy
        for episode in tqdm(range(self.num_episodes), desc="Evaluating policy"):
            
            # Generate a random episode
            states, actions, rewards = zip(*self.generate_random_episode(self.policy))
            # 暂停等待输入
            # input("Press Enter to continue...")
            G = 0
            for t in reversed(range(len(rewards))):
                G = rewards[t] + self.gamma * G
                state = states[t]
                action = actions[t]
                self.returns[state, action] += G
                self.state_action_counts[state, action] += 1
                # Update Q value as the average of returns
                self.Q[state, action] = (
                    self.returns[state, action] / self.state_action_counts[state, action]
                )
                
                if greedy is False:
                    # Update policy to be greedy with respect to Q values
                    best_action = np.argmax(self.Q[state])
                    self.policy[state] = np.zeros(self.env.num_actions)
                    self.policy[state][best_action] = 1.0
                else:
                    # Epsilon-greedy policy update
                    epsilon = self.epsilon * (1 - episode / self.num_episodes)
                    epsilon = max(epsilon, 0.1)  # Ensure epsilon does not go below 0.1
                    # Update policy to be epsilon-greedy
                    # print(f"Episode {episode + 1}: state={state}, action={action}, G={G}, Q={self.Q[state]}")
                    action_values = self.Q[state]
                    best_action = np.argmax(action_values)
                    self.policy[state] = np.full(self.env.num_actions, epsilon / self.env.num_actions)
                    self.policy[state][best_action] = 1 - epsilon + (epsilon / self.env.num_actions)

            
                # # Update policy based on Q values  MC Exploration Starts method
                # if greedy is False:
                #     for state in range(self.env.num_states):
                #         best_action = np.argmax(self.Q[state])
                #         self.policy[state] = np.zeros(self.env.num_actions)
                #         self.policy[state][best_action] = 1.0
                
                # else:
                #     # Epsilon-greedy policy update
                #     epsilon = self.epsilon * (1 - episode / self.num_episodes)  # Decay epsilon
                #     epsilon = max(epsilon, 0.1)  # Ensure epsilon does not go below 0.01
                #     # Update policy to be epsilon-greedy
                #     for state in range(self.env.num_states):
                #         action_values = self.Q[state]
                #         best_action = np.argmax(action_values)
                #         self.policy[state] = np.full(self.env.num_actions, epsilon / self.env.num_actions)
                #         self.policy[state][best_action] = 1 - epsilon + (epsilon / self.env.num_actions)
            # env.add_policy(self.policy)  # Add the current policy to the environment for visualization
            # env.add_state_values(self.Q.max(axis=1))   
            # env.render(animation_interval=10)  

        # calculate the state value function
        self.V = np.zeros(self.env.num_states)
        for state in range(self.env.num_states):
            self.V[state] = np.mean(self.returns[state])  # Use the maximum Q value for each state as the state value
        return self.V, self.policy



if __name__ == "__main__":
    # Example usage
    env = GridWorld()
    agent = MonteCarloAgent(env, gamma=0.9, epsilon=1.0, num_episodes=1000)

    state = env.reset()
    env.render(animation_interval=2)
    policy = agent.generate_random_policy()
    state_values, optimal_policy= agent.evaluate_policy(policy, greedy=True)
    #将最大值作为最优策略，即最大的action的p为1，其余为0
    optimal_policy = np.argmax(optimal_policy, axis=1)  # Convert policy to action indices
    optimal_policy = np.eye(env.num_actions)[optimal_policy]  # Convert to one-hot encoding
    print("State Value Function:", state_values)
    env.add_state_values(state_values)
    env.add_policy(optimal_policy)
    env.render(animation_interval=100)
    print("Final Policy:", agent.policy)

