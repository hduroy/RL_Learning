import numpy as np
import sys
import os

# 获取当前文件所在目录的上三层目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print("Project root directory:", project_root)
sys.path.append(project_root)


from src.grid_world import GridWorld
from utils.tool import calc_state_value
from examples.arguments import args  # 再上一层目录


class PolicyEvaluator:
    """
    A class to evaluate a given policy in a GridWorld environment.
    """

    def __init__(self, env, max_iteration=args.max_iteration, env_size=args.env_size, start_state=args.start_state, target_state=args.target_state, forbidden_states=args.forbidden_states):
        """
        Initialize the PolicyEvaluator with environment and policy parameters.
        """
        if env is not None:
            self.env = env
        else:
            # Initialize the GridWorld environment
            self.env = GridWorld(env_size, start_state, target_state, forbidden_states)
            _ = self.env.reset()  # Reset the environment to initialize it
        self.policy = None
        self.max_iteration = max_iteration
        self.state_value = None 
        self.gamma = 0.9  # Default discount factor

    def set_policy(self, policy):
        """
        Set the policy to be evaluated.

        Parameters:
            policy (np.ndarray): The policy to evaluate, where each row corresponds to a state and each column corresponds to an action.
        """
        self.policy = policy

    def calc_state_value(self, env : GridWorld, policy, theta=1e-6):
        """
        Calculate the state value function for a given policy in the environment.

        Parameters:
            env (GridWorld): The environment in which the policy is evaluated.
            policy (np.ndarray): The policy to evaluate, where each row corresponds to a state and each column corresponds to an action.
            theta (float): Convergence threshold for state value updates.

        Returns:
            np.ndarray: The state value function for the given policy.
        """
        #init the state value
        if self.state_value is None:
            self.state_value = np.zeros(env.num_states)

        for iteration in range(self.max_iteration):
            delta = 0
            for state in range(env.num_states):
                # if state == env.pos_2_index(env.target_state):
                #     self.state_value[state] = 1.0  # Target state value is always 0
                #     continue
                v_old = self.state_value[state]
                v_new = 0.0
                for action in range(len(env.action_space)):
                    action_prob = policy[state, action]
                    if action_prob == 0:
                        continue
                    env.agent_state = env.index_2_pos(state)  # Set the agent's state to the current state
                    next_state_pos, reward, _, _ = env.only_step(env.action_space[action])
                    # print(f"State: {state}, Action: {action}, Next State: {next_state_pos}, Reward: {reward}")
                    next_state = env.pos_2_index(next_state_pos)
                    v_new += action_prob * (reward + self.gamma * self.state_value[next_state])
                self.state_value[state] = v_new
                delta = max(delta, abs(v_new - v_old))
            if delta < theta:
                break
        if iteration == self.max_iteration - 1:
            print(f"Warning: Maximum iterations ({self.max_iteration}) reached without convergence.")

        return self.state_value

    def evaluate_policy(self):
        """
        Evaluate the current policy and return the state values.

        Returns:
            np.ndarray: The state value function for the current policy.
        """
        if self.policy is None:
            raise ValueError("Policy not set.")
        
        state_values = self.calc_state_value(self.env, self.policy)
        return state_values
    


if __name__ == "__main__":

    # Example usage
    env = GridWorld()
    state = env.reset()
    evaluator = PolicyEvaluator(env)
    env.render(animation_interval=2)
    # Create a random policy
    policy_matrix = np.random.rand(env.num_states, len(env.action_space))
    # 每行随机选一个值为1
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    for i in range(env.num_states):
        action_index = np.random.choice(len(env.action_space))
        policy_matrix[i, action_index] = 1.0  # Set one action to 1, others to 0
    
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # Normalize to make it a valid probability distribution

    evaluator.set_policy(policy_matrix)
    print("Policy Matrix:\n", policy_matrix)
    # Evaluate the policy
    state_values = evaluator.evaluate_policy()
    env.add_state_values(state_values)
    env.add_policy(policy_matrix)
    print("State Values:\n", state_values)
    env.render(animation_interval=100)