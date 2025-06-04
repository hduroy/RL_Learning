
import sys
sys.path.append("..")
from src.grid_world import GridWorld
import random
import numpy as np

# Example usage:
if __name__ == "__main__":
    env = GridWorld()
    state = env.reset()

    # # 初始化策略和值函数
    # policy_matrix = np.random.rand(env.num_states, len(env.action_space))
    # policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]

    # values = np.random.uniform(0, 10, (env.num_states,))

    # # 第一次 render 创建绘图上下文
    # env.render()

    # # 添加策略和值函数
    # env.add_policy(policy_matrix)
    # env.add_state_values(values)

    # 开始交互循环
    for t in range(1000):
        env.render()

        action = random.choice(env.action_space)
        next_state, reward, done, info = env.step(action)
        print(f"Step: {t}, Action: {action}, State: {next_state + np.array([1,1])}, Reward: {reward}, Done: {done}")

        policy_matrix = np.random.rand(env.num_states, len(env.action_space))
        policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]

        values = np.random.uniform(0, 10, (env.num_states,))


        # 添加策略和值函数
        env.add_policy(policy_matrix)
        env.add_state_values(values)
        
        # if done:
        #     break

    # 最后再渲染一次，展示最终结果
    env.render(animation_interval=2)


    
    # # Add policy
    # policy_matrix=np.random.rand(env.num_states,len(env.action_space))                                            
    # policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1

    # env.add_policy(policy_matrix)

    
    # # Add state values
    # values = np.random.uniform(0,10,(env.num_states,))
    # env.add_state_values(values)

    # # Render the environment
    # env.render(animation_interval=2)