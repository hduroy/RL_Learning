import numpy as np
import matplotlib.pyplot as plt



def normalize_array(arr):
    """
    Normalize a numpy array to the range [0, 1].

    Parameters:
        arr (np.ndarray): Input array.

    Returns:
        np.ndarray: Normalized array.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val)


def save_to_file(data, filename):
    """
    Save data to a file.

    Parameters:
        data (any): Data to save.
        filename (str): Path to the file.

    Returns:
        None
    """
    try:
        with open(filename, "w") as f:
            f.write(str(data))
    except Exception as e:
        raise IOError(f"Failed to save data to {filename}: {e}")



def calc_state_value(pi, P, R, gamma=0.9, theta=1e-6):
    num_states = pi.shape[0]
    num_actions = pi.shape[1]
    V = np.zeros(num_states)

    while True:
        delta = 0
        for s in range(num_states):
            v_old = V[s]
            v_new = 0.0
            for a in range(num_actions):
                action_prob = pi[s, a]
                expected_reward = R[s, a]
                future_value = np.sum(P[s, a, :] * V)
                v_new += action_prob * (expected_reward + gamma * future_value)
            V[s] = v_new
            delta = max(delta, abs(V[s] - v_old))
        if delta < theta:
            break
    return V


def test_complex_case():
    grid_size = 5
    gamma = 0.95
    theta = 1e-6

    # Actions: up, down, left, right
    actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    num_actions = len(actions)
    num_states = grid_size * grid_size

    # Initialize policy, transition probabilities, and rewards
    pi = np.ones((num_states, num_actions)) / num_actions  # 默认随机策略

    # 非均匀策略：让某些状态更偏好某些动作
    # 示例：在某些区域偏向向右或向下
    for s in range(num_states):
        row, col = divmod(s, grid_size)
        if row == 1 and col == 1:
            pi[s] = [0.1, 0.1, 0.1, 0.7]  # 偏好向右
        elif row == 3 and col == 3:
            pi[s] = [0.7, 0.1, 0.1, 0.1]  # 偏好向上

    P = np.zeros((num_states, num_actions, num_states))
    R = np.zeros((num_states, num_actions))

    # 填充状态转移与奖励
    for state in range(num_states):
        r, c = divmod(state, grid_size)

        for action, (dr, dc) in actions.items():
            # 主方向
            nr, nc = r + dr, c + dc
            next_states = []

            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                new_state = nr * grid_size + nc
                next_states.append((new_state, 0.8))
            else:
                next_states.append((state, 0.8))  # 碰壁则留在原地

            # 左右偏移作为扰动（模拟不确定环境）
            offsets = [(dc, -dr), (-dc, dr)]  # 左右偏移
            for i, (drr, dcc) in enumerate(offsets):
                nrr, ncc = r + drr, c + dcc
                if 0 <= nrr < grid_size and 0 <= ncc < grid_size:
                    new_offset_state = nrr * grid_size + ncc
                    next_states.append((new_offset_state, 0.1))
                else:
                    next_states.append((state, 0.1))  # 偏移碰壁也留在原地

            # 设置转移概率
            for ns, prob in next_states:
                P[state, action, ns] += prob

            # 设置奖励
            for ns, prob in next_states:
                # 如果是主动作的目标状态
                if (nr, nc) == divmod(ns, grid_size):
                    if (r, c) == (0, 0):  # 起点
                        R[state, action] = 5
                    elif (r, c) == (4, 4):  # 终点
                        R[state, action] = 10
                    elif (r, c) == (2, 2):  # 陷阱
                        R[state, action] = -10
                    else:
                        if nr != r + dr or nc != c + dc:
                            R[state, action] = -1  # 碰壁惩罚
                        else:
                            R[state, action] = -1  # 正常移动
                else:
                    if (divmod(ns, grid_size) == (2, 2)):
                        R[state, action] -= 10 * prob  # 可能进入陷阱的期望惩罚

    # 计算状态价值
    V = calc_state_value(pi, P, R, gamma, theta)

    # 可视化
    V_grid = V.reshape(grid_size, grid_size)
    plt.figure(figsize=(8, 8))
    plt.imshow(V_grid, cmap='coolwarm', origin='upper')
    plt.colorbar(label='State Value')
    plt.title("Complex State Value Function")
    plt.xlabel("Column")
    plt.ylabel("Row")

# 标记每个格子的状态值
    for r in range(grid_size):
        for c in range(grid_size):
            value = V_grid[r, c]
            plt.text(c, r, f"{value:.2f}", ha='center', va='center', color='black', fontsize=8,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # 标记特殊位置
    special_states = {
        (0, 0): 'Start (+10)',
        (4, 4): 'Goal (+5)',
        (2, 2): 'Trap (-10)'
    }
    for (r, c), label in special_states.items():
        plt.text(c, r, label, ha='center', va='center', color='black', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    plt.show()

    return V


if __name__ == "__main__":
    test_complex_case()
