__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]

import sys

sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from examples.arguments import args


class GridWorld:

    def __init__(
        self,
        env_size=args.env_size,
        start_state=args.start_state,
        target_state=args.target_state,
        forbidden_states=args.forbidden_states,
    ):

        self.env_size = env_size
        self.num_states = env_size[0] * env_size[1]
        self.start_state = start_state
        self.target_state = target_state
        self.forbidden_states = forbidden_states

        self.agent_state = start_state
        self.action_space = args.action_space
        self.num_actions = len(self.action_space)
        self.reward_target = args.reward_target
        self.reward_forbidden = args.reward_forbidden
        self.reward_step = args.reward_step
        self.reward_stay = args.reward_stay

        self.canvas = None
        self.animation_interval = args.animation_interval

        self.color_forbid = (0.9290, 0.6940, 0.125)
        self.color_target = (0.3010, 0.7450, 0.9330)
        self.color_policy = (0.4660, 0.6740, 0.1880)
        self.color_trajectory = (0, 1, 0)
        self.color_agent = (0, 0, 1)

    def reset(self):
        self.agent_state = self.start_state
        self.traj = [self.agent_state]
        return self.agent_state, {}

    def set_state(self, state):
        """
        Set the agent's state to a specific position.
        This is useful for resetting the environment or evaluating policies.
        """
        assert (
            isinstance(state, tuple) and len(state) == 2
        ), "State must be a tuple of (x, y)"
        assert (
            0 <= state[0] < self.env_size[0] and 0 <= state[1] < self.env_size[1]
        ), "State out of bounds"

        self.agent_state = state
        self.traj = [self.agent_state]
        return self.agent_state, {}

    def step(self, action):
        assert action in self.action_space, "Invalid action"

        next_state, reward = self._get_next_state_and_reward(self.agent_state, action)
        done = self._is_done(next_state)

        x_store = next_state[0] + 0.03 * np.random.randn()
        y_store = next_state[1] + 0.03 * np.random.randn()
        state_store = tuple(np.array((x_store, y_store)) + 0.2 * np.array(action))
        state_store_2 = (next_state[0], next_state[1])

        self.agent_state = next_state

        self.traj.append(state_store)
        self.traj.append(state_store_2)
        return self.agent_state, reward, done, {}

    def only_step(self, action):
        """
        Perform a step in the environment without updating the trajectory.
        This is useful for evaluating policies without visualizing the trajectory.
        """
        assert action in self.action_space, "Invalid action"

        next_state, reward = self._get_next_state_and_reward(self.agent_state, action)
        done = self._is_done(next_state)

        self.agent_state = next_state
        return self.agent_state, reward, done, {}

    def _get_next_state_and_reward(self, state, action):
        x, y = state
        new_state = tuple(np.array(state) + np.array(action))
        if y + 1 > self.env_size[1] - 1 and action == (0, 1):  # down
            y = self.env_size[1] - 1
            reward = self.reward_forbidden
        elif x + 1 > self.env_size[0] - 1 and action == (1, 0):  # right
            x = self.env_size[0] - 1
            reward = self.reward_forbidden
        elif y - 1 < 0 and action == (0, -1):  # up
            y = 0
            reward = self.reward_forbidden
        elif x - 1 < 0 and action == (-1, 0):  # left
            x = 0
            reward = self.reward_forbidden
        elif new_state == self.target_state:  # stay
            x, y = self.target_state
            reward = self.reward_target
        elif new_state in self.forbidden_states:  # stay
            x, y = state
            # 即使遇到禁止态，依然可走
            # x, y = new_state
            reward = self.reward_forbidden
        elif new_state == state:  # stay
            x, y = state
            reward = self.reward_stay
        else:
            x, y = new_state
            reward = self.reward_step

        return (x, y), reward

    def _is_done(self, state):
        return state == self.target_state

    def render(self, animation_interval=args.animation_interval):
        if self.canvas is None:
            plt.ion()
            self.canvas, self.ax = plt.subplots()
            self.ax.set_xlim(-0.5, self.env_size[0] - 0.5)
            self.ax.set_ylim(-0.5, self.env_size[1] - 0.5)
            self.ax.xaxis.set_ticks(np.arange(-0.5, self.env_size[0], 1))
            self.ax.yaxis.set_ticks(np.arange(-0.5, self.env_size[1], 1))
            self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis="both")
            self.ax.set_aspect("equal")
            self.ax.invert_yaxis()
            self.ax.xaxis.set_ticks_position("top")

            idx_labels_x = [i for i in range(self.env_size[0])]
            idx_labels_y = [i for i in range(self.env_size[1])]
            for lb in idx_labels_x:
                self.ax.text(
                    lb,
                    -0.75,
                    str(lb + 1),
                    size=10,
                    ha="center",
                    va="center",
                    color="black",
                )
            for lb in idx_labels_y:
                self.ax.text(
                    -0.75,
                    lb,
                    str(lb + 1),
                    size=10,
                    ha="center",
                    va="center",
                    color="black",
                )
            self.ax.tick_params(
                bottom=False,
                left=False,
                right=False,
                top=False,
                labelbottom=False,
                labelleft=False,
                labeltop=False,
            )

            self.target_rect = patches.Rectangle(
                (self.target_state[0] - 0.5, self.target_state[1] - 0.5),
                1,
                1,
                linewidth=1,
                edgecolor=self.color_target,
                facecolor=self.color_target,
            )
            self.ax.add_patch(self.target_rect)

            for forbidden_state in self.forbidden_states:
                rect = patches.Rectangle(
                    (forbidden_state[0] - 0.5, forbidden_state[1] - 0.5),
                    1,
                    1,
                    linewidth=1,
                    edgecolor=self.color_forbid,
                    facecolor=self.color_forbid,
                )
                self.ax.add_patch(rect)

            (self.agent_star,) = self.ax.plot(
                [], [], marker="*", color=self.color_agent, markersize=20, linewidth=0.5
            )
            (self.traj_obj,) = self.ax.plot(
                [], [], color=self.color_trajectory, linewidth=0.5
            )

        # self.agent_circle.center = (self.agent_state[0], self.agent_state[1])
        self.agent_star.set_data([self.agent_state[0]], [self.agent_state[1]])
        traj_x, traj_y = zip(*self.traj)
        self.traj_obj.set_data(traj_x, traj_y)

        plt.draw()
        plt.pause(animation_interval)
        if args.debug:
            input("press Enter to continue...")

    # def add_policy(self, policy_matrix):
    #     # 在清除前加入以下调试信息

    #     for state, state_action_group in enumerate(policy_matrix):
    #         x = state % self.env_size[0]
    #         y = state // self.env_size[0]
    #         for i, action_probability in enumerate(state_action_group):
    #             if action_probability !=0:
    #                 dx, dy = self.action_space[i]
    #                 if (dx, dy) != (0,0):
    #                     self.ax.add_patch(patches.FancyArrow(x, y, dx=(0.1+action_probability/2)*dx, dy=(0.1+action_probability/2)*dy, color=self.color_policy, width=0.001, head_width=0.05))
    #                 else:
    #                     self.ax.add_patch(patches.Circle((x, y), radius=0.07, facecolor=self.color_policy, edgecolor=self.color_policy, linewidth=1, fill=False))
    def add_policy(self, policy_matrix):
        # 清除所有策略相关的图形元素（使用类型 + 颜色前三个通道近似匹配）
        removed_count = 0
        for patch in self.ax.patches:
            # 判断是否是 FancyArrow 或 Circle 类型
            if not (
                isinstance(patch, patches.Polygon) or isinstance(patch, patches.Circle)
            ):
                continue

            # 获取 facecolor 和 edgecolor（只比较前三个通道，忽略透明度）
            fc = (
                patch.get_facecolor()[:3]
                if len(patch.get_facecolor()) >= 3
                else patch.get_facecolor()
            )
            ec = (
                patch.get_edgecolor()[:3]
                if len(patch.get_edgecolor()) >= 3
                else patch.get_edgecolor()
            )

            target_color = self.color_policy[:3]  # 只保留 RGB 三通道

            if np.allclose(fc, target_color, atol=1e-2) or np.allclose(
                ec, target_color, atol=1e-2
            ):
                patch.remove()
                removed_count += 1

        print(f"Removed {removed_count} old policy elements.")

        # 再重新绘制新的策略
        for state, state_action_group in enumerate(policy_matrix):
            x = state % self.env_size[0]
            y = state // self.env_size[0]
            for i, action_probability in enumerate(state_action_group):
                if action_probability != 0:
                    dx, dy = self.action_space[i]
                    if (dx, dy) != (0, 0):
                        arrow = patches.FancyArrow(
                            x,
                            y,
                            dx=(0.1 + action_probability / 4) * dx,
                            dy=(0.1 + action_probability / 4) * dy,
                            color=self.color_policy,
                            width=0.001,
                            head_width=0.05,
                        )
                        self.ax.add_patch(arrow)
                    else:
                        circle = patches.Circle(
                            (x, y),
                            radius=0.07,
                            facecolor=self.color_policy,
                            edgecolor=self.color_policy,
                            linewidth=1,
                            fill=False,
                        )
                        self.ax.add_patch(circle)

        # plt.draw()
        # plt.pause(0.001)

    def add_state_values(self, values, precision=1):
        """
        values: iterable
        """
        # 先清空已有文本
        for text in self.ax.texts:
            if text.get_color() == "black":
                text.remove()

        values = np.round(values, precision)
        for i, value in enumerate(values):
            x = i % self.env_size[0]
            y = i // self.env_size[0]
            self.ax.text(
                x, y, str(value), ha="center", va="center", fontsize=10, color="black"
            )

    def pos_2_index(self, pos):
        """
        Convert a position (x, y) to a state index.
        """
        if isinstance(pos, tuple) or isinstance(pos, list):
            return pos[0] + pos[1] * self.env_size[0]
        else:
            raise ValueError("Position must be a tuple or list.")

    def index_2_pos(self, index):
        """
        Convert a state index to a position (x, y).
        """
        if isinstance(index, int):
            x = index % self.env_size[0]
            y = index // self.env_size[0]
            return (x, y)
        else:
            raise ValueError("Index must be an integer.")

    def sample_action(self, state, policy=None):
        """
        Sample an action from the policy for a given state.
        If no policy is provided, a random action is sampled.
        """
        if policy is None:
            return self.action_space[np.random.randint(self.num_actions)]
        else:
            action_probabilities = policy[state]
            return self.action_space[
                np.random.choice(self.num_actions, p=action_probabilities)
            ]
