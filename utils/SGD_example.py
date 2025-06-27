import random
import matplotlib.pyplot as plt
import numpy as np

# 采样函数
def sample_in_circle(n_samples=100, radius=1.0, seed=None):
    if seed is None:
        seed = (0, 0)
    rootx, rooty = seed
    points = []
    while len(points) < n_samples:
        x = random.uniform(-radius, radius) + rootx
        y = random.uniform(-radius, radius) + rooty
        if (x - rootx)**2 + (y - rooty)**2 <= radius**2:
            points.append((x, y))
    return np.array(points)

###----------------------------------- GD --------------------------
def gd(data, lr=0.01, iterations=100):
    w = np.zeros(2)
    history = [w.copy()]
    for k in range(iterations):
        grad = (w - data).mean(axis=0)
        w -= lr * grad
        history.append(w.copy())
    return w, history

###----------------------------------- MBGD --------------------------
def get_mini_batches(data, batch_size):
    n = len(data)
    indices = np.random.permutation(n)
    for i in range(0, n, batch_size):
        yield data[indices[i:i+batch_size]]

def mbgd(data, batch_size=32, lr=0.01, iterations=100):
    w = np.zeros(2)
    history = [w.copy()]
    for k in range(iterations):
        for batch in get_mini_batches(data, batch_size):
            grad = (w - batch).mean(axis=0)
            w -= lr * grad
        history.append(w.copy())
    return w, history

###----------------------------------- SGD --------------------------
def sgd(data, lr=0.01, iterations=100):
    w = np.zeros(2)
    n = len(data)
    history = [w.copy()]
    for k in range(iterations):
        i = random.randint(0, n - 1)
        grad = (w - data[i])
        w -= lr * grad
        history.append(w.copy())
    return w, history

###----------------------------------- RM Algorithm --------------------------
def g(w, data):
    return (w - data).mean(axis=0)

def rm_algorithm(func, data, iterations=50, w_0=np.zeros(2)):
    best_w = w_0.copy()
    min_error = float('inf')
    w_current = w_0.copy()
    history = [w_current.copy()]
    for k in range(iterations):
        theta = np.random.randn(2)
        alpha = 1 / (k + 2)
        w_next = w_current - alpha * (func(w_current, data) + theta)
        history.append(w_next.copy())
        error = np.linalg.norm(func(w_next, data))
        if error < min_error:
            min_error = error
            best_w = w_next
        w_current = w_next
    return best_w, history

###------------------------------------------------------------
### 🚀 主函数 + 绘图部分
###------------------------------------------------------------

def main():
    # 生成数据
    data = sample_in_circle(n_samples=200, radius=1.0, seed=(2, 2))
    true_mean = data.mean(axis=0)

    print("True Mean:", true_mean)

    # 调用各算法
    iterations = 100
    w_gd, hist_gd = gd(data, lr=0.1, iterations = iterations)
    w_sgd, hist_sgd = sgd(data, lr=0.1, iterations=iterations)
    w_mbgd, hist_mbgd = mbgd(data, batch_size=50, lr=0.1, iterations=iterations)
    w_rm, hist_rm = rm_algorithm(g, data, iterations=iterations)

    print("\nFinal Estimates:")
    print(f"GD:     {w_gd}")
    print(f"SGD:    {w_sgd}")
    print(f"MBGD:   {w_mbgd}")
    print(f"RM:     {w_rm}")

    # 可视化
    plt.figure(figsize=(10, 8))

    # 所有点
    plt.scatter(data[:, 0], data[:, 1], c='lightgray', label='Sample Points', alpha=0.5)

    # 真实均值
    plt.scatter(true_mean[0], true_mean[1], c='black', marker='x', s=100, label='True Mean')

    # 初始点
    plt.scatter(0, 0, c='purple', marker='o', s=100, label='Initial Point')

    # 路径绘制
    def plot_path(history, color, label):
        history = np.array(history)
        plt.plot(history[:, 0], history[:, 1], '-o', color=color, label=label, markersize=4)

    plot_path(hist_gd, 'blue', 'GD Path')
    plot_path(hist_sgd, 'red', 'SGD Path')
    plot_path(hist_mbgd, 'green', 'MBGD Path')
    plot_path(hist_rm, 'orange', 'RM Path')

    plt.title("Convergence Paths of Optimization Algorithms")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # 保证坐标轴比例一致
    plt.show()

    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot([np.linalg.norm(g(w, data)) for w in hist_gd], label='GD Convergence', color='blue')
    plt.plot([np.linalg.norm(g(w, data)) for w in hist_sgd], label='SGD Convergence', color='red')
    plt.plot([np.linalg.norm(g(w, data)) for w in hist_mbgd], label='MBGD Convergence', color='green')
    plt.plot([np.linalg.norm(g(w, data)) for w in hist_rm], label='RM Convergence', color='orange')
    plt.title("Convergence of Optimization Algorithms")
    plt.xlabel("Iteration")
    plt.ylabel("Norm of Gradient")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()