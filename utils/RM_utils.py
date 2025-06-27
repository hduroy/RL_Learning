import random
import matplotlib.pyplot as plt
import numpy as np


def sample_in_circle(n_samples=100, radius=1.0, seed=None):
    if seed is None:
        seed = (0, 0)
    rootx, rooty = seed
    points = []
    while len(points) < n_samples:
        x = random.uniform(-radius, radius) + rootx
        y = random.uniform(-radius, radius) + rooty
        if (x - rootx) ** 2 + (y - rooty) ** 2 <= radius ** 2:
            points.append((x, y))
        
    return np.array(points)

# 示例数据
# data = sample_in_circle(n_samples=100, radius=1.0)
###----------------------------------- GD --------------------------
def gd(data, lr=0.01, iterations=100):
    w = np.zeros(2)  # 初始值
    for k in range(iterations):
        grad = (w - data).mean(axis=0)
        w -= lr * grad
        if k % 10 == 0:
            loss = np.mean(np.sum((w - data) ** 2, axis=1))
            print(f"GD Iteration {k}: w = {w}, Loss = {loss:.6f}")
    return w

###----------------------------------- MBGD --------------------------
def get_mini_batches(data, batch_size):
    n = len(data)
    indices = np.random.permutation(n)
    for i in range(0, n, batch_size):
        yield data[indices[i:i+batch_size]]

def mbgd(data, batch_size=32, lr=0.01, iterations=100):
    w = np.zeros(2)  # 初始值
    n = len(data)
    for k in range(iterations):
        for batch in get_mini_batches(data, batch_size):
            grad = 2 * (w - batch).mean(axis=0)
            w -= lr * grad
        if k % 10 == 0:
            loss = np.mean(np.sum((w - data) ** 2, axis=1))
            print(f"MBGD Iteration {k}: w = {w}, Loss = {loss:.6f}")
    return w

###----------------------------------- SGD --------------------------
def sgd(data, lr=0.01, iterations=100):
    w = np.zeros(2)
    n = len(data)
    for k in range(iterations):
        i = random.randint(0, n - 1)
        grad = 2 * (w - data[i])
        w -= lr * grad
        if k % 10 == 0:
            loss = np.mean(np.sum((w - data) ** 2, axis=1))
            print(f"SGD Iteration {k}: w = {w}, Loss = {loss:.6f}")
    return w


###----------------------------------- RM算法 --------------------------
# def g(w):
#     return w**3 - 5

# # RM算法求解 g(w) = 0
# def rm_algorithm(func, iterations=50, w_0=0):

#     best_w = None
#     min_error = float('inf')
#     w_current = w_0
#     for k in range(iterations):
#         # 生成满足0-1分布的噪音
#         theta = random.uniform(0, 1)
#         # 根据噪音生成新的w
#         alpha = 1 / (k + 2)
#         w_next = w_current - alpha * (g(w_current) + theta)
#         # 计算误差
#         error = abs(func(w_next))
        
#         if error < min_error:
#             min_error = error
#             best_w = w_next
#         # 更新当前w
#         w_current = w_next
        
       
#         # if k % 1 == 0:
#         #     print(f"Iteration {k}: w = {w_current:.8f}, theta = {theta:.4f}, error = {error:.2e}")
            
#     return best_w, min_error


def g(w, data):
    return (w - data).mean(axis=0)

def rm_algorithm(func, data, iterations=50, w_0=np.zeros(2)):
    best_w = w_0.copy()
    min_error = float('inf')
    w_current = w_0.copy()

    for k in range(iterations):
        theta = np.random.randn(2)  # 噪音向量
        alpha = 1 / (k + 2)
        w_next = w_current - alpha * (func(w_current, data) + theta)

        error = np.linalg.norm(func(w_next, data))

        if error < min_error:
            min_error = error
            best_w = w_next

        w_current = w_next

        if k % 10 == 0:
            print(f"RM Iteration {k}: w = {w_current}, Error = {error:.6f}")

    return best_w, min_error


if __name__ == "__main__":
    # 示例数据
    data = sample_in_circle(n_samples=100, radius=1.0, seed=(1, 1))
    
    print("=== GD ===")
    w_gd = gd(data, iterations=100)
    print("GD Estimate:", w_gd)

    print("\n=== SGD ===")
    w_sgd = sgd(data, iterations=100)
    print("SGD Estimate:", w_sgd)

    print("\n=== MBGD ===")
    w_mbgd = mbgd(data, batch_size=10, iterations=100)
    print("MBGD Estimate:", w_mbgd)

    print("\n=== RM Algorithm ===")
    w_rm, _ = rm_algorithm(g, data, iterations=100)
    print("RM Estimate:", w_rm)

    print("\n=== True Mean ===")
    print("True Mean:", data.mean(axis=0))
        
    # RM算法结果已经在上面输出