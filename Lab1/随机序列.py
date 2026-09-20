import numpy as np

# 随机序列长度
N = 10000000

# 服从[-1, 1]均匀分布的随机序列
rng = np.random.default_rng()
X = rng.uniform(-1, 1, N)
Y = rng.uniform(-1, 1, N)

# 蒙特卡洛投点法
inside_count = 0

for i in range(N):
    if pow(X[i], 2) + pow(Y[i], 2) <= 1:
        inside_count += 1

# 输出结果
print(inside_count * 4 / N)