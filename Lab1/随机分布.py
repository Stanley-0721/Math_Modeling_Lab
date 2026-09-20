import numpy as np
import math
import matplotlib.pyplot as plt

# 样本总数
N = 1000

# 样本，均值，方差
X = []
mean = []
var = []
num = [i for i in range(1, N+1)]

rng = np.random.default_rng()
for i in range(N):
    x = rng.normal(10, math.sqrt(5))
    X.append(x)
    mean.append(np.mean(X))
    var.append(np.var(X))

# 作图
plt.subplot(1, 2, 1)
plt.plot(num, mean)
plt.title('mean')

plt.subplot(1, 2, 2)
plt.plot(num, var)
plt.title('var')
plt.show()

# 模拟坦克到达的数量
for i in range(3):
    print(f'minute {i+1}: {np.random.poisson(4)}')


# 模拟坦克
time = 0
time += np.random.exponential(0.25)
num = 1
print(f'tank {num}: {time}')
while time < 3:
    time += np.random.exponential(0.25)
    num += 1
    if time < 3:
        print(f'tank {num}: {time}')
