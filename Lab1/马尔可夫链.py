import numpy as np

# 参数设置
N = 1000
a = 1/2
b = 1/3

# 转移矩阵 P: P[i][j] 从状态i转移到j (状态0代表state1，状态1代表state2)
P = np.array([
    [a,   1-a],
    [1-b, b]
])

rng = np.random.default_rng()

# 马尔可夫链模拟
state = 0  # 初始状态，0=状态1，1=状态2
count = np.zeros(2)
count[state] += 1

for _ in range(N-1):
    # 根据当前状态，按转移概率抽样下一状态
    state = rng.choice([0,1], p=P[state])
    count[state] += 1

# 计算模拟得到的频率
freq = count / N
print("各状态出现次数：", count)
print("模拟得到的频率：", freq)

# 理论平稳分布
pi_theo = np.array([(1-b)/(2-a-b), (1-a)/(2-a-b)])
print("理论的平稳分布：", pi_theo)
