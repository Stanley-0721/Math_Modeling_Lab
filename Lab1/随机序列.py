import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 1. 蒙特卡洛计算 π ----------------------
N_calc = 10000000  # 你原来的计算用样本量
rng = np.random.default_rng()
X_calc = rng.uniform(-1, 1, N_calc)
Y_calc = rng.uniform(-1, 1, N_calc)
inside_count = np.sum(X_calc**2 + Y_calc**2 <= 1)
pi_estimate = inside_count * 4 / N_calc
print(f"【计算结果】N={N_calc}, 蒙特卡洛估算 π = {pi_estimate:.8f}")

# ---------------------- 2. 散点图（少量点用于可视化） ----------------------
N_plot = 20000  # 绘图点数，不要太大
X_plot = rng.uniform(-1, 1, N_plot)
Y_plot = rng.uniform(-1, 1, N_plot)
in_mask = X_plot**2 + Y_plot**2 <= 1

# 拆分圆内、圆外点
X_in, Y_in = X_plot[in_mask], Y_plot[in_mask]
X_out, Y_out = X_plot[~in_mask], Y_plot[~in_mask]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(6,6))
plt.scatter(X_in, Y_in, s=1, c="#1f77b4", label="圆内点")
plt.scatter(X_out, Y_out, s=1, c="#ff7f0e", label="圆外点")

# 绘制单位圆轮廓
theta = np.linspace(0, 2*np.pi, 300)
plt.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1.5)

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axis("equal")
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"蒙特卡洛投点法，可视化散点图\n估算π ≈ {pi_estimate:.5f}")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
