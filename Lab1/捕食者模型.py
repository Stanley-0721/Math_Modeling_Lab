import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ========== 解决matplotlib中文乱码 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示方框

# ========== 原始观测数据 ==========
t_data = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18])
x_data = np.array([60, 63, 64, 63, 61, 58, 53, 44, 39, 38, 41, 46, 53]) # 狼 x
y_data = np.array([30, 34, 38, 44, 50, 55, 58, 56, 47, 38, 30, 27, 26]) # 羊 y

# 定义Lotka-Volterra微分方程组
def lv_model(t, z, a, b, c, d):
    x, y = z
    dxdt = -a * x + b * x * y
    dydt = c * y - d * x * y
    return [dxdt, dydt]

# 残差函数：给定参数，求解ODE，返回预测值与观测值之差
def residual(params):
    a, b, c, d = params
    # 初值取t=0的观测值
    sol = solve_ivp(lv_model, [t_data[0], t_data[-1]], [x_data[0], y_data[0]],
                    args=(a,b,c,d), t_eval=t_data, method='RK45')
    x_pred, y_pred = sol.y
    res = np.concatenate([x_pred - x_data, y_pred - y_data])
    return res

# 初始猜测参数 [a,b,c,d]，参数均>0
p0 = np.array([0.1, 0.01, 0.1, 0.01])
# 最小二乘拟合，加边界保证参数>0
bounds = ((0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf))
res_fit = least_squares(residual, p0, bounds=bounds)
a_fit, b_fit, c_fit, d_fit = res_fit.x
print(f"拟合得到参数：")
print(f"a = {a_fit:.4f}")
print(f"b = {b_fit:.4f}")
print(f"c = {c_fit:.4f}")
print(f"d = {d_fit:.4f}")

# ========== 仿真：更长时间范围模拟生态演化 ==========
t_span = [0, 30]
t_sim = np.linspace(0,30,300)
sol_sim = solve_ivp(lv_model, t_span, [x_data[0], y_data[0]],
                    args=(a_fit,b_fit,c_fit,d_fit), t_eval=t_sim)
x_sim, y_sim = sol_sim.y

# ========== 绘图 ==========
plt.figure(figsize=(12,5))
plt.plot(t_data, x_data, 'ro', label='狼(观测)')
plt.plot(t_data, y_data, 'go', label='羊(观测)')
plt.plot(t_sim, x_sim, 'r-', label='狼(模型拟合)')
plt.plot(t_sim, y_sim, 'g-', label='羊(模型拟合)')
plt.xlabel("时间 t")
plt.ylabel("种群数量")
plt.title("Lotka-Volterra捕食者猎物模型拟合与仿真")
plt.legend()
plt.grid(True)
plt.show()
