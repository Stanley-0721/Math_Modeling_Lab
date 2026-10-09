import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# ========== 中文乱码修复 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== PPT中的30条水果数据 密度、含糖率 ==========
data_raw = [
    [0.697, 0.460],
    [0.774, 0.376],
    [0.634, 0.264],
    [0.608, 0.318],
    [0.556, 0.215],
    [0.403, 0.237],
    [0.481, 0.149],
    [0.437, 0.211],
    [0.666, 0.091],
    [0.243, 0.267],
    [0.245, 0.057],
    [0.343, 0.099],
    [0.639, 0.161],
    [0.657, 0.198],
    [0.360, 0.370],
    [0.593, 0.042],
    [0.719, 0.103],
    [0.359, 0.188],
    [0.339, 0.241],
    [0.282, 0.257],
    [0.748, 0.232],
    [0.714, 0.346],
    [0.483, 0.312],
    [0.478, 0.437],
    [0.525, 0.369],
    [0.751, 0.489],
    [0.532, 0.472],
    [0.473, 0.376],
    [0.725, 0.445],
    [0.446, 0.459]
]

df = pd.DataFrame(data_raw, columns=["密度", "含糖率"])
X = np.array(data_raw)
print("===== 原始水果数据（前5行） =====")
print(df.head())

# ========== GMM高斯混合模型，k=3，EM迭代 ==========
n_components = 3  # 类别数k=3
gmm = GaussianMixture(n_components=n_components, max_iter=200)
labels = gmm.fit_predict(X)   # fit做EM迭代，predict得到每个样本聚类标签0/1/2

df['聚类标签'] = labels

print("\n===== GMM模型参数 =====")
print("各高斯权重(混合系数 π):\n", gmm.weights_)
print("\n各高斯分布均值 μ：")
for i, mu in enumerate(gmm.means_):
    print(f"类别{i} 均值：{mu}")

print("\n各高斯协方差矩阵 Σ：")
for i, sigma in enumerate(gmm.covariances_):
    print(f"类别{i} 协方差：\n{sigma}\n")

print("EM迭代收敛时迭代次数：", gmm.n_iter_)
print("是否收敛：", gmm.converged_)

print("\n===== 每个样本聚类结果（编号从1开始） =====")
df_show = df.copy()
df_show.index = np.arange(1, len(df_show)+1)
print(df_show)

# ========== 绘制聚类结果散点图 ==========
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:,0], X[:,1], c=labels, cmap="coolwarm", s=70, alpha=0.8)
plt.xlabel("密度")
plt.ylabel("含糖率")
plt.title("GMM‑EM水果聚类 k=3（密度‑含糖率）")
plt.grid(True, alpha=0.3)
plt.legend(*scatter.legend_elements(), title="聚类类别")
plt.show()

# ========== 输出每一类样本数量统计 ==========
print("\n===== 每一类样本数量 =====")
print(df['聚类标签'].value_counts().sort_index())
