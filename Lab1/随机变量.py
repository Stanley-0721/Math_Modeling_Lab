import numpy as np
import matplotlib.pyplot as plt

# ========== 中文乱码修复 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 1. 复现PPT中4种线性同余LCG伪随机数 ======================
def lcg(mode, M, param, y1=5, length=1000):
    """
    mode: 1加法同余；2乘法同余；3普通混合；4改良混合
    param: 参数
    """
    y = np.zeros(length, dtype=int)
    x = np.zeros(length, dtype=float)
    y[0] = y1
    x[0] = y[0]/M
    for i in range(1, length):
        prev = y[i-1]
        if mode == 1:
            # 加法同余 y(i) = mod(y(i‑1)+c, M)
            c = param["c"]
            y[i] = (prev + c) % M
        elif mode == 2:
            # 乘法同余 y(i) = mod(y(i‑1)*a, M)
            a = param["a"]
            y[i] = (prev * a) % M
        elif mode == 3:
            # 普通混合 y(i) = mod(y(i‑1)*a + c, M)
            a, c = param["a"], param["c"]
            y[i] = (prev * a + c) % M
        elif mode ==4:
            # 改良版本 mod(y(i‑1)*(4*a+1)+(2*b+1), M)
            a, b = param["a"], param["b"]
            coeff_a = 4*a + 1
            coeff_b = 2*b + 1
            y[i] = (prev * coeff_a + coeff_b) % M
        x[i] = y[i]/M
    return x

plt.figure(figsize=(12,9))

# case1 加法同余 M=1000, c=4235 PPT第5页
plt.subplot(2,2,1)
x1 = lcg(mode=1, M=1000, param={"c":4235})
plt.plot(x1)
plt.title("case1 加法同余 M=1000 c=4235")
plt.ylim(0,1)

# case2 乘法同余 M=2^16 a=4235 PPT第6页
plt.subplot(2,2,2)
x2 = lcg(mode=2, M=2**16, param={"a":4235})
plt.plot(x2)
plt.title("case2 乘法同余 M=2^16 a=4235")
plt.ylim(0,1)

# case3 普通混合 LCG，坏参数，发生退化 PPT第7页
plt.subplot(2,2,3)
x3 = lcg(mode=3, M=2**16, param={"a":1234,"c":4235})
plt.plot(x3)
plt.title("case3 普通混合LCG(坏参数，序列退化)")
plt.ylim(0,1)

# case4 改良版LCG PPT第8页
plt.subplot(2,2,4)
x4 = lcg(mode=4, M=2**16, param={"a":1234,"b":4235})
plt.plot(x4)
plt.title("case4 改良版LCG")
plt.ylim(0,1)

plt.tight_layout()
plt.show()