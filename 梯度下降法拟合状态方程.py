import numpy as np
import matplotlib.pyplot as plt

# 定义晶格常数 R（或原子间距）的实验数据，单位Bhor半径
R = np.array([2.850000, 2.875000, 2.900000, 2.925000, 2.950000, 2.975000, 3.000000, 3.025000, 3.050000, 3.075000, 3.100000, 3.125000, 3.150000, 3.175000, 3.200000])
# 定义与各个 R 对应的能量 E 数据，用于拟合理论方程，单位Ry
E = np.array([-0.407909, -0.418714, -0.428745, -0.437251, -0.444351, -0.450373, -0.454622, -0.457922, -0.459465, -0.460778, -0.460919, -0.459735, -0.458370, -0.456056, -0.453357])
# 将 R 转换为体积 V，这里采用 V = R³ 的关系
V = R**3

lr = 0.01           # 设置梯度下降的学习率，决定每次迭代时参数的更新幅度
max_step = 150000   # 设置最大迭代次数，限制梯度下降循环的执行次数

# 设置待拟合参数的初始值
E0_initial = -0.4
V0_initial = 29.0
B0_initial = 0.06
dB0_initial = 4.0

# 定义 Birch-Murnaghan（BM）状态方程，用于计算给定体积 V 下的理论能量
# V：输入体积
# E0：平衡体积对应的能量
# V0：平衡体积
# B0：平衡体积处的体积模量
# dB0：体积模量对压力的导数
def BM_equation(V,E0,V0,B0,dB0):
    x=(V0/V)**(2/3)
    return E0 + (9/16)*B0*V0 * ((x-1)**3*dB0 - (x-1)**2*(4*x-6))


# 定义残差平方和函数，用于衡量理论计算值与实验数据之间的偏差
# 残差平方和越小，说明理论曲线整体上越接近实验数据
def residual_square_sum(E0,V0,B0,dB0):
    residual_square_s = 0            # 初始化残差平方和为 0
    for i in range(len(V)):          # 遍历所有实验数据点
        residual_square = E[i]-BM_equation(V[i],E0,V0,B0,dB0)       # 计算第 i 个数据点的残差：实验能量减去理论能量
        residual_square_s += residual_square**2         # 将该数据点的残差平方累加到总残差平方和中
    return residual_square_s


# 定义残差平方和对各个拟合参数的偏导数计算函数
# 使用中心差分法近似计算数值偏导数
def partial_rsm(E0,V0,B0,dB0):
    h = 1e-6            # 设置用于计算数值偏导数的微小扰动量
    # 计算残差平方和对 E0 的偏导数
    p_rsm_E0 = (residual_square_sum(E0+h,V0,B0,dB0)-residual_square_sum(E0-h,V0,B0,dB0))/(2*h)
    p_rsm_V0 = (residual_square_sum(E0,V0+h,B0,dB0)-residual_square_sum(E0,V0-h,B0,dB0))/(2*h)
    p_rsm_B0 = (residual_square_sum(E0,V0,B0+h,dB0)-residual_square_sum(E0,V0,B0-h,dB0))/(2*h)
    p_rsm_dB0 = (residual_square_sum(E0,V0,B0,dB0+h)-residual_square_sum(E0,V0,B0,dB0-h))/(2*h)
    return np.array([p_rsm_E0,p_rsm_V0,p_rsm_B0,p_rsm_dB0])

# 将四个参数的初始值存储在同一个 NumPy 数组中，后续可以通过索引分别访问和更新各个参数
parameter = np.array([E0_initial,V0_initial,B0_initial,dB0_initial])

# 使用梯度下降法迭代优化四个拟合参数
for step in range(max_step):
    # 按照梯度下降规则更新参数：
    # 新参数 = 旧参数 - 学习率 × 残差平方和的梯度
    # parameter[0]、parameter[1]、parameter[2]、parameter[3]分别对应 E0、V0、B0、dB0
    parameter = parameter - lr*partial_rsm(parameter[0],parameter[1],parameter[2],parameter[3])
if step == max_step - 1:
    print(f"迭代{max_step}次，已达到最大迭代次数")

# 输出梯度下降迭代结束后的四个拟合参数
print(f"E0 = {parameter[0]}")
print(f"V0 = {parameter[1]}")
print(f"B0 = {parameter[2]}")
print(f"dB0 = {parameter[3]}")


# 在实验体积范围之外适当扩展区间，生成用于绘制拟合曲线的体积数据，从最小体积的 0.9 倍到最大体积的 1.1 倍，均匀取 1000 个点
x = np.linspace(V.min()*0.9,V.max()*1.1,1000)

# 将拟合得到的参数代入 BM 状态方程，计算各个体积对应的理论能量
y = BM_equation(x,parameter[0],parameter[1],parameter[2],parameter[3])

plt.plot(x,y)           # 绘制 BM 状态方程得到的理论拟合曲线
plt.scatter(V,E)        # 将原始实验数据以散点形式绘制在图中，用于与拟合曲线进行比较
plt.show()              # 显示图像，观察理论曲线与实验数据的拟合情况