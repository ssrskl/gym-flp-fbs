from FbsEnv.utils import FBSUtil
import FbsEnv
import gym
from FbsEnv.envs.FBSModel import FBSModel
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FbsEnv-v0", instance="AB20-ar3")
matrix = env.F

# 绘制矩阵图
fig, ax = plt.subplots()
cax = ax.matshow(matrix, cmap="YlOrRd")
# 设置坐标轴刻度从 1 开始
ax.set_xticks(np.arange(matrix.shape[1]))  # 设置 x 轴刻度位置
ax.set_yticks(np.arange(matrix.shape[0]))  # 设置 y 轴刻度位置
ax.set_xticklabels(np.arange(1, matrix.shape[1] + 1))  # 设置 x 轴刻度标签为 1, 2, ...
ax.set_yticklabels(np.arange(1, matrix.shape[0] + 1))  # 设置 y 轴刻度标签为 1, 2, ...


# 在每个单元格中显示具体数值
# for (i, j), val in np.ndenumerate(matrix):
#     ax.text(j, i, f"{val}", ha="center", va="center", color="black")

# 添加颜色条
plt.colorbar(cax)
plt.title("Matrix Visualization with 1-based Index")
plt.show()
