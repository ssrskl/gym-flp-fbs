# 输出模型的相关信息

from FbsEnv.utils import FBSUtil
import FbsEnv
import gym
from FbsEnv.envs.FBSModel import FBSModel
import numpy as np
import matplotlib.pyplot as plt

instance_name = "O7-maoyan"
env = gym.make("FbsEnv-v0", instance=instance_name)
permutation = [3,5,7,1,4,6,2]
bay = [0,0,1,0,0,0,1]
fbsModel = FBSModel(permutation, bay)
env.reset(fbs_model = fbsModel)
print(env.fac_x)
print(env.fac_y)
print(env.fac_b)
print(env.fac_h)
sources = np.sum(env.TM, axis=1)
sinks = np.sum(env.TM, axis=0)
permutation = env.fbs_model.permutation
R = np.array(
    ((permutation - np.min(permutation)) / (np.max(permutation) - np.min(permutation)))
    * 255
).astype(np.uint8)
G = np.array(
    ((sources - np.min(sources)) / (np.max(sources) - np.min(sources)))
    * 255
).astype(np.uint8)
B = np.array(
    ((sinks - np.min(sinks)) / (np.max(sinks) - np.min(sinks)))
    * 255
).astype(np.uint8)
print(sources)
print(sinks)
print(permutation)
print(R)
print(G)
print(B)
# 显示图像
image = np.dstack((R, G, B))
print(image)
plt.imshow(image)
plt.show()
for i in range(len(R)):
    print(f"\033[38;2;{R[i]};{G[i]};{B[i]}m████\033[0m 颜色值：RGB({R[i]}, {G[i]}, {B[i]})")
env.render()

data = FBSUtil.constructState(env.fac_x,env.fac_y,env.fac_b,env.fac_h,env.W,env.H,env.fbs_model,env.TM)
plt.imshow(data)
plt.show()

