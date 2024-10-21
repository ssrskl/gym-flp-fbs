# 环境注册测试
import numpy as np
import FbsEnv
import gym
import FbsEnv.utils.FBSUtil as FBSUtil


env = gym.make("FbsEnv-v0", instance="O7-maoyan")
permutation = np.array([3, 5, 7, 1, 4, 6, 2])
bay = np.array([0, 0, 1, 0, 0, 0, 1])
layout = (permutation, bay)
env.reset(layout=layout)
env.render()