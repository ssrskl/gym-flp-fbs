# 环境注册测试
import numpy as np
import FbsEnv
import gym
from FbsEnv.envs.FBSModel import FBSModel


env = gym.make("FbsEnv-v0", instance="O7-maoyan")
permutation = [3, 5, 7, 1, 4, 6, 2]
bay = [0, 0, 1, 0, 0, 0, 1]
fbs_model = FBSModel(permutation, bay)

env.reset(fbs_model=fbs_model)
env.render()
