# 用于测试深拷贝的使用
import numpy as np
import FbsEnv
import gym
from FbsEnv.envs.FBSModel import FBSModel
import logging
import copy

env = gym.make("FbsEnv-v0", instance="O7-maoyan")
permutation = [3, 5, 7, 1, 4, 6, 2]
bay = [0, 0, 1, 0, 0, 0, 1]
fbs_model = FBSModel(permutation, bay)
env.reset(fbs_model = fbs_model)
logging.info(env.fbs_model.permutation)
logging.info(env.fbs_model.bay)
# ----------------赋值一个对象----------------

env_1 = copy.deepcopy(env)

env.fbs_model.permutation = [3, 5, 7, 2, 4, 6, 1]

logging.info(env.fbs_model.permutation)
logging.info(env_1.fbs_model.permutation)
