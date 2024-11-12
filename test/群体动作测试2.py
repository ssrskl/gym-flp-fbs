# 群体动作空间测试
import gym
import matplotlib.pyplot as plt
import numpy as np
import logging
from FbsEnv.utils.FBSUtil import FBSUtils
from FbsEnv.envs.FBSModel import FBSModel

instance = "O9-maoyan"

parent1 = gym.make("FbsEnv-v0", instance=instance)
parent1.reset()

parent2 = gym.make("FbsEnv-v0", instance=instance)
parent2.reset()

logging.info("打印亲本信息")
logging.info(parent1.fbs_model.permutation)
logging.info(parent1.fbs_model.bay)
logging.info(parent2.fbs_model.permutation)
logging.info(parent2.fbs_model.bay)

offspring1_permutation, offspring2_permutation = (
    FBSUtils.CrossoverActions.order_crossover(parent1.fbs_model.permutation,
                                              parent2.fbs_model.permutation))

logging.info(offspring1_permutation)
logging.info(offspring2_permutation)
